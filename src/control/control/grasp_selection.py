"""파지 후보 최종 선택 — 실행할 후보 하나를 로봇 제약과 vision 정보로 고른다.

**왜 여기(control)인가.** 2026-09-07 이전에는 planner의 `validator._best_grasp`가
GraspNet score 최고 후보 하나를 골라 그대로 Pick으로 보냈다. 그 하나가 실행 불가능하면
(개폭 미상, IK 안 풀림) pick 전체가 실패했고, 나머지 후보는 쓰이지 못한 채 버려졌다.
후보를 고르려면 "이 자세로 갈 수 있는가"를 알아야 하는데 그건 로봇에 붙어 있어야만
답할 수 있다(ikin 서비스) — planner는 ROS2를 모르는 별도 HTTP 서비스라 물어볼 수 없다.
그래서 planner는 후보 **목록**을 순위대로 넘기고, 최종 선택은 여기서 한다.

**역할 분담(NFR-03a)은 그대로다.**
- LLM: object_id / bin_id / 순서
- GraspNet: 6DoF 후보 생성
- 여기: 그 후보들 중 **로봇 제약 + vision 정보**로 실행할 하나 선택
좌표·개폭·힘은 어느 단계에서도 LLM이 정하지 않는다.

**이 모듈은 ROS2를 import하지 않는다.** IK 질의와 좌표 계산은 호출부(pick_server)가
콜러블로 넘긴다 — 그래야 로봇 없이 단위 테스트로 판정 규칙 전체를 돌릴 수 있다.

**구현하지 않은 것**(있는 척하지 않기 위해 명시한다):
- 3D collision checker / collision map / 주변 물체와의 충돌 예측. 하지 않는다.
  작업 환경은 사람이 안전하게 배치한다는 전제다.
- singularity 판정. dsr_msgs2에 특이점을 **질의**하는 서비스가 없다
  (`SetSingularityHandling`/`SetSingularHandlingForce`는 회피 동작 설정이지 질의가 아니다).
  J5 값은 참고용으로 기록만 하고 점수에는 넣지 않는다.
- 경로(path) 검사. 검사하는 것은 접근점·파지점 **두 끝점**뿐이다. 그 사이를 movel이
  실제로 어떻게 지나가는지는 보지 않는다.
"""
import math
from dataclasses import dataclass, field

# --- 후보 상태 ---------------------------------------------------------------
# raw는 "아직 검사 안 함"이다. 나머지는 검사 결과이고, selected는 valid 중 최종 1등.
STATUS_RAW = "raw"
STATUS_VALID = "valid"
STATUS_WIDTH_INVALID = "width_invalid"
STATUS_IK_FAILED = "ik_failed"
STATUS_JOINT_LIMIT = "joint_limit"
STATUS_SAFETY_INVALID = "safety_invalid"
STATUS_SELECTED = "selected"

# dsr_motion.IK_* 와 같은 문자열이어야 한다. 이 모듈이 ROS2를 import하지 않으려고
# 값을 복제했으므로, 어긋나면 조용히 오분류된다 — test_grasp_selection이 두 값을 비교한다.
IK_OK = "ok"
IK_JOINT_LIMIT = "joint_limit"


def _has_real_solution(verdict) -> bool:
    """이 판정의 관절값이 의미 있는 수인가.

    ikin은 해가 없어도 관절값을 채워 돌려준다 — 실측 2026-09-07: 팔 밖 목표에
    `conv_posj=[-7203, -2186, ...]`도. 그 값으로 여유나 J5를 계산해 로그에 남기면
    (실제로 `joint_margin=-14960deg`가 찍혔다) 사람이 "관절 한계 문제"로 오해한다.
    왕복 검증을 통과한 판정(ok / joint_limit)의 관절값만 쓴다.
    """
    return bool(verdict.known and (verdict.ok or verdict.status == IK_JOINT_LIMIT))


@dataclass(frozen=True)
class Candidate:
    """검사 대상 후보 하나. sort_msgs/GraspCandidate에서 필요한 것만 옮겨온 형태."""

    candidate_id: str
    rank: int                 # 입력 순서(= GraspNet score 내림차순). 로그·동점 처리에 쓴다
    pose: object              # geometry_msgs/Pose 형태 (position.x…, orientation.x…)
    score: float              # GraspCandidate.score — 계약상 0.0~1.0이고 전략들이 clip한다
    gripper_width_mm: float   # 0이면 "전략이 폭을 못 냈다"(미상)
    strategy: str = ""


@dataclass(frozen=True)
class ObjectContext:
    """랭킹에 쓰는 물체 정보. DetectedObject에서 Pick.action으로 실려 온 값 그대로."""

    object_id: str
    center_mm: tuple | None = None   # (x, y, z) base mm. 없으면 중심 근접도·높이 항을 못 쓴다
    height_mm: float = 0.0           # 0이면 미상 — 지지면 검사와 높이 항을 건너뛴다
    depth_valid_ratio: float = 0.0

    @property
    def support_z_mm(self) -> float | None:
        """물체가 놓인 면의 z. perception이 실측한 값들로만 만든다(DetectedObject.height_mm
        주석 참조: 지지면 z = position_base_mm.z - height_mm). 둘 중 하나라도 없으면 None."""
        if self.center_mm is None or not (self.height_mm > 0.0):
            return None
        return float(self.center_mm[2]) - float(self.height_mm)


@dataclass(frozen=True)
class PickGeometry:
    """후보 하나를 실제로 실행할 때의 좌표. pick_server가 dsr_motion으로 만들어 넘긴다.

    **실행이 쓰는 것과 같은 값이어야 한다.** 검사용으로 따로 계산하면 둘이 갈라져
    "검사는 통과했는데 실행은 못 가는" 후보가 생긴다.
    """

    target_posx: list      # 하강(파지) 목표 TCP [x,y,z,rx,ry,rz]
    approach_posx: list    # 접근 목표 TCP (같은 회전, 접근축으로 물러난 위치)
    approach_axis: list    # base 기준 접근 방향 단위벡터
    pad_reference_mm: list # 그 자세에서 손끝(모델값)이 가는 base 좌표 (x, y, z)


@dataclass
class SelectionParams:
    """판정·점수 파라미터. skill_params.yaml의 `grasp_selection` 블록에서 만든다.

    개폭 관련 값은 pick_server가 이미 쓰던 것을 그대로 받는다 — 검사 기준과 실행 기준이
    다르면 "통과시켜 놓고 실행에서 다르게 동작하는" 후보가 생긴다.
    """

    # --- hard filter 기준 (전부 실행 경로에 이미 있던 값) ---
    max_opening_mm: float = 110.0      # motion.gripper_open_m x 1000
    min_grip_width_mm: float = 5.0     # gripper.min_grip_width_mm
    width_margin_mm: float = 30.0      # gripper.width_margin_mm
    # 지지면보다 이만큼까지 내려가는 것은 허용한다. 0이면 "지지면 아래로 한 mm도 안 된다".
    # 손끝 기준값 자체가 모델(tool.grasp_center_offset_mm)이라 여유를 0으로 두면
    # 모델 오차만으로 후보가 탈락한다.
    support_tolerance_mm: float = 5.0
    # 절대 바닥 높이(TCP z). 작업대 높이를 접촉으로 재본 적이 없어 기본값은 None(검사 안 함)이다.
    # 재고 나면 여기에 넣으면 물체 높이를 모르는 후보까지 걸러진다.
    min_tcp_z_mm: float | None = None
    # 검사할 후보 수 상한. ikin 한 번이 실측 약 31ms라(2026-09-07) 후보당 왕복 2회 =
    # 약 66ms — 10개면 0.7초다. 늘리면 그만큼 pick 시작이 늦어진다.
    max_evaluated: int = 10

    # --- 점수 정규화 기준 ---
    # 이보다 좁은 폭은 "손끝이 스칠 위험"으로 보고 점수를 깎는다. **측정 근거 없는
    # 튜닝값이다** — 실물에서 얇은 물체를 놓치면 올리고, 멀쩡한 후보가 밀리면 내린다.
    comfort_min_width_mm: float = 15.0
    # 중심 근접도의 정규화 길이 하한. 보통은 후보들의 최대 개폭 절반(= 물체 반폭 추정)을
    # 쓰고, 그게 이보다 작으면 이 값을 쓴다 — hand-eye 잔차가 mm 단위라 너무 작은
    # 기준으로 나누면 잡음이 순위를 뒤집는다.
    center_scale_floor_mm: float = 20.0
    # 관절 여유가 이 이상이면 만점. M0609는 J3만 ±150도이고 나머지는 ±360도라
    # 실질적으로 J3 여유를 보는 항이 된다.
    joint_margin_ref_deg: float = 30.0

    # --- 가중치 (합이 1이 아니어도 된다 — 사용 가능한 항만 모아 정규화한다) ---
    weights: dict = field(default_factory=lambda: {
        "grasp_score": 0.35,       # GraspNet 품질. 여러 판단 요소 중 하나일 뿐이다
        "width_fit": 0.20,         # RG2 개폭 적합성
        "center_proximity": 0.20,  # 파지점이 물체 중심에 가까운가
        "height_fit": 0.10,        # 물체 높이의 중간쯤을 무는가 (윗모서리만 물면 미끄러진다)
        "joint_margin": 0.10,      # 관절 한계까지 여유
        "depth_quality": 0.05,     # depth 신뢰도. 물체 단위 값이라 같은 물체 안에서는 동점이다
    })


@dataclass
class Evaluation:
    """후보 하나의 검사 결과 전체. 로그·보고가 이 객체 하나만 보면 되도록 다 담는다."""

    candidate: Candidate
    status: str = STATUS_RAW
    rejection_reason: str = ""
    width_valid: bool | None = None
    approach_ik: bool | None = None      # None = 확인 못 함(ikin 무응답)
    grasp_ik: bool | None = None
    joint_ok: bool | None = None
    joint_margin_deg: float | None = None
    wrist_j5_deg: float | None = None    # 참고 기록용. 특이점 판정은 미구현이다
    safety_ok: bool | None = None
    geometry: PickGeometry | None = None
    terms: dict = field(default_factory=dict)
    total_score: float | None = None

    @property
    def rejected_before_motion(self) -> bool:
        """실행 전에 탈락한 후보인가. 이 상태들은 로봇을 전혀 움직이지 않았다는 뜻이다."""
        return self.status in (STATUS_WIDTH_INVALID, STATUS_IK_FAILED,
                               STATUS_JOINT_LIMIT, STATUS_SAFETY_INVALID)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


# --- 점수 항 ------------------------------------------------------------------
# 전부 [0,1]로 정규화해 돌려준다(클수록 좋다). 판단할 정보가 없으면 None —
# combined_score가 그 항을 빼고 나머지 가중치로 다시 정규화한다. 정보가 없는 후보에게
# 0점을 주면 "모른다"가 "나쁘다"로 둔갑한다.

def score_grasp_quality(candidate: Candidate) -> float:
    """GraspCandidate.score. 계약상 이미 0~1이고 두 전략 모두 clip해서 낸다
    (graspnet_baseline.py / heuristic_pca.py) — 그래서 추가 정규화를 하지 않는다."""
    return _clamp01(candidate.score)


def score_width_fit(width_mm: float, params: SelectionParams) -> float:
    """RG2로 물기 좋은 폭인가.

    양쪽 끝이 나쁘다는 것은 실행 코드가 이미 알고 있는 사실이다:
    - 너무 넓으면(`폭 + width_margin_mm > 최대개폭`) pick_server가 목표 개폭을 못 만들고
      최대로 열어버린다 — 접근 중 옆 물체를 건드리기 쉽고 닫는 시간도 길어진다.
    - 너무 좁으면 손끝이 물체를 스치고 지나갈 위험이 커진다(grasp_params.yaml
      gripper.min_width_mm 주석과 같은 이유).
    """
    width = float(width_mm)
    comfort_hi = params.max_opening_mm - params.width_margin_mm
    comfort_lo = params.comfort_min_width_mm
    if comfort_hi <= comfort_lo:
        # 설정이 뒤집힌 경우(여유가 최대개폭에 가깝다). 폭으로는 우열을 못 가린다.
        return 1.0
    if width <= comfort_lo:
        span = comfort_lo - params.min_grip_width_mm
        return _clamp01((width - params.min_grip_width_mm) / span) if span > 0 else 0.0
    if width <= comfort_hi:
        return 1.0
    span = params.max_opening_mm - comfort_hi
    return _clamp01((params.max_opening_mm - width) / span) if span > 0 else 0.0


def score_center_proximity(candidate: Candidate, obj: ObjectContext,
                           scale_mm: float) -> float | None:
    """파지점이 물체 중심(xy)에서 얼마나 벗어났는가. 가까울수록 1.

    z는 보지 않는다 — 위아래 위치는 `score_height_fit`이 따로 본다. 중심에서 먼 파지는
    물체를 돌리거나 미끄러뜨리기 쉽다.
    """
    if obj.center_mm is None or scale_mm <= 0.0:
        return None
    position = candidate.pose.position
    distance = math.hypot(float(position.x) - float(obj.center_mm[0]),
                          float(position.y) - float(obj.center_mm[1]))
    return 1.0 - _clamp01(distance / scale_mm)


def center_scale_mm(candidates, params: SelectionParams) -> float:
    """중심 근접도를 나눌 길이(mm).

    world_state에 물체의 가로 크기가 없다(height_mm뿐). 대신 후보들이 낸 개폭의 최댓값
    절반을 **물체 반폭의 추정치**로 쓴다 — 그래야 작은 물체와 큰 물체에서 같은 mm 차이가
    같은 감점이 되지 않는다. 개폭을 아무도 못 냈으면 하한값을 그대로 쓴다.
    """
    widths = [float(c.gripper_width_mm) for c in candidates if c.gripper_width_mm > 0.0]
    estimate = 0.5 * max(widths) if widths else 0.0
    return max(estimate, params.center_scale_floor_mm)


def score_height_fit(candidate: Candidate, obj: ObjectContext) -> float | None:
    """물체 높이의 어디쯤을 무는가. 중간(0.5)이 1점, 윗면/바닥이 0점.

    윗모서리만 물면 들어올릴 때 미끄러지기 쉽고, 지지면 높이를 물면 작업대를 긁는다.
    perception이 depth로 실측한 값(윗면 z, height_mm)만 쓴다 — 둘 중 하나라도 미상이면 None.
    """
    support_z = obj.support_z_mm
    if support_z is None:
        return None
    relative = (float(candidate.pose.position.z) - support_z) / float(obj.height_mm)
    return 1.0 - 2.0 * abs(_clamp01(relative) - 0.5)


def score_joint_margin(margin_deg: float | None, params: SelectionParams) -> float | None:
    """관절 한계까지 남은 여유(도)를 0~1로. 확인 못 했으면 None."""
    if margin_deg is None:
        return None
    if params.joint_margin_ref_deg <= 0.0:
        return 1.0
    return _clamp01(margin_deg / params.joint_margin_ref_deg)


def score_depth_quality(obj: ObjectContext) -> float | None:
    """DetectedObject.depth_valid_ratio. 낮으면 좌표 자체의 신뢰도가 낮다.

    **물체 단위 값이라 같은 물체의 후보들 사이에서는 순위를 바꾸지 못한다.** 그래도 남겨
    두는 이유는 총점을 물체 사이에 비교할 수 있게 하고, 로그에 신뢰도가 남기 때문이다.
    """
    if obj.depth_valid_ratio <= 0.0:
        return None
    return _clamp01(obj.depth_valid_ratio)


def combined_score(terms: dict, weights: dict) -> float:
    """항들을 가중 평균한다. 값이 None인 항은 빼고 **남은 가중치로 다시 정규화**한다.

    빼지 않고 0으로 채우면 "정보가 없는 후보"가 "나쁜 후보"로 바뀐다 — 예를 들어
    height_mm을 못 잰 물체의 모든 후보가 일제히 감점돼, 후보 사이 우열과 무관하게
    총점만 낮아진다.
    """
    present = {name: value for name, value in terms.items() if value is not None}
    total_weight = sum(float(weights.get(name, 0.0)) for name in present)
    if total_weight <= 0.0:
        return 0.0
    return sum(float(weights.get(name, 0.0)) * float(value)
               for name, value in present.items()) / total_weight


# --- hard filter ---------------------------------------------------------------

def check_width(candidate: Candidate, params: SelectionParams) -> str:
    """개폭이 실행 가능한 값인지. 통과면 빈 문자열, 아니면 거부 사유.

    **width_mm == 0(미상)을 반드시 막아야 한다.** 그 경우 pick_server는 완전 닫기("c")로
    닫는데, 그러면 손가락끼리 부딪혀 힘 한계에 도달해 **빈 그리퍼에서도 Grip detected가
    켜진다** — 2026-09-07 실물에서 아무것도 안 잡고 `Grip detected=True (width=0.0mm)`로
    성공 판정이 났다. 그 판정으로 place_into가 이어지면 빈 손으로 목적지까지 간다.
    폭을 못 믿는 후보는 쓰지 않고 다음 후보로 넘어간다.
    """
    width = float(candidate.gripper_width_mm)
    if not math.isfinite(width) or width <= 0.0:
        return ("개폭 미상(width_mm=0) — 완전 닫기로 떨어져 빈 그리퍼도 파지 성공으로 "
                "오판될 수 있다")
    if width < params.min_grip_width_mm:
        # 닫기 목표는 max(폭 x grip_close_ratio, min_grip_width_mm)이라, 폭이 이 값보다
        # 얇으면 그리퍼가 물체에 닿기 전에 멈춘다 — 물리적으로 물 수 없는 후보다.
        return (f"개폭 {width:.1f}mm가 최소 파지폭 {params.min_grip_width_mm:.1f}mm 미만 — "
                "그리퍼가 물체에 닿기 전에 멈춘다")
    if width > params.max_opening_mm:
        return f"개폭 {width:.1f}mm가 RG2 최대 개폭 {params.max_opening_mm:.1f}mm 초과"
    return ""


def check_min_safety(geometry: PickGeometry, obj: ObjectContext,
                     params: SelectionParams) -> str:
    """최소 안전 검사. 통과면 빈 문자열, 아니면 거부 사유.

    **여기서 하는 것은 두 가지뿐이다.**
    1. 손끝(모델값)이 물체의 지지면을 명백히 뚫는가 — 지지면은 perception이 실측한
       `윗면 z - height_mm`이다(DetectedObject.height_mm).
    2. 명령할 TCP가 설정된 절대 바닥(min_tcp_z_mm) 아래인가 — 기본값은 미설정이라 안 본다.

    **하지 않는 것**: 주변 물체와의 충돌, 그리퍼 몸통/팔의 충돌, 경로 상의 간섭.
    작업 환경은 사람이 안전하게 배치한다는 전제이고, 이 단계에서 collision checker를
    도입하지 않기로 했다. 강한 필터로 후보를 과하게 죽이지 않는 것이 여기 목적이다.
    """
    support_z = obj.support_z_mm
    if support_z is not None:
        pad_z = float(geometry.pad_reference_mm[2])
        floor = support_z - params.support_tolerance_mm
        if pad_z < floor:
            return (f"손끝 예상 z {pad_z:.1f}mm가 지지면 {support_z:.1f}mm보다 "
                    f"{support_z - pad_z:.1f}mm 아래 (허용 {params.support_tolerance_mm:.1f}mm)")
    if params.min_tcp_z_mm is not None:
        for name, posx in (("파지", geometry.target_posx), ("접근", geometry.approach_posx)):
            if float(posx[2]) < params.min_tcp_z_mm:
                return (f"{name} TCP z {float(posx[2]):.1f}mm가 작업대 하한 "
                        f"{params.min_tcp_z_mm:.1f}mm 아래")
    return ""


# --- 평가 · 선택 ----------------------------------------------------------------

def evaluate_candidates(candidates, obj: ObjectContext, params: SelectionParams,
                        geometry_of, ik_verdict_of) -> list:
    """후보를 하나씩 검사해 Evaluation 목록을 돌려준다. 로봇은 움직이지 않는다.

    `geometry_of(pose) -> PickGeometry`, `ik_verdict_of(posx) -> verdict`는 호출부가 준다
    (`verdict`는 `.ok` / `.known` / `.status` / `.joint_margin_deg` / `.posj`를 갖는
    dsr_motion.IkVerdict 형태). 검사 순서는 **개폭 → IK → 관절 → 최소 안전**이다.

    **IK를 확인하지 못한 경우(ikin 무응답)는 탈락시키지 않는다.** ikin이 죽었다는 이유로
    모든 후보를 버리면 pick이 통째로 멈춘다 — 그건 이 작업이 없애려는 실패 모드
    (후보 하나 때문에 전체가 실패)와 같은 종류다. 대신 `approach_ik`/`grasp_ik`가
    None으로 남아 로그에 "확인 못 함"으로 찍힌다.
    """
    scale_mm = center_scale_mm(candidates, params)
    evaluations = []
    for candidate in candidates[:max(1, int(params.max_evaluated))]:
        evaluation = Evaluation(candidate=candidate)
        evaluations.append(evaluation)

        reason = check_width(candidate, params)
        evaluation.width_valid = not reason
        if reason:
            evaluation.status = STATUS_WIDTH_INVALID
            evaluation.rejection_reason = reason
            continue

        geometry = geometry_of(candidate.pose)
        evaluation.geometry = geometry

        # 접근점부터 본다 — 실행에서도 거기가 첫 이동이다.
        approach = ik_verdict_of(geometry.approach_posx)
        grasp = ik_verdict_of(geometry.target_posx)
        evaluation.approach_ik = approach.ok if approach.known else None
        evaluation.grasp_ik = grasp.ok if grasp.known else None
        margins = [v.joint_margin_deg for v in (approach, grasp)
                   if _has_real_solution(v) and v.joint_margin_deg is not None]
        evaluation.joint_margin_deg = min(margins) if margins else None
        if _has_real_solution(grasp) and grasp.posj is not None and len(grasp.posj) >= 5:
            # 참고 기록. 6축 로봇의 손목 특이점은 J5=0 근처지만, **판정에는 쓰지 않는다** —
            # 드라이버에 특이점을 질의하는 서비스가 없어 기준값을 검증할 방법이 없다.
            evaluation.wrist_j5_deg = float(grasp.posj[4])

        failed = [(name, verdict) for name, verdict in (("접근", approach), ("파지", grasp))
                  if verdict.known and not verdict.ok]
        if failed:
            name, verdict = failed[0]
            if verdict.status == IK_JOINT_LIMIT:
                evaluation.joint_ok = False
                evaluation.status = STATUS_JOINT_LIMIT
                evaluation.rejection_reason = (
                    f"{name} 자세의 관절해가 한계를 "
                    f"{abs(verdict.joint_margin_deg or 0.0):.1f}도 넘는다")
            else:
                evaluation.status = STATUS_IK_FAILED
                evaluation.rejection_reason = f"{name} 자세의 IK 해가 없다"
            continue
        evaluation.joint_ok = True if (approach.known and grasp.known) else None

        reason = check_min_safety(geometry, obj, params)
        evaluation.safety_ok = not reason
        if reason:
            evaluation.status = STATUS_SAFETY_INVALID
            evaluation.rejection_reason = reason
            continue

        evaluation.status = STATUS_VALID
        evaluation.terms = {
            "grasp_score": score_grasp_quality(candidate),
            "width_fit": score_width_fit(candidate.gripper_width_mm, params),
            "center_proximity": score_center_proximity(candidate, obj, scale_mm),
            "height_fit": score_height_fit(candidate, obj),
            "joint_margin": score_joint_margin(evaluation.joint_margin_deg, params),
            "depth_quality": score_depth_quality(obj),
        }
        evaluation.total_score = combined_score(evaluation.terms, params.weights)

    return evaluations


def select(evaluations) -> Evaluation | None:
    """valid 후보 중 총점이 가장 높은 하나를 selected로 표시하고 돌려준다. 없으면 None.

    동점이면 입력 순위(GraspNet score 순)가 앞선 쪽 — 순서가 실행마다 흔들리지 않아야
    같은 장면에서 같은 후보가 나온다.
    """
    valid = [e for e in evaluations if e.status == STATUS_VALID]
    if not valid:
        return None
    best = min(valid, key=lambda e: (-(e.total_score or 0.0), e.candidate.rank))
    best.status = STATUS_SELECTED
    return best


def counts(evaluations) -> dict:
    """상태별 후보 수. 보고·로그가 그대로 쓴다."""
    tally = {status: 0 for status in (STATUS_RAW, STATUS_VALID, STATUS_WIDTH_INVALID,
                                      STATUS_IK_FAILED, STATUS_JOINT_LIMIT,
                                      STATUS_SAFETY_INVALID, STATUS_SELECTED)}
    for evaluation in evaluations:
        tally[evaluation.status] = tally.get(evaluation.status, 0) + 1
    return tally


def _flag(value: bool | None) -> str:
    return "unknown" if value is None else ("true" if value else "false")


def log_line(evaluation: Evaluation) -> str:
    """후보 하나의 판정을 한 줄로. 웹에는 이 숫자들을 내보내지 않는다 — 로그 전용이다."""
    candidate = evaluation.candidate
    parts = [f"candidate={candidate.candidate_id or f'#{candidate.rank}'}",
             f"score={candidate.score:.3f}",
             f"width={candidate.gripper_width_mm:.1f}",
             f"width_valid={_flag(evaluation.width_valid)}",
             f"approach_ik={_flag(evaluation.approach_ik)}",
             f"grasp_ik={_flag(evaluation.grasp_ik)}",
             f"joint_limit={_flag(None if evaluation.joint_ok is None else not evaluation.joint_ok)}"]
    if evaluation.joint_margin_deg is not None:
        parts.append(f"joint_margin={evaluation.joint_margin_deg:.1f}deg")
    if evaluation.wrist_j5_deg is not None:
        # 특이점 판정은 미구현 — 사람이 로그에서 보고 판단하라는 뜻으로만 남긴다.
        parts.append(f"j5={evaluation.wrist_j5_deg:.1f}deg")
    parts.append(f"safety={_flag(evaluation.safety_ok)}")
    if evaluation.total_score is not None:
        terms = " ".join(f"{name}={value:.2f}" for name, value in evaluation.terms.items()
                         if value is not None)
        parts.append(f"total={evaluation.total_score:.3f} [{terms}]")
    parts.append(f"status={evaluation.status}")
    if evaluation.rejection_reason:
        parts.append(f"reason={evaluation.rejection_reason}")
    return " ".join(parts)

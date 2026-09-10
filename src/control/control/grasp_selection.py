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
- 경로(path) 검사. **이 모듈이** 보는 것은 접근점·파지점 두 끝점뿐이다. 그 사이를
  movel이 어떻게 지나가는지는 STEP 2(2026-09-10)에서 `motion_feasibility.py`로 옮겼고,
  pick_server가 랭킹 순서대로 후보를 꺼내며 그것으로 거른다 — 여기의 점수에는 들어오지
  않는다(점수는 "이 자세가 좋은가", 경로 검사는 "거기까지 갈 수 있는가"로 나눠 둔다).
"""
import math
from dataclasses import dataclass, field

# --- 후보 상태 ---------------------------------------------------------------
# raw는 "아직 검사 안 함"이다. 나머지는 검사 결과이고, selected는 valid 중 최종 1등.
#
# STATUS_GEOMETRY_INVALID(2026-09-08, 2.6차)는 "물체와 안 맞는 자세"다. 주된 판정은
# grasp 쪽에서 포인트클라우드로 한다(graspnet_baseline: cloud_mismatch /
# no_grip_material — control에는 클라우드가 없다). 여기서는 클라우드 없이도 확실히
# 알 수 있는 한 가지, **파지점이 물체 윗면보다 위에 있는 경우**만 본다.
STATUS_RAW = "raw"
STATUS_VALID = "valid"
STATUS_WIDTH_INVALID = "width_invalid"
STATUS_IK_FAILED = "ik_failed"
STATUS_JOINT_LIMIT = "joint_limit"
STATUS_SAFETY_INVALID = "safety_invalid"
STATUS_GEOMETRY_INVALID = "geometry_invalid"
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
    # 손가락 사이 실측 재료 지지도 0~1 (grasp의 되잡기가 계산, GraspCandidate.msg).
    # **-1은 미상**이고 0.0(지지 없음)과 뜻이 다르다 — score_contact_support가 -1을
    # None으로 바꿔 랭킹에서 빼고, 0.0은 그대로 낮은 점수로 쓴다.
    contact_support_score: float = -1.0


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

    # --- 접근 적합도 (2026-09-08, 2.5차) ---
    # 접근각은 이제 grasp 단계에서 hard filter로 자르지 않는다(grasp_params.yaml
    # approach_angle_hard_max_deg 주석 참조) — "불가"가 아니라 "덜 좋다"이므로 여기서
    # 점수로 다룬다. 세 값은 그 감점 곡선의 꼭짓점이고, 예전 hard filter의 숫자를
    # 그대로 물려받아 추적이 되게 했다.
    #   prefer(15) : 수직에 가깝다 — 만점. 예전 approach_angle_threshold_deg와 같은 값
    #   soft(30)   : 여기서 0.5점. 예전 approach_angle_max_deg(= 잘리던 선)와 같은 값이라,
    #                "예전에 잘리던 각도 = 절반짜리 후보"로 읽으면 된다
    #   zero(75)   : 여기서 0점. grasp의 hard 상한과 같은 값 — 그 위는 애초에 안 온다
    approach_angle_prefer_deg: float = 15.0
    approach_angle_soft_max_deg: float = 30.0
    approach_angle_zero_deg: float = 75.0
    # 현재 TCP에서 접근 지점까지의 이동량/회전량 정규화 기준. 이만큼 움직여야 하면 0점.
    # **판정이 아니라 선호도다** — 도달 가능성은 IK가 이미 봤다. 같은 값이면 지금 자세에서
    # 덜 움직여도 되는 후보를 고르자는 뜻이고, 그만큼 사이클이 짧고 경로가 단순하다.
    approach_travel_ref_mm: float = 400.0
    approach_rotation_ref_deg: float = 120.0

    # --- robot comfort (STEP 1, 2026-09-10) ---
    # 접근 IK 해와 파지 IK 해 사이의 관절 이동량(도, 성분별 최대)이 이 값 이상이면 0점.
    # 두 자세는 접근축을 따라 몇 cm 떨어져 있을 뿐이라 관절해도 이어져야 정상인데, 크게
    # 갈리면 그 근방에서 관절 공간이 불연속이라는 뜻이다. **판정이 아니라 선호도다** —
    # 도달 가능성은 IK가 이미 봤다. 근거 없는 초기값이고, 추가 서비스 호출은 하지 않는다
    # (이미 받아 둔 IK 결과의 posj만 쓴다).
    comfort_joint_delta_ref_deg: float = 60.0

    # --- 가중치 (합이 1이 아니어도 된다 — 사용 가능한 항만 모아 정규화한다) ---
    # **enhanced 랭킹의 가중치다** (STEP 1). 기존 release 랭킹은 아래 LEGACY_WEIGHTS로
    # 회귀 비교용 legacy 값도 계산하지만 production 실행은 pick_server가 enhanced로 고정한다.
    # enhanced에는 approach_angle/approach_travel을 **중복해서 넣지 않는다** — 접근각
    # hard 상한(45도)이 grasp에 그대로 있고, 이 STEP의 관심사는 contact/robot_comfort다.
    weights: dict = field(default_factory=lambda: {
        "grasp_score": 0.30,       # GraspNet 품질. 여러 판단 요소 중 하나일 뿐이다
        "width_fit": 0.15,         # RG2 개폭 적합성
        "center_proximity": 0.10,  # 파지점이 물체 중심에 가까운가 (보조 heuristic)
        "height_fit": 0.10,        # 물체 높이의 중간쯤을 무는가 (윗모서리만 물면 미끄러진다)
        "depth_quality": 0.05,     # depth 신뢰도. 물체 단위 값이라 같은 물체 안에서는 동점이다
        "joint_margin": 0.10,      # 관절 한계까지 여유
        "contact_support": 0.15,   # 손가락 사이에 실제로 물릴 재료가 있는가 (STEP 1)
        "robot_comfort": 0.05,     # 접근→파지 관절 이동이 자연스럽게 이어지는가 (STEP 1)
    })


# 기존 release(2026-09-09까지)의 랭킹을 회귀 테스트용으로 보존한 스냅샷이다.
LEGACY_WEIGHTS = {
    "grasp_score": 0.30,
    "width_fit": 0.15,
    "center_proximity": 0.15,
    "height_fit": 0.10,
    "joint_margin": 0.10,
    "depth_quality": 0.05,
    "approach_angle": 0.10,
    "approach_travel": 0.05,
}


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
    # 접근 적합도의 원시값. 점수로 환산하기 전 숫자를 그대로 남겨야 로그에서 "왜 이
    # 후보가 밀렸나"를 각도/이동량으로 바로 읽을 수 있다.
    approach_angle_deg: float | None = None
    approach_travel_mm: float | None = None
    approach_rotation_deg: float | None = None
    safety_ok: bool | None = None
    geometry: PickGeometry | None = None
    # 접근 IK 해와 파지 IK 해 사이 관절 이동량(도). robot_comfort의 원시값 — 추가 서비스
    # 호출 없이 이미 받은 두 IK 결과의 posj 차이로 낸다. 둘 중 하나라도 없으면 None.
    joint_delta_deg: float | None = None
    # enhanced production 랭킹(신규 항 포함).
    terms: dict = field(default_factory=dict)
    total_score: float | None = None
    # legacy 회귀 비교 랭킹(기존 release 그대로, LEGACY_WEIGHTS).
    legacy_terms: dict = field(default_factory=dict)
    legacy_score: float | None = None

    @property
    def rejected_before_motion(self) -> bool:
        """실행 전에 탈락한 후보인가. 이 상태들은 로봇을 전혀 움직이지 않았다는 뜻이다."""
        return self.status in (STATUS_WIDTH_INVALID, STATUS_IK_FAILED,
                               STATUS_JOINT_LIMIT, STATUS_SAFETY_INVALID,
                               STATUS_GEOMETRY_INVALID)


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


def approach_angle_deg(approach_axis) -> float | None:
    """base 접근축이 **수직 아래(0,0,-1)** 에서 몇 도 벌어져 있는지. 0도가 바로 위에서 내려오는 것.

    **`abs()`로 재지 않는다** — 그러면 아래에서 위로 찌르는 자세(작업대를 뚫는 방향)가
    0도로 나온다. 내적을 그대로 쓰므로 90도 초과는 "작업대 아래에서 올라옴"을 뜻한다.

    grasp이 낸 `approach_angle_deg`를 그대로 쓰지 않고 여기서 다시 재는 이유: 이 축은
    **실행이 실제로 쓰는 접근축**(dsr_motion.approach_axis_from_pose → PickGeometry)이라,
    검사와 실행이 갈라질 여지가 없다. 두 값은 같아야 하고 다르면 그게 버그다.
    """
    if approach_axis is None or len(approach_axis) < 3:
        return None
    x, y, z = (float(v) for v in approach_axis[:3])
    norm = math.sqrt(x * x + y * y + z * z)
    if norm < 1e-9:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, -z / norm))))


def score_approach_angle(angle_deg: float | None, params: SelectionParams) -> float | None:
    """접근각을 0~1로. 모르면 None.

    prefer까지 1.0, soft에서 0.5, zero에서 0.0인 꺾은선이다. 왜 한 번 꺾는가 —
    직선 하나로 15도에서 75도까지 떨어뜨리면 예전에 "통과/탈락"을 갈랐던 30도 부근의
    차이가 뭉개진다. 30도를 0.5로 못박아 두면 그 선의 의미가 점수에 남는다.
    """
    if angle_deg is None:
        return None
    prefer = float(params.approach_angle_prefer_deg)
    soft = float(params.approach_angle_soft_max_deg)
    zero = float(params.approach_angle_zero_deg)
    if angle_deg <= prefer:
        return 1.0
    if angle_deg >= zero:
        return 0.0
    if soft > prefer and angle_deg <= soft:
        return 1.0 - 0.5 * (angle_deg - prefer) / (soft - prefer)
    if zero > soft:
        return 0.5 * (1.0 - (angle_deg - soft) / (zero - soft))
    return 0.0


def score_approach_travel(travel_mm: float | None, rotation_deg: float | None,
                          params: SelectionParams) -> float | None:
    """지금 자세에서 접근 지점까지의 이동량·회전량을 0~1로. 둘 다 모르면 None.

    **도달 가능성 판정이 아니다** — 그건 IK가 이미 봤다. 여기는 "같은 값이면 덜 움직이는
    쪽"이라는 선호도이고, 둘 중 아는 것만 평균한다(현재 자세를 못 읽었으면 이 항은 없다).
    """
    parts = []
    if travel_mm is not None and params.approach_travel_ref_mm > 0.0:
        parts.append(1.0 - _clamp01(travel_mm / params.approach_travel_ref_mm))
    if rotation_deg is not None and params.approach_rotation_ref_deg > 0.0:
        parts.append(1.0 - _clamp01(rotation_deg / params.approach_rotation_ref_deg))
    if not parts:
        return None
    return _clamp01(sum(parts) / len(parts))


def score_contact_support(candidate: Candidate) -> float | None:
    """GraspCandidate.contact_support_score — 손가락 사이에 실제로 물릴 재료가 있는가.

    grasp의 되잡기가 실측 클라우드로 계산해 실어 보낸 값이다(별도 클라우드 패스 없음).
    **"물체 중심에 가까운가"의 대체가 아니라 보완이다** — 물체 끝을 잡아도 양쪽 손가락에
    재료가 충분하면 높게 나온다. 명백히 재료가 없는 후보(cloud_mismatch/no_grip_material)는
    grasp 단계에서 이미 hard reject되므로 여기서는 soft 점수로만 쓴다.

    -1(미상: 되잡기를 못 한 전략/경로)이면 None — 정보 없음이 감점이 되면 안 된다.
    """
    value = float(candidate.contact_support_score)
    if value < 0.0:
        return None
    return _clamp01(value)


def joint_delta_deg(approach_verdict, grasp_verdict) -> float | None:
    """접근 IK 해 → 파지 IK 해의 관절 이동량(도, 성분별 최대). 둘 중 하나라도 없으면 None.

    **추가 서비스 호출을 만들지 않는다** — 후보 검사에서 이미 받아 둔 두 IkVerdict의
    posj만 뺀다. 두 자세는 접근축을 따라 몇 cm 떨어져 있을 뿐이라 관절해도 이어지는 것이
    정상이고, 크게 갈리면 그 근방에서 관절 공간이 불연속이라는 뜻이다.
    """
    approach_posj = getattr(approach_verdict, "posj", None)
    grasp_posj = getattr(grasp_verdict, "posj", None)
    if not approach_posj or not grasp_posj:
        return None
    return max(abs(float(a) - float(g)) for a, g in zip(approach_posj, grasp_posj))


def score_robot_comfort(delta_deg: float | None, params: SelectionParams) -> float | None:
    """관절 이동이 자연스럽게 이어지는 후보를 선호한다. 확인 못 했으면 None.

    **hard reject가 아니다** — IK가 풀리는지는 이미 앞에서 봤고, 여기는 "둘 다 되면 어느
    쪽이 편한가"만 본다. J5 단독 특이점 판정·다중 solution space·경로 계획은 이 STEP에서
    하지 않는다(STEP 2의 motion feasibility에서 하나로 다룬다).

    joint_margin(관절 한계까지 여유)과는 다른 것을 본다 — 그쪽은 "한계에 얼마나 가깝나",
    이쪽은 "접근에서 파지로 가며 얼마나 크게 움직이나"다.
    """
    if delta_deg is None:
        return None
    reference = float(params.comfort_joint_delta_ref_deg)
    if reference <= 0.0:
        return 1.0
    return _clamp01(1.0 - float(delta_deg) / reference)


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


def check_grasp_geometry(candidate: Candidate, obj: ObjectContext,
                        params: SelectionParams) -> str:
    """파지점이 물체 안에 있는가. 통과면 빈 문자열, 아니면 거부 사유.

    **여기서 보는 것은 하나뿐이다: 파지점이 물체 윗면보다 위에 있는가.** 그러면 손가락이
    물체 위 허공에서 닫힌다. `check_min_safety`가 지지면(아래쪽)을 보므로 이 함수는 그
    반대쪽을 막아 물체의 두께 구간을 위아래로 닫는다.

    **가로 방향(옆 허공)은 여기서 못 본다** — 그 판정에는 포인트클라우드가 필요하고
    그건 grasp에 있다(graspnet_baseline._refine_on_cloud: cloud_mismatch). 같은 검사를
    control에서 근사로 흉내내면 두 곳의 기준이 갈라지므로 하지 않는다.

    여유는 `support_tolerance_mm`을 그대로 쓴다 — 손끝 위치가 모델값이라(
    tool.grasp_center_offset_mm) 아래쪽에 주는 것과 같은 크기의 모델 오차 여유를
    위쪽에도 준다. 새 상수를 만들면 근거 없는 값이 하나 더 생긴다.
    """
    if obj.center_mm is None or not (obj.height_mm > 0.0):
        return ""            # 물체 두께를 모르면 판단하지 않는다
    top_z = float(obj.center_mm[2])
    grasp_z = float(candidate.pose.position.z)
    ceiling = top_z + params.support_tolerance_mm
    if grasp_z > ceiling:
        return (f"파지점 z {grasp_z:.1f}mm가 물체 윗면 {top_z:.1f}mm보다 "
                f"{grasp_z - top_z:.1f}mm 위 (허용 {params.support_tolerance_mm:.1f}mm) — "
                "손가락이 물체 위 허공에서 닫힌다")
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
                        geometry_of, ik_verdict_of,
                        current_posx=None, rotation_diff=None) -> list:
    """후보를 하나씩 검사해 Evaluation 목록을 돌려준다. 로봇은 움직이지 않는다.

    `geometry_of(pose) -> PickGeometry`, `ik_verdict_of(posx) -> verdict`는 호출부가 준다
    (`verdict`는 `.ok` / `.known` / `.status` / `.joint_margin_deg` / `.posj`를 갖는
    dsr_motion.IkVerdict 형태). 검사 순서는 **개폭 → IK → 관절 → 최소 안전**이다.

    `current_posx`(지금 TCP [x,y,z,rx,ry,rz])와 `rotation_diff(a_zyz, b_zyz) -> deg`를
    주면 접근 적합도에 이동량·회전량이 들어간다. 둘 다 선택 인자다 — 현재 자세를 못
    읽는 상황(movel 직후 aux_control 무응답 구간)에서도 나머지 판정은 그대로 돌아야 한다.
    회전 비교를 콜러블로 받는 이유: ZYZ는 ry가 180도 근처면 같은 방향이 다른 (rx,rz)로
    나오므로 성분 차로 못 재는데, 그 계산(dsr_motion.rotation_diff_deg)을 여기로 들이면
    이 모듈이 ROS2/perception_common에 묶인다.

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

        # 물체 밖 자세는 IK를 묻기 전에 거른다 — ikin 왕복이 후보당 약 66ms라
        # 어차피 못 쓸 후보에 그 시간을 쓸 이유가 없다.
        reason = check_grasp_geometry(candidate, obj, params)
        if reason:
            evaluation.status = STATUS_GEOMETRY_INVALID
            evaluation.rejection_reason = reason
            continue

        geometry = geometry_of(candidate.pose)
        evaluation.geometry = geometry
        # **탈락한 후보에도 각도를 남긴다.** IK로 떨어진 후보가 몇 도였는지가 보여야
        # hard 상한(grasp_params.yaml)을 실측으로 조일지 풀지 판단할 수 있다.
        evaluation.approach_angle_deg = approach_angle_deg(geometry.approach_axis)
        if current_posx is not None and len(current_posx) >= 3:
            evaluation.approach_travel_mm = math.dist(
                [float(v) for v in current_posx[:3]],
                [float(v) for v in geometry.approach_posx[:3]])
            if rotation_diff is not None and len(current_posx) >= 6:
                evaluation.approach_rotation_deg = float(
                    rotation_diff(current_posx[3:6], geometry.approach_posx[3:6]))

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
        # 항 값은 **한 번만** 계산하고, legacy/enhanced는 그것을 다른 가중치로 합치기만
        # 한다 — IK도 기하 검사도 두 번 돌리지 않는다(STEP 1 §14).
        shared = {
            "grasp_score": score_grasp_quality(candidate),
            "width_fit": score_width_fit(candidate.gripper_width_mm, params),
            "center_proximity": score_center_proximity(candidate, obj, scale_mm),
            "height_fit": score_height_fit(candidate, obj),
            "joint_margin": score_joint_margin(evaluation.joint_margin_deg, params),
            "depth_quality": score_depth_quality(obj),
        }
        evaluation.legacy_terms = {
            **shared,
            "approach_angle": score_approach_angle(evaluation.approach_angle_deg, params),
            "approach_travel": score_approach_travel(
                evaluation.approach_travel_mm, evaluation.approach_rotation_deg, params),
        }
        evaluation.legacy_score = combined_score(evaluation.legacy_terms, LEGACY_WEIGHTS)
        evaluation.joint_delta_deg = joint_delta_deg(approach, grasp)
        evaluation.terms = {
            **shared,
            "contact_support": score_contact_support(candidate),
            "robot_comfort": score_robot_comfort(evaluation.joint_delta_deg, params),
        }
        evaluation.total_score = combined_score(evaluation.terms, params.weights)

    return evaluations


def select(evaluations, mode: str = "legacy"):
    """valid 후보 중 하나를 selected로 표시한다.

    반환: `(실제로 고른 Evaluation 또는 None, legacy_best, enhanced_best)`.

    - `mode="legacy"`(기본) — 기존 release 랭킹(LEGACY_WEIGHTS)으로만 고른다. mode를 주지
      않는 호출부·테스트가 전부 이 경로를 타므로, 이 함수만 놓고 보면 STEP 1 이전과 선택
      결과가 같다.
    - `mode="enhanced"` — contact_support/robot_comfort가 들어간 total_score로 고른다.
    `legacy`와 `log_only`는 회귀 테스트 API로만 남아 있고 runtime에서는 호출하지 않는다.

    두 랭킹 모두 동점이면 입력 순위(rank)가 앞선 쪽 — 순서가 실행마다 흔들리지 않아야
    같은 장면에서 같은 후보가 나온다.
    """
    valid = [e for e in evaluations if e.status == STATUS_VALID]
    if not valid:
        return None, None, None
    legacy_best = min(valid, key=lambda e: (-(e.legacy_score or 0.0), e.candidate.rank))
    enhanced_best = min(valid, key=lambda e: (-(e.total_score or 0.0), e.candidate.rank))
    chosen = enhanced_best if mode == "enhanced" else legacy_best
    chosen.status = STATUS_SELECTED
    return chosen, legacy_best, enhanced_best


def counts(evaluations) -> dict:
    """상태별 후보 수. 보고·로그가 그대로 쓴다."""
    tally = {status: 0 for status in (STATUS_RAW, STATUS_VALID, STATUS_WIDTH_INVALID,
                                      STATUS_IK_FAILED, STATUS_JOINT_LIMIT,
                                      STATUS_SAFETY_INVALID, STATUS_GEOMETRY_INVALID,
                                      STATUS_SELECTED)}
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
    if evaluation.approach_angle_deg is not None:
        parts.append(f"angle={evaluation.approach_angle_deg:.1f}deg")
    if evaluation.approach_travel_mm is not None:
        parts.append(f"travel={evaluation.approach_travel_mm:.0f}mm")
    if evaluation.approach_rotation_deg is not None:
        parts.append(f"rot={evaluation.approach_rotation_deg:.0f}deg")
    if evaluation.joint_delta_deg is not None:
        parts.append(f"joint_delta={evaluation.joint_delta_deg:.0f}deg")
    parts.append(f"safety={_flag(evaluation.safety_ok)}")
    # legacy(실제 실행이 쓰는 값)와 enhanced(비교용)를 나란히 남긴다 — contact_support와
    # robot_comfort는 enhanced 쪽 terms에 있다(STEP 1 §18).
    if evaluation.legacy_score is not None:
        terms = " ".join(f"{name}={value:.2f}"
                         for name, value in evaluation.legacy_terms.items()
                         if value is not None)
        parts.append(f"legacy={evaluation.legacy_score:.3f} [{terms}]")
    if evaluation.total_score is not None:
        terms = " ".join(f"{name}={value:.2f}" for name, value in evaluation.terms.items()
                         if value is not None)
        parts.append(f"enhanced={evaluation.total_score:.3f} [{terms}]")
    parts.append(f"status={evaluation.status}")
    if evaluation.rejection_reason:
        parts.append(f"reason={evaluation.rejection_reason}")
    return " ".join(parts)

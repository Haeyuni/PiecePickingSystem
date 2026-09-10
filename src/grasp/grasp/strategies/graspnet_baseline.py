"""GraspNet-baseline Docker adapter.

GraspNet 공식 demo는 RGB-D 카메라 좌표계의 meter point cloud를 입력으로 쓰고, grasp의
translation/rotation도 그 camera frame으로 돌려준다. ROS grasp 노드는 기존 PCA용 base-mm
point cloud도 유지하되, 이 전략에만 context의 camera-mm points와 base2camera를 제공한다.
컨테이너는 camera frame 추론만 하고, 이 파일이 `T_base_camera_mm @ T_camera_graspnet_mm
@ T_graspnet_tcp_mm`으로 최종 TCP Pose를 base-mm로 단 한 번 변환한다.

추론은 **상주 추론 서버**(compose의 `graspnet` 서비스)에 HTTP로 보낸다 — `endpoint`가
그 주소다. 예전에는 `endpoint`가 비어 있으면 `docker run`으로 1회용 컨테이너를 띄우는
경로가 있었지만 제거했다: 호출마다 모델을 다시 로드해 추론 하나가 수십~수백 초 걸렸고,
grasp 노드에 Docker socket/CLI를 요구해 compose 컨테이너(지금의 실행 환경) 안에서는
애초에 돌 수 없었다.
"""
import io
import json
import threading
import urllib.error
import urllib.parse
import urllib.request
import numpy as np

from .exceptions import InferenceBusy


class _NoUprightCandidate(RuntimeError):
    """추론은 됐지만 임계각 안에 드는 후보가 없다. node가 사유를 로그로 남길 수 있게
    일반 실패와 구분한다 — 이 경우는 "추론 불가"가 아니라 "쓸 만한 자세가 없음"이다."""

STRATEGY = "graspnet_baseline"

# grasp_node는 ReentrantCallbackGroup을 쓴다 — 한 추론(수십 초, 콜드 스타트 포함 컨테이너
# 하나)이 끝나기 전에 새 관측이 들어오면 그 스레드가 별도로 또 docker run을 쏜다. 실물로
# 확인한 사고(2026-09-05): 관측 주기(~2s)가 추론 시간보다 짧아 컨테이너가 수십 개까지
# 동시에 쌓였고, GPU 메모리가 바닥나 perception까지 OOM으로 죽었다. 전략 함수 자체가
# per-call 상태를 못 갖는 순수 함수 계약이라(heuristic_pca와 동일 시그니처) 락은 모듈
# 전역에 둔다 — 이 프로세스 안에서 도는 grasp_node는 하나뿐이므로 충분하다.
_INFERENCE_LOCK = threading.Lock()


def _quaternion_from_matrix(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """오른손 3x3 회전행렬을 geometry_msgs 순서(x, y, z, w)로 바꾼다."""
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            x, y, z, w = 0.25 * scale, (matrix[0, 1] + matrix[1, 0]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale, (matrix[2, 1] - matrix[1, 2]) / scale
        elif index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            x, y, z, w = (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale, (matrix[1, 2] + matrix[2, 1]) / scale, (matrix[0, 2] - matrix[2, 0]) / scale
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            x, y, z, w = (matrix[0, 2] + matrix[2, 0]) / scale, (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale, (matrix[1, 0] - matrix[0, 1]) / scale
    return tuple(float(value) for value in (x, y, z, w))


# base 좌표계에서 "아래로 똑바로" 내려가는 방향. 접근축을 이 벡터와 비교해 기울기를 잰다.
_BASE_DOWN = np.array([0.0, 0.0, -1.0])

# 접근각 hard 상한의 기본값(도). 설정에 `approach_angle_hard_max_deg`가 없을 때 쓴다.
# 근거: 접근 후퇴는 접근축을 따라 80mm다(skill_params.yaml motion.approach_height_mm).
# 75도에서 그 후퇴가 주는 **수직 여유는 80 x cos(75도) = 20.7mm**뿐이라, 이보다 더
# 기울면 그리퍼가 작업대와 거의 나란히 쓸고 들어온다 — control의 최소 안전 검사는
# 파지점 한 점의 z만 보므로(check_min_safety) 그 진입 경로를 못 본다.
# 90도를 넘으면 작업대 아래에서 위로 찌르는 자세다(각도를 abs()로 재지 않는 이유).
_DEFAULT_HARD_MAX_DEG = 45.0


def _camera_offset(params: dict) -> tuple[float, float, float]:
    """설정에서 카메라 좌표계 오프셋(mm) 3축을 읽는다.

    `camera_frame_offset_mm`(x/y/z)가 있으면 그걸 쓰고, 없으면 예전 키
    `camera_axis_offset_mm`(시선축 하나)을 z로 해석한다 — 그 값만 튜닝해 둔 설정이
    조용히 무시되면 로봇이 갑자기 다른 데로 간다."""
    offset = params.get("camera_frame_offset_mm")
    if isinstance(offset, dict):
        return (float(offset.get("x", 0.0)), float(offset.get("y", 0.0)),
                float(offset.get("z", 0.0)))
    if isinstance(offset, (list, tuple)) and len(offset) == 3:
        return tuple(float(v) for v in offset)
    return (0.0, 0.0, float(params.get("camera_axis_offset_mm", 0.0)))



# --- 되잡기 창 크기 (mm) ------------------------------------------------------
# RG2 손가락 패드의 대략적인 크기다. 정확한 기구 치수가 아니라 "손가락이 실제로 닿는
# 범위"를 대변하는 창이고, 이 안의 점만 보고 파지 깊이·개폭을 정한다.
_PAD_HALF_MM = 12.0
# 접근축 위 국소 표면을 잴 때 그리퍼 중심선 주변에서 볼 반경. 점이 모자라면 넓힌다.
_AXIS_RADIUS_MM = 12.0
_MIN_LOCAL_POINTS = 40

# --- 후보 기하 판정 (2026-09-08, 2.6차) ---------------------------------------
GEOMETRY_OK = "ok"
# 후보의 그리퍼 중심선 근처에 물체 점이 없다 = 물체 **옆 허공**을 잡는 자세.
# 중심선은 물체 위인데 손가락 창 안에 물릴 재료가 없다(스치듯 지나간다).
GEOMETRY_NO_MATERIAL = "no_grip_material"
# 되잡기(_refine_on_cloud)가 실측 클라우드로 다시 잰 개폭이 그리퍼 한계를 벗어난다 —
# GraspNet raw width는 추론 서버 쪽 min/max_width_m으로 이미 걸러지지만, 되잡기는 그
# 필터를 거치지 않은 채 실제 물체 폭으로 덮어써서 110mm를 다시 넘을 수 있다
# (2026-09-09: control/grasp_selection.check_width가 121.2mm짜리를 걸러내는 걸 확인 —
# raw는 필터를 통과했는데 되잡은 폭이 다시 넘은 경우). control이 어차피 못 쓸 후보를
# Top-K 자리에 남겨두느니 여기서 걸러 그 자리를 실행 가능한 후보에게 준다.
GEOMETRY_WIDTH_INVALID = "width_invalid"

# 손가락 창 안에 있어야 하는 최소 점 수. 창을 풀어 줄 때 쓰던 기준과 같은 값이다.
_MIN_GRIP_POINTS = 20
# 기하 검사에서 버려지는 후보를 감안해 refine_pool_size의 몇 배까지 살펴볼지. 되잡기가 후보마다
# 클라우드 전체를 투영하므로(실측 173k점) 무한정 늘릴 수 없다.
_GEOMETRY_EXAMINE_FACTOR = 3


def _local_surface(proj_a, lateral, radius_mm: float) -> float:
    """그리퍼 **중심선 주변**의 점들만 보고 접근축 위 표면 위치를 낸다.

    반경 안에 점이 너무 적으면(작은 물체, 성긴 depth) 충분해질 때까지 넓힌다 — 그래도
    모자라면 전체를 쓴다. 백분위수 10%는 잡음 한 점이 표면이 되지 않게 하는 것이다.
    """
    while radius_mm <= 60.0:
        selected = proj_a[lateral <= radius_mm]
        if selected.size >= _MIN_LOCAL_POINTS:
            return float(np.percentile(selected, 10.0))
        radius_mm *= 2.0
    return float(np.percentile(proj_a, 3.0))


def _refine_on_cloud(T_base_tcp: np.ndarray, points_base: np.ndarray,
                     grasp_depth_mm: float,
                     min_grip_points: int = _MIN_GRIP_POINTS
                     ) -> tuple[np.ndarray, float, dict]:
    """GraspNet이 준 **자세는 유지**하고, 위치와 개폭은 실측 포인트클라우드로 다시 잡는다.

    **왜 필요한가.** GraspNet은 원래 장면 전체의 포인트클라우드로 학습됐는데 이 시스템은
    물체 하나를 마스크로 잘라낸 클라우드만 준다(node._candidates_for). 그 입력은 모델의
    학습 분포 밖이라 예측이 신뢰할 수 없다 — 2026-09-07 실측:
      * 후보 위치가 물체 클라우드에서 **8~34mm 떨어져** 있었다(= 허공)
      * 폭이 실제와 크게 달랐다(네일: 실측 4.3mm vs GraspNet 57.1mm)
    반면 **자세(접근축/닫힘축)는 쓸 만하다** — 그건 국소 형상에서 나오는 값이라 잘린
    클라우드로도 의미가 있다. 그래서 자세만 쓰고 위치·폭은 측정값으로 확정한다.

    ## 2026-09-08 (2.6차): 가로 위치를 GraspNet 것으로 두려다 되돌렸다

    Top-K 후보가 최종 위치 2곳으로 뭉치는 문제 때문에, 가로 위치를 GraspNet 것으로 두고
    깊이·폭만 보정하도록 바꿔 봤다. **실물에서 더 나빠져서 되돌렸다**(같은 날 obj_015):

        raw_w=100.0 (10개 전부 — GraspNet 폭 출력이 포화값이다)
        → 후보 자리에서 잰 폭 73~141mm, 그중 4개가 RG2 최대 개폭 110mm 초과로 탈락
        → 파지점이 물체 중심에서 65~88mm 벗어남 (center_proximity 0.00~0.33)

    즉 **GraspNet의 가로 위치는 지켜줄 만한 정보가 아니었다.** 위 docstring이 이미 말하고
    있던 그대로다 — 잘린 클라우드를 받는 이 구성에서는 위치·폭 예측이 신뢰할 수 없고,
    "후보마다 위치가 다르다"는 것도 실제 파지 지점의 다양성이 아니라 그 잡음이다.
    후보가 2곳으로 뭉치는 것은 되잡기의 결함이 아니라 **GraspNet에 물체 하나만 잘라
    넣고 있다는 사실의 증상**이다. 그건 여기가 아니라 입력을 바꿔야 풀린다.

    **2.6차에서 남긴 것**: 손가락 창에 물릴 재료가 없으면 끌어오지 않고 버리는 판정
    (`no_grip_material`)과 진단값들. 위치를 되돌린 뒤에도 그 검사는 유효하다.

    계산(모두 후보 자신의 축 기준):
      1) 가로 중심 (닫힘축 c, 나머지축 o) → 클라우드 중앙값. 물체의 중심이다.
      2) 접근축 a → **그 중심선 주변**의 국소 표면 + grasp_depth_mm
      3) 개폭 → 손가락 패드가 지나는 창 안에서 잰 닫힘축 실측 폭
      4) 최종 가로 위치 → 그 창 안의 실측 범위 중앙 (비대칭 물체에서 중앙값보다 정확)
    """
    closing, other, approach = T_base_tcp[:3, 0], T_base_tcp[:3, 1], T_base_tcp[:3, 2]
    origin = T_base_tcp[:3, 3]
    # **투영을 한 번의 행렬곱으로 묶는다.** 축마다 따로 돌리면 클라우드(실측 170k점)를
    # 세 번 훑는다 — 같은 값이 나오지만 메모리 대역폭만 3배로 쓴다(실측 14.4ms → 5ms대).
    projections = points_base @ np.column_stack([closing, other, approach])
    proj_c, proj_o, proj_a = projections[:, 0], projections[:, 1], projections[:, 2]

    # 1) 그리퍼 중심선: 물체의 가로 중앙값(이상치에 끌리지 않게 median)
    c0, o0 = float(np.median(proj_c)), float(np.median(proj_o))

    # 2) 그 중심선 주변의 국소 표면. 접근축은 물체를 향하므로 가까운 면이 투영 최소다.
    lateral = np.hypot(proj_c - c0, proj_o - o0)
    a_grasp = _local_surface(proj_a, lateral, _AXIS_RADIUS_MM) + float(grasp_depth_mm)

    # 후보 자신의 중심선이 물체에서 얼마나 떨어져 있었는지 — **진단으로만 남긴다.**
    # 2.6차에서 이 값으로 후보를 버려 봤는데, 그러면 되잡기가 구제하던 후보까지 죽어
    # 파지가 통째로 나빠졌다(위 docstring). 값은 GraspNet 위치 품질의 지표로 유용하다.
    candidate_lateral = float(np.hypot(
        float(origin @ closing) - c0, float(origin @ other) - o0))
    diag = {
        "cloud_points": int(len(points_base)),
        "candidate_lateral_mm": round(candidate_lateral, 1),
        "closing_extent_mm": float(np.subtract(*np.percentile(proj_c, [98.0, 2.0]))),
    }

    # 3) 손가락 패드가 실제로 지나는 창. 여기 있는 점만이 닫을 때 실제로 물리는 재료다.
    #    마스크가 배경으로 새어 나온 점은 깊이가 더 깊어 이 창 밖으로 자연히 빠진다.
    window = ((np.abs(proj_o - o0) <= _PAD_HALF_MM)
              & (np.abs(proj_a - a_grasp) <= _PAD_HALF_MM))
    if int(window.sum()) < int(min_grip_points):
        # 얇거나 성긴 물체 — 창을 접근축 방향으로만 풀어 준다(가로 제한은 유지).
        window = np.abs(proj_o - o0) <= _PAD_HALF_MM
    diag["window_points"] = int(window.sum())
    if int(window.sum()) < int(min_grip_points):
        # **전체 클라우드로는 풀지 않는다** (2.6차에서 남긴 것). 예전에는 여기서 전체를
        # 썼는데, 그러면 물체 반대쪽 끝 점으로 폭을 재서 말도 안 되는 개폭이 나왔다.
        diag["status"] = GEOMETRY_NO_MATERIAL
        return origin.copy(), 0.0, diag

    # 남은 이상치에 폭이 끌려가지 않게 min/max 대신 백분위수를 쓴다.
    c_lo, c_hi = np.percentile(proj_c[window], [2.0, 98.0])
    o_lo, o_hi = np.percentile(proj_o[window], [2.0, 98.0])
    position = (float(c_lo + c_hi) / 2.0 * closing
                + float(o_lo + o_hi) / 2.0 * other
                + a_grasp * approach)
    diag.update({
        "status": GEOMETRY_OK,
        "window_width_mm": round(float(c_hi - c_lo), 1),
        "depth_shift_mm": round(float(a_grasp - (origin @ approach)), 1),
        "contact_support_score": _contact_support(
            proj_c, proj_a, window, float(c_lo + c_hi) / 2.0, int(min_grip_points)),
    })
    return position, float(c_hi - c_lo), diag


def _contact_support(proj_c: np.ndarray, proj_a: np.ndarray, window: np.ndarray,
                     grip_center_c: float, min_grip_points: int) -> float:
    """"이 자세에서 두 손가락 사이에 안정적으로 물릴 재료가 있는가"를 0~1로 (STEP 1).

    **새 클라우드 패스를 만들지 않는다.** `_refine_on_cloud`가 이미 계산해 둔 축 투영
    (`proj_c`/`proj_a`)과 손가락 창(`window`), 그리고 그 창에서 잰 파지 중심만 다시 읽는다.

    세 가지를 섞는다 — 전부 기존 기준을 재사용해 새 임계값을 늘리지 않았다.
      1. **좌우 균형**: 닫힘축 기준 파지 중심의 양쪽에 재료가 고르게 있는가. 한쪽이 비어
         있으면 닫는 순간 물체가 그쪽으로 밀려 미끄러진다. **"중심에서 멀다"와는 다르다** —
         물체 끝을 잡아도 양쪽 손가락에 재료만 있으면 여기서는 높은 점수가 나온다.
      2. **창 재료량**: 창 안 점 수를 `min_grip_material_points`(하드 리젝트 기준선)의
         3배를 "넉넉함"으로 보고 정규화. 그 기준선 자체는 이미 no_grip_material이 쓴다.
      3. **두께**: 창 안 재료가 접근축 방향으로 얼마나 두꺼운지를 창 반폭(`_PAD_HALF_MM`)
         으로 정규화. 표면 잡음 한 겹인지 실제 두께가 있는지를 가른다.

    **soft feature다.** 명백히 물릴 재료가 없는 경우(cloud_mismatch/no_grip_material)는
    이 함수에 오기 전에 이미 hard reject된다 — 여기서 낮은 점수가 나온다고 후보를 버리지
    않는다. 가중치 배분(0.5/0.3/0.2)은 실물 근거가 아직 없는 초기값이다.
    """
    selected_c = proj_c[window]
    if selected_c.size == 0:
        return -1.0
    left = int((selected_c < grip_center_c).sum())
    right = int(selected_c.size - left)
    balance = min(left, right) / max(left, right, 1)
    coverage = _clamp01(selected_c.size / max(1.0, 3.0 * float(min_grip_points)))
    selected_a = proj_a[window]
    thickness_mm = float(selected_a.max() - selected_a.min()) if selected_a.size else 0.0
    thickness = _clamp01(thickness_mm / _PAD_HALF_MM)
    return round(float(_clamp01(0.5 * balance + 0.3 * coverage + 0.2 * thickness)), 3)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _select_candidates(raw_candidates: list, T_base_camera_mm: np.ndarray,
                       T_graspnet_tcp_mm: np.ndarray, hard_max_deg: float,
                       legacy_max_deg: float = 30.0,
                       camera_offset_mm=(0.0, 0.0, 0.0),
                       points_base=None, refine_depth_mm: float = 8.0,
                       refine_pool_size: int = 12,
                       min_grip_points: int = _MIN_GRIP_POINTS,
                       min_width_mm: float = 5.0,
                       max_opening_mm: float = 110.0) -> tuple[list, dict]:
    """camera frame GraspNet 후보들을 base로 한 번에 옮기고, **명백히 위험한 접근각만**
    걷어낸 뒤, 되잡기(refine)를 통과한 후보로 `refine_pool_size`개짜리 pool을 채워
    **점수 순서 그대로** 돌려준다. 다양성 선택은 여기가 아니라 node.py가 최종 pose를
    확정한 뒤에 한다(STEP 1). (후보 리스트, 진단정보)를 반환한다.

    **왜 base로 옮긴 뒤에 각도를 재는가.** GraspNet의 접근축은 `rotation_matrix[:, 0]`인데
    그건 **카메라 좌표** 기준이다. 카메라가 손목에 달려 있어(eye-in-hand) 로봇 자세마다
    카메라가 기울어지므로, 카메라 기준 각도는 "작업대에 대해 수직인가"와 아무 관계가 없다.
    base로 옮긴 뒤 `(0, 0, -1)`과 비교해야 의미가 있다.

    **여기서 각도로 거르는 것은 최소한으로 한다 (2026-09-08, 2.5차).**

    원래 이 필터는 두 가지를 한꺼번에 막고 있었다.
      (1) IK가 안 풀리는 자세 — 2026-09-07 실물에서 64.4도/48.7도 후보가 그랬다.
      (2) RG2로 물리적으로 물기 어려운 자세 — 평평한 물체를 옆에서 찌르는 것.
    그런데 (1)은 이제 control이 **직접** 본다(grasp_selection: 접근/파지 IK + 관절 한계).
    각도는 IK 실패의 대리 지표일 뿐이라, 로봇에 물어볼 수 있게 된 지금은 여기서 각도로
    미리 자르면 **실행 가능한 후보까지 같이 죽는다.** (2)는 애초에 "불가"가 아니라
    "덜 좋다"이므로 hard filter가 아니라 랭킹에서 다룰 문제다
    (control/grasp_selection.score_approach_angle).

    그래서 여기 남기는 것은 **명백히 위험한 접근각뿐**이다(`hard_max_deg`). 그 위는
    접근 후퇴(80mm)로 얻는 수직 여유가 거의 없어 그리퍼가 작업대와 나란히 쓸고
    들어오고, 90도를 넘으면 아예 작업대를 뚫고 아래에서 올라오는 자세다.

    **예전의 "넓혀가며 재시도"는 없앴다.** `threshold_deg`(15)부터 `step_deg`(5)씩
    넓히다 **처음 통과자가 나오는 순간 멈추는** 구조라, 실제 상한은 30도가 아니라
    "후보가 하나라도 있는 가장 좁은 5도 구간"이었다. 12도짜리 후보 하나가 있으면
    15도에서 멈춰 20~30도의 멀쩡한 후보가 통째로 버려졌다 — 2026-09-08 실물 로그에서
    raw=50인데 filtered=1로 나온 물체들이 이것이다. Top-K를 유지하려는 1차 작업과
    정면으로 어긋나므로 제거한다.

    각도는 `abs()`로 재지 **않는다** — 그러면 아래에서 위로 찌르는 자세(작업대를 뚫는
    방향)가 0도로 통과한다. `(0,0,-1)`과의 내적을 그대로 쓴다.
    """
    diagnostics: dict = {"raw_count": len(raw_candidates)}
    if not raw_candidates:
        return [], diagnostics

    # --- 벡터화: 파싱 + 유효성 ---------------------------------------------
    n = len(raw_candidates)
    rot_cam = np.zeros((n, 3, 3))
    trans_mm = np.zeros((n, 3))
    depth_mm = np.zeros(n)
    width_mm = np.zeros(n)
    score = np.zeros(n)
    parsed = np.zeros(n, dtype=bool)
    for i, raw in enumerate(raw_candidates):
        try:
            rot_cam[i] = np.asarray(raw["rotation_matrix"], dtype=float).reshape(3, 3)
            trans_mm[i] = np.asarray(raw["translation_m"], dtype=float).reshape(3) * 1000.0
            width_mm[i] = float(raw["width_m"]) * 1000.0
            # depth가 없으면 0 — 구버전 runner(2026-09-07 이전)와 섞여도 죽지 않게.
            depth_mm[i] = float(raw.get("depth_m", 0.0)) * 1000.0
            score[i] = float(raw["score"])
            parsed[i] = True
        except (KeyError, TypeError, ValueError):
            continue

    finite = (parsed
              & np.isfinite(rot_cam).all(axis=(1, 2))
              & np.isfinite(trans_mm).all(axis=1)
              & np.isfinite(width_mm) & np.isfinite(score))
    # 회전행렬이 실제로 직교이고 오른손계인지 — 한 번에 검사한다.
    gram = np.einsum("nij,nik->njk", rot_cam, rot_cam)
    orthonormal = np.isclose(gram, np.eye(3), atol=1e-3).all(axis=(1, 2))
    right_handed = np.linalg.det(rot_cam) > 0.0
    valid = finite & orthonormal & right_handed
    diagnostics["valid_count"] = int(valid.sum())
    if not valid.any():
        return [], diagnostics

    index = np.flatnonzero(valid)
    rot_cam, trans_mm = rot_cam[index], trans_mm[index]
    width_mm, score, depth_mm = width_mm[index], score[index], depth_mm[index]

    # **translation은 무는 지점이 아니다.** graspnetAPI의 그리퍼 기하(plot_gripper_pro_max)
    # 에서 손가락 상자는 center(=translation)에서 접근축(+X)으로 depth만큼 뻗는다. 실제로
    # 물체를 무는 곳은 그 끝이므로 접근축을 따라 depth만큼 전진시켜야 한다.
    # 이 값을 버리면 파지점이 10~40mm 앞(허공)에 찍힌다 — 2026-09-07에 그 부족분을
    # camera_frame_offset_mm.z로 덮고 있었는데, 그건 카메라 시선축이라 기울어진 파지에서
    # 방향이 어긋난다. 여기서 제대로 접근축을 따라 보정한다.
    trans_mm = trans_mm + depth_mm[:, None] * rot_cam[:, :, 0]

    # --- 벡터화: camera → base 변환 ----------------------------------------
    # T_base_tcp = T_base_camera @ T_camera_graspnet @ T_graspnet_tcp  (후보마다 하나)
    T_camera_graspnet = np.tile(np.eye(4), (len(index), 1, 1))
    T_camera_graspnet[:, :3, :3] = rot_cam
    T_camera_graspnet[:, :3, 3] = trans_mm
    T_base_graspnet = T_base_camera_mm @ T_camera_graspnet          # (N,4,4) 브로드캐스트
    T_base_tcp = T_base_graspnet @ T_graspnet_tcp_mm

    # --- 카메라 시선축 보정 -------------------------------------------------
    # depth가 실제보다 가깝게(=물체가 높게) 나오는 계통 오차를 카메라가 보는 방향으로
    # 밀어서 상쇄한다. **base Z나 파지 접근축이 아니라 카메라 시선축인 이유**: 오차의
    # 출처가 depth 측정이라, 어긋나는 방향은 카메라에서 물체를 향하는 선이다. 다른 축으로
    # 보정하면 그 축과 시선축이 이루는 각만큼 옆으로 밀린다.
    #
    # 2026-09-07 실측 근거: perception이 보고한 작업대 z는 314~318인데, 로봇은 z=283까지
    # 아무 저항 없이 내려갔다([파지오차] d오차 0.2mm 이내, 즉 막힌 게 아니라 그냥 도달).
    # 실제 작업대가 283보다 낮다는 뜻이므로 perception z가 최소 30~35mm 높다.
    # 이것이 예전 pick_depth_extra_mm=40이 경험적으로 필요했던 이유이기도 하다 — 그 값은
    # 접근축을 따라 밀어서 기울어진 파지에서는 옆으로도 밀렸다. 여기로 옮겨 바로잡는다.
    # 카메라 좌표계 3축을 base로 옮긴 것. optical frame 규약(backproject와 같다):
    #   [:,0] = 이미지 오른쪽,  [:,1] = 이미지 아래,  [:,2] = 시선(깊이) 방향
    camera_axes_base = T_base_camera_mm[:3, :3]
    offset_cam = np.asarray(camera_offset_mm, dtype=float).reshape(3)
    if np.any(offset_cam):
        # 각 축을 정규화해서 더한다 — 회전행렬이라 이미 단위지만, 캘리브레이션 결과에
        # 미세한 스케일이 섞여 있어도 mm 값이 그대로 mm가 되게 한다.
        axes = camera_axes_base / np.maximum(
            np.linalg.norm(camera_axes_base, axis=0, keepdims=True), 1e-9)
        T_base_tcp[:, :3, 3] += axes @ offset_cam
    camera_axis_base = camera_axes_base[:, 2]
    norm_axis = float(np.linalg.norm(camera_axis_base)) or 1.0

    # --- 벡터화: 접근축 기울기 ---------------------------------------------
    # GraspNet 접근축 = rotation_matrix[:, 0] (카메라 기준) → base로 회전만 적용.
    # (R @ v) 를 행벡터로 하면 v @ R.T 이므로 아래와 같이 한 번에 돈다.
    approach_base = rot_cam[:, :, 0] @ T_base_camera_mm[:3, :3].T   # (N,3)
    norm = np.linalg.norm(approach_base, axis=1)
    norm[norm < 1e-9] = 1.0
    approach_base = approach_base / norm[:, None]
    cos_from_down = np.clip(approach_base @ _BASE_DOWN, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_from_down))
    diagnostics["angles_deg"] = [round(float(a), 1) for a in angle_deg]

    # --- 접근각: 명백히 위험한 것만 자른다 -----------------------------------
    # 비교용으로 **예전 정책이 통과시켰을 수**도 함께 남긴다 — 정책을 바꿨을 때 후보가
    # 실제로 얼마나 살아났는지 로그 한 줄로 보이지 않으면 튜닝할 근거가 없다.
    # (예전 정책은 15도부터 5도씩 넓히다 처음 통과자에서 멈췄으므로, 아래 legacy 수는
    #  "30도 이하 개수"이지 예전에 실제로 통과한 수가 아니다 — 예전 것은 이보다 적거나 같다.)
    diagnostics["legacy_max_deg"] = round(float(legacy_max_deg), 1)
    diagnostics["legacy_pass_count"] = int((angle_deg <= float(legacy_max_deg)).sum())
    diagnostics["hard_max_deg"] = round(float(hard_max_deg), 1)
    keep = angle_deg <= float(hard_max_deg)
    diagnostics["passed_count"] = int(keep.sum())
    # STEP 1의 lifecycle 보고가 쓰는 이름. passed_count와 같은 값이지만 "각도에서 몇 개가
    # 살아남았나"를 단계 이름으로 남겨 두면 뒤 단계(refine/diverse)와 나란히 읽힌다.
    diagnostics["angle_pass_count"] = int(keep.sum())
    if not keep.any():
        return [], diagnostics

    # --- 통과 후보를 점수순으로 Top-K ---------------------------------------
    # **하나만 돌려주지 않는다.** 예전에는 최고점 하나만 내보내 웹 시각화와 이후 선택
    # 알고리즘이 쓸 후보가 남지 않았다. 서버가 이미 NMS + sort_by_score를 마친 것을
    # 주므로(docker/graspnet_baseline/runner.py: GraspGroup.nms/sort_by_score) 여기서는
    # 기울기 필터를 통과한 것 중 점수 상위 K개를 그대로 유지한다.
    #
    # 실제로 어느 후보를 집을지는 control이 정한다(control/grasp_selection.py) —
    # 개폭 유효성·접근/파지 IK·관절 한계·최소 안전·**접근 적합도**를 보고 통과한 것들 중
    # 랭킹으로 고른다. planner는 그 사이에서 작업반경만 걸러 목록을 그대로 넘긴다.
    # 이 단계는 "선택지를 잃지 않는 것"까지만 한다.
    #
    # **hard_max_deg를 넓혀도 여기서 나가는 수는 refine_pool_size로 묶인다.** 각도
    # 정책을 풀면 통과 후보가 수십 개가 되는데, 그걸 그대로 되잡으면 시간이 그만큼 들고
    # world_state도 커진다. 최종 개수는 그 pool에서 다양성으로 고른
    # execution_pool_size개다(node.py).
    kept = np.flatnonzero(keep)
    ranked = kept[np.argsort(-score[kept])]
    # **Top-K를 '점수 상위 K개'가 아니라 '기하 검사를 통과한 상위 K개'로 채운다**
    # (2026-09-08, 2.6차). 예전에는 상위 K개만 되잡고 끝냈는데, 이제 되잡기가 허공
    # 후보를 끌어오지 않고 **버리므로** 그대로 두면 published가 K보다 적어진다. 뒤에
    # 남은 후보로 채우는 것이 Top-K를 둔 이유(선택지를 남긴다)에 맞는다.
    #
    # **여기서 실행 후보 수(10)까지만 채우고 멈추지 않는다** (STEP 1, 2026-09-10).
    # 다양성 선택은 이 함수 뒤에서 **최종 pose 기준으로** 하는데(node._candidates_for가
    # _fit_grasp_depth까지 끝낸 뒤 diverse_pool로 고른다), 그 전에 점수 상위 10개만
    # 남겨 버리면 고를 대상 자체가 이미 한쪽으로 몰려 있을 수 있다. 그래서 여기서는
    # **refined valid pool을 refine_pool_size(12)까지 채우는 것**까지만 한다.
    #
    # GraspNet score는 그대로 쓴다 — 후보의 기본 품질이자 **검사 순서**다(점수 높은 것부터
    # 되잡기를 시도한다). 바뀐 것은 "score Top-10에서 끝내지 않는다"는 것뿐이다.
    #
    # 살펴보는 수에 상한을 둔다 — 되잡기는 후보마다 클라우드 전체를 투영하므로
    # (물티슈 실측 173k점) 전부 돌리면 관측당 시간이 눈에 띄게 늘어난다. 상한은 pool
    # 크기(refine_pool_size, 현재 12)의 3배지만, raw 자체가 max_candidates(50)로 묶여
    # 있어 실제 상한은 50이다.
    examine_limit = min(len(ranked), max(1, int(refine_pool_size)) * _GEOMETRY_EXAMINE_FACTOR)
    order = ranked[:examine_limit]
    diagnostics["examined_count"] = int(len(order))

    results = []
    geometry_rejects: dict = {}
    refine_attempted = 0
    for best in (int(v) for v in order):
        if len(results) >= max(1, int(refine_pool_size)):
            break
        refine_attempted += 1
        rank = len(results)
        T_best = T_base_tcp[best]
        chosen_width_mm = float(width_mm[best])
        refined_shift = None
        # **단계 분리용 원본.** "후보가 이상하다"가 GraspNet raw부터인지 되잡기에서
        # 생긴 것인지는 이 둘을 나란히 봐야만 갈린다(2026-09-08).
        raw_xyz = [round(float(v), 1) for v in T_best[:3, 3]]
        raw_width = round(float(width_mm[best]), 1)
        refine_diag: dict = {}
        lateral_shift = None
        if points_base is not None and len(points_base) >= 30:
            position, chosen_width_mm, refine_diag = _refine_on_cloud(
                T_best, points_base, refine_depth_mm,
                min_grip_points)
            status = refine_diag.get("status", GEOMETRY_OK)
            if status != GEOMETRY_OK:
                # **끌어오지 않고 버린다.** 이 후보를 물체 중심으로 옮겨 살리면 Top-K가
                # 같은 자리로 뭉치고(2.6차가 없앤 문제), 애초에 물체와 안 맞는 자세를
                # 억지로 실행하게 된다. 다음 후보로 넘어간다.
                geometry_rejects[status] = geometry_rejects.get(status, 0) + 1
                continue
            delta = position - T_best[:3, 3]
            refined_shift = round(float(np.linalg.norm(delta)), 1)
            # 접근축 성분은 **의도된** 깊이 보정이다. 문제가 되는 것은 그것과 직교하는
            # 가로 이동 — 그만큼 파지점이 GraspNet이 고른 자리에서 옆으로 끌려간 것이다.
            approach_axis = T_best[:3, 2]
            along = float(delta @ approach_axis)
            lateral_shift = round(float(np.linalg.norm(delta - along * approach_axis)), 1)
            T_best = T_best.copy()
            T_best[:3, 3] = position
        # raw든 되잡은 값이든 최종 개폭이 그리퍼 한계를 벗어나면 버린다 — GEOMETRY_WIDTH_INVALID
        # 주석 참조. raw만 걸러진 채 되잡기가 다시 넘긴 경우가 실물에서 나왔다.
        if not (min_width_mm <= chosen_width_mm <= max_opening_mm):
            geometry_rejects[GEOMETRY_WIDTH_INVALID] = (
                geometry_rejects.get(GEOMETRY_WIDTH_INVALID, 0) + 1)
            continue
        qx, qy, qz, qw = _quaternion_from_matrix(T_best[:3, :3])
        entry = {
            "pose": {
                "position": {"x": float(T_best[0, 3]), "y": float(T_best[1, 3]),
                             "z": float(T_best[2, 3])},
                "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
            },
            "width_mm": chosen_width_mm,
            "score": float(np.clip(score[best], 0.0, 1.0)),
            "grasp_depth_mm": float(depth_mm[best]),
            # 진단·로그 전용. control은 이 값을 쓰지 않고 **실행에 쓰는 접근축**에서 다시
            # 잰다(dsr_motion.approach_axis_from_pose) — 검사와 실행이 갈라지지 않게.
            # 두 값은 같아야 한다: 여기 pose의 Z축이 곧 GraspNet 접근축이다
            # (T_graspnet_tcp_mm이 GraspNet X(접근) → TCP Z(접근)로 맞바꾼다).
            "approach_angle_deg": float(angle_deg[best]),
            # 손가락 사이 실측 재료 지지도 0~1. 되잡기를 못 한 경우(-1)는 "미상"이고
            # 0(지지 없음)과 뜻이 다르다 — control이 None으로 바꿔 랭킹에서 뺀다.
            "contact_support_score": float(
                refine_diag.get("contact_support_score", -1.0)),
            # 후보 정합성 진단(로그 전용, 메시지로 나가지 않는다). node가 물체 중심과의
            # 거리까지 붙여 한 줄로 찍는다.
            "geometry_debug": {
                "raw_xyz": raw_xyz,
                "raw_width_mm": raw_width,
                "refined_shift_mm": refined_shift,
                "lateral_shift_mm": lateral_shift,
                **refine_diag,
            },
            "strategy": STRATEGY,
        }
        if rank == 0:
            # 1순위에만 좌표 변환 중간값을 실어 보낸다 — node가 [좌표체인] 로그로 찍는다.
            # 후보마다 붙이면 로그가 K배로 늘어나기만 한다.
            diagnostics.update({
                "chosen_angle_deg": round(float(angle_deg[best]), 1),
                "chosen_score": round(float(score[best]), 3),
                "graspnet_translation_cam_mm": [round(float(v), 2) for v in trans_mm[best]],
                "graspnet_depth_mm": round(float(depth_mm[best]), 1),
                "graspnet_approach_cam": [round(float(v), 3) for v in rot_cam[best][:, 0]],
                "T_graspnet_tcp_translation_mm": [round(float(v), 2)
                                                  for v in T_graspnet_tcp_mm[:3, 3]],
                "offset_frame": "GraspNet gripper frame (X=접근, Y=닫힘, Z=나머지)",
                "graspnet_point_base_mm": [round(float(v), 2)
                                           for v in T_base_graspnet[best][:3, 3]],
                "tcp_target_base_mm": [round(float(v), 2) for v in T_best[:3, 3]],
                "tcp_approach_axis_base": [round(float(v), 3) for v in T_best[:3, 2]],
                "camera_offset_mm": [round(float(v), 1) for v in offset_cam],
                "camera_axis_base": [round(float(v), 3) for v in camera_axis_base / norm_axis],
                "refined_shift_mm": refined_shift,
                "graspnet_width_mm": round(float(width_mm[best]), 1),
            })
            entry["debug"] = diagnostics
        results.append(entry)
    diagnostics["geometry_rejects"] = geometry_rejects
    diagnostics["refine_attempted"] = refine_attempted
    diagnostics["refine_valid"] = len(results)
    return results, diagnostics


def _hard_max_deg(params: dict) -> float:
    """접근각 hard 상한(도). 이 위의 후보는 아예 만들지 않는다.

    **예전 키(`approach_angle_max_deg`, 기본 30)를 그대로 상한으로 쓰지 않는다.** 그 값은
    "이보다 기울면 IK가 안 풀리거나 물기 어렵다"는 뜻이었는데, IK는 이제 control이 직접
    확인하고(grasp_selection) 물기 어려움은 랭킹이 다룬다. 새 키가 없는 설정 파일에서
    예전 값이 그대로 hard 상한이 되면 2.5차 변경이 아무 효과가 없으므로, 예전 키는
    **읽지 않는다** — 새 키가 없으면 보수적 기본값을 쓴다.
    """
    return float(params.get("approach_angle_hard_max_deg", _DEFAULT_HARD_MAX_DEG))


def _infer_via_endpoint(endpoint: str, points_cam_mm: np.ndarray, params: dict) -> list:
    """상주 추론 서버(compose의 graspnet)에 HTTP로 보낸다.

    점 배열은 `.npy` 바이트로 그대로 싣는다 — 20000x3 float32를 JSON 숫자로 풀면 본문이
    수 MB의 텍스트가 되고 양쪽에서 파싱 비용만 든다. stdlib만 쓴다(이 ROS 이미지에
    requests 같은 걸 새로 넣지 않는다).
    """
    query = urllib.parse.urlencode({
        "num_points": int(params.get("num_points", 20000)),
        "min_width_m": float(params.get("min_width_mm", 5.0)) / 1000.0,
        "max_width_m": float(params.get("max_opening_mm", 110.0)) / 1000.0,
        "max_candidates": int(params.get("max_candidates", 5)),
    })
    buffer = io.BytesIO()
    np.save(buffer, points_cam_mm.astype(np.float32) / 1000.0, allow_pickle=False)
    request = urllib.request.Request(
        f"{endpoint.rstrip('/')}/infer?{query}", data=buffer.getvalue(),
        headers={"Content-Type": "application/octet-stream"}, method="POST")
    timeout_s = float(params.get("endpoint_timeout_s", 30.0))
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # 409 = 서버가 이미 다른 추론 중. 관측 주기가 추론보다 빠를 때 정상적으로 나는
        # 상태이므로 오류가 아니라 "이번 관측 건너뜀"으로 호출부에 알린다.
        if exc.code == 409:
            raise InferenceBusy("GraspNet 서버가 이전 추론을 처리 중입니다") from exc
        detail = exc.read().decode("utf-8", "replace").strip()
        raise RuntimeError(f"GraspNet 서버 오류 {exc.code}: {detail[:200]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"GraspNet 서버({endpoint})에 연결하지 못했습니다: {exc.reason}. "
            "`docker compose up -d graspnet` 상태와 healthcheck를 확인하세요") from exc
    if result.get("input_frame") != "camera" or result.get("input_unit") != "m":
        raise RuntimeError("GraspNet 결과의 camera/m 입력 계약이 다릅니다")
    try:
        return result["candidates"]
    except KeyError as exc:
        raise RuntimeError("GraspNet 결과 형식이 올바르지 않습니다") from exc


def plan(points_base: np.ndarray, params: dict, context: dict | None = None) -> list[dict]:
    """Camera-mm point cloud를 GraspNet에 보내고 base-mm 후보로 돌려준다."""
    # points_base는 GraspNet 입력에는 안 쓰지만(모델은 camera frame을 요구한다) 결과를
    # 실측 클라우드로 되잡는 데 쓴다(_refine_on_cloud).
    context = context or {}
    points_cam_mm = np.asarray(context.get("points_cam_mm"), dtype=float)
    T_base_camera_mm = np.asarray(context.get("T_base_camera_mm"), dtype=float)
    T_graspnet_tcp_mm = np.asarray(params.get("T_graspnet_tcp_mm"), dtype=float)
    if points_cam_mm.ndim != 2 or points_cam_mm.shape[1:] != (3,) or not len(points_cam_mm):
        raise RuntimeError("camera-frame point cloud가 없습니다")
    if T_base_camera_mm.shape != (4, 4):
        raise RuntimeError("T_base_camera 변환이 없습니다")
    if T_graspnet_tcp_mm.shape != (4, 4):
        raise RuntimeError(
            "검증된 graspnet_baseline.T_graspnet_tcp_mm(GraspNet frame에서 RG2 TCP frame으로의 mm 변환)이 필요합니다")
    endpoint = str(params.get("endpoint", "") or "").strip()
    if not endpoint:
        raise RuntimeError(
            "graspnet_baseline.endpoint가 비어 있습니다. compose의 graspnet 서비스 주소를 "
            "넣으세요(기본 http://localhost:8200). 이 전략은 상주 추론 서버로만 동작합니다")

    min_points = int(params.get("min_points", 80))
    if len(points_cam_mm) < min_points:
        return []

    # 서버 경로에도 이 락을 유지한다. grasp_node는 ReentrantCallbackGroup을 써서 이전
    # 추론이 끝나기 전에 다음 관측 스레드가 들어올 수 있는데, 그때마다 요청을 새로 쏘면
    # 서버는 409로 돌려주겠지만 낡은 point cloud를 들고 왕복만 하는 셈이다. 여기서 먼저
    # 끊는 편이 싸다 — 빈 후보는 planner가 실제 파지 불가로 해석하므로, node가 이번
    # 관측 publish를 보류할 수 있게 일반 RuntimeError와 구분되는 상태로 알린다.
    if not _INFERENCE_LOCK.acquire(blocking=False):
        raise InferenceBusy("이전 GraspNet 추론이 진행 중입니다")
    try:
        raw_candidates = _infer_via_endpoint(endpoint, points_cam_mm, params)
    finally:
        _INFERENCE_LOCK.release()

    hard_max_deg = _hard_max_deg(params)
    candidates, diagnostics = _select_candidates(
        raw_candidates, T_base_camera_mm, T_graspnet_tcp_mm,
        hard_max_deg,
        float(params.get("approach_angle_soft_max_deg",
                         params.get("approach_angle_max_deg", 30.0))),
        _camera_offset(params),
        np.asarray(points_base, dtype=float) if points_base is not None else None,
        float(params.get("refine_grasp_depth_mm", 8.0)),
        # refined valid pool 목표치. 실행 후보 수(execution_pool_size)와 **다른 값**이다 —
        # 다양성 선택은 이 pool에서 최종 pose 기준으로 고른다(node._candidates_for).
        int(params.get("refine_pool_size", 12)),
        int(params.get("min_grip_material_points", _MIN_GRIP_POINTS)),
        float(params.get("min_width_mm", 5.0)),
        float(params.get("max_opening_mm", 110.0)))
    if not candidates:
        # 추론은 됐는데 hard 상한 안에 드는 후보가 없다. 빈 리스트로 돌려주면 node가
        # "후보 없음"으로 발행하고 planner는 파지 불가로 읽는데, **왜** 걸러졌는지가
        # 로그에 없으면 추적이 안 된다 — 일반 실패와 구분되는 예외로 알린다.
        raise _NoUprightCandidate(
            f"수직 대비 {hard_max_deg}도(hard 상한) 이내 후보 없음 "
            f"(기울기 {diagnostics.get('angles_deg')})")
    return candidates

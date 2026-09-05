"""전략: 포인트클라우드 PCA 주축 + 윗면 기준 수직 접근 (학습 불필요 베이스라인).

FR-07이 요구하는 "기하학적 중심 기반" 전략이다. 3단계 비교 실험(BR 4.1절)에서 학습 기반
전략들의 기준점이 된다 — 이것보다 나아지지 않는 학습 모델은 쓸 이유가 없다.

**어떻게 잡는가**: 물체의 윗면 점들을 XY로 투영해 PCA를 돌리고, 짧은 축 방향으로 손가락을
닫는다. 긴 축을 가로질러 잡으면 벌어지는 폭이 그리퍼 개폭을 넘기 쉽고, 잡히더라도 접촉면이
얕아 미끄러진다. 접근은 위에서 수직으로 내려온다(base -Z).

**이 전략이 못 하는 것**: 기울어진 물체(7단계)와 클러터(2·9단계). 윗면이 수평이라고 가정하고
접근 방향을 -Z로 고정하기 때문이다. 그것이 학습 기반 전략을 함께 두는 이유다.
"""
import numpy as np

from ..pointcloud_utils import principal_axes_xy, top_surface

STRATEGY = "heuristic_pca"          # sort_msgs/GraspCandidate.msg의 STRATEGY_HEURISTIC_PCA


def _quaternion_from_matrix(R: np.ndarray) -> tuple[float, float, float, float]:
    """회전행렬 → 쿼터니언 (x, y, z, w). Shepperd 방식 — 가장 큰 성분부터 구해 수치안정.

    trace만 보고 나누면 180° 근처에서 0으로 나누게 된다. 위에서 내려오는 파지는 정확히
    그 근처(base 기준 X축 180° 회전)라 이 분기가 실제로 쓰인다.
    """
    trace = np.trace(R)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)


def _pose_from_closing_axis(position, closing_axis: np.ndarray) -> dict:
    """닫는 방향(수평)과 위치로 파지 자세를 만든다. 접근은 항상 base -Z(수직 하강).

    그리퍼 좌표계: X = 손가락이 닫히는 방향, Z = 접근 방향, Y = 나머지(오른손 좌표계).
    """
    z_axis = np.array([0.0, 0.0, -1.0])
    x_axis = np.array([closing_axis[0], closing_axis[1], 0.0])
    norm = np.linalg.norm(x_axis)
    if norm < 1e-9:                      # 축이 거의 수직이면 닫는 방향을 정할 수 없다
        x_axis = np.array([1.0, 0.0, 0.0])
    else:
        x_axis = x_axis / norm
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.column_stack([x_axis, y_axis, z_axis])
    qx, qy, qz, qw = _quaternion_from_matrix(rotation)
    return {
        "position": {"x": float(position[0]), "y": float(position[1]), "z": float(position[2])},
        "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
    }


def _width_score(width_mm: float, max_opening_mm: float, min_width_mm: float) -> float:
    """개폭 대비 여유. 너무 넓으면 0, 너무 얇으면 감점(손가락이 스쳐 지나간다)."""
    if width_mm >= max_opening_mm or width_mm <= 0.0:
        return 0.0
    if width_mm < min_width_mm:
        return 0.35
    # 개폭의 절반쯤일 때가 가장 여유롭다. 꽉 차게 잡으면 오차 몇 mm에 실패한다.
    ratio = width_mm / max_opening_mm
    return float(np.clip(1.0 - abs(ratio - 0.5) * 1.4, 0.1, 1.0))


def plan(points_base: np.ndarray, params: dict, context: dict | None = None) -> list[dict]:
    """물체 하나의 점들(base 좌표계 mm)로 파지 후보 목록을 만든다.

    반환: [{"pose": {...}, "score": float, "strategy": "heuristic_pca"}], 점수 내림차순.
    후보가 없으면 빈 리스트 — planner의 검증기가 "유효한 파지 후보가 없습니다"로 거부한다.
    """
    max_opening_mm = float(params.get("max_opening_mm", 110.0))
    min_width_mm = float(params.get("min_width_mm", 5.0))
    min_points = int(params.get("min_points", 80))
    band_mm = float(params.get("top_surface_band_mm", 15.0))
    grasp_depth_mm = float(params.get("grasp_depth_mm", 8.0))

    if len(points_base) < min_points:
        return []

    top = top_surface(points_base, band_mm=band_mm)
    if len(top) < min_points:
        return []

    major, minor, extents = principal_axes_xy(top)
    major_extent, minor_extent = float(extents[0]), float(extents[1])

    center_xy = top[:, :2].mean(axis=0)
    top_z = float(np.percentile(top[:, 2], 97.0))
    # 윗면보다 조금 내려간 곳을 잡는다. 정확히 윗면을 노리면 손가락이 물체를 스치고 지나간다.
    position = (center_xy[0], center_xy[1], top_z - grasp_depth_mm)

    # 윗면이 평평할수록 수직 접근 가정이 잘 맞는다 (기울어진 물체는 7단계 확장)
    flatness = float(np.clip(1.0 - np.std(top[:, 2]) / max(band_mm, 1e-6), 0.0, 1.0))
    support = float(np.clip(len(top) / (min_points * 6.0), 0.3, 1.0))

    candidates = []
    # 1순위: 짧은 축을 가로질러 닫는다
    score = _width_score(minor_extent, max_opening_mm, min_width_mm) * flatness * support
    if score > 0.0:
        candidates.append({"pose": _pose_from_closing_axis(position, minor),
                           "score": round(float(np.clip(score, 0.0, 1.0)), 4),
                           "strategy": STRATEGY,
                           "width_mm": round(minor_extent, 1)})

    # 2순위: 긴 축 쪽. 짧은 축이 너무 얇아 손가락이 헛도는 물체에서는 이쪽이 낫다.
    # 개폭을 넘으면 _width_score가 0을 돌려 자연히 빠진다.
    score = _width_score(major_extent, max_opening_mm, min_width_mm) * flatness * support * 0.8
    if score > 0.0:
        candidates.append({"pose": _pose_from_closing_axis(position, major),
                           "score": round(float(np.clip(score, 0.0, 1.0)), 4),
                           "strategy": STRATEGY,
                           "width_mm": round(major_extent, 1)})

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates

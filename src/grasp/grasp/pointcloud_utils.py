"""InstanceMasks + depth → 물체별 포인트클라우드 (인터페이스_정의서.md 2.0절).

**이 재구성은 grasp 프로세스 안에서만 일어난다.** 포인트클라우드를 토픽으로 내보내지
않는 이유는 대역폭 때문만이 아니라, 그것이 grasp의 내부 표현이기 때문이다 — 밖으로
나가는 것은 `GraspCandidate`뿐이고, 그래야 전략을 바꿔도 인터페이스가 그대로다.

좌표 단위는 mm로 통일한다(1.1절). depth 이미지가 mm 정수이므로 환산 없이 그대로 쓴다.
"""
import numpy as np


def backproject(mask_bool: np.ndarray, depth_mm: np.ndarray,
                intrinsics: dict) -> np.ndarray:
    """마스크 안쪽 유효 depth 픽셀을 카메라 좌표계 점(mm) Nx3으로 편다.

    depth가 0인 픽셀은 "측정 실패"이지 "거리 0"이 아니다 — 빼지 않으면 카메라 원점에
    점 무더기가 생겨 PCA 축이 통째로 틀어진다.
    """
    ys, xs = np.nonzero(mask_bool)
    if xs.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    z = depth_mm[ys, xs].astype(np.float64)
    valid = z > 0
    if not valid.any():
        return np.empty((0, 3), dtype=np.float64)
    z, xs, ys = z[valid], xs[valid].astype(np.float64), ys[valid].astype(np.float64)
    x = (xs - intrinsics["cx"]) * z / intrinsics["fx"]
    y = (ys - intrinsics["cy"]) * z / intrinsics["fy"]
    return np.stack([x, y, z], axis=1)


def transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Nx3 점들에 4x4 동차변환을 적용한다."""
    if points.size == 0:
        return points
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def remove_outliers(points: np.ndarray, z_percentile: float = 2.0,
                    max_radius_mm: float = 250.0) -> np.ndarray:
    """마스크 가장자리에서 배경으로 새어 나온 점을 걷어낸다.

    세그멘테이션 마스크는 물체 경계에서 몇 픽셀씩 넘치고, 그 픽셀의 depth는 물체가 아니라
    **뒤쪽 바닥**을 잰 값이다. 그대로 두면 물체가 실제보다 훨씬 두꺼운 것처럼 보인다.
    깊이 방향 양끝 퍼센타일을 자르고, 중앙값에서 지나치게 먼 점을 버린다.
    """
    if len(points) < 10:
        return points
    low, high = np.percentile(points[:, 2], [z_percentile, 100.0 - z_percentile])
    kept = points[(points[:, 2] >= low) & (points[:, 2] <= high)]
    if len(kept) < 10:
        return points
    center = np.median(kept, axis=0)
    distance = np.linalg.norm(kept - center, axis=1)
    return kept[distance <= max_radius_mm]


def top_surface(points_base: np.ndarray, band_mm: float = 15.0) -> np.ndarray:
    """base 좌표계에서 위쪽 표면에 해당하는 점들.

    그리퍼는 위에서 내려와 닿는다. 물체 전체의 무게중심이 아니라 **손가락이 실제로 만나는
    윗면**을 기준으로 자세를 잡아야 한다 — 높이가 있는 물체에서 둘은 다르다.

    최고점 하나가 아니라 밴드를 쓰는 것은 depth 잡음 때문이다. 한 점을 기준으로 삼으면
    튀어 오른 잡음 픽셀이 그대로 파지 높이가 된다.
    """
    if points_base.size == 0:
        return points_base
    top_z = np.percentile(points_base[:, 2], 97.0)
    band = points_base[points_base[:, 2] >= top_z - band_mm]
    return band if len(band) >= 10 else points_base


def principal_axes_xy(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """XY 평면에 투영한 뒤 PCA. (major_axis, minor_axis, extents) 반환.

    Z를 빼고 보는 이유는 위에서 내려오는 파지에서 **손가락이 닫히는 방향은 수평면 안**이기
    때문이다. 높이는 접근 깊이를 정할 뿐 닫는 방향과 무관하다.
    """
    xy = points[:, :2] - points[:, :2].mean(axis=0)
    # 공분산 고유분해. 2x2라 SVD보다 값이 직관적이다(고유값 = 축별 분산).
    covariance = np.cov(xy, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]           # 큰 분산 = 긴 축
    major = np.array([*eigenvectors[:, order[0]], 0.0])
    minor = np.array([*eigenvectors[:, order[1]], 0.0])

    # 실제 폭은 고유값이 아니라 축 방향 투영 길이로 잰다 — 그리퍼 개폭과 비교할 값이다.
    projection_major = points[:, :2] @ major[:2]
    projection_minor = points[:, :2] @ minor[:2]
    extents = np.array([projection_major.max() - projection_major.min(),
                        projection_minor.max() - projection_minor.min()])
    return major, minor, extents

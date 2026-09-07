"""세그멘테이션 마스크 처리: 3D 환산, InstanceMasks 인코딩, 신규 물체 크롭.

`InstanceMasks.masks`는 mono8 0/255 단일 채널이다(인터페이스_정의서 3.3절). `grasp`가
이 마스크로 물체별 포인트클라우드를 재구성하므로, **마스크 해상도는 depth·color와 같아야
한다** — 모델 입력 크기(640)로 나온 마스크를 그대로 실으면 grasp가 엉뚱한 픽셀을 집는다.
"""
import numpy as np


def resize_mask(mask_bool: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    """마스크를 (h, w)로 맞춘다. 최근접 보간 — 마스크는 0/1이라 보간하면 안 된다."""
    if mask_bool.shape == shape_hw:
        return mask_bool
    import cv2

    height, width = shape_hw
    resized = cv2.resize(mask_bool.astype(np.uint8), (width, height),
                         interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def mask_3d(mask_bool: np.ndarray, depth_mm: np.ndarray,
            intrinsics: dict) -> tuple[tuple[float, float, float] | None, float]:
    """마스크 안쪽 유효 depth로 카메라 좌표계 3D 점(mm)과 유효 비율을 낸다.

    중심 픽셀 하나의 depth 대신 **유효 픽셀 depth의 median**을 쓴다. 물체 가장자리에서는
    depth가 배경으로 튀어 한 픽셀만 보면 수십 cm씩 어긋난다.

    `depth_valid_ratio`는 마스크 안에서 depth가 0이 아닌 픽셀의 비율이다. 투명·반사 물체는
    이 값이 낮게 나오므로 파지 가능 판정과 8단계(투명 물체) 확장의 입력이 된다.
    """
    if mask_bool.shape != depth_mm.shape:
        return None, 0.0
    ys, xs = np.nonzero(mask_bool)
    if xs.size == 0:
        return None, 0.0
    z = depth_mm[ys, xs].astype(np.float32)
    valid = z > 0
    ratio = float(valid.mean())
    if not valid.any():
        return None, ratio
    zc = float(np.median(z[valid]))
    uc, vc = float(xs[valid].mean()), float(ys[valid].mean())
    x = (uc - intrinsics["cx"]) * zc / intrinsics["fx"]
    y = (vc - intrinsics["cy"]) * zc / intrinsics["fy"]
    return (x, y, zc), ratio


def mask_extent_3d(mask_bool: np.ndarray, depth_mm: np.ndarray, intrinsics: dict,
                   count: int = 24) -> list[tuple[float, float, float]]:
    """마스크 **테두리**를 따라 뽑은 카메라 좌표계 3D 점들(mm). 물체가 차지한 범위다.

    `mask_3d`는 중심 한 점만 준다. 온디맨드 재관측(D-1)은 그것으로 부족하다 — 다음 프레임에서
    SAM에 줄 **박스**를 만들어야 하는데(점 프롬프트는 물체 대신 무늬를 잡는다,
    sam_marks.segment_at_boxes 주석) 박스에는 범위가 필요하다.

    테두리를 쓰는 이유는 그것이 곧 박스의 경계이기 때문이다. depth가 무효인 점은 뺀다 —
    지어낸 깊이로 만든 박스는 다음 프레임에서 엉뚱한 곳을 감싼다. 각 점은 **자기 픽셀의
    depth**로 역투영하므로 기울어진 물체도 범위가 유지된다.
    """
    if mask_bool.shape != depth_mm.shape:
        return []
    ys, xs = np.nonzero(mask_bool)
    if xs.size == 0:
        return []

    # 행마다 좌우 끝 픽셀 = 마스크의 테두리. contour를 쓰지 않는 이유는 cv2 의존을 늘리지
    # 않기 위해서다(이 모듈은 numpy만 쓴다).
    rows = np.unique(ys)
    if rows.size > count // 2:
        rows = rows[np.linspace(0, rows.size - 1, count // 2).astype(int)]

    points = []
    for row in rows:
        columns = xs[ys == row]
        for column in (columns.min(), columns.max()):
            z = float(depth_mm[row, column])
            if z <= 0:
                continue
            points.append((
                (float(column) - intrinsics["cx"]) * z / intrinsics["fx"],
                (float(row) - intrinsics["cy"]) * z / intrinsics["fy"],
                z,
            ))
    return points


def mask_to_image_msg(mask_bool: np.ndarray, header):
    """mono8 0/255 단일 채널 Image (인터페이스_정의서 3.3절)."""
    from sensor_msgs.msg import Image

    data = (mask_bool.astype(np.uint8) * 255)
    msg = Image()
    msg.header = header
    msg.height, msg.width = data.shape
    msg.encoding = "mono8"
    msg.is_bigendian = 0
    msg.step = msg.width
    msg.data = data.tobytes()
    return msg


def crop_bgr(bgr: np.ndarray, mask_bool: np.ndarray, margin_px: int = 12) -> np.ndarray | None:
    """마스크 바운딩박스로 잘라낸 이미지. 신규 클래스 확인용 크롭 (시스템명세서 4.4절).

    마스크 밖을 지우지 않고 여백까지 남기는 이유는, 사람과 VLM 둘 다 **주변 맥락이 있어야**
    무엇인지 판단하기 때문이다. 배경을 검게 칠한 크롭은 오히려 알아보기 어렵다.
    """
    ys, xs = np.nonzero(mask_bool)
    if xs.size == 0:
        return None
    height, width = bgr.shape[:2]
    x0 = max(int(xs.min()) - margin_px, 0)
    x1 = min(int(xs.max()) + margin_px + 1, width)
    y0 = max(int(ys.min()) - margin_px, 0)
    y1 = min(int(ys.max()) + margin_px + 1, height)
    return bgr[y0:y1, x0:x1].copy()


def support_3d(mask_bool: np.ndarray, depth_mm: np.ndarray, intrinsics: dict,
               dilate_px: int = 15, percentile: float = 70.0
               ) -> tuple[float, float, float] | None:
    """물체가 **얹혀 있는 면**(작업대 등)의 카메라 좌표 3D 점(mm)을 낸다.

    마스크를 조금 부풀려 만든 바깥 링에서 depth를 읽는다. 링은 물체 바로 옆이라 그 물체를
    받치고 있는 면일 가능성이 높다.

    **xy는 물체 마스크의 중심을 그대로 쓰고 depth만 링에서 가져온다** — `mask_3d`와 같은
    (u, v)를 쓰므로 두 점의 차이가 순수하게 "물체 윗면과 지지면의 높이차"가 된다. 링의
    무게중심을 쓰면 물체 옆으로 밀린 지점이 되어 그 차이에 xy 성분이 섞인다.

    depth는 median이 아니라 **높은 백분위수**(기본 70%)를 쓴다. depth는 카메라에서 먼
    쪽이 큰 값이라, 링에 옆 물체가 걸쳐 들어와도 더 먼(=더 낮은) 면 쪽으로 치우친다.
    물체가 다른 물체 위에 얹힌 경우에는 지지면을 작업대로 잡아 물체 높이를 **크게**
    보게 되는데, 그 방향의 오차는 place에서 "더 높은 데서 놓는다"가 되므로 안전한 쪽이다.

    링에 유효 depth가 없으면 None.
    """
    if mask_bool.shape != depth_mm.shape:
        return None
    ys, xs = np.nonzero(mask_bool)
    if xs.size == 0:
        return None
    z_obj = depth_mm[ys, xs].astype(np.float32)
    valid_obj = z_obj > 0
    if not valid_obj.any():
        return None
    uc, vc = float(xs[valid_obj].mean()), float(ys[valid_obj].mean())

    import cv2

    kernel = np.ones((2 * int(dilate_px) + 1,) * 2, np.uint8)
    dilated = cv2.dilate(mask_bool.astype(np.uint8), kernel, iterations=1).astype(bool)
    ring = dilated & ~mask_bool
    ring_z = depth_mm[ring].astype(np.float32)
    ring_z = ring_z[ring_z > 0]
    if ring_z.size == 0:
        return None
    zs = float(np.percentile(ring_z, percentile))
    x = (uc - intrinsics["cx"]) * zs / intrinsics["fx"]
    y = (vc - intrinsics["cy"]) * zs / intrinsics["fy"]
    return (x, y, zs)

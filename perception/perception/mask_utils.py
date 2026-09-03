"""세그멘테이션 마스크 처리: 3D 환산, InstanceMasks 인코딩, 신규 물체 크롭.

`InstanceMasks.masks`는 mono8 0/255 단일 채널이다(인터페이스_정의서 3.3절). `grasp`가
이 마스크로 물체별 포인트클라우드를 재구성하므로, **마스크 해상도는 depth·color와 같아야
한다** — 모델 입력 크기(640)로 나온 마스크를 그대로 실으면 grasp가 엉뚱한 픽셀을 집는다.
"""
import numpy as np
from sensor_msgs.msg import Image


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


def mask_to_image_msg(mask_bool: np.ndarray, header) -> Image:
    """mono8 0/255 단일 채널 Image (인터페이스_정의서 3.3절)."""
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

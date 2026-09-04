"""sensor_msgs/Image ↔ ndarray 변환.

**cv_bridge를 쓰지 않는다.** venv의 opencv-python과 시스템 cv_bridge가 섞이면 ABI가 어긋나
import 단계에서 깨진다(`perception/README.md`). 지원해야 할 인코딩이 몇 개 안 되므로 직접
변환하는 편이 의존성 하나를 줄인다.
"""
import numpy as np
from sensor_msgs.msg import Image

_ENCODINGS = {
    "bgr8": (np.uint8, 3),
    "rgb8": (np.uint8, 3),
    "mono8": (np.uint8, 1),
    "16UC1": (np.uint16, 1),
    "mono16": (np.uint16, 1),
}


def image_to_numpy(msg: Image) -> np.ndarray:
    """지원 인코딩: bgr8 / rgb8 / mono8 / 16UC1 / mono16. 결과는 항상 BGR 기준."""
    try:
        dtype, channels = _ENCODINGS[msg.encoding]
    except KeyError:
        raise ValueError(f"지원하지 않는 인코딩: {msg.encoding}") from None
    buffer = np.frombuffer(msg.data, dtype=dtype)
    # step은 바이트 단위 행 길이다. 패딩이 붙는 드라이버가 있어 width로 자른다.
    array = buffer.reshape(msg.height, msg.step // np.dtype(dtype).itemsize)
    array = array[:, : msg.width * channels].reshape(msg.height, msg.width, channels)
    if msg.encoding == "rgb8":
        array = array[:, :, ::-1]
    return array if channels > 1 else array[:, :, 0]

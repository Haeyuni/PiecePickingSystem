"""카메라 좌표 → base 좌표 변환 (인터페이스_정의서.md 1.1절, 개발계획.md D-2).

`DetectedObject.position_base_mm`은 이름 그대로 **base 좌표계 mm**다. 카메라는 그리퍼에
붙어 있으므로(eye-in-hand) 변환은 현재 TCP 자세에 따라 매 프레임 달라진다:

    base ← gripper ← camera
    T_base2cam = T_base2gripper(현재 TCP 자세) @ T_gripper2camera(캘리브레이션 결과)

**TF를 쓰지 않는 이유**: 리얼센스 드라이버가 발행하는 TF 트리는 `camera_link`를 뿌리로
자기 프레임들만 담고 있고, 로봇 트리와 이어져 있지 않다. 이어 붙이려면 결국 여기서 만드는
것과 같은 hand-eye 변환을 static transform으로 발행해야 한다. 더 중요한 이유는 **캘리브레이션이
`get_current_posx()` 값으로 풀렸다는 것**이다 — 같은 정의의 자세를 써야 T_gripper2camera가
의미를 갖는다. TF의 `link6`/`tool0`가 그 TCP와 같다는 보장이 없고, 다르면 툴 오프셋만큼
조용히 어긋난다.
"""
import pathlib

import numpy as np

from .paths import find_repo_path

def handeye_path() -> pathlib.Path:
    """캘리브레이션 결과 위치 (개발계획.md D-2).

    `HANDEYE_PATH`가 우선이지만, 그 경로가 없으면 저장소에서 다시 찾는다 — `.env`의 값은
    컨테이너 기준(`/data/...`)이고 이 노드는 호스트에서도 돈다(`paths.find_repo_path`).
    """
    return find_repo_path("data", "calibration", "T_gripper2camera.npy", env_var="HANDEYE_PATH")


def load_handeye(path: pathlib.Path | None = None) -> np.ndarray:
    """T_gripper2camera (4x4). 없으면 좌표를 낼 수 없으므로 그대로 예외를 올린다."""
    path = path or handeye_path()
    matrix = np.load(path)
    if matrix.shape != (4, 4):
        raise ValueError(f"hand-eye 행렬 형태가 4x4가 아닙니다: {matrix.shape} ({path})")
    return matrix


def _rot_z(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_y(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def posx_to_matrix(posx) -> np.ndarray:
    """두산 `get_current_posx()` 반환값 → T_base2gripper (4x4).

    `posx`는 `[x, y, z, rx, ry, rz]`이고 위치는 mm, 회전은 **ZYZ 오일러각(도)**이다.
    `calibration/handeye_calibration.py`·`verify.py`가 쓴 것과 같은 해석이어야 한다 —
    여기가 어긋나면 캘리브레이션 결과 전체가 무의미해진다.

    scipy를 쓰지 않고 직접 만든다. 이 워크스테이션의 scipy는 설치된 numpy 버전을 지원하지
    않는다고 경고하는데, 좌표 변환은 조용히 틀리면 로봇이 엉뚱한 곳으로 가는 계산이라
    경고가 붙은 의존성에 맡기지 않는다. 회전행렬 3개 곱이라 직접 쓰는 비용도 없다.
    """
    x, y, z, rx, ry, rz = (float(v) for v in posx[:6])
    matrix = np.eye(4)
    matrix[:3, :3] = _rot_z(rx) @ _rot_y(ry) @ _rot_z(rz)   # 내재 ZYZ
    matrix[:3, 3] = (x, y, z)
    return matrix


def base_from_camera(point_cam_mm, base2gripper: np.ndarray,
                     gripper2camera: np.ndarray) -> tuple[float, float, float]:
    """카메라 좌표계 점(mm) → base 좌표계 점(mm)."""
    homogeneous = np.append(np.asarray(point_cam_mm, dtype=float), 1.0)
    base_point = base2gripper @ gripper2camera @ homogeneous
    return tuple(float(v) for v in base_point[:3])

"""GraspNet-baseline Docker adapter.

GraspNet 공식 demo는 RGB-D 카메라 좌표계의 meter point cloud를 입력으로 쓰고, grasp의
translation/rotation도 그 camera frame으로 돌려준다. ROS grasp 노드는 기존 PCA용 base-mm
point cloud도 유지하되, 이 전략에만 context의 camera-mm points와 base2camera를 제공한다.
컨테이너는 camera frame 추론만 하고, 이 파일이 `T_base_camera_mm @ T_camera_graspnet_mm
@ T_graspnet_tcp_mm`으로 최종 TCP Pose를 base-mm로 단 한 번 변환한다.
"""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

STRATEGY = "graspnet_baseline"
_CHECKPOINT_MOUNT = "/checkpoint.tar"


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


def _docker_directory() -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory

        installed = Path(get_package_share_directory("grasp")) / "docker" / "graspnet_baseline"
        if (installed / "Dockerfile").is_file():
            return installed
    except Exception:
        pass
    return Path(__file__).resolve().parents[2] / "docker" / "graspnet_baseline"


def _require_image(image: str) -> None:
    if shutil.which("docker") is None:
        raise RuntimeError("docker 명령을 찾지 못했습니다")
    exists = subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0
    if not exists:
        raise RuntimeError(
            f"GraspNet Docker image가 없습니다: {image}. 테스트 PC에서 "
            "bash src/grasp/scripts/setup_graspnet_runtime.sh 를 먼저 실행하세요.")


def _base_candidate(raw: dict, T_base_camera_mm: np.ndarray,
                    T_graspnet_tcp_mm: np.ndarray) -> dict | None:
    """camera/m GraspNet pose를 보정된 base/mm RG2 TCP 후보로 바꾼다."""
    try:
        width_mm = float(raw["width_m"]) * 1000.0
        translation_mm = np.asarray(raw["translation_m"], dtype=float).reshape(3) * 1000.0
        rotation_cam = np.asarray(raw["rotation_matrix"], dtype=float).reshape(3, 3)
        score = float(raw["score"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (np.isfinite(width_mm) and np.isfinite(translation_mm).all()
            and np.isfinite(rotation_cam).all() and np.isfinite(score)):
        return None
    if not np.allclose(rotation_cam.T @ rotation_cam, np.eye(3), atol=1e-3):
        return None
    if np.linalg.det(rotation_cam) <= 0.0:
        return None

    T_camera_graspnet_mm = np.eye(4)
    T_camera_graspnet_mm[:3, :3] = rotation_cam
    T_camera_graspnet_mm[:3, 3] = translation_mm
    # T_graspnet_tcp_mm maps TCP-frame coordinates into GraspNet's predicted gripper frame.
    T_base_tcp_mm = T_base_camera_mm @ T_camera_graspnet_mm @ T_graspnet_tcp_mm
    qx, qy, qz, qw = _quaternion_from_matrix(T_base_tcp_mm[:3, :3])
    return {
        "pose": {
            "position": {"x": float(T_base_tcp_mm[0, 3]), "y": float(T_base_tcp_mm[1, 3]), "z": float(T_base_tcp_mm[2, 3])},
            "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
        },
        "width_mm": width_mm,
        "score": float(np.clip(score, 0.0, 1.0)),
        "strategy": STRATEGY,
    }


def plan(points_base: np.ndarray, params: dict, context: dict | None = None) -> list[dict]:
    """Camera-mm point cloud를 GraspNet Docker에 보내고 base-mm 후보로 돌려준다."""
    del points_base  # PCA 호환 전략 계약을 유지하지만 GraspNet은 camera frame 입력을 요구한다.
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

    checkpoint = Path(str(params.get("checkpoint_path", ""))).expanduser()
    if not checkpoint.is_file():
        raise RuntimeError(
            "graspnet_baseline.checkpoint_path가 없습니다. 테스트 PC에서 "
            "setup_graspnet_runtime.sh를 실행해 checkpoint.tar를 준비하세요")
    device = str(params.get("device", "cuda:0"))
    if not device.startswith("cuda"):
        raise RuntimeError("GraspNet-baseline은 CUDA device가 필요합니다")
    min_points = int(params.get("min_points", 80))
    if len(points_cam_mm) < min_points:
        return []

    image = str(params.get("image", "piece-picking-graspnet-baseline:0.1.0"))
    _require_image(image)
    num_points = int(params.get("num_points", 20000))
    max_candidates = int(params.get("max_candidates", 5))
    timeout_s = float(params.get("timeout_s", 60.0))
    min_width_m = float(params.get("min_width_mm", 5.0)) / 1000.0
    max_width_m = float(params.get("max_opening_mm", 110.0)) / 1000.0

    with tempfile.TemporaryDirectory(prefix="graspnet_baseline_") as temp_dir:
        temp = Path(temp_dir)
        np.savez_compressed(temp / "input.npz", points_cam_m=points_cam_mm.astype(np.float32) / 1000.0)
        command = [
            "docker", "run", "--rm", "--gpus", "all",
            "-v", f"{temp}:/io",
            "-v", f"{checkpoint.resolve()}:{_CHECKPOINT_MOUNT}:ro",
            image, "/io/input.npz", "/io/output.json", _CHECKPOINT_MOUNT, device,
            str(num_points), str(min_width_m), str(max_width_m), str(max_candidates),
        ]
        try:
            run = subprocess.run(command, text=True, capture_output=True, timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"GraspNet 추론 timeout ({timeout_s:.0f}s)") from exc
        output = temp / "output.json"
        if run.returncode or not output.is_file():
            detail = (run.stderr or run.stdout).strip().splitlines()
            raise RuntimeError(f"GraspNet container 실행 실패: {detail[-1] if detail else run.returncode}")
        try:
            result = json.loads(output.read_text(encoding="utf-8"))
            if result.get("input_frame") != "camera" or result.get("input_unit") != "m":
                raise RuntimeError("GraspNet 결과의 camera/m 입력 계약이 다릅니다")
            raw_candidates = result["candidates"]
        except (OSError, ValueError, KeyError) as exc:
            raise RuntimeError("GraspNet 결과 형식이 올바르지 않습니다") from exc

    return [candidate for raw in raw_candidates
            if (candidate := _base_candidate(raw, T_base_camera_mm, T_graspnet_tcp_mm)) is not None]

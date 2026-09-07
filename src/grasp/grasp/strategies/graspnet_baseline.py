"""GraspNet-baseline Docker adapter.

GraspNet 공식 demo는 RGB-D 카메라 좌표계의 meter point cloud를 입력으로 쓰고, grasp의
translation/rotation도 그 camera frame으로 돌려준다. ROS grasp 노드는 기존 PCA용 base-mm
point cloud도 유지하되, 이 전략에만 context의 camera-mm points와 base2camera를 제공한다.
컨테이너는 camera frame 추론만 하고, 이 파일이 `T_base_camera_mm @ T_camera_graspnet_mm
@ T_graspnet_tcp_mm`으로 최종 TCP Pose를 base-mm로 단 한 번 변환한다.

추론 요청 경로는 둘이다. `endpoint`가 설정돼 있으면 **상주 추론 서버**(compose의 graspnet
서비스)에 HTTP로 보내고, 비어 있으면 예전처럼 `docker run`으로 1회용 컨테이너를 띄운다.
서버 쪽이 기본이자 권장 경로다 — 1회용은 호출마다 모델 로드를 다시 해서 추론 하나가
수십~수백 초 걸리고, grasp 노드에 Docker socket/CLI를 요구해 compose 컨테이너 안에서는
아예 못 돈다. `docker run` 경로는 grasp 노드를 호스트에서 네이티브로 돌릴 때를 위해 남겨둔다.
"""
import io
import json
import shutil
import subprocess
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import numpy as np

from .exceptions import InferenceBusy


class _NoUprightCandidate(RuntimeError):
    """추론은 됐지만 임계각 안에 드는 후보가 없다. node가 사유를 로그로 남길 수 있게
    일반 실패와 구분한다 — 이 경우는 "추론 불가"가 아니라 "쓸 만한 자세가 없음"이다."""

STRATEGY = "graspnet_baseline"
_CHECKPOINT_MOUNT = "/checkpoint.tar"

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


# base 좌표계에서 "아래로 똑바로" 내려가는 방향. 접근축을 이 벡터와 비교해 기울기를 잰다.
_BASE_DOWN = np.array([0.0, 0.0, -1.0])


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
                     grasp_depth_mm: float) -> tuple[np.ndarray, float]:
    """GraspNet이 준 **자세는 유지**하고, 위치와 개폭은 실제 포인트클라우드로 다시 잡는다.

    **왜 필요한가.** GraspNet은 원래 장면 전체의 포인트클라우드로 학습됐는데 이 시스템은
    물체 하나를 마스크로 잘라낸 클라우드만 준다(node._candidates_for). 그 입력은 모델의
    학습 분포 밖이라 예측이 신뢰할 수 없다 — 2026-09-07 실측:
      * 후보 위치가 물체 클라우드에서 **8~34mm 떨어져** 있었다(= 허공)
      * 폭이 실제와 크게 달랐다(네일: 실측 4.3mm vs GraspNet 57.1mm)
    반면 **자세(접근축/닫힘축)는 쓸 만하다** — 그건 국소 형상에서 나오는 값이라 잘린
    클라우드로도 의미가 있다. 그래서 자세만 쓰고 위치·폭은 측정값으로 확정한다.

    **모든 측정은 그리퍼 중심선 주변의 국소 창에서 한다.** 예전에는 표면을 클라우드
    **전체의 접근축 극값**(`percentile(proj_a, 3)`)으로 잡고 거기서 15mm 밴드를 떴는데,
    접근축이 기울면 그 극값이 파지 중심이 아니라 **먼 모서리**가 된다. 그러면 밴드가
    윗면이 아니라 모서리를 지나는 대각 슬랩이 되어 중심과 폭이 통째로 그리로 끌려갔다.
    2026-09-07 실측(물체는 가만히 둔 채 접근축만 기울여 봄):

        기울기  0도 → 파지점 (321.1, 123.2, 327.1)  폭 97.8mm
        기울기 20도 → 파지점 (300.8, 134.6, 326.6)  폭 58.5mm   (가로 18.1mm 이동)
        기울기 30도 → 파지점 (287.7, 146.0, 323.8)  폭 31.3mm   (가로 35.2mm 이동)

    물체는 움직이지 않았다. GraspNet은 거의 항상 15~30도 기울어진 자세를 내므로 이
    미끄러짐이 매 파지마다 작용했고, 같은 물체의 폭이 프레임마다 72~94mm로 널뛴 것도
    이것이다. 아래처럼 국소 창에서 재면 기울기와 무관하게 같은 값이 나온다.

    계산(모두 후보 자신의 축 기준):
      1) 가로 중심 (닫힘축 c, 나머지축 o) → 클라우드 중앙값. 물체의 중심이다.
      2) 접근축 a → **그 중심선 주변**의 국소 표면 + grasp_depth_mm
      3) 개폭 → 손가락 패드가 지나는 창 안에서 잰 닫힘축 실측 폭
      4) 최종 가로 위치 → 그 창 안의 실측 범위 중앙 (비대칭 물체에서 중앙값보다 정확)
    """
    closing, other, approach = T_base_tcp[:3, 0], T_base_tcp[:3, 1], T_base_tcp[:3, 2]
    proj_c = points_base @ closing
    proj_o = points_base @ other
    proj_a = points_base @ approach

    # 1) 그리퍼 중심선: 물체의 가로 중앙값(이상치에 끌리지 않게 median)
    c0, o0 = float(np.median(proj_c)), float(np.median(proj_o))

    # 2) 그 중심선 주변의 국소 표면. 접근축은 물체를 향하므로 가까운 면이 투영 최소다.
    lateral = np.hypot(proj_c - c0, proj_o - o0)
    a_grasp = _local_surface(proj_a, lateral, _AXIS_RADIUS_MM) + float(grasp_depth_mm)

    # 3) 손가락 패드가 실제로 지나는 창. 여기 있는 점만이 닫을 때 실제로 물리는 재료다.
    #    마스크가 배경으로 새어 나온 점은 깊이가 더 깊어 이 창 밖으로 자연히 빠진다.
    window = ((np.abs(proj_o - o0) <= _PAD_HALF_MM)
              & (np.abs(proj_a - a_grasp) <= _PAD_HALF_MM))
    if int(window.sum()) < 20:
        # 얇거나 성긴 물체 — 창을 접근축 방향으로만 풀어 준다(가로 제한은 유지).
        window = np.abs(proj_o - o0) <= _PAD_HALF_MM
    if int(window.sum()) < 20:
        window = np.ones(len(points_base), dtype=bool)

    # 남은 이상치에 폭이 끌려가지 않게 min/max 대신 백분위수를 쓴다.
    c_lo, c_hi = np.percentile(proj_c[window], [2.0, 98.0])
    o_lo, o_hi = np.percentile(proj_o[window], [2.0, 98.0])
    position = (float(c_lo + c_hi) / 2.0 * closing
                + float(o_lo + o_hi) / 2.0 * other
                + a_grasp * approach)
    return position, float(c_hi - c_lo)


def _select_best(raw_candidates: list, T_base_camera_mm: np.ndarray,
                 T_graspnet_tcp_mm: np.ndarray, threshold_deg: float,
                 max_deg: float, step_deg: float,
                 camera_offset_mm=(0.0, 0.0, 0.0),
                 points_base=None, refine_depth_mm: float = 8.0) -> tuple[dict | None, dict]:
    """camera frame GraspNet 후보들을 base로 한 번에 옮기고, **접근축이 수직에 가까운**
    후보만 남겨 점수가 가장 높은 하나를 돌려준다. (선택된 후보, 진단정보)를 반환한다.

    **왜 base로 옮긴 뒤에 각도를 재는가.** GraspNet의 접근축은 `rotation_matrix[:, 0]`인데
    그건 **카메라 좌표** 기준이다. 카메라가 손목에 달려 있어(eye-in-hand) 로봇 자세마다
    카메라가 기울어지므로, 카메라 기준 각도는 "작업대에 대해 수직인가"와 아무 관계가 없다.
    base로 옮긴 뒤 `(0, 0, -1)`과 비교해야 의미가 있다.

    **왜 기울기를 거르는가.** 실물 5회(2026-09-07): 후보 기울기가 34~64도로 나왔고 그중
    64.4도와 48.7도는 **IK가 안 풀려 이동 자체가 실패**했다. 나머지도 평평한 물체를 옆에서
    찌르는 자세라 RG2로 물리적으로 물기 어려웠다. GraspNet은 후보를 여러 개 내므로,
    그중 수직에 가까운 것을 고르면 같은 추론 결과로 실행 가능한 파지를 얻을 수 있다.

    **넓혀가며 재시도하는 이유.** 임계각을 좁게 잡으면 어떤 장면에서는 후보가 0개가 되어
    그 물체를 아예 못 집는다. 좁은 값부터 시작해 필요한 만큼만 넓히면, 가능한 한 수직에
    가까운 후보를 쓰되 "후보 없음"으로 끝나지는 않는다.

    각도는 `abs()`로 재지 **않는다** — 그러면 아래에서 위로 찌르는 자세(작업대를 뚫는
    방향)가 0도로 통과한다. `(0,0,-1)`과의 내적을 그대로 쓴다.
    """
    diagnostics: dict = {"raw_count": len(raw_candidates)}
    if not raw_candidates:
        return None, diagnostics

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
        return None, diagnostics

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

    # --- 임계각을 넓혀가며 통과 후보 찾기 -----------------------------------
    used_deg = None
    keep = np.zeros(len(index), dtype=bool)
    limit = float(threshold_deg)
    while limit <= float(max_deg) + 1e-9:
        keep = angle_deg <= limit
        if keep.any():
            used_deg = limit
            break
        limit += float(step_deg)
    if used_deg is None:
        diagnostics["used_threshold_deg"] = None
        return None, diagnostics
    diagnostics["used_threshold_deg"] = round(used_deg, 1)
    diagnostics["passed_count"] = int(keep.sum())

    # --- 통과 후보 중 최고 점수 하나 ----------------------------------------
    kept = np.flatnonzero(keep)
    best = int(kept[int(np.argmax(score[kept]))])
    T_best = T_base_tcp[best]
    chosen_width_mm = float(width_mm[best])
    if points_base is not None and len(points_base) >= 30:
        position, chosen_width_mm = _refine_on_cloud(T_best, points_base, refine_depth_mm)
        diagnostics["refined_from_mm"] = [round(float(v), 1) for v in T_best[:3, 3]]
        diagnostics["refined_shift_mm"] = round(
            float(np.linalg.norm(position - T_best[:3, 3])), 1)
        diagnostics["graspnet_width_mm"] = round(float(width_mm[best]), 1)
        T_best = T_best.copy()
        T_best[:3, 3] = position
    qx, qy, qz, qw = _quaternion_from_matrix(T_best[:3, :3])
    diagnostics.update({
        "chosen_angle_deg": round(float(angle_deg[best]), 1),
        "chosen_score": round(float(score[best]), 3),
        "graspnet_translation_cam_mm": [round(float(v), 2) for v in trans_mm[best]],
        "graspnet_depth_mm": round(float(depth_mm[best]), 1),
        "graspnet_approach_cam": [round(float(v), 3) for v in rot_cam[best][:, 0]],
        "T_graspnet_tcp_translation_mm": [round(float(v), 2) for v in T_graspnet_tcp_mm[:3, 3]],
        "offset_frame": "GraspNet gripper frame (X=접근, Y=닫힘, Z=나머지)",
        "graspnet_point_base_mm": [round(float(v), 2) for v in T_base_graspnet[best][:3, 3]],
        "tcp_target_base_mm": [round(float(v), 2) for v in T_best[:3, 3]],
        "tcp_approach_axis_base": [round(float(v), 3) for v in T_best[:3, 2]],
        "camera_offset_mm": [round(float(v), 1) for v in offset_cam],
        "camera_axis_base": [round(float(v), 3) for v in camera_axis_base / norm_axis],
    })
    return {
        "debug": diagnostics,
        "pose": {
            "position": {"x": float(T_best[0, 3]), "y": float(T_best[1, 3]),
                         "z": float(T_best[2, 3])},
            "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
        },
        "width_mm": chosen_width_mm,
        "score": float(np.clip(score[best], 0.0, 1.0)),
        "strategy": STRATEGY,
    }, diagnostics


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


def _infer_via_docker_run(points_cam_mm: np.ndarray, params: dict) -> list:
    """1회용 컨테이너 경로(fallback). grasp 노드가 호스트에서 네이티브로 돌 때만 쓴다 —
    compose의 grasp 컨테이너에는 Docker socket/CLI가 없다(src/grasp/Dockerfile 주석)."""
    checkpoint = Path(str(params.get("checkpoint_path", ""))).expanduser()
    if not checkpoint.is_file():
        raise RuntimeError(
            "graspnet_baseline.checkpoint_path가 없습니다. 테스트 PC에서 "
            "setup_graspnet_runtime.sh를 실행해 checkpoint.tar를 준비하세요 "
            "(또는 endpoint를 설정해 상주 서버를 쓰세요)")
    device = str(params.get("device", "cuda:0"))
    if not device.startswith("cuda"):
        raise RuntimeError("GraspNet-baseline은 CUDA device가 필요합니다")

    image = str(params.get("image", "piece-picking-graspnet-baseline:0.2.0"))
    _require_image(image)
    num_points = int(params.get("num_points", 20000))
    max_candidates = int(params.get("max_candidates", 5))
    timeout_s = float(params.get("timeout_s", 60.0))
    min_width_m = float(params.get("min_width_mm", 5.0)) / 1000.0
    max_width_m = float(params.get("max_opening_mm", 110.0)) / 1000.0

    container_name = f"graspnet-baseline-{uuid.uuid4().hex[:12]}"
    with tempfile.TemporaryDirectory(prefix="graspnet_baseline_") as temp_dir:
        temp = Path(temp_dir)
        np.savez_compressed(temp / "input.npz",
                            points_cam_m=points_cam_mm.astype(np.float32) / 1000.0)
        command = [
            "docker", "run", "--rm", "--gpus", "all", "--name", container_name,
            "-v", f"{temp}:/io",
            "-v", f"{checkpoint.resolve()}:{_CHECKPOINT_MOUNT}:ro",
            image, "/io/input.npz", "/io/output.json", _CHECKPOINT_MOUNT, device,
            str(num_points), str(min_width_m), str(max_width_m), str(max_candidates),
        ]
        try:
            run = subprocess.run(command, text=True, capture_output=True, timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            # subprocess timeout은 `docker` CLI(클라이언트) 프로세스만 죽인다 — 컨테이너
            # 본체는 데몬에서 계속 돌며 GPU 메모리를 쥔 채 고아로 남는다(2026-09-05 실물
            # 확인: 컨테이너 20개 넘게 쌓여 GPU를 다 먹고 perception까지 OOM으로 죽음).
            # --name으로 명시적으로 죽여야 실제로 회수된다.
            subprocess.run(["docker", "kill", container_name], capture_output=True, timeout=10.0)
            raise RuntimeError(f"GraspNet 추론 timeout ({timeout_s:.0f}s)") from exc
        output = temp / "output.json"
        if run.returncode or not output.is_file():
            detail = (run.stderr or run.stdout).strip().splitlines()
            raise RuntimeError(f"GraspNet container 실행 실패: {detail[-1] if detail else run.returncode}")
        try:
            result = json.loads(output.read_text(encoding="utf-8"))
            if result.get("input_frame") != "camera" or result.get("input_unit") != "m":
                raise RuntimeError("GraspNet 결과의 camera/m 입력 계약이 다릅니다")
            return result["candidates"]
        except (OSError, ValueError, KeyError) as exc:
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
        endpoint = str(params.get("endpoint", "") or "").strip()
        if endpoint:
            raw_candidates = _infer_via_endpoint(endpoint, points_cam_mm, params)
        else:
            raw_candidates = _infer_via_docker_run(points_cam_mm, params)
    finally:
        _INFERENCE_LOCK.release()

    best, diagnostics = _select_best(
        raw_candidates, T_base_camera_mm, T_graspnet_tcp_mm,
        float(params.get("approach_angle_threshold_deg", 15.0)),
        float(params.get("approach_angle_max_deg", 45.0)),
        float(params.get("approach_angle_step_deg", 5.0)),
        _camera_offset(params),
        np.asarray(points_base, dtype=float) if points_base is not None else None,
        float(params.get("refine_grasp_depth_mm", 8.0)))
    if best is None:
        # 추론은 됐는데 임계각 안에 드는 후보가 없다. 빈 리스트로 돌려주면 node가
        # "후보 없음"으로 발행하고 planner는 파지 불가로 읽는데, **왜** 걸러졌는지가
        # 로그에 없으면 추적이 안 된다 — 일반 실패와 구분되는 예외로 알린다.
        max_deg = params.get("approach_angle_max_deg", 45.0)
        raise _NoUprightCandidate(
            f"수직 대비 {max_deg}도 이내 후보 없음 (기울기 {diagnostics.get('angles_deg')})")
    return [best]

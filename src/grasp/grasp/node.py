"""grasp 노드: world_state_raw + instance_masks → 파지 후보를 채워 /world_state 발행.

참조: 인터페이스_정의서.md 2.0절(릴레이 구조), 3.1절(GraspCandidate)

한 관측의 처리 순서:

    /perception/world_state_raw + /perception/instance_masks (같은 stamp)
      + 그 시각에 가장 가까운 depth 프레임
      → 물체별 마스크 backprojection (카메라 좌표계 mm)
      → eye-in-hand 변환으로 base 좌표계
      → 전략(strategies/)이 파지 후보 생성
      → grasp_candidates를 채워 /world_state 발행

**`/world_state`의 발행자는 이 노드 하나다.** perception은 `world_state_raw`까지만 낸다 —
두 노드가 같은 토픽에 발행하면 구독자가 "빈 버전"과 "완성 버전"을 구분할 수 없다(2.0절).

**message_filters를 쓰지 않는다.** `WorldState`/`InstanceMasks`에는 `header`가 없고 `stamp`가
직접 필드로 있어 동기화기가 요구하는 모양이 아니다. 게다가 perception이 두 메시지를 같은
호출에서 같은 stamp로 내보내므로, 근사 동기화가 아니라 **정확 일치**로 짝지을 수 있다.

**어느 쪽이 먼저 도착할지는 보장되지 않는다.** perception이 world_state_raw를 먼저, 그
직후 instance_masks를 publish하지만(node.py), 후자는 마스크 이미지를 실어 직렬화·전송이
더 오래 걸린다 — 실측으로도 world_state_raw가 먼저 도착하는 쪽이 우세했다. 그래서 두 콜백
(`_on_world_state`/`_on_masks`) 모두 "내 짝이 이미 와 있으면 바로 처리, 없으면 내가 대신
대기열에 남는다"는 대칭 구조다. 한쪽만 대기열을 두면(예: masks만) 반대 순서로 도착하는
쪽에서 매번 유실된다.
"""
import bisect
import math
import pathlib
import time

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Point32
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

from sort_msgs.msg import GraspCandidate, InstanceMasks, SelectedGrasp, WorldState

# perception_common(공용 패키지)의 좌표 변환·이미지 변환을 그대로 쓴다.
#
# **복제하지 않는 이유**: eye-in-hand 변환이 두 곳에 있으면 한쪽만 고쳐진 채로 돌 수 있고,
# 그때 어긋나는 것은 로봇이 가는 좌표다(control/control/units.py의 mm 규칙과 같은 논리).
# 캘리브레이션이 `get_current_posx()` 기준으로 풀렸기 때문에 TF로 우회할 수도 없다 —
# 로봇 TF의 루트는 `base_link`이고 그것이 posx의 base와 같은 원점이라는 보장이 없다.
#
# perception/grasp를 각자 컨테이너로 분리하면서(시스템명세서 5.2절) geometry/image_utils/
# paths/robot_pose를 perception_common으로 뺐다 — grasp가 컨테이너 안에서 perception
# 패키지(ultralytics 등 무거운 의존성 포함) 전체를 끌고 올 이유가 없어졌다.
from perception_common import geometry
from perception_common.image_utils import image_to_numpy
from perception_common.paths import find_repo_path
from perception_common.robot_pose import RobotPoseClient

from . import pointcloud_utils, strategies
from .config_utils import asset_path
from .strategies.exceptions import InferenceBusy
from .strategies.graspnet_baseline import _NoUprightCandidate

SCHEMA_VERSION = "1.0.0"
DEPTH_BUFFER_SIZE = 60          # 30fps 기준 2초. 관측 stamp가 조금 뒤처져도 같은 프레임을 찾는다
DEPTH_ENCODINGS = ("16UC1", "mono16")
# 디버그 오버레이용. heuristic_pca는 거의 즉시 처리해 depth와 같은 2초 버퍼로 충분했지만,
# graspnet_baseline은 컨테이너 콜드 스타트 포함 관측당 5~8초가 걸린다(2026-09-05 실물 확인) —
# 그만큼 오래된 컬러 프레임도 버퍼에 남아 있어야 디버그 이미지를 낼 수 있다. 30fps 기준
# 15초치. 프레임당 컬러 이미지 하나를 통째로 들고 있어 depth 버퍼보다 메모리를 더 쓰지만,
# 디버그 뷰 용도라 감수한다.
COLOR_BUFFER_SIZE = 450
COLOR_ENCODINGS = ("bgr8", "rgb8")

# perception_test_live.py --show / ultralytics res.plot()과 눈에 익도록 비슷한 팔레트를 쓴다.
# 물체별로 고정 색을 배정해 프레임 간에도 같은 object_id는 같은 색으로 보이게 한다.
_PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
    (10, 249, 72), (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0),
]


def _color_for(object_id: str) -> tuple[int, int, int]:
    return _PALETTE[hash(object_id) % len(_PALETTE)]


def config_path() -> pathlib.Path:
    """grasp_params.yaml 위치. 설치본과 소스 양쪽에서 찾는다."""
    try:
        from ament_index_python.packages import get_package_share_directory

        installed = pathlib.Path(get_package_share_directory("grasp")) / "config" / "grasp_params.yaml"
        if installed.is_file():
            return installed
    except Exception:
        pass
    return pathlib.Path(__file__).resolve().parents[1] / "config" / "grasp_params.yaml"


def stamp_seconds(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def stamp_key(stamp) -> tuple[int, int]:
    return (stamp.sec, stamp.nanosec)


def observation_key(msg) -> tuple[str, str, int, int]:
    """trace/Observe/stamp를 모두 묶은 내부 join key."""
    return (str(msg.trace_id), str(msg.observation_id), msg.stamp.sec, msg.stamp.nanosec)


class GraspNode(Node):
    def __init__(self):
        super().__init__("grasp_node")

        self.declare_parameter("config_path", "")
        self.declare_parameter("strategy", "")
        self.declare_parameter("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("info_topic", "/camera/color/camera_info")
        self.declare_parameter("color_topic", "/camera/color/image_raw")
        # perception_test_live.py --show와 같은 검출 오버레이 + 파지 후보점을 web이 보여줄 수
        # 있게 발행한다(화면정의서 2.2.4절 개정 — 사용자 요청으로 원본 스트림 대신 이걸 보여준다).
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("debug_topic", "/grasp/debug_image")
        self.declare_parameter("pose_max_age_s", 1.0)
        # 관측 stamp와 depth 프레임 stamp가 이보다 벌어지면 그 관측은 버린다.
        # 다른 순간의 depth로 만든 포인트클라우드는 물체가 그때 있던 자리를 가리킨다.
        self.declare_parameter("max_depth_age_s", 0.5)
        # 디버그 이미지용 컬러 프레임 허용 오차. depth와 달리 3D 재구성 정확도와 무관한
        # 시각화 용도라 훨씬 느슨하다 — graspnet_baseline의 관측당 수 초짜리 지연을
        # 감안한 값이다(COLOR_BUFFER_SIZE 주석 참조).
        self.declare_parameter("debug_color_max_age_s", 15.0)

        path = pathlib.Path(self.get_parameter("config_path").value or config_path())
        with path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        self._config = config
        self._assets = config.get("assets") or {}
        self._strategy_name = (self.get_parameter("strategy").value
                               or (config.get("strategy") or {}).get("name")
                               # 설정에도 없을 때의 최후 기본값. grasp_params.yaml의
                               # strategy.name과 같아야 한다 — 셋(런치 인자/설정/여기)이
                               # 어긋나면 실행 방식에 따라 다른 전략이 돈다.
                               or "graspnet_baseline")
        self._plan = strategies.get(self._strategy_name)
        self._strategy_params = {
            **(config.get("gripper") or {}),
            **(config.get(self._strategy_name) or {}),
        }
        self._pointcloud_params = config.get("pointcloud") or {}
        # graspnet_baseline이 실제로 추론에 넣은 물체 단일 포인트클라우드를 남기는 곳.
        # fine-tuning에는 grasp_pose+성공/실패만으로 부족하고 그 판정의 근거인 입력 자체가
        # 필요하다 — perception._observations_dir(D-6)와 같은 원칙.
        self._point_clouds_dir = find_repo_path("data", env_var="DATA_DIR") / "point_clouds"
        self._pose_max_age_s = float(self.get_parameter("pose_max_age_s").value)
        self._max_depth_age_s = float(self.get_parameter("max_depth_age_s").value)
        self._debug_color_max_age_s = float(self.get_parameter("debug_color_max_age_s").value)

        calibration = asset_path(self._assets, "calibration_path")
        self._gripper2camera = geometry.load_handeye(calibration)
        self._intrinsics = None
        self._depth_frames: list[tuple[float, np.ndarray, str]] = []
        self._color_frames: list[tuple[float, np.ndarray, str]] = []
        # object_id → 그 물체 1순위 후보의 좌표 변환 중간값(graspnet_baseline만 채운다).
        # 좌표 오차를 쫓을 때 어느 단계에서 틀어졌는지 보려면 최종값만으로는 부족하다.
        self._chain_debug: dict[str, dict] = {}
        # object_id → (전략이 낸 후보 수, 발행한 후보 수). 아래 [후보수] 로그가 쓴다.
        self._candidate_counts: dict[str, tuple[int, int]] = {}
        self._pending_masks: dict[tuple[str, str, int, int], InstanceMasks] = {}
        self._pending_worlds: dict[tuple[str, str, int, int], WorldState] = {}
        self._publish_debug = bool(self.get_parameter("publish_debug_image").value)
        self._last_debug_snapshot = None

        callbacks = ReentrantCallbackGroup()
        image_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, self.get_parameter("depth_topic").value,
                                 self._on_depth, image_qos, callback_group=callbacks)
        self.create_subscription(CameraInfo, self.get_parameter("info_topic").value,
                                 self._on_info, 10, callback_group=callbacks)
        self.create_subscription(InstanceMasks, "/perception/instance_masks",
                                 self._on_masks, 10, callback_group=callbacks)
        self.create_subscription(WorldState, "/perception/world_state_raw",
                                 self._on_world_state, 10, callback_group=callbacks)
        if self._publish_debug:
            self.create_subscription(Image, self.get_parameter("color_topic").value,
                                     self._on_color, image_qos, callback_group=callbacks)
            self._debug_pub = self.create_publisher(
                Image, self.get_parameter("debug_topic").value, 1)

        self._pub = self.create_publisher(WorldState, "/world_state", 10)
        self._pose_client = RobotPoseClient(self, callback_group=callbacks)

        # control이 **실제로 실행하기로 고른** 후보. 디버그 오버레이에서 그 하나만 다른
        # 색으로 그린다 — 예전에는 1순위(rank 0)를 초록으로 칠했는데, 이제 실행할 후보는
        # control이 개폭·IK·관절·안전을 보고 고르므로 1순위가 아닐 수 있다. 화면이
        # 실행과 다른 것을 강조하면 안 된다.
        # 최신 하나만 의미가 있고 늦게 붙어도 받아야 하므로 pick_server와 같은
        # transient local QoS를 쓴다.
        self._selected_grasp: tuple[float, str, object] | None = None
        self.create_subscription(
            SelectedGrasp, "/control/selected_grasp", self._on_selected_grasp,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
            callback_group=callbacks)

        self.get_logger().info(
            f"grasp 기동 완료 (전략 {self._strategy_name}, 설정 {path})")

    # --- 입력 ---------------------------------------------------------------
    def _on_selected_grasp(self, msg: SelectedGrasp) -> None:
        # candidate_id가 비면 "고른 것이 없다"(전 후보 탈락)는 뜻이라 강조를 지운다.
        self._selected_grasp = ((time.monotonic(), msg.observation_id, msg.candidate)
                                if msg.candidate.candidate_id else None)
        # on-demand 화면은 다음 관측을 기다리지 않는다. 방금 고른 후보를 같은 frozen
        # observation 이미지에 다시 그려 selected 표시만 갱신한다.
        if self._publish_debug and self._last_debug_snapshot is not None:
            world, mask_by_id, base2camera = self._last_debug_snapshot
            if msg.observation_id == world.observation_id:
                self._publish_debug_image(world, mask_by_id, base2camera)

    def _on_info(self, msg: CameraInfo) -> None:
        if msg.k[0] <= 0.0 or msg.k[4] <= 0.0:
            self._intrinsics = None
            self.get_logger().warning("CameraInfo K가 유효하지 않다", throttle_duration_sec=5.0)
            return
        self._intrinsics = {"fx": msg.k[0], "fy": msg.k[4], "cx": msg.k[2], "cy": msg.k[5],
                            "width": msg.width, "height": msg.height,
                            "frame_id": msg.header.frame_id}

    def _on_depth(self, msg: Image) -> None:
        if msg.encoding not in DEPTH_ENCODINGS:
            self.get_logger().warning(
                f"지원하지 않는 aligned depth encoding: {msg.encoding}",
                throttle_duration_sec=5.0)
            return
        frame = (stamp_seconds(msg.header.stamp), image_to_numpy(msg), msg.header.frame_id)
        index = bisect.bisect_right([item[0] for item in self._depth_frames], frame[0])
        self._depth_frames.insert(index, frame)
        if len(self._depth_frames) > DEPTH_BUFFER_SIZE:
            self._depth_frames.pop(0)

    def _on_color(self, msg: Image) -> None:
        if msg.encoding not in COLOR_ENCODINGS:
            self.get_logger().warning(
                f"디버그 이미지: 지원하지 않는 컬러 encoding {msg.encoding}",
                throttle_duration_sec=10.0)
            return
        frame = (stamp_seconds(msg.header.stamp), image_to_numpy(msg), msg.header.frame_id)
        index = bisect.bisect_right([item[0] for item in self._color_frames], frame[0])
        self._color_frames.insert(index, frame)
        if len(self._color_frames) > COLOR_BUFFER_SIZE:
            self._color_frames.pop(0)

    def _nearest_color(self, when: float) -> tuple[np.ndarray, str] | None:
        """`when`에 가장 가까운 컬러 프레임. `_nearest_depth`와 같은 탐색 방식이지만
        허용 오차는 별도다 — 디버그 이미지는 3D 재구성이 아니라 시각화라 depth만큼
        엄격할 필요가 없다(debug_color_max_age_s 선언부 참조)."""
        if not self._color_frames:
            return None
        times = [item[0] for item in self._color_frames]
        index = bisect.bisect_left(times, when)
        best, best_gap = None, float("inf")
        for candidate in (index - 1, index, index + 1):
            if 0 <= candidate < len(times):
                gap = abs(times[candidate] - when)
                if gap < best_gap:
                    best, best_gap = candidate, gap
        if best is None or best_gap > self._debug_color_max_age_s:
            return None
        _, frame, frame_id = self._color_frames[best]
        return frame, frame_id

    def _on_masks(self, msg: InstanceMasks) -> None:
        key = observation_key(msg)
        world = self._pending_worlds.pop(key, None)
        if world is not None:
            self._process(world, msg)
            return
        self._pending_masks[key] = msg
        if len(self._pending_masks) > 10:
            self._pending_masks.pop(next(iter(self._pending_masks)))

    def _nearest_depth(self, when: float) -> tuple[np.ndarray, str] | None:
        """`when`에 가장 가까운 depth 프레임. 너무 멀면 None.

        "가장 최근"이 아니라 "가장 가까운"이다 — 관측 stamp는 카메라 취득 시각이라 지금보다
        조금 과거이고, 그 사이 프레임이 여러 장 들어와 있다.
        """
        if not self._depth_frames:
            return None
        times = [item[0] for item in self._depth_frames]
        index = bisect.bisect_left(times, when)
        best, best_gap = None, float("inf")
        for candidate in (index - 1, index, index + 1):
            if 0 <= candidate < len(self._depth_frames):
                gap = abs(self._depth_frames[candidate][0] - when)
                if gap < best_gap:
                    best, best_gap = candidate, gap
        if best is None or best_gap > self._max_depth_age_s:
            return None
        _, depth, frame_id = self._depth_frames[best]
        return depth, frame_id

    # --- 한 관측 ------------------------------------------------------------
    def _on_world_state(self, world: WorldState) -> None:
        key = observation_key(world)
        masks = self._pending_masks.pop(key, None)
        if masks is not None:
            self._process(world, masks)
            return
        self._pending_worlds[key] = world
        if len(self._pending_worlds) > 10:
            self._pending_worlds.pop(next(iter(self._pending_worlds)))

    def _process(self, world: WorldState, masks: InstanceMasks) -> None:
        blockers = self._blockers(world, masks)
        if blockers:
            # 후보를 못 채운 world_state를 내보내지 않는다. planner는 `/world_state`가
            # 완성본이라는 전제로 파지 자세를 고르므로(2.0절), 빈 후보를 실어 보내면
            # "파지 후보가 없는 물체"로 읽혀 멀쩡한 물체가 거부된다.
            self.get_logger().warning(f"건너뜀: {', '.join(blockers)}",
                                      throttle_duration_sec=5.0)
            return

        depth, depth_frame_id = self._nearest_depth(stamp_seconds(world.stamp))
        base2gripper = geometry.posx_to_matrix(self._pose_client.posx(self._pose_max_age_s))
        T_base_camera_mm = base2gripper @ self._gripper2camera
        mask_by_id = {object_id: image for object_id, image in zip(masks.object_ids, masks.masks)}

        filled = 0
        for obj in world.objects:
            image = mask_by_id.get(obj.object_id)
            if image is None:
                continue
            try:
                candidates = self._candidates_for(obj, image, depth, depth_frame_id,
                                                  T_base_camera_mm, world.trace_id,
                                                  world.observation_id)
            except InferenceBusy:
                # busy는 파지 실패가 아니다. 빈 후보 WorldState를 내보내 planner가 정상 물체를
                # 거부하지 않게 하고, web은 마지막으로 완성된 관측을 계속 보여준다.
                self.get_logger().info(
                    "GraspNet 추론 진행 중: 이번 관측 publish 보류",
                    throttle_duration_sec=2.0)
                return
            obj.grasp_candidates = candidates
            if candidates:
                d = self._chain_debug.get(obj.object_id)
                if d:
                    self.get_logger().info(
                        f"[좌표체인] {obj.object_id} "
                        f"GraspNet translation(camera,mm)={d['graspnet_translation_cam_mm']} "
                        f"접근축(camera)={d['graspnet_approach_cam']} "
                        f"depth={d.get('graspnet_depth_mm')}mm(접근축 전진 반영) "
                        f"→ base 파지점={d['graspnet_point_base_mm']} "
                        f"| offset={d['T_graspnet_tcp_translation_mm']} ({d['offset_frame']}) "
                        f"→ TCP 목표(base,mm)={d['tcp_target_base_mm']} "
                        f"접근축(base)={d['tcp_approach_axis_base']} "
                        f"| 클라우드 되잡기 {d.get('refined_shift_mm')}mm "
                        f"(폭 {d.get('graspnet_width_mm')}→실측) "
                        f"| 기울기 {d.get('chosen_angle_deg')}deg "
                        f"(후보 {d.get('angles_deg')} 중 hard 상한 "
                        f"{d.get('hard_max_deg')}deg로 {d.get('passed_count')}개 통과, "
                        f"예전 {d.get('legacy_max_deg')}deg 기준이면 "
                        f"{d.get('legacy_pass_count')}개)",
                        throttle_duration_sec=5.0)
            filled += bool(candidates)

        self._pub.publish(world)
        # **후보 수를 물체별로 남긴다.** Top-K가 어디서 줄어드는지 추적하려면 발행 시점의
        # 개수가 로그에 있어야 한다(2026-09-07). raw는 전략이 돌려준 수,
        # published는 world_state에 실제로 실린 수다 — 둘이 다르면 노드가 깎은 것이다.
        detail = ", ".join(
            f"{oid}: raw={c['raw']} valid={c['valid']} "
            + (f"old_angle_pass_{c['legacy_deg']:.0f}={c['legacy']} "
               if c.get('legacy') is not None and c.get('legacy_deg') is not None else "")
            + (f"new_angle_pass_{c['hard_deg']:.0f}={c['passed']} "
               if c.get('hard_deg') is not None else f"filtered={c['passed']} ")
            + (f"검사={c['examined']} " if c.get('examined') is not None else "")
            + (f"geom탈락={dict(c['geom_rejects'])} " if c.get('geom_rejects') else "")
            + f"published={c['published']}"
            + (f" angles={c['angles']}" if c.get('angles') else "")
            for oid, c in self._candidate_counts.items())
        total_published = sum(c["published"] for c in self._candidate_counts.values())
        self.get_logger().info(
            f"[후보수] {len(world.objects)}개 중 {filled}개에 파지 후보 생성 "
            f"({self._strategy_name}) | 총 published={total_published}"
            + (f" | {detail}" if detail else ""), throttle_duration_sec=2.0)
        self._candidate_counts.clear()

        if self._publish_debug:
            self._last_debug_snapshot = (world, mask_by_id, T_base_camera_mm.copy())
            self._publish_debug_image(world, mask_by_id, T_base_camera_mm)

    def _blockers(self, world: WorldState, masks: InstanceMasks) -> list[str]:
        blockers = []
        if self._intrinsics is None:
            blockers.append("camera_info 미수신")
        if self._nearest_depth(stamp_seconds(world.stamp)) is None:
            blockers.append("관측 시각에 맞는 depth 프레임 없음")
        if self._pose_client.posx(self._pose_max_age_s) is None:
            blockers.append("최신 TCP 자세 없음")
        return blockers

    def _save_point_cloud(self, trace_id: str, observation_id: str, object_id: str,
                         points_cam_mm: np.ndarray, T_base_camera_mm: np.ndarray) -> str:
        """GraspNet에 실제로 넣은 물체 단일 포인트클라우드(camera frame, mm)를 파일로 남긴다.

        **왜 필요한가.** execution_logs에 이미 쌓이는 grasp_pose+result(성공/실패)만으로는
        GraspNet fine-tuning이 안 된다 — PointNet++ 백본은 포인트클라우드 자체로 학습하는데,
        그 판정의 근거였던 입력이 지금까지 어디에도 남지 않았다(추론 서버에 한 번 던지고
        버려짐). T_base_camera_mm도 함께 남긴다 — eye-in-hand라 관측마다 카메라 자세가
        달라, 나중에 base 좌표와 맞춰 보려면 그 순간의 변환이 있어야 한다.

        perception._save_observation_image(D-6)와 같은 원칙 — 실패해도 추론 자체를
        막지 않는다(로그만 남기고 빈 문자열 반환).
        """
        if not trace_id or not observation_id:
            return ""
        try:
            directory = self._point_clouds_dir / trace_id
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{observation_id}_{object_id}.npz"
            np.savez_compressed(
                path,
                points_cam_mm=points_cam_mm.astype(np.float32),
                T_base_camera_mm=T_base_camera_mm.astype(np.float64))
            return str(path.relative_to(self._point_clouds_dir.parent.parent))
        except OSError as e:
            self.get_logger().warning(f"포인트클라우드 저장 실패({e}) — 추론은 계속한다")
            return ""

    def _candidates_for(self, obj, mask_image: Image, depth: np.ndarray, depth_frame_id: str,
                        T_base_camera_mm: np.ndarray, trace_id: str,
                        observation_id: str) -> list[GraspCandidate]:
        object_id = obj.object_id
        mask = image_to_numpy(mask_image) > 0
        if mask.shape != depth.shape:
            self.get_logger().warning(
                f"마스크 {mask.shape}와 depth {depth.shape}의 해상도가 다르다",
                throttle_duration_sec=10.0)
            return []
        camera_grid = (self._intrinsics["height"], self._intrinsics["width"])
        if depth.shape != camera_grid:
            self.get_logger().warning("depth와 CameraInfo grid가 다르다", throttle_duration_sec=10.0)
            return []
        if (mask_image.header.frame_id != depth_frame_id
                or depth_frame_id != self._intrinsics["frame_id"]):
            self.get_logger().warning("mask/depth/CameraInfo frame이 다르다", throttle_duration_sec=10.0)
            return []

        points_cam = pointcloud_utils.backproject(mask, depth, self._intrinsics)
        points_cam = pointcloud_utils.remove_outliers(
            points_cam,
            z_percentile=float(self._pointcloud_params.get("z_percentile", 2.0)),
            max_radius_mm=float(self._pointcloud_params.get("max_radius_mm", 250.0)))
        points_base = pointcloud_utils.transform(points_cam, T_base_camera_mm)
        # Place footprint는 perception의 분류/마스크 품질 로직을 바꾸지 않고, grasp가 이미
        # 만든 object-only cloud의 XY convex hull만 전달한다. 보이지 않는 면의 불확실성은
        # box geometry의 별도 wall margin이 담당한다.
        obj.footprint_base_mm.points = []
        if len(points_base) >= 3:
            hull = cv2.convexHull(np.asarray(points_base[:, :2], dtype=np.float32)).reshape(-1, 2)
            if len(hull) >= 3:
                obj.footprint_base_mm.points = [
                    Point32(x=float(x), y=float(y), z=0.0) for x, y in hull
                ]

        min_points = int(self._strategy_params.get("min_points", 0))
        if min_points and len(points_base) < min_points:
            self.get_logger().warning(
                f"{self._strategy_name} 유효 point 부족: {len(points_base)} < {min_points}",
                throttle_duration_sec=10.0)
            return []

        try:
            candidates = self._plan(
                points_base, self._strategy_params,
                context={"points_cam_mm": points_cam, "T_base_camera_mm": T_base_camera_mm})
        except InferenceBusy:
            # busy는 `_process`가 따로 처리한다(publish 보류) — 여기서 삼키면 RuntimeError의
            # 서브클래스라 아래 handler에 잡혀 "추론 불가" 에러로 오인되고 빈 후보가 나간다.
            raise
        except _NoUprightCandidate as exc:
            # 추론은 됐고 후보도 나왔는데 전부 너무 기울어 실행 불가였던 경우.
            # "추론 불가"와 구분해야 원인 추적이 된다(전자는 GPU/서버 문제, 이건 자세 문제).
            self.get_logger().warning(f"{self._strategy_name} 쓸 만한 자세 없음: {exc}",
                                      throttle_duration_sec=5.0)
            return []
        except RuntimeError as exc:
            self.get_logger().error(f"{self._strategy_name} 추론 불가: {exc}",
                                    throttle_duration_sec=10.0)
            return []
        if self._strategy_name == "graspnet_baseline" and candidates:
            point_cloud_path = self._save_point_cloud(
                trace_id, observation_id, object_id, points_cam, T_base_camera_mm)
            if point_cloud_path:
                for c in candidates:
                    if isinstance(c, dict):
                        c["point_cloud_path"] = point_cloud_path
        # 전략이 중간 변환값을 실어 보냈으면(graspnet_baseline) 물체별로 보관한다 —
        # 아래 _process가 로그로 찍는다. GraspCandidate.msg에는 이 필드가 없으므로
        # 메시지로 변환하기 전에 여기서 빼둬야 한다.
        debug = None
        if candidates and isinstance(candidates[0], dict) and candidates[0].get("debug"):
            debug = candidates[0]["debug"]
            self._chain_debug[object_id] = debug
        # 파이프라인 각 단계의 후보 수. 어디서 줄어드는지 추적하려면 전부 있어야 한다.
        #   raw        추론 서버가 돌려준 원본 수 (서버가 이미 NMS + sort_by_score를 마친 것)
        #   valid      회전행렬·수치가 정상인 것
        #   legacy     **예전 30도 정책이었으면** 통과했을 수 (2.5차 비교용)
        #   passed     새 정책(접근각 hard 상한)을 통과한 것
        #   published  world_state에 실린 것 (= min(passed, top_k))
        # graspnet_baseline만 진단을 실어 보내므로, 없으면(PCA) 전략 반환 수로 채운다.
        stage = {
            "raw": (debug or {}).get("raw_count", len(candidates)),
            "valid": (debug or {}).get("valid_count", len(candidates)),
            "passed": (debug or {}).get("passed_count", len(candidates)),
            "legacy": (debug or {}).get("legacy_pass_count"),
            # 기하 검사(2.6차)에서 버린 후보. 사유별 수 — 물체에 안 걸친 후보를
            # 물체 중심으로 끌어오는 대신 버리므로, 몇 개가 왜 빠졌는지가 보여야 한다.
            "geom_rejects": (debug or {}).get("geometry_rejects") or {},
            "examined": (debug or {}).get("examined_count"),
            "legacy_deg": (debug or {}).get("legacy_max_deg"),
            "hard_deg": (debug or {}).get("hard_max_deg"),
            # 후보별 접근각(도). 각도 정책을 튜닝하려면 "몇 개가 걸렸나"만으로는 부족하고
            # **어디에 몰려 있나**를 봐야 한다 — 물티슈처럼 전부 60~77도인 물체가 있다.
            "angles": [round(float(c.get("approach_angle_deg", 0.0) or 0.0), 1)
                       for c in candidates
                       if isinstance(c, dict) and c.get("approach_angle_deg") is not None],
        }
        before_depth_fit = [
            list((c.get("pose") or {}).get("position", {}).values())
            if isinstance(c, dict) else None
            for c in candidates]
        candidates = [self._fit_grasp_depth(c, obj) for c in candidates]
        self._log_candidate_geometry(object_id, obj, candidates, before_depth_fit)
        # candidate_id는 "<object_id>#<순위>"다. 순위 0이 1순위(점수 최고)지만, **실행할
        # 후보는 control이 고른다**(control/grasp_selection.py — 개폭·IK·관절·최소안전을
        # 보고 랭킹). 로그·웹·planner·control이 같은 후보를 가리킬 수 있어야 Top-K를
        # 내보내는 의미가 생긴다.
        messages = [self._to_msg(c, f"{object_id}#{rank}")
                    for rank, c in enumerate(candidates)]
        stage["published"] = len(messages)
        self._candidate_counts[object_id] = stage
        return messages

    def _log_candidate_geometry(self, object_id, obj, candidates, before_depth_fit) -> None:
        """후보가 **실제로 그 물체에 걸쳐 있는지**를 단계별로 한 줄씩 남긴다 (2026-09-08).

        "후보가 물체 끝단에 잡힌다"를 볼 때 알아야 하는 것은 네 가지가 갈리는 지점이다.
          A. GraspNet raw부터 이상한가        → raw_xyz / raw_w
          B. 되잡기에서 옆으로 끌려갔나       → lat(접근축과 직교한 이동량)
          C. 깊이 맞춤에서 움직였나           → dfit(그 단계의 이동량, 접근축 방향이어야 정상)
          D. 자세는 멀쩡한데 웹 투영만 이상한가 → 이 로그가 정상인데 화면이 이상하면 D다

        끝단 치우침은 `edge`로 본다 — 되잡기의 패드 창이 잡은 중앙이 물체 전체 닫힘축
        중앙에서 얼마나 벗어났는지다. `extent`(물체 전체 폭) 대비 크면 모서리를 문 것이다.
        `win`이 20 미만이면 창이 비어 풀린 경우라 그 후보의 위치·폭을 믿으면 안 된다.

        **로그 전용이다.** 이 숫자들은 GraspCandidate.msg에도 웹에도 나가지 않는다.
        """
        center = getattr(obj, "position_base_mm", None)
        center_xyz = ([float(center.x), float(center.y), float(center.z)]
                      if center is not None else None)
        for rank, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                continue
            debug = candidate.get("geometry_debug") or {}
            position = (candidate.get("pose") or {}).get("position") or {}
            final = [float(position.get(k, 0.0)) for k in ("x", "y", "z")]
            parts = [f"[후보기하] {object_id}#{rank}",
                     f"raw_xyz={debug.get('raw_xyz')}",
                     f"raw_w={debug.get('raw_width_mm')}"]
            if debug.get("refined_shift_mm") is not None:
                parts.append(f"refine={debug['refined_shift_mm']}mm "
                             f"(lat={debug.get('lateral_shift_mm')}mm)")
            prior = before_depth_fit[rank] if rank < len(before_depth_fit) else None
            if prior and len(prior) == 3:
                moved = math.dist([float(v) for v in prior], final)
                parts.append(f"dfit={moved:.1f}mm")
            parts.append(f"final_xyz={[round(v, 1) for v in final]}")
            parts.append(f"w={float(candidate.get('width_mm', 0.0) or 0.0):.1f}mm")
            if debug.get("closing_extent_mm") is not None:
                parts.append(f"extent={debug['closing_extent_mm']:.1f}mm")
                # 2.6차 이후 lat은 **0이어야 정상**이다 — 되잡기가 가로 위치를 안 옮긴다.
                # 0이 아니면 어딘가에서 다시 끌어오고 있다는 뜻이라 바로 눈에 띈다.
                # GraspNet 위치가 클라우드 중심에서 얼마나 벗어나 있었는지 —
                # 되잡기가 그만큼 구제한 것이다(버리는 기준이 아니라 품질 지표).
                parts.append(f"cand_lat={debug.get('candidate_lateral_mm')}mm")
                if debug.get("depth_shift_mm") is not None:
                    parts.append(f"depth_shift={debug['depth_shift_mm']}mm")
                parts.append(f"win={debug.get('window_points')}/{debug.get('cloud_points')}")
                parts.append(f"geom={debug.get('status')}")
            if center_xyz is not None:
                parts.append(f"obj중심거리={math.dist(center_xyz, final):.1f}mm")
            parts.append(f"angle={float(candidate.get('approach_angle_deg', 0.0) or 0.0):.1f}deg")
            self.get_logger().info(" ".join(parts), throttle_duration_sec=0.0)

    # Provisional empirical-pad clearance, not a validated fingertip safety margin.
    # Source repeatability reaches range 3.9mm / std 0.85mm (obj_179);
    # plane p95 ~1.5mm is a trimmed residual, not absolute error. TCP dz=+2.4mm
    # does not prove contact. None of these establishes 2.5mm as a safe bound.
    _SUPPORT_CLEARANCE_MM = 2.5

    # Thin-object compromise: min(2.5, 0.4*h), not guaranteed physical clearance.
    _SUPPORT_CLEARANCE_HEIGHT_FRACTION = 0.4
    # Provisional target for candidates that are too shallow. It does not override a strategy
    # candidate that is already deeper, and is not validated collision geometry.
    _FINGER_REACH_MM = 45.0

    def _fit_grasp_depth(self, candidate: dict, obj) -> dict:
        """Fit the empirical pad reference along the candidate's downward axis.

        Support is estimated as top-height. A shallow candidate is lowered toward
        max(support+clearance, top-reach); an already-deep candidate is preserved unless
        it violates the support floor. This is not a physical fingertip or collision
        guarantee. Unknown/invalid inputs are not fitted.
        """
        height_mm = float(getattr(obj, "height_mm", 0.0) or 0.0)
        if not np.isfinite(height_mm) or height_mm <= 0.0 or not isinstance(candidate, dict):
            self.get_logger().warning(f"[파지깊이] {obj.object_id} skipped: invalid/unknown height or candidate")
            return candidate
        pose = candidate.get("pose") or {}
        position, orientation = pose.get("position"), pose.get("orientation")
        if not position or not orientation:
            self.get_logger().warning(f"[파지깊이] {obj.object_id} skipped: missing pose")
            return candidate

        top_z = float(obj.position_base_mm.z)
        original = np.array([position[k] for k in ("x", "y", "z")], dtype=float)
        quaternion = tuple(float(orientation[k]) for k in ("x", "y", "z", "w"))
        if (not np.isfinite(top_z) or not np.all(np.isfinite(original))
                or not np.all(np.isfinite(quaternion)) or not any(quaternion)):
            self.get_logger().warning(f"[파지깊이] {obj.object_id} skipped: nonfinite/invalid pose")
            return candidate
        approach = geometry.quaternion_to_matrix(*quaternion)[:, 2]
        if not np.all(np.isfinite(approach)) or float(approach[2]) >= -1e-6:
            self.get_logger().warning(f"[파지깊이] {obj.object_id} skipped: axis not finite/downward")
            return candidate

        support_z = top_z - height_mm
        clearance = min(self._SUPPORT_CLEARANCE_MM,
                        height_mm * self._SUPPORT_CLEARANCE_HEIGHT_FRACTION)
        floor_z = support_z + clearance
        current_z = float(original[2])
        reach_z = top_z - self._FINGER_REACH_MM
        old_z = max(min(current_z, max(support_z, reach_z)), support_z)
        target_z = max(min(current_z, max(floor_z, reach_z)), floor_z)
        moved = original + (
            (target_z - current_z) / float(approach[2])) * approach
        moved[2] = target_z  # Avoid roundoff below the numerical floor.
        self.get_logger().info(
            f"[파지깊이] {obj.object_id} empirical_pad "
            f"h={height_mm!r} top_z={top_z!r} support_z=top-h={support_z!r} "
            f"candidate_z={current_z!r} reach={self._FINGER_REACH_MM!r} "
            f"| OLD zero-clearance: max(min(candidate_z, max(support_z+0, top_z-reach)), "
            f"support_z+0)={old_z!r} (legacy deadband: abs(OLD-candidate_z)<0.5; "
            f"returned_z={current_z if abs(old_z - current_z) < 0.5 else old_z!r}) "
            f"| NEW clearance=min({self._SUPPORT_CLEARANCE_MM!r}, "
            f"{self._SUPPORT_CLEARANCE_HEIGHT_FRACTION!r}*h)={clearance!r}; "
            f"max(min(candidate_z, max(support_z+clearance={floor_z!r}, "
            f"top_z-reach={reach_z!r})), support_z+clearance)={target_z!r} "
            f"| fitted_z={float(moved[2])!r} fitted_xyz={tuple(float(v) for v in moved)!r} "
            f"quaternion_xyzw={quaternion!r} unchanged={target_z == current_z}")
        candidate = dict(candidate)
        candidate["pose"] = {**pose, "position": {"x": float(moved[0]),
                                                  "y": float(moved[1]),
                                                  "z": float(moved[2])}}
        return candidate

    # --- 디버그 오버레이 (perception_test_live.py --show와 같은 그림 + 파지 후보) ----------
    def _publish_debug_image(self, world: WorldState, mask_by_id: dict,
                             base2camera: np.ndarray) -> None:
        found = self._nearest_color(stamp_seconds(world.stamp))
        if found is None:
            self.get_logger().warning(
                "디버그 이미지: 관측 시각에 맞는 컬러 프레임 없음", throttle_duration_sec=5.0)
            return
        color, frame_id = found
        vis = color.copy()
        camera2base = np.linalg.inv(base2camera)

        for obj in world.objects:
            box_color = _color_for(obj.object_id)
            mask_image = mask_by_id.get(obj.object_id)
            if mask_image is not None:
                mask = image_to_numpy(mask_image) > 0
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, contours, -1, box_color, 2)
                ys, xs = np.nonzero(mask)
                if xs.size:
                    self._draw_label(vis, (int(xs.min()), int(ys.min())),
                                     f"{obj.class_name} {obj.confidence:.2f}", box_color)
            self._draw_grasp_candidates(
                vis, obj.grasp_candidates, camera2base, world.observation_id)

        self._publish_image(vis, world.stamp, frame_id)

    # 손끝이 물체를 물기 전 대략 이만큼 뒤에서 다가온다고 보고 그린다 — 실제 손가락
    # 길이 데이터는 없어서(GraspCandidate에 안 실림) 순수 시각화 상수다. graspnetAPI의
    # plot_gripper_pro_max가 그리는 "ㄷ"자 그리퍼 스케치와 같은 구조(손끝 두 점 + 그
    # 뒤 몸통)를 우리 좌표계로 다시 그린다.
    _FINGER_LENGTH_MM = 40.0
    _DEFAULT_WIDTH_MM = 60.0

    # control이 고른 후보를 이 시간까지만 강조한다. pick 한 번이 수십 초라 그보다는
    # 길어야 하고, 다음 사이클까지 남아 엉뚱한 후보를 강조하면 안 된다.
    _SELECTED_MAX_AGE_S = 90.0
    # 선택 메시지의 자세와 이만큼 안쪽이면 "같은 후보"로 본다. candidate_id의 순위 부분은
    # 프레임마다 바뀔 수 있어(점수 순위가 흔들린다) id만으로는 못 맞춘다 — 자세로 맞춘다.
    _SELECTED_MATCH_TOL_MM = 5.0

    def _selected_index(self, candidates, observation_id: str) -> int | None:
        """이 후보 목록에서 control이 고른 것의 인덱스. 없으면 None.

        object_id(candidate_id의 '#' 앞부분)가 같고 파지점이 가장 가까운 후보를 고른다.
        서로 다른 물체의 선택이 남아 있어도 강조되지 않게 하려는 것이다.
        """
        if self._selected_grasp is None or not candidates:
            return None
        received, selected_observation_id, selected = self._selected_grasp
        if time.monotonic() - received > self._SELECTED_MAX_AGE_S:
            return None
        if selected_observation_id != observation_id:
            return None
        object_id = selected.candidate_id.split("#")[0]
        best, best_distance = None, self._SELECTED_MATCH_TOL_MM
        for index, candidate in enumerate(candidates):
            if candidate.candidate_id.split("#")[0] != object_id:
                continue
            a, b = candidate.pose.position, selected.pose.position
            distance = float(np.linalg.norm([a.x - b.x, a.y - b.y, a.z - b.z]))
            if distance <= best_distance:
                best, best_distance = index, distance
        return best

    def _draw_grasp_candidates(self, vis: np.ndarray, candidates, camera2base: np.ndarray,
                               observation_id: str) -> None:
        """파지 후보를 base→camera 역변환·투영해 실제 개폭·접근축을 반영한 그리퍼
        스케치로 그린다 — "닫는 축 선 하나"만 그리면 개폭·접근 방향이 안 보여
        PCA 결과와 구분이 안 됐다(2026-09-05, 사용자 피드백).

        전략이 점수 내림차순으로 돌려주므로(heuristic_pca.py / graspnet_baseline.py) 앞에서부터
        그린다. 2026-09-07 이전에는 상위 3개만 그렸는데, Top-K를 유지하도록 바꾸면서
        **받은 후보를 전부** 그린다 — 시각화의 목적이 "선택지가 몇 개나 있는지"를 보는
        것이기 때문이다.

        **초록으로 강조하는 것은 control이 실제로 고른 후보 하나뿐이다**(/control/selected_grasp).
        예전에는 1순위(점수 최고)를 초록으로 칠했는데, 이제 실행할 후보는 control이
        개폭·IK·관절·최소안전을 보고 고르므로 1순위가 아닐 수 있다 — 화면이 실행과 다른
        것을 강조하면 사람이 잘못된 후보를 보고 판단하게 된다. 아직 아무것도 고르지 않은
        상태(첫 pick 전)에서는 전부 주황이다.

        숫자(점수·개폭)는 그리지 않는다 — 화면을 깨끗하게 유지한다(2026-09-07 요구사항).
        """
        selected_index = self._selected_index(candidates, observation_id)
        for rank, candidate in enumerate(candidates):
            p = candidate.pose.position
            point_cam = camera2base @ np.array([p.x, p.y, p.z, 1.0])
            if point_cam[2] <= 0:
                continue
            center = pointcloud_utils.project(point_cam[np.newaxis, :3], self._intrinsics)[0]
            cx, cy = int(round(center[0])), int(round(center[1]))
            if not (0 <= cx < vis.shape[1] and 0 <= cy < vis.shape[0]):
                continue

            q = candidate.pose.orientation
            rotation = geometry.quaternion_to_matrix(q.x, q.y, q.z, q.w)
            closing_axis, approach_axis = rotation[:, 0], rotation[:, 2]
            half_width = 0.5 * float(candidate.gripper_width_mm or self._DEFAULT_WIDTH_MM)
            base_point = np.array([p.x, p.y, p.z])

            # 접근축(approach_axis)은 그리퍼가 물체를 향해 다가가는 방향이라, 그리퍼 몸통은
            # 그 반대(-approach_axis)에 있다 — 손끝(contact)에서 몸통(base)으로 선을 긋는다.
            tip_left = base_point + half_width * closing_axis
            tip_right = base_point - half_width * closing_axis
            body_left = tip_left - self._FINGER_LENGTH_MM * approach_axis
            body_right = tip_right - self._FINGER_LENGTH_MM * approach_axis

            pixels = self._project_points(
                [tip_left, tip_right, body_left, body_right], camera2base)
            if pixels is None:
                continue
            (tlx, tly), (trx, try_), (blx, bly), (brx, bry) = pixels

            is_selected = rank == selected_index
            marker_color = (0, 220, 0) if is_selected else (0, 180, 255)
            # 색만으로는 겹쳤을 때 구분이 어렵다 — 선택된 것만 굵게 그린다.
            thickness = 3 if is_selected else 2
            # 손가락 두 개(손끝→몸통) + 몸통을 잇는 선(팜) = graspnetAPI 그리퍼 스케치의
            # "ㄷ"자와 같은 형태. 손끝에 작은 원을 찍어 실제로 물체를 무는 지점을 강조한다.
            cv2.line(vis, (blx, bly), (brx, bry), marker_color, thickness)   # 팜(base)
            cv2.line(vis, (tlx, tly), (blx, bly), marker_color, thickness)   # 왼손가락
            cv2.line(vis, (trx, try_), (brx, bry), marker_color, thickness)  # 오른손가락
            cv2.circle(vis, (tlx, tly), 4 + thickness - 2, marker_color, -1)
            cv2.circle(vis, (trx, try_), 4 + thickness - 2, marker_color, -1)
            # 전략명(예: [graspnet_baseline]) 라벨은 화면을 깨끗하게 유지하려고 뺐다
            # (2026-09-08 요구사항). score/width/전략은 메시지·로그에 그대로 있다.

    def _project_points(self, points_base: list[np.ndarray],
                        camera2base: np.ndarray) -> list[tuple[int, int]] | None:
        """base 좌표계 점들을 한꺼번에 카메라 픽셀로 투영한다. 하나라도 카메라 뒤쪽
        (point_cam[2]<=0)이면 그리퍼 스케치 전체를 그리지 않는다 — 절반만 그리면
        방향을 오해하기 쉽다."""
        pixels = []
        for point in points_base:
            point_cam = camera2base @ np.append(point, 1.0)
            if point_cam[2] <= 0:
                return None
            u, v = pointcloud_utils.project(point_cam[np.newaxis, :3], self._intrinsics)[0]
            pixels.append((int(round(u)), int(round(v))))
        return pixels

    @staticmethod
    def _draw_label(vis: np.ndarray, origin: tuple[int, int], text: str,
                    color: tuple[int, int, int]) -> None:
        (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x, y = origin
        y = max(y, h + baseline)
        cv2.rectangle(vis, (x, y - h - baseline), (x + w, y), color, -1)
        cv2.putText(vis, text, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (255, 255, 255), 1, cv2.LINE_AA)

    def _publish_image(self, vis: np.ndarray, stamp, frame_id: str) -> None:
        msg = Image()
        msg.header = Header(stamp=stamp, frame_id=frame_id)
        msg.height, msg.width = vis.shape[:2]
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = msg.width * 3
        msg.data = vis.tobytes()
        self._debug_pub.publish(msg)

    @staticmethod
    def _to_msg(candidate: dict, candidate_id: str = "") -> GraspCandidate:
        from geometry_msgs.msg import Pose

        msg = GraspCandidate()
        pose = Pose()
        pose.position.x = candidate["pose"]["position"]["x"]
        pose.position.y = candidate["pose"]["position"]["y"]
        pose.position.z = candidate["pose"]["position"]["z"]
        pose.orientation.x = candidate["pose"]["orientation"]["x"]
        pose.orientation.y = candidate["pose"]["orientation"]["y"]
        pose.orientation.z = candidate["pose"]["orientation"]["z"]
        pose.orientation.w = candidate["pose"]["orientation"]["w"]
        msg.pose = pose
        msg.score = float(candidate["score"])
        msg.strategy = candidate["strategy"]
        msg.candidate_id = candidate_id
        msg.grasp_depth_mm = float(candidate.get("grasp_depth_mm", 0.0) or 0.0)
        msg.gripper_width_mm = float(candidate.get("width_mm") or 0.0)
        msg.point_cloud_path = str(candidate.get("point_cloud_path") or "")
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = GraspNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

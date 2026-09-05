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
import pathlib

import numpy as np
import rclpy
import yaml
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

from sort_msgs.msg import GraspCandidate, InstanceMasks, WorldState

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

SCHEMA_VERSION = "1.0.0"
DEPTH_BUFFER_SIZE = 60          # 30fps 기준 2초. 관측 stamp가 조금 뒤처져도 같은 프레임을 찾는다
DEPTH_ENCODINGS = ("16UC1", "mono16")


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


class GraspNode(Node):
    def __init__(self):
        super().__init__("grasp_node")

        self.declare_parameter("config_path", "")
        self.declare_parameter("strategy", "")
        self.declare_parameter("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("info_topic", "/camera/color/camera_info")
        self.declare_parameter("pose_max_age_s", 1.0)
        # 관측 stamp와 depth 프레임 stamp가 이보다 벌어지면 그 관측은 버린다.
        # 다른 순간의 depth로 만든 포인트클라우드는 물체가 그때 있던 자리를 가리킨다.
        self.declare_parameter("max_depth_age_s", 0.5)

        path = pathlib.Path(self.get_parameter("config_path").value or config_path())
        with path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        self._config = config
        self._assets = config.get("assets") or {}
        self._strategy_name = (self.get_parameter("strategy").value
                               or (config.get("strategy") or {}).get("name")
                               or "heuristic_pca")
        self._plan = strategies.get(self._strategy_name)
        self._strategy_params = {
            **(config.get("gripper") or {}),
            **(config.get(self._strategy_name) or {}),
        }
        checkpoint = asset_path(self._assets, "graspnet_checkpoint_path")
        if checkpoint is not None:
            self._strategy_params["checkpoint_path"] = str(checkpoint)
        self._pointcloud_params = config.get("pointcloud") or {}
        self._pose_max_age_s = float(self.get_parameter("pose_max_age_s").value)
        self._max_depth_age_s = float(self.get_parameter("max_depth_age_s").value)

        calibration = asset_path(self._assets, "calibration_path")
        self._gripper2camera = geometry.load_handeye(calibration)
        self._intrinsics = None
        self._depth_frames: list[tuple[float, np.ndarray, str]] = []
        self._pending_masks: dict[tuple[int, int], InstanceMasks] = {}
        self._pending_worlds: dict[tuple[int, int], WorldState] = {}

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

        self._pub = self.create_publisher(WorldState, "/world_state", 10)
        self._pose_client = RobotPoseClient(self, callback_group=callbacks)

        self.get_logger().info(
            f"grasp 기동 완료 (전략 {self._strategy_name}, 설정 {path})")

    # --- 입력 ---------------------------------------------------------------
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

    def _on_masks(self, msg: InstanceMasks) -> None:
        key = stamp_key(msg.stamp)
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
        key = stamp_key(world.stamp)
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
            candidates = self._candidates_for(image, depth, depth_frame_id, T_base_camera_mm)
            obj.grasp_candidates = candidates
            filled += bool(candidates)

        self._pub.publish(world)
        self.get_logger().info(
            f"{len(world.objects)}개 중 {filled}개에 파지 후보 생성 "
            f"({self._strategy_name})", throttle_duration_sec=2.0)

    def _blockers(self, world: WorldState, masks: InstanceMasks) -> list[str]:
        blockers = []
        if self._intrinsics is None:
            blockers.append("camera_info 미수신")
        if self._nearest_depth(stamp_seconds(world.stamp)) is None:
            blockers.append("관측 시각에 맞는 depth 프레임 없음")
        if self._pose_client.posx(self._pose_max_age_s) is None:
            blockers.append("최신 TCP 자세 없음")
        return blockers

    def _candidates_for(self, mask_image: Image, depth: np.ndarray, depth_frame_id: str,
                        T_base_camera_mm: np.ndarray) -> list[GraspCandidate]:
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
        except RuntimeError as exc:
            self.get_logger().error(f"{self._strategy_name} 추론 불가: {exc}",
                                    throttle_duration_sec=10.0)
            return []
        return [self._to_msg(c) for c in candidates]

    @staticmethod
    def _to_msg(candidate: dict) -> GraspCandidate:
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
        msg.gripper_width_mm = float(candidate.get("width_mm", 0.0))
        msg.strategy = candidate["strategy"]
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

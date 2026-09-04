"""perception 노드: 검출·세그멘테이션 → world_state_raw / instance_masks 발행.

참조: 인터페이스_정의서.md 2.0절(릴레이 구조), 3.2~3.4절(메시지 스키마), 1.1절(단위 mm)

한 프레임의 처리 순서:

    color/depth/camera_info + 현재 TCP 자세
      → YOLO-seg 추론
      → 모델 라벨을 class_name으로 (objects.yaml의 model_labels)
      → 마스크 안쪽 median depth로 카메라 좌표 3D → base 좌표 (eye-in-hand 변환)
      → 속성 조회 (object_attributes → objects.yaml → fallback)
      → object_id 유지 (프레임 간 추적)
      → /perception/world_state_raw + /perception/instance_masks 동시 발행

`grasp_candidates`는 **비운 채로** 낸다. 채우는 것은 grasp의 일이고, 채워진 최종본만
`/world_state`로 나간다(2.0절). 두 노드가 같은 토픽에 발행하면 구독자가 어느 버전을 받았는지
구분할 수 없다.

**두 토픽의 stamp는 반드시 같다.** grasp가 그 값으로 마스크와 물체를 짝짓기 때문이다.
"""
import pathlib

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

from sort_msgs.msg import DetectedObject, InstanceMasks, WorldState

from perception_common import geometry
from perception_common.image_utils import image_to_numpy
from perception_common.paths import find_repo_path
from perception_common.robot_pose import RobotPoseClient

from . import mask_utils
from .attribute_db import AttributeSource
from .tracker import ObjectTracker

SCHEMA_VERSION = "1.0.0"
BASE_FRAME = "base"

# graspable=false 사유. mock 픽스처(data/mock)가 쓰는 값과 같아야 한다.
REASON_DEPTH_INVALID = "depth_invalid"
REASON_MASK_TOO_SMALL = "mask_too_small"
REASON_NO_MASK = "no_mask"


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")

        self.declare_parameter("model_path", "")
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("device", "")
        self.declare_parameter("period_s", 0.5)
        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("info_topic", "/camera/color/camera_info")
        # 마스크 안쪽 depth가 이보다 적게 유효하면 파지 대상에서 뺀다. 투명·반사 물체가
        # 주로 여기 걸리고, 그 물체들은 needs_reobserve로 올라가 능동 재촬영 대상이 된다(FR-03).
        self.declare_parameter("min_depth_valid_ratio", 0.35)
        self.declare_parameter("min_mask_pixels", 200)
        # 이보다 오래된 TCP 자세로는 좌표를 만들지 않는다 (robot_pose.posx 주석 참조)
        self.declare_parameter("pose_max_age_s", 1.0)
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("require_robot_pose", True)

        self._conf = float(self.get_parameter("conf").value)
        self._imgsz = int(self.get_parameter("imgsz").value)
        self._device = self.get_parameter("device").value or None
        self._min_ratio = float(self.get_parameter("min_depth_valid_ratio").value)
        self._min_pixels = int(self.get_parameter("min_mask_pixels").value)
        self._pose_max_age_s = float(self.get_parameter("pose_max_age_s").value)
        self._publish_debug = bool(self.get_parameter("publish_debug_image").value)
        self._require_pose = bool(self.get_parameter("require_robot_pose").value)

        model_path = self.get_parameter("model_path").value or str(
            find_repo_path("models", "best.pt", env_var="PERCEPTION_MODEL"))
        from ultralytics import YOLO      # import 비용이 커서 노드 생성 시점에만 낸다

        self._model = YOLO(model_path)
        self.get_logger().info(
            f"모델 {model_path} (task={self._model.task}, names={self._model.names})")

        self._attributes = AttributeSource()
        self._tracker = ObjectTracker()
        self._gripper2camera = geometry.load_handeye()
        self.get_logger().info(f"hand-eye {geometry.handeye_path()} 적재 완료")

        self._unknown_crops_dir = find_repo_path("data", env_var="DATA_DIR") / "unknown_crops"
        self._saved_unknown_crops: set[str] = set()

        self._color = None
        self._color_stamp = None
        self._depth = None
        self._intrinsics = None
        self._observation_count = 0

        callbacks = ReentrantCallbackGroup()
        image_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, self.get_parameter("color_topic").value,
                                 self._on_color, image_qos, callback_group=callbacks)
        self.create_subscription(Image, self.get_parameter("depth_topic").value,
                                 self._on_depth, image_qos, callback_group=callbacks)
        self.create_subscription(CameraInfo, self.get_parameter("info_topic").value,
                                 self._on_info, 10, callback_group=callbacks)

        self._world_state_pub = self.create_publisher(WorldState, "/perception/world_state_raw", 10)
        self._instance_masks_pub = self.create_publisher(InstanceMasks, "/perception/instance_masks", 10)
        self._debug_pub = self.create_publisher(Image, "/perception/debug_image", 1)

        self._pose_client = RobotPoseClient(self, callback_group=callbacks)
        self.create_timer(float(self.get_parameter("period_s").value), self._tick,
                          callback_group=callbacks)
        self.get_logger().info("perception 기동 완료")

    # --- 입력 ---------------------------------------------------------------
    def _on_color(self, msg: Image) -> None:
        self._color = image_to_numpy(msg)
        # 관측 시각을 함께 들고 있는다. 발행 시각이 아니라 **이 프레임이 찍힌 시각**이
        # WorldState.stamp가 되어야 grasp가 같은 depth 프레임을 다시 찾을 수 있다
        # (ROS 관례이기도 하다 — stamp는 데이터 취득 시각이다).
        self._color_stamp = msg.header.stamp

    def _on_depth(self, msg: Image) -> None:
        self._depth = image_to_numpy(msg)

    def _on_info(self, msg: CameraInfo) -> None:
        self._intrinsics = {"fx": msg.k[0], "fy": msg.k[4], "cx": msg.k[2], "cy": msg.k[5]}

    # --- 한 프레임 ----------------------------------------------------------
    def _tick(self) -> None:
        missing = self._missing_inputs()
        if missing:
            # 입력이 안 갖춰졌을 때 **빈 world_state를 내보내지 않는다.** 빈 목록은
            # "아무것도 없다"는 뜻이고, 그것은 여기서 할 수 있는 주장이 아니다.
            self.get_logger().warning(f"입력 대기 중: {', '.join(missing)}",
                                      throttle_duration_sec=5.0)
            return

        color, depth = self._color, self._depth
        posx = self._pose_client.posx(self._pose_max_age_s)
        base2gripper = geometry.posx_to_matrix(posx) if posx else None

        result = self._model.predict(color, conf=self._conf, imgsz=self._imgsz,
                                     device=self._device, verbose=False)[0]
        detections = self._build_detections(result, color, depth, base2gripper)

        # 관측 시각(컬러 프레임의 stamp). 아직 못 받았으면 현재 시각으로 대신한다.
        stamp = self._color_stamp or self.get_clock().now().to_msg()
        self._observation_count += 1
        # trace_id는 web이 명령 접수 시 발급한다(인터페이스_정의서 1절). perception은 명령과
        # 무관하게 주기 발행하므로 발급받을 trace가 없어 관측 일련번호를 넣는다. web은 명령을
        # 받을 때 자기 trace_id를 새로 만들므로 이 값을 쓰지 않는다.
        trace_id = f"obs-{self._observation_count:06d}"

        self._publish(trace_id, stamp, detections)
        if self._publish_debug:
            self._publish_debug_image(result, stamp)

    def _missing_inputs(self) -> list[str]:
        missing = []
        if self._color is None:
            missing.append("color")
        if self._depth is None:
            missing.append("depth")
        if self._intrinsics is None:
            missing.append("camera_info")
        if self._require_pose and self._pose_client.posx(self._pose_max_age_s) is None:
            missing.append("robot_pose(get_current_posx)")
        return missing

    def _build_detections(self, result, color: np.ndarray, depth: np.ndarray,
                          base2gripper) -> list[dict]:
        count = 0 if result.boxes is None else len(result.boxes)
        shape_hw = color.shape[:2]
        detections: list[dict] = []

        for index in range(count):
            model_label = result.names[int(result.boxes.cls[index])]
            class_name = self._attributes.class_name(model_label)
            confidence = float(result.boxes.conf[index])

            if result.masks is None:
                detections.append(self._detection(class_name, confidence, None, None, 0.0,
                                                  REASON_NO_MASK))
                continue

            mask = mask_utils.resize_mask(
                result.masks.data[index].cpu().numpy() > 0.5, shape_hw)
            point_cam, valid_ratio = mask_utils.mask_3d(mask, depth, self._intrinsics)

            position = None
            if point_cam is not None and base2gripper is not None:
                position = geometry.base_from_camera(point_cam, base2gripper, self._gripper2camera)

            reason = self._not_graspable_reason(mask, point_cam, valid_ratio, position)
            detections.append(self._detection(class_name, confidence, position, mask,
                                              valid_ratio, reason))
            self._maybe_save_unknown_crop(class_name, color, mask)

        return detections

    def _not_graspable_reason(self, mask, point_cam, valid_ratio, position) -> str:
        if int(mask.sum()) < self._min_pixels:
            return REASON_MASK_TOO_SMALL
        if point_cam is None or position is None or valid_ratio < self._min_ratio:
            return REASON_DEPTH_INVALID
        return ""

    @staticmethod
    def _detection(class_name, confidence, position, mask, valid_ratio, reason) -> dict:
        return {
            "class_name": class_name,
            "confidence": confidence,
            "position": position,
            "mask": mask,
            "depth_valid_ratio": valid_ratio,
            "not_graspable_reason": reason,
        }

    # --- 발행 ---------------------------------------------------------------
    def _publish(self, trace_id: str, stamp: Time, detections: list[dict]) -> None:
        object_ids = self._tracker.assign(detections)

        world = WorldState()
        world.schema_version = SCHEMA_VERSION
        world.trace_id = trace_id
        world.stamp = stamp
        world.frame_id = BASE_FRAME

        masks = InstanceMasks()
        masks.schema_version = SCHEMA_VERSION
        masks.trace_id = trace_id
        masks.stamp = stamp

        header = self._mask_header(stamp)
        needs_reobserve = []

        for object_id, detection in zip(object_ids, detections):
            attributes = self._attributes.attributes(detection["class_name"])
            graspable = not detection["not_graspable_reason"]

            obj = DetectedObject()
            obj.object_id = object_id
            obj.class_name = detection["class_name"]
            obj.name_ko = attributes["name_ko"]
            obj.confidence = detection["confidence"]
            position = detection["position"] or (0.0, 0.0, 0.0)
            obj.position_base_mm = Point(x=position[0], y=position[1], z=position[2])
            obj.depth_valid_ratio = detection["depth_valid_ratio"]
            obj.graspable = graspable
            obj.not_graspable_reason = detection["not_graspable_reason"]
            obj.mass_g = attributes["mass_g"]
            obj.fragile = attributes["fragile"]
            obj.deformable = attributes["deformable"]
            obj.transparent = attributes["transparent"]
            obj.profile = attributes["profile"]
            obj.attr_source = attributes["attr_source"]
            obj.needs_confirmation = attributes["needs_confirmation"]
            obj.grasp_candidates = []      # grasp가 채운다 (2.0절)
            world.objects.append(obj)

            if not graspable:
                needs_reobserve.append(object_id)

            if detection["mask"] is not None:
                masks.object_ids.append(object_id)
                masks.masks.append(mask_utils.mask_to_image_msg(detection["mask"], header))

        world.needs_reobserve = needs_reobserve
        self._world_state_pub.publish(world)
        self._instance_masks_pub.publish(masks)
        self.get_logger().info(
            f"{len(world.objects)}개 발행 "
            f"({', '.join(f'{o.object_id}:{o.class_name}' for o in world.objects) or '없음'})"
            + (f" | 재촬영 대상 {needs_reobserve}" if needs_reobserve else ""),
            throttle_duration_sec=2.0)

    def _mask_header(self, stamp: Time):
        from std_msgs.msg import Header

        header = Header()
        header.stamp = stamp
        header.frame_id = BASE_FRAME
        return header

    def _publish_debug_image(self, result, stamp: Time) -> None:
        vis = result.plot()
        msg = Image()
        msg.header = self._mask_header(stamp)
        msg.height, msg.width = vis.shape[:2]
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = msg.width * 3
        msg.data = vis.tobytes()
        self._debug_pub.publish(msg)

    def _maybe_save_unknown_crop(self, class_name: str, color: np.ndarray, mask) -> None:
        """신규 클래스는 크롭을 남긴다 (시스템명세서 4.4절 — VLM 제안·사람 확인의 입력).

        클래스당 한 번만 저장한다. 프레임마다 쓰면 같은 물체로 디스크가 찬다.
        """
        if mask is None or class_name in self._saved_unknown_crops:
            return
        if not self._attributes.attributes(class_name)["needs_confirmation"]:
            return
        crop = mask_utils.crop_bgr(color, mask)
        if crop is None:
            return
        import cv2

        # 크롭 저장은 사람 확인을 돕는 부수 기능이지 발행의 전제조건이 아니다 — 여기서
        # 실패해도(예: /data가 읽기전용) world_state_raw/instance_masks 발행은 계속돼야
        # 한다. 크래시로 노드 전체가 죽으면 그때부터는 알던 물체까지 전부 안 보이게 된다
        # (실제로 겪음 — DB 장애를 objects.yaml seed로 넘기는 attribute_db.py와 같은 이유).
        try:
            self._unknown_crops_dir.mkdir(parents=True, exist_ok=True)
            path = self._unknown_crops_dir / f"{class_name}.png"
            cv2.imwrite(str(path), crop)
            self.get_logger().warning(f"신규 클래스 '{class_name}' 크롭 저장: {path}")
        except OSError as e:
            self.get_logger().warning(
                f"신규 클래스 '{class_name}' 크롭 저장 실패({e}) — 검출은 계속한다")
        self._saved_unknown_crops.add(class_name)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    # 추론이 도는 동안에도 자세 서비스 응답과 이미지 콜백이 들어와야 한다
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

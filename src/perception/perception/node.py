"""perception 노드: 온디맨드 관측(`observe` 액션) → world_state_raw / instance_masks 발행.

참조: 인터페이스_정의서.md 2.0절(릴레이 구조), 3.2~3.4절(메시지 스키마), 1.1절(단위 mm),
docs/on-demand-perception.md(전환 설계, D-1~D-8)

**주기 발행은 하지 않는다.** `observe` 액션 goal을 받았을 때만 검출한다. 두 가지 모드가 있다
(`Observe.action`, docs/on-demand-perception.md 2절):

    MODE_FULL (self._observe_full)
      color/depth/camera_info + 현재 TCP 자세
        → 검출기 추론 전체 스캔 (detector 파라미터: yolo | vlm_sam — detectors/base.py)
        → (vlm_sam이면) 라벨링이 끝난 뒤 새 프레임을 다시 찍어 재투영 박스로 SAM 1패스 (D-8 —
          그래야 발행 stamp가 촬영 시각에서 10초씩 벌어지지 않는다)
        → 마스크 안쪽 median depth로 카메라 좌표 3D → base 좌표 (eye-in-hand 변환)
        → 속성 조회 (object_attributes → objects.yaml → fallback)
        → object_id 유지 (프레임 간 추적)
        → /perception/world_state_raw + /perception/instance_masks 동시 발행

    MODE_REPROMPT (self._observe_reprompt)
      직전 관측이 남긴 물체들의 3D 범위를 지금 TCP 자세로 재투영해 박스 프롬프트로 SAM만
      1패스 돌린다. VLM은 부르지 않는다(D-1) — pick/place_into 스텝 사이마다 이걸 쓴다.

`grasp_candidates`는 **비운 채로** 낸다. 채우는 것은 grasp의 일이고, 채워진 최종본만
`/world_state`로 나간다(2.0절). 두 노드가 같은 토픽에 발행하면 구독자가 어느 버전을 받았는지
구분할 수 없다.

**두 토픽의 stamp는 반드시 같다.** grasp가 그 값으로 마스크와 물체를 짝짓기 때문이다.
"""
import os
import threading
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

from sort_msgs.action import Observe
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
DEPTH_ENCODINGS = ("16UC1", "mono16")

# graspable=false 사유. mock 픽스처(data/mock)가 쓰는 값과 같아야 한다.
REASON_DEPTH_INVALID = "depth_invalid"
REASON_MASK_TOO_SMALL = "mask_too_small"
REASON_NO_MASK = "no_mask"

# detector 파라미터로 고를 수 있는 검출기 (detectors/base.py의 계약을 지키는 것들)
DETECTORS = ("yolo", "vlm_sam")


class _LoggerAdapter:
    """검출기는 ROS를 모르므로 logging 스타일(.info/.warning)만 기대한다. rclpy 로거에 얹는다."""

    def __init__(self, ros_logger):
        self._logger = ros_logger

    def info(self, message, *args):
        self._logger.info(message % args if args else message)

    def warning(self, message, *args):
        self._logger.warning(message % args if args else message)

    def exception(self, message, *args):
        self._logger.error(message % args if args else message)


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")

        # 검출기 선택. yolo는 학습한 7클래스만 알고, vlm_sam은 처음 보는 물건도 이름을
        # 붙인다(docs/vlm_sam_pipeline.md). 스위치로 둔 이유는 물러설 곳을 남기고 같은
        # 프레임에서 둘을 비교할 수 있게 하기 위해서다.
        self.declare_parameter("detector", "yolo")
        self.declare_parameter("model_path", "")
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("device", "")
        # --- detector=vlm_sam 전용 ---
        # 가중치는 이미지에 굽지 않는다(*.pt는 .gitignore) — models/를 마운트해 쓴다.
        self.declare_parameter("sam_model_path", "")
        # planner는 network_mode: host에서 게시된 포트로 부른다 (web_ros와 같은 이유)
        self.declare_parameter("planner_url", "")
        self.declare_parameter("sam_points_stride", 16)
        self.declare_parameter("sam_min_area", 0.002)
        self.declare_parameter("sam_max_area", 0.35)
        self.declare_parameter("sam_max_marks", 20)
        # VLM 왕복이 8~10초다. planner_client의 120초와 같은 판단 (실시간 제어와 분리된 경로)
        self.declare_parameter("label_timeout_s", 120.0)
        # frame B를 새로 찍을 때까지 기다리는 한도(D-8). 카메라가 계속 흐르고 있으면
        # 사실상 즉시 통과한다 — 통과하지 못하면 카메라가 멎은 것이므로 관측을 실패시킨다.
        self.declare_parameter("recapture_timeout_s", 2.0)
        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("info_topic", "/camera/color/camera_info")
        self.declare_parameter("max_frame_age_s", 0.5)
        # 마스크 안쪽 depth가 이보다 적게 유효하면 파지 대상에서 뺀다. 투명·반사 물체가
        # 주로 여기 걸리고, 그 물체들은 needs_reobserve로 올라가 능동 재촬영 대상이 된다(FR-03).
        self.declare_parameter("min_depth_valid_ratio", 0.35)
        # 기본 valid ratio 0.35와 함께 최소 84개 유효점을 요구한다. PCA의 최소 80점보다
        # 작으면 perception은 graspable인데 grasp는 후보가 없는 모순이 생긴다.
        self.declare_parameter("min_mask_pixels", 240)
        # 이보다 오래된 TCP 자세로는 좌표를 만들지 않는다 (robot_pose.posx 주석 참조)
        self.declare_parameter("pose_max_age_s", 1.0)
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("require_robot_pose", True)

        self._min_ratio = float(self.get_parameter("min_depth_valid_ratio").value)
        self._min_pixels = int(self.get_parameter("min_mask_pixels").value)
        self._max_frame_age_s = float(self.get_parameter("max_frame_age_s").value)
        self._pose_max_age_s = float(self.get_parameter("pose_max_age_s").value)
        self._publish_debug = bool(self.get_parameter("publish_debug_image").value)
        self._require_pose = bool(self.get_parameter("require_robot_pose").value)
        self._recapture_timeout_s = float(self.get_parameter("recapture_timeout_s").value)

        self._attributes = AttributeSource()
        self._detector_name = self.get_parameter("detector").value
        self._detector = self._build_detector()
        # vlm_sam일 때만 쓰는 재관측 전용 검출기(D-1). 같은 SAM 가중치를 쓰지만 프롬프트가
        # 없으면 아무것도 안 하므로(segment_at_boxes([]) → []) 항상 만들어 둬도 무해하다 —
        # 매번 detector 이름으로 분기하지 않게 하기 위해서다.
        self._reprompt = self._build_reprompt_detector()
        self._tracker = ObjectTracker()
        self._gripper2camera = geometry.load_handeye()
        self.get_logger().info(f"hand-eye {geometry.handeye_path()} 적재 완료")

        self._unknown_crops_dir = find_repo_path("data", env_var="DATA_DIR") / "unknown_crops"
        self._saved_unknown_crops: set[str] = set()
        # 관측 사진 보관 (docs/on-demand-perception.md D-6). "왜 이걸 집었나"를 나중에
        # 따지려면 그 순간의 그림이 필요하다 — DB(execution_logs.observation_image_path,
        # 마이그레이션 002)에는 경로만 넣고 이미지 자체는 파일로 둔다.
        self._observations_dir = find_repo_path("data", env_var="DATA_DIR") / "observations"

        self._color = None
        self._color_stamp = None
        self._color_frame_id = ""
        self._depth = None
        self._depth_stamp = None
        self._depth_frame_id = ""
        self._intrinsics = None

        # MODE_REPROMPT가 되돌릴 기준. _publish()가 매 관측 끝에 갱신한다. class_name/
        # confidence/position_base_mm/extent_base_mm만 있으면 된다 — SamRepromptDetector.prime이
        # 요구하는 것이 그것뿐이다.
        self._last_objects: list[dict] = []
        # goal_callback과 execute_callback이 서로 다른 스레드에서 돈다(MultiThreadedExecutor).
        # "이미 관측 중이면 거절"을 검사와 표시가 한 원자적 동작이어야 두 goal이 동시에
        # 통과하지 않는다.
        self._observing_lock = threading.Lock()
        self._observing = False

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

        self._action_server = ActionServer(
            self, Observe, "observe", self.execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=callbacks,
        )
        self.get_logger().info(f"perception 기동 완료 (detector={self._detector_name})")

    # --- 입력 ---------------------------------------------------------------
    def _on_color(self, msg: Image) -> None:
        self._color = image_to_numpy(msg)
        # 관측 시각을 함께 들고 있는다. 발행 시각이 아니라 **이 프레임이 찍힌 시각**이
        # WorldState.stamp가 되어야 grasp가 같은 depth 프레임을 다시 찾을 수 있다
        # (ROS 관례이기도 하다 — stamp는 데이터 취득 시각이다).
        self._color_stamp = msg.header.stamp
        self._color_frame_id = msg.header.frame_id

    def _on_depth(self, msg: Image) -> None:
        if msg.encoding not in DEPTH_ENCODINGS:
            self._depth = None
            self.get_logger().warning(
                f"지원하지 않는 aligned depth encoding: {msg.encoding}",
                throttle_duration_sec=5.0)
            return
        self._depth = image_to_numpy(msg)
        self._depth_stamp = msg.header.stamp
        self._depth_frame_id = msg.header.frame_id

    def _on_info(self, msg: CameraInfo) -> None:
        if msg.k[0] <= 0.0 or msg.k[4] <= 0.0:
            self._intrinsics = None
            self.get_logger().warning("CameraInfo K가 유효하지 않다", throttle_duration_sec=5.0)
            return
        self._intrinsics = {"fx": msg.k[0], "fy": msg.k[4], "cx": msg.k[2], "cy": msg.k[5],
                            "width": msg.width, "height": msg.height,
                            "frame_id": msg.header.frame_id}

    def _build_detector(self):
        name = self.get_parameter("detector").value
        if name == "yolo":
            from .detectors.yolo import YoloDetector

            model_path = self.get_parameter("model_path").value or str(
                find_repo_path("models", "best.pt", env_var="PERCEPTION_MODEL"))
            detector = YoloDetector(
                model_path, self._attributes,
                conf=float(self.get_parameter("conf").value),
                imgsz=int(self.get_parameter("imgsz").value),
                device=self.get_parameter("device").value or None,
            )
            self.get_logger().info(f"검출기 yolo — 모델 {detector.describe()}")
            return detector

        if name == "vlm_sam":
            from .detectors.sam_vlm import SamVlmDetector

            weights = self.get_parameter("sam_model_path").value or str(
                find_repo_path("models", "mobile_sam.pt", env_var="SAM_MODEL"))
            planner_url = (self.get_parameter("planner_url").value
                           or os.environ.get("PLANNER_URL", "http://localhost:8100"))
            detector = SamVlmDetector(
                weights, planner_url,
                device=self.get_parameter("device").value or None,
                points_stride=int(self.get_parameter("sam_points_stride").value),
                min_area=float(self.get_parameter("sam_min_area").value),
                max_area=float(self.get_parameter("sam_max_area").value),
                max_marks=int(self.get_parameter("sam_max_marks").value),
                timeout_s=float(self.get_parameter("label_timeout_s").value),
                logger_=_LoggerAdapter(self.get_logger()),
            )
            self.get_logger().info(f"검출기 vlm_sam — {detector.describe()}")
            return detector

        raise ValueError(f"알 수 없는 detector: {name!r} — 쓸 수 있는 값: {DETECTORS}")

    def _build_reprompt_detector(self):
        """vlm_sam일 때만 실체를 만든다. yolo는 이미 빨라서(수십ms) 재관측 트릭이
        필요 없다 — MODE_REPROMPT도 그냥 다시 한 번 전체 추론한다(_observe_instant)."""
        if self._detector_name != "vlm_sam":
            return None
        from .detectors.sam_reprompt import SamRepromptDetector

        weights = self.get_parameter("sam_model_path").value or str(
            find_repo_path("models", "mobile_sam.pt", env_var="SAM_MODEL"))
        detector = SamRepromptDetector(
            weights, device=self.get_parameter("device").value or None,
            logger_=_LoggerAdapter(self.get_logger()))
        self.get_logger().info(f"재관측 검출기 — {detector.describe()}")
        return detector

    # --- observe 액션 --------------------------------------------------------
    def _goal_callback(self, goal_request):
        """동시에 두 관측이 도는 것만 막는다. 다시 찍는 편이 안전하므로(관측은 로봇을
        움직이지 않는다) RequestCache 같은 중복 실행 방지는 두지 않는다 — pick/place와
        달리 재실행에 부작용이 없다."""
        with self._observing_lock:
            if self._observing:
                self.get_logger().warning("이미 관측이 진행 중이다 — 새 goal을 거절한다")
                return GoalResponse.REJECT
            self._observing = True
            return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        self.get_logger().warning("observe 취소 요청 수신")
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        goal = goal_handle.request
        started = time.monotonic()
        trace_id = goal.trace_id or f"obs-{int(started * 1000) % 1_000_000:06d}"

        try:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return self._observe_result(False, Observe.Result.REASON_NONE, 0, None, started)

            def feedback(phase: str) -> None:
                msg = Observe.Feedback()
                msg.phase = phase
                goal_handle.publish_feedback(msg)

            if self._detector_name != "vlm_sam":
                # yolo는 이미 빠르다 — 두 모드를 가를 이유가 없다. 매번 새로 찍고 새로 본다.
                outcome = self._observe_instant(goal_handle, trace_id, feedback)
            elif goal.mode == Observe.Goal.MODE_REPROMPT:
                outcome = self._observe_reprompt(goal_handle, trace_id, feedback)
            else:
                outcome = self._observe_full(goal_handle, trace_id, feedback)

            success, reason, count, stamp = outcome
            if success:
                goal_handle.succeed()
            elif goal_handle.is_cancel_requested:
                goal_handle.canceled()
            else:
                goal_handle.abort()
            return self._observe_result(success, reason, count, stamp, started)

        except Exception as e:
            self.get_logger().error(f"observe 실패: {e}")
            goal_handle.abort()
            return self._observe_result(False, Observe.Result.REASON_NO_FRAME, 0, None, started)
        finally:
            with self._observing_lock:
                self._observing = False

    def _observe_result(self, success: bool, reason: str, object_count: int,
                        stamp: Time | None, started: float) -> "Observe.Result":
        result = Observe.Result()
        result.success = success
        result.failure_reason = reason
        result.object_count = object_count
        result.stamp = stamp or self.get_clock().now().to_msg()
        result.cycle_time_ms = (time.monotonic() - started) * 1000
        return result

    def _frame_failure_reason(self) -> str:
        """지금 상태로는 관측할 수 없는 이유. 관측 가능하면 빈 문자열."""
        missing = self._missing_inputs()
        pose_only = missing == ["robot_pose(get_current_posx)"]
        if missing and pose_only:
            return Observe.Result.REASON_NO_ROBOT_POSE
        if missing:
            return Observe.Result.REASON_NO_FRAME
        problem = self._rgbd_contract_problem()
        if problem:
            self.get_logger().warning(f"RGB-D 입력 불일치: {problem}")
            return Observe.Result.REASON_NO_FRAME
        return ""

    def _capture(self):
        """지금 캐시된 프레임 + TCP 자세로 (color, depth, base2gripper, stamp)를 만든다.
        `_frame_failure_reason()`으로 먼저 관측 가능 여부를 확인한 뒤에만 부른다."""
        posx = self._pose_client.posx(self._pose_max_age_s)
        base2gripper = geometry.posx_to_matrix(posx) if posx else None
        stamp = self._color_stamp or self.get_clock().now().to_msg()
        return self._color, self._depth, base2gripper, stamp

    def _observe_instant(self, goal_handle, trace_id: str, feedback):
        """detector=yolo. 두 모드가 같다 — 이미 수십 ms라 재관측 트릭이 필요 없다."""
        reason = self._frame_failure_reason()
        if reason:
            return False, reason, 0, None

        feedback(Observe.Feedback.PHASE_CAPTURING)
        color, depth, base2gripper, stamp = self._capture()
        raw, debug_bgr = self._detector.detect(
            color, trace_id, on_phase=lambda p: feedback(self._phase_token(p)))
        detections = self._enrich(raw, color, depth, base2gripper)

        feedback(Observe.Feedback.PHASE_PUBLISHING)
        count = self._publish(trace_id, stamp, detections)
        if self._publish_debug and debug_bgr is not None:
            self._publish_debug_image(debug_bgr, stamp, trace_id)
        return True, Observe.Result.REASON_NONE, count, stamp

    def _observe_full(self, goal_handle, trace_id: str, feedback):
        """detector=vlm_sam, MODE_FULL. 프레임 A로 everything+VLM 라벨링(수 초~10초) →
        끝나면 프레임 B를 새로 찍어 재투영 박스로 SAM 1패스(D-8) → B의 stamp로 발행.

        **왜 두 프레임인가.** stamp는 촬영 시각이어야 하는데(grasp가 그 값으로 depth
        프레임을 찾는다), 프레임 A의 stamp로 그대로 발행하면 VLM이 걸린 시간만큼 이미
        낡아 있다(허용치 0.5초, docs/on-demand-perception.md D-8). 로봇은 home에서 멈춰
        있으므로 라벨은 그대로 옮겨간다.
        """
        reason = self._frame_failure_reason()
        if reason:
            return False, reason, 0, None

        feedback(Observe.Feedback.PHASE_CAPTURING)
        color_a, depth_a, base2gripper_a, stamp_a = self._capture()

        raw_a, debug_bgr_a = self._detector.detect(
            color_a, trace_id, on_phase=lambda p: feedback(self._phase_token(p)))
        detections_a = self._enrich(raw_a, color_a, depth_a, base2gripper_a)

        if goal_handle.is_cancel_requested:
            return False, Observe.Result.REASON_NONE, 0, None

        previous = [
            {"class_name": d["class_name"], "confidence": d["confidence"],
             "position_base_mm": d["position"], "extent_base_mm": d["extent_base_mm"]}
            for d in detections_a if d["position"] is not None and d["extent_base_mm"]
        ]
        if not previous or self._reprompt is None:
            # 라벨링은 됐지만 되돌릴 3D가 없다(전부 depth 무효 등) — D-8을 건너뛰고 프레임
            # A를 그대로 낸다. stamp가 낡았을 위험을 감수하는 것이 아무것도 못 내는 것보다
            # 낫다 — grasp가 그 판단은 알아서 한다(오래됐으면 스스로 거른다).
            feedback(Observe.Feedback.PHASE_PUBLISHING)
            count = self._publish(trace_id, stamp_a, detections_a)
            if self._publish_debug and debug_bgr_a is not None:
                self._publish_debug_image(debug_bgr_a, stamp_a, trace_id)
            return True, Observe.Result.REASON_NONE, count, stamp_a

        feedback(Observe.Feedback.PHASE_RECAPTURING)
        if not self._wait_for_new_color_frame(stamp_a):
            self.get_logger().warning("재촬영(D-8) 대기 시간 초과 — 카메라 스트림을 확인한다")
            return False, Observe.Result.REASON_NO_FRAME, 0, None

        reason = self._frame_failure_reason()
        if reason:
            return False, reason, 0, None
        color_b, depth_b, base2gripper_b, stamp_b = self._capture()

        count_reprojected = self._reprompt.prime(
            previous, base2gripper_b, self._gripper2camera, self._intrinsics)
        if count_reprojected == 0:
            # 되돌릴 수 있는 물체가 하나도 없다 — 전부 화면 밖/카메라 뒤로 나갔다는 뜻이라
            # 있을 법하지 않다(로봇이 안 움직였으므로). 그래도 프레임 A로 낸다.
            self.get_logger().warning("재투영할 물체가 없다 — 프레임 A를 그대로 낸다")
            feedback(Observe.Feedback.PHASE_PUBLISHING)
            count = self._publish(trace_id, stamp_a, detections_a)
            if self._publish_debug and debug_bgr_a is not None:
                self._publish_debug_image(debug_bgr_a, stamp_a, trace_id)
            return True, Observe.Result.REASON_NONE, count, stamp_a

        raw_b, debug_bgr = self._reprompt.detect(
            color_b, trace_id, on_phase=lambda p: feedback(self._phase_token(p)))
        detections_b = self._enrich(raw_b, color_b, depth_b, base2gripper_b)

        feedback(Observe.Feedback.PHASE_PUBLISHING)
        count = self._publish(trace_id, stamp_b, detections_b)
        if self._publish_debug and debug_bgr is not None:
            self._publish_debug_image(debug_bgr, stamp_b, trace_id)
        return True, Observe.Result.REASON_NONE, count, stamp_b

    def _observe_reprompt(self, goal_handle, trace_id: str, feedback):
        """detector=vlm_sam, MODE_REPROMPT. VLM은 부르지 않는다(D-1, 명령당 1회 규칙) —
        직전 관측(`_last_objects`)을 지금 프레임으로 재투영해 SAM만 1패스 돌린다.
        박스 프롬프트 1패스는 CPU에서도 0.2초대라 D-8의 재촬영 트릭이 필요 없다.
        """
        if not self._last_objects:
            return False, Observe.Result.REASON_NOTHING_TO_REPROMPT, 0, None

        reason = self._frame_failure_reason()
        if reason:
            return False, reason, 0, None

        feedback(Observe.Feedback.PHASE_CAPTURING)
        color, depth, base2gripper, stamp = self._capture()

        count_reprojected = self._reprompt.prime(
            self._last_objects, base2gripper, self._gripper2camera, self._intrinsics)
        if count_reprojected == 0:
            return False, Observe.Result.REASON_NOTHING_TO_REPROMPT, 0, None

        raw, debug_bgr = self._reprompt.detect(
            color, trace_id, on_phase=lambda p: feedback(self._phase_token(p)))
        detections = self._enrich(raw, color, depth, base2gripper)

        feedback(Observe.Feedback.PHASE_PUBLISHING)
        count = self._publish(trace_id, stamp, detections)
        if self._publish_debug and debug_bgr is not None:
            self._publish_debug_image(debug_bgr, stamp, trace_id)
        return True, Observe.Result.REASON_NONE, count, stamp

    def _wait_for_new_color_frame(self, since_stamp: Time) -> bool:
        """D-8: 카메라가 계속 흐르고 있으면 사실상 즉시 통과한다(다른 스레드가 콜백을
        계속 처리하므로 — MultiThreadedExecutor + ReentrantCallbackGroup, home_server의
        블로킹 대기와 같은 전제). 통과 못 하면 카메라가 멎었다는 뜻이라 관측을 실패시킨다."""
        deadline = time.monotonic() + self._recapture_timeout_s
        while time.monotonic() < deadline:
            if self._color_stamp is not None and (
                self._color_stamp.sec != since_stamp.sec
                or self._color_stamp.nanosec != since_stamp.nanosec
            ):
                return True
            time.sleep(0.02)
        return False

    @staticmethod
    def _phase_token(token: str) -> str:
        """검출기가 주는 토큰("segmenting"/"labeling")을 Observe.Feedback 상수로 옮긴다.
        검출기는 액션 메시지를 몰라도 되게 하기 위한 어댑터다(detectors/base.py 참조)."""
        return {
            "segmenting": Observe.Feedback.PHASE_SEGMENTING,
            "labeling": Observe.Feedback.PHASE_LABELING,
        }.get(token, Observe.Feedback.PHASE_SEGMENTING)

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

    def _rgbd_contract_problem(self) -> str:
        if self._color.shape[:2] != self._depth.shape:
            return f"color{self._color.shape[:2]} depth{self._depth.shape}"
        if self._color_frame_id != self._depth_frame_id:
            return f"color frame={self._color_frame_id} depth frame={self._depth_frame_id}"
        if (self._intrinsics["height"], self._intrinsics["width"]) != self._color.shape[:2]:
            return "CameraInfo grid"
        if self._intrinsics["frame_id"] != self._color_frame_id:
            return f"CameraInfo frame={self._intrinsics['frame_id']}"
        color_time = self._color_stamp.sec + self._color_stamp.nanosec * 1e-9
        depth_time = self._depth_stamp.sec + self._depth_stamp.nanosec * 1e-9
        if abs(color_time - depth_time) > self._max_frame_age_s:
            return "color/depth timestamp"
        return ""

    def _enrich(self, raw: list[dict], color: np.ndarray, depth: np.ndarray,
                base2gripper) -> list[dict]:
        """검출기가 낸 (class_name, confidence, mask)에 3D·높이·파지 가능 여부를 붙인다.

        여기서부터는 검출 방법과 무관하다 — YOLO든 SAM+VLM이든 마스크가 나온 뒤는 같은 계산이다.
        """
        detections: list[dict] = []

        for item in raw:
            class_name = item["class_name"]
            confidence = item["confidence"]
            mask = item["mask"]

            if mask is None:
                detections.append(self._detection(class_name, confidence, None, None, 0.0,
                                                  REASON_NO_MASK, 0.0))
                continue

            point_cam, valid_ratio = mask_utils.mask_3d(mask, depth, self._intrinsics)

            position = None
            if point_cam is not None and base2gripper is not None:
                position = geometry.base_from_camera(point_cam, base2gripper, self._gripper2camera)

            # 물체가 지지면 위로 얼마나 솟아 있는지. place_into가 놓는 높이를 정하는 데
            # 쓴다(DetectedObject.msg의 height_mm 주석 참조). 마스크 바깥 링에서 지지면
            # depth를 읽어(support_3d) 같은 (u,v)의 물체 점과 base z를 비교한다 — 두 점이
            # 같은 픽셀이라 차이가 순수한 높이차가 된다. base 변환이 필요하므로 TCP 자세가
            # 없으면 잴 수 없다(0=미상).
            height_mm = 0.0
            if position is not None:
                support_cam = mask_utils.support_3d(mask, depth, self._intrinsics)
                if support_cam is not None:
                    support = geometry.base_from_camera(
                        support_cam, base2gripper, self._gripper2camera)
                    height_mm = max(0.0, float(position[2]) - float(support[2]))

            # 재관측(D-1)이 다음 프레임에서 이 물체를 감쌀 박스를 만들 때 쓴다. 마스크
            # 테두리를 각자의 depth로 역투영한 base 점들이라, 시점이 바뀌어도 되돌려진다.
            extent = []
            if position is not None:
                extent = [geometry.base_from_camera(p, base2gripper, self._gripper2camera)
                          for p in mask_utils.mask_extent_3d(mask, depth, self._intrinsics)]

            reason = self._not_graspable_reason(mask, point_cam, valid_ratio, position)
            detections.append(self._detection(class_name, confidence, position, mask,
                                              valid_ratio, reason, height_mm, extent))
            self._maybe_save_unknown_crop(class_name, color, mask)

        return detections

    def _not_graspable_reason(self, mask, point_cam, valid_ratio, position) -> str:
        if int(mask.sum()) < self._min_pixels:
            return REASON_MASK_TOO_SMALL
        if point_cam is None or position is None or valid_ratio < self._min_ratio:
            return REASON_DEPTH_INVALID
        return ""

    @staticmethod
    def _detection(class_name, confidence, position, mask, valid_ratio, reason,
                   height_mm, extent=()) -> dict:
        return {
            "class_name": class_name,
            "confidence": confidence,
            "position": position,
            "mask": mask,
            "depth_valid_ratio": valid_ratio,
            "not_graspable_reason": reason,
            "height_mm": height_mm,
            # base 좌표 mm. 메시지에는 안 나가고 다음 관측의 프롬프트로만 쓴다.
            "extent_base_mm": list(extent),
        }

    # --- 발행 ---------------------------------------------------------------
    def _publish(self, trace_id: str, stamp: Time, detections: list[dict]) -> int:
        """world_state_raw + instance_masks를 발행하고, 낸 물체 수를 돌려준다
        (Observe.Result.object_count). 재관측(MODE_REPROMPT)이 되돌릴 기준(`_last_objects`)도
        여기서 함께 갱신한다 — 발행한 것과 다음에 되찾을 것이 어긋나면 안 되기 때문이다.
        """
        object_ids = self._tracker.assign(detections)
        last_objects = []

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
            obj.height_mm = float(detection["height_mm"])
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

            if detection["position"] is not None and detection["extent_base_mm"]:
                last_objects.append({
                    "class_name": detection["class_name"],
                    "confidence": detection["confidence"],
                    "position_base_mm": detection["position"],
                    "extent_base_mm": detection["extent_base_mm"],
                })

        world.needs_reobserve = needs_reobserve
        self._last_objects = last_objects
        self._world_state_pub.publish(world)
        self._instance_masks_pub.publish(masks)
        self.get_logger().info(
            f"{len(world.objects)}개 발행 "
            f"({', '.join(f'{o.object_id}:{o.class_name}' for o in world.objects) or '없음'})"
            + (f" | 재촬영 대상 {needs_reobserve}" if needs_reobserve else ""),
            throttle_duration_sec=2.0)
        return len(world.objects)

    def _mask_header(self, stamp: Time):
        from std_msgs.msg import Header

        header = Header()
        header.stamp = stamp
        # mask의 픽셀 좌표는 aligned color/depth grid 기준이다. base는 WorldState와
        # GraspCandidate의 좌표계이며 Image header에 넣으면 안 된다.
        header.frame_id = self._color_frame_id
        return header

    def _publish_debug_image(self, vis: np.ndarray, stamp: Time, trace_id: str = "") -> None:
        msg = Image()
        msg.header = self._mask_header(stamp)
        msg.height, msg.width = vis.shape[:2]
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = msg.width * 3
        msg.data = vis.tobytes()
        self._debug_pub.publish(msg)
        self._save_observation_image(vis, stamp, trace_id)

    def _save_observation_image(self, vis: np.ndarray, stamp: Time, trace_id: str) -> None:
        """`data/observations/<trace_id>/<stamp>.png`에 남긴다 (D-6).

        `/perception/debug_image`로 나가는 것과 **같은 그림**이다 — 캡처 사진 위에
        마스크 윤곽과 파지 후보 그립 모양까지 합성된 것(grasp가 아니라 여기서 만드는
        이유는 grasp의 오버레이는 grasp_candidates가 채워진 *이후* 시점이라 perception이
        직접 낼 수 있는 게 마스크까지이기 때문이다 — grasp_candidates가 빈 채로 나가는
        이 시점의 그림은 파지 후보 없이 마스크만 보인다. 파지 후보까지 합쳐진 최종
        오버레이는 `/grasp/debug_image`이고, D-5에 따라 그건 이미 UI가 보여준다).

        크롭 저장(_maybe_save_unknown_crop)과 같은 원칙 — 실패해도 발행 자체를 막지 않는다.
        """
        if not trace_id:
            return
        import cv2

        try:
            directory = self._observations_dir / trace_id
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{stamp.sec}_{stamp.nanosec:09d}.png"
            cv2.imwrite(str(path), vis)
        except OSError as e:
            self.get_logger().warning(f"관측 사진 저장 실패({e}) — 발행은 계속한다")

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

"""rclpy 격리 계층. main.py(순수 HTTP)가 ROS2를 직접 알지 않도록 여기서만 rclpy를 다룬다.

/control/robot_state, /control/safety_events 구독과 pick/place_into 액션 클라이언트 호출을 담당.

`executor.Executor`와 같은 시그니처를 구현하므로, `MOCK_MODE=0`으로 바꾸는 것만으로
mock과 교체된다. **실제 로봇/ROS2 환경에서의 검증은 B4에서 한다** — 그전까지 이 파일은
인터페이스가 실물에도 맞는지 확인하는 용도이며, 컨테이너에 rclpy가 없으면 import되지 않는다
(`executor.create_executor()`가 mock일 때 이 모듈을 건드리지 않는다).
"""
import asyncio
import logging
import threading
import time
import uuid
from typing import Awaitable, Callable

import cv2
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from sort_msgs.action import Home, Pick, PlaceInto
from sort_msgs.msg import RobotState, SafetyEvent, WorldState

from .executor import SkillGoal, SkillResult

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0.0"


# --- 카메라·뎁스 뷰 (화면정의서 2.2.4절: RGB 원본 스트림, 마스크 오버레이 없음) -----------
#
# perception_common.image_utils를 그대로 쓰지 않는 이유는 web이 ROS2 패키지가 아니라서다
# (rclpy는 여기서만 import되는 순수 pip 의존성 — executor.py 주석 참조). 필요한 인코딩
# 두 개(bgr8, 16UC1)만 이 파일 안에서 직접 변환한다 — perception_capture.py가 cv_bridge
# 대신 자기만의 image_to_numpy를 갖는 것과 같은 이유(중복이지만 결합을 만들지 않는다).

def _color_to_bgr(msg: Image) -> np.ndarray:
    """bgr8/rgb8 → BGR ndarray. 리얼센스는 rgb8로 낸다(cv2가 기대하는 건 bgr8)."""
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    arr = buf.reshape(msg.height, msg.step)[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
    return arr[:, :, ::-1] if msg.encoding == "rgb8" else arr


def _depth_to_colormap(msg: Image) -> np.ndarray:
    """16UC1 depth(mm) → JET 컬러맵. 0(무효 픽셀)은 검게 남긴다."""
    buf = np.frombuffer(msg.data, dtype=np.uint16)
    depth = buf.reshape(msg.height, msg.step // 2)[:, : msg.width]
    valid = depth > 0
    if not valid.any():
        return np.zeros((msg.height, msg.width, 3), dtype=np.uint8)
    low, high = int(depth[valid].min()), int(depth[valid].max())
    span = max(high - low, 1)
    normalized = np.clip((depth.astype(np.float32) - low) * (255.0 / span), 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    colored[~valid] = 0
    return colored


def _encode_jpeg(bgr: np.ndarray) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", bgr)
    return buf.tobytes() if ok else None


def _pose_to_msg(pose: dict):
    """JSON pose(mm) → geometry_msgs/Pose.

    단위는 mm 그대로 둔다 — 인터페이스_정의서 1.1절대로 계층 경계는 전부 mm이고,
    m 환산은 control이 MoveIt2에 넘길 때 한 곳에서만 한다.
    """
    from geometry_msgs.msg import Pose

    msg = Pose()
    msg.position.x = float(pose["position"]["x"])
    msg.position.y = float(pose["position"]["y"])
    msg.position.z = float(pose["position"]["z"])
    msg.orientation.x = float(pose["orientation"]["x"])
    msg.orientation.y = float(pose["orientation"]["y"])
    msg.orientation.z = float(pose["orientation"]["z"])
    msg.orientation.w = float(pose["orientation"]["w"])
    return msg


def _world_state_to_dict(msg: WorldState) -> dict:
    """WorldState.msg → JSON. planner에 넘기는 형태이자 mock 픽스처와 같은 구조다."""
    return {
        "schema_version": msg.schema_version,
        "trace_id": msg.trace_id,
        "stamp": {"sec": msg.stamp.sec, "nanosec": msg.stamp.nanosec},
        "frame_id": msg.frame_id,
        "objects": [
            {
                "object_id": o.object_id,
                "class_name": o.class_name,
                "name_ko": o.name_ko,
                "confidence": o.confidence,
                "position_base_mm": {
                    "x": o.position_base_mm.x,
                    "y": o.position_base_mm.y,
                    "z": o.position_base_mm.z,
                },
                "depth_valid_ratio": o.depth_valid_ratio,
                "graspable": o.graspable,
                "not_graspable_reason": o.not_graspable_reason,
                "mass_g": o.mass_g,
                "fragile": o.fragile,
                "deformable": o.deformable,
                "transparent": o.transparent,
                "profile": o.profile,
                "attr_source": o.attr_source,
                "needs_confirmation": o.needs_confirmation,
                "grasp_candidates": [
                    {
                        "pose": {
                            "position": {"x": c.pose.position.x, "y": c.pose.position.y,
                                         "z": c.pose.position.z},
                            "orientation": {"x": c.pose.orientation.x, "y": c.pose.orientation.y,
                                            "z": c.pose.orientation.z, "w": c.pose.orientation.w},
                        },
                        "score": c.score,
                        "strategy": c.strategy,
                    }
                    for c in o.grasp_candidates
                ],
            }
            for o in msg.objects
        ],
        "needs_reobserve": list(msg.needs_reobserve),
    }


class _BridgeNode(Node):
    """구독·액션 클라이언트를 들고 있는 ROS2 노드. 자체 스레드에서 spin된다."""

    def __init__(self):
        super().__init__("web_ros_bridge")
        self.latest_world_state: dict | None = None
        self.latest_robot_state: dict = {"schema_version": SCHEMA_VERSION, "mode": "idle",
                                         "current_skill": "none", "gripper_width_mm": 0.0}
        self.latest_color_jpeg: bytes | None = None
        self.latest_depth_jpeg: bytes | None = None
        self.on_event: Callable[[dict], None] | None = None

        self.create_subscription(WorldState, "/world_state", self._on_world_state, 10)
        self.create_subscription(RobotState, "/control/robot_state", self._on_robot_state, 10)
        self.create_subscription(SafetyEvent, "/control/safety_events", self._on_safety_event, 10)

        # 카메라·뎁스 뷰. perception이 아니라 리얼센스 드라이버가 직접 내는 원본을 구독한다
        # (화면정의서 2.2.4절 — 마스크 오버레이 없는 원본). 이미지 토픽은 대역폭이 커서
        # BEST_EFFORT — 화면 프레임 하나 놓쳐도 다음 프레임이 금방 오므로 재전송을 기다릴
        # 이유가 없다(grasp의 depth 구독과 같은 QoS 선택, grasp/node.py 참조).
        image_qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, "/camera/color/image_raw", self._on_color_image,
                                 image_qos)
        self.create_subscription(Image, "/camera/aligned_depth_to_color/image_raw",
                                 self._on_depth_image, image_qos)

        self.pick_client = ActionClient(self, Pick, "pick")
        self.place_client = ActionClient(self, PlaceInto, "place_into")
        self.home_client = ActionClient(self, Home, "home")

    def _on_world_state(self, msg: WorldState) -> None:
        self.latest_world_state = _world_state_to_dict(msg)

    def _on_robot_state(self, msg: RobotState) -> None:
        """최신값은 항상 갱신하고, **변화가 있을 때만** web으로 올린다.

        control은 상태를 5Hz로 주기 발행한다(주기 발행이라야 구독자가 언제 붙어도 현재
        상태를 알 수 있다). 그것을 그대로 WebSocket에 중계하면 아무 일도 일어나지 않는
        동안에도 모든 브라우저에 초당 5건이 나간다 — 실제로 B4 E2E 로그가 동일한
        `mode=idle` 이벤트로 뒤덮였다. 화면이 필요한 것은 전이이지 주기가 아니다.

        폴백 경로(`GET /health`, `GET /api/world-state`)는 `latest_robot_state`를 읽으므로
        방송을 줄여도 새로 접속한 화면이 상태를 못 보는 일은 없다.
        """
        state = {
            "schema_version": msg.schema_version,
            "mode": msg.mode,
            "current_skill": msg.current_skill,
            "gripper_width_mm": msg.gripper_width_mm,
        }
        changed = state != self.latest_robot_state
        self.latest_robot_state = state
        if changed and self.on_event:
            self.on_event({"type": "robot_state", **state})

    def _on_color_image(self, msg: Image) -> None:
        if msg.encoding not in ("bgr8", "rgb8"):
            self.get_logger().warning(f"카메라 뷰: 지원하지 않는 인코딩 {msg.encoding}",
                                      throttle_duration_sec=10.0)
            return
        self.latest_color_jpeg = _encode_jpeg(_color_to_bgr(msg))

    def _on_depth_image(self, msg: Image) -> None:
        if msg.encoding not in ("16UC1", "mono16"):
            self.get_logger().warning(f"뎁스 뷰: 지원하지 않는 인코딩 {msg.encoding}",
                                      throttle_duration_sec=10.0)
            return
        self.latest_depth_jpeg = _encode_jpeg(_depth_to_colormap(msg))

    def _on_safety_event(self, msg: SafetyEvent) -> None:
        # 안전 이벤트는 지연 없이 즉시 올린다(4절)
        if self.on_event:
            self.on_event({
                "type": "safety_event",
                "event_type": msg.event_type,
                "severity": msg.severity,
                "description": msg.description,
                "requires_manual_reset": msg.requires_manual_reset,
            })


class RosExecutor:
    """executor.Executor의 ROS2 구현."""

    def __init__(self):
        self._node: _BridgeNode | None = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._active_goal_handle = None
        self._active_request_id: str | None = None

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        if not rclpy.ok():
            rclpy.init()
        self._node = _BridgeNode()
        self._thread = threading.Thread(target=rclpy.spin, args=(self._node,), daemon=True)
        self._thread.start()
        logger.info("ROS2 브리지 노드 시작")

    async def close(self) -> None:
        if self._node:
            self._node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def get_latest_world_state(self) -> dict | None:
        return self._node.latest_world_state if self._node else None

    def robot_state(self) -> dict:
        return self._node.latest_robot_state if self._node else {"mode": "error"}

    def latest_color_jpeg(self) -> bytes | None:
        return self._node.latest_color_jpeg if self._node else None

    def latest_depth_jpeg(self) -> bytes | None:
        return self._node.latest_depth_jpeg if self._node else None

    def subscribe_state(self, on_event: Callable[[dict], Awaitable[None]]) -> None:
        loop = self._loop

        def bridge(event: dict) -> None:
            # ROS2 콜백은 spin 스레드에서 온다 — 이벤트 루프로 넘겨야 한다
            if loop and loop.is_running():
                asyncio.run_coroutine_threadsafe(on_event(event), loop)

        if self._node:
            self._node.on_event = bridge

    async def _send(self, client: ActionClient, goal_msg, request_id: str,
                    on_feedback) -> SkillResult:
        started = time.monotonic()
        loop = asyncio.get_running_loop()

        if not client.wait_for_server(timeout_sec=5.0):
            logger.error("액션 서버에 연결하지 못했습니다 (%s)", request_id)
            return SkillResult(success=False, failure_reason="unreachable")

        def feedback_cb(feedback):
            phase = feedback.feedback.phase
            asyncio.run_coroutine_threadsafe(on_feedback(request_id, phase), loop)

        send_future = client.send_goal_async(goal_msg, feedback_callback=feedback_cb)
        goal_handle = await _await_ros_future(send_future)

        if not goal_handle.accepted:
            return SkillResult(success=False, failure_reason="unreachable")

        self._active_goal_handle = goal_handle
        self._active_request_id = request_id
        try:
            result = (await _await_ros_future(goal_handle.get_result_async())).result
        finally:
            self._active_goal_handle = None
            self._active_request_id = None

        # 액션마다 Result 필드가 다르다 (Home에는 재시도·토크 개념이 없다).
        # 공통 필드만 직접 읽고 나머지는 getattr로 낮춘다.
        return SkillResult(
            success=result.success,
            failure_reason=result.failure_reason,
            retries_used=getattr(result, "retries_used", 0),
            cycle_time_ms=result.cycle_time_ms or (time.monotonic() - started) * 1000,
            visual_verification_passed=getattr(result, "visual_verification_passed", None),
            torque_trace=list(getattr(result, "torque_trace_summary", []) or []),
        )

    async def call_pick(self, goal: SkillGoal, on_feedback) -> SkillResult:
        msg = Pick.Goal()
        msg.schema_version = SCHEMA_VERSION
        msg.trace_id = goal.trace_id
        msg.request_id = goal.request_id
        msg.object_id = goal.object_id
        msg.profile = goal.profile
        msg.grasp_pose = _pose_to_msg(goal.grasp_pose)
        msg.max_retries = goal.max_retries
        return await self._send(self._node.pick_client, msg, goal.request_id, on_feedback)

    async def call_place_into(self, goal: SkillGoal, on_feedback) -> SkillResult:
        msg = PlaceInto.Goal()
        msg.schema_version = SCHEMA_VERSION
        msg.trace_id = goal.trace_id
        msg.request_id = goal.request_id
        msg.object_id = goal.object_id
        msg.profile = goal.profile
        msg.bin_id = goal.bin_id or ""
        msg.use_pose_override = False
        msg.max_retries = goal.max_retries
        return await self._send(self._node.place_client, msg, goal.request_id, on_feedback)

    async def stop(self) -> str | None:
        """진행 중인 goal을 취소한다. busy일 때 가장 필요한 동작이라 상태를 보지 않는다."""
        handle, request_id = self._active_goal_handle, self._active_request_id
        if handle is None:
            return None
        await _await_ros_future(handle.cancel_goal_async())
        return request_id

    async def home(self) -> None:
        """홈 자세 복귀 (인터페이스_정의서 4.3절 Home.action).

        pick/place_into와 같은 `_send`를 지나므로 `_active_goal_handle`에 등록되고,
        따라서 `stop()`으로 취소된다 — 홈이동이 서비스가 아니라 액션인 이유다.

        `request_id`는 여기서 만든다. 홈이동은 planner를 거치지 않아 발급해 줄 상위 계층이
        없지만, 중복 실행 방지는 control 쪽 계약이라 빈 값으로 보내면 캐시가 동작하지 않는다.
        """
        msg = Home.Goal()
        msg.schema_version = SCHEMA_VERSION
        msg.request_id = f"rq-home-{uuid.uuid4().hex[:8]}"

        async def _ignore_feedback(_request_id: str, _phase: str) -> None:
            """홈이동은 trace에 속하지 않아 중계할 진행률 화면이 없다 (2.6절)."""

        result = await self._send(self._node.home_client, msg, msg.request_id,
                                  _ignore_feedback)
        if not result.success:
            raise RuntimeError(f"home 이동 실패: {result.failure_reason}")


async def _await_ros_future(ros_future):
    """rclpy Future를 asyncio에서 기다린다. spin은 별도 스레드가 돌리고 있다."""
    loop = asyncio.get_running_loop()
    asyncio_future = loop.create_future()

    def done(f):
        if not asyncio_future.done():
            loop.call_soon_threadsafe(asyncio_future.set_result, f.result())

    ros_future.add_done_callback(done)
    return await asyncio_future

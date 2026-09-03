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
from typing import Awaitable, Callable

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from sort_msgs.action import Pick, PlaceInto
from sort_msgs.msg import RobotState, SafetyEvent, WorldState

from .executor import SkillGoal, SkillResult

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0.0"


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
        self.on_event: Callable[[dict], None] | None = None

        self.create_subscription(WorldState, "/world_state", self._on_world_state, 10)
        self.create_subscription(RobotState, "/control/robot_state", self._on_robot_state, 10)
        self.create_subscription(SafetyEvent, "/control/safety_events", self._on_safety_event, 10)

        self.pick_client = ActionClient(self, Pick, "pick")
        self.place_client = ActionClient(self, PlaceInto, "place_into")

    def _on_world_state(self, msg: WorldState) -> None:
        self.latest_world_state = _world_state_to_dict(msg)

    def _on_robot_state(self, msg: RobotState) -> None:
        self.latest_robot_state = {
            "schema_version": msg.schema_version,
            "mode": msg.mode,
            "current_skill": msg.current_skill,
            "gripper_width_mm": msg.gripper_width_mm,
        }
        if self.on_event:
            self.on_event({"type": "robot_state", **self.latest_robot_state})

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

        return SkillResult(
            success=result.success,
            failure_reason=result.failure_reason,
            retries_used=result.retries_used,
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
        # TODO(B4): control에 home 이동 인터페이스가 없다. 스킬로 추가할지 별도 서비스로 둘지
        # control 착수 시점에 결정한다 (웹_인터페이스_정의서 2.6절).
        raise NotImplementedError("home 이동은 control 인터페이스 확정 후 연결한다 (B4)")


async def _await_ros_future(ros_future):
    """rclpy Future를 asyncio에서 기다린다. spin은 별도 스레드가 돌리고 있다."""
    loop = asyncio.get_running_loop()
    asyncio_future = loop.create_future()

    def done(f):
        if not asyncio_future.done():
            loop.call_soon_threadsafe(asyncio_future.set_result, f.result())

    ros_future.add_done_callback(done)
    return await asyncio_future

"""/control/robot_state 주기 발행.

참조: 인터페이스_정의서.md 3.5절 — 구독자는 web (planner가 아님, NFR-04 유지)

mode 전이(idle↔busy)는 액션 서버가 알려준다. 상태의 소유자를 한 곳으로 모으기 위해
`RobotStateStore`를 두고, pick/place 서버가 여기에 쓰고 이 노드가 읽어서 발행한다.
두 액션 서버가 각자 발행하면 어느 쪽 상태가 최신인지 알 수 없어진다.
"""
import os
import threading

import rclpy
from rclpy.node import Node

from sort_msgs.msg import RobotState

SCHEMA_VERSION = "1.0.0"
PUBLISH_PERIOD_S = 0.2


class RobotStateStore:
    """프로세스 공유 로봇 상태. 액션 서버가 쓰고 발행 노드가 읽는다."""

    def __init__(self):
        self._lock = threading.Lock()
        self._mode = RobotState.MODE_IDLE
        self._current_skill = "none"
        self._gripper_width_mm = 0.0
        self._gripper_closed = False

    def set_busy(self, skill: str) -> None:
        with self._lock:
            self._mode = RobotState.MODE_BUSY
            self._current_skill = skill

    def set_idle(self) -> None:
        with self._lock:
            self._mode = RobotState.MODE_IDLE
            self._current_skill = "none"

    def set_error(self) -> None:
        with self._lock:
            self._mode = RobotState.MODE_ERROR
            self._current_skill = "none"

    def set_gripper(self, width_mm: float, closed: bool) -> None:
        with self._lock:
            self._gripper_width_mm = width_mm
            self._gripper_closed = closed

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "mode": self._mode,
                "current_skill": self._current_skill,
                "gripper_width_mm": self._gripper_width_mm,
                "gripper_closed": self._gripper_closed,
            }


# 같은 프로세스(control_launch로 함께 뜨는 노드들)가 공유하는 단일 인스턴스
store = RobotStateStore()


def is_fake_robot() -> bool:
    """FAKE_ROBOT=1이면 실물 로봇 없이 ROS2 경로만 검증한다 (개발계획 B4)."""
    return os.environ.get("FAKE_ROBOT", "0") == "1"


class RobotStatePublisherNode(Node):
    def __init__(self):
        super().__init__('robot_state_publisher_node')
        self._pub = self.create_publisher(RobotState, '/control/robot_state', 10)
        self.create_timer(PUBLISH_PERIOD_S, self._publish)
        self.get_logger().info(
            f"robot_state 발행 시작 ({'fake' if is_fake_robot() else '실물'} 모드)"
        )
        # TODO(실물): 조인트 상태·그리퍼 상태를 doosan-robot2 드라이버에서 받아 store에 반영

    def _publish(self) -> None:
        state = store.snapshot()
        msg = RobotState()
        msg.schema_version = SCHEMA_VERSION
        msg.stamp = self.get_clock().now().to_msg()
        msg.mode = state["mode"]
        msg.current_skill = state["current_skill"]
        msg.gripper_width_mm = state["gripper_width_mm"]
        msg.gripper_closed = state["gripper_closed"]
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = RobotStatePublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

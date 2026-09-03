"""/control/robot_state 주기 발행.

참조: 인터페이스_정의서.md 3.5절 — 구독자는 web (planner가 아님, NFR-04 유지)
"""
import rclpy
from rclpy.node import Node

from sort_msgs.msg import RobotState


class RobotStatePublisherNode(Node):
    def __init__(self):
        super().__init__('robot_state_publisher_node')
        self._pub = self.create_publisher(RobotState, '/control/robot_state', 10)
        # TODO: 주기 타이머 + 조인트 상태·그리퍼 상태 소스 연결


def main(args=None):
    rclpy.init(args=args)
    node = RobotStatePublisherNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

"""/control/safety_events 발행 (비상정지·충돌·제한 위반).

참조: 인터페이스_정의서.md 3.6절 — 검증기(FR-12)의 게이트 조건으로 사용됨
"""
import rclpy
from rclpy.node import Node

from sort_msgs.msg import SafetyEvent


class SafetyMonitorNode(Node):
    def __init__(self):
        super().__init__('safety_monitor')
        self._pub = self.create_publisher(SafetyEvent, '/control/safety_events', 10)
        # TODO: 비상정지/충돌/제한 위반 감지 소스 연결


def main(args=None):
    rclpy.init(args=args)
    node = SafetyMonitorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

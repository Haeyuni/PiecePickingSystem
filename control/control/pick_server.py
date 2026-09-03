"""pick 액션 서버. compliance(토크 판정) AND visual_verification 통과 시에만 success=true.

참조: 인터페이스_정의서.md 4.1절 (Pick.action)
"""
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node

from sort_msgs.action import Pick


class PickServer(Node):
    def __init__(self):
        super().__init__('pick_server')
        self._action_server = ActionServer(
            self, Pick, 'pick', self.execute_callback)

    def execute_callback(self, goal_handle):
        # TODO: compliance.py + visual_verification.py 연동, request_id 중복 실행 방지 캐시
        goal_handle.succeed()
        return Pick.Result()


def main(args=None):
    rclpy.init(args=args)
    node = PickServer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

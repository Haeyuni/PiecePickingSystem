"""place_into 액션 서버. bin_id → bins.yaml 좌표 변환.

참조: 인터페이스_정의서.md 4.2절 (PlaceInto.action)
"""
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node

from sort_msgs.action import PlaceInto


class PlaceServer(Node):
    def __init__(self):
        super().__init__('place_server')
        self._action_server = ActionServer(
            self, PlaceInto, 'place_into', self.execute_callback)
        # TODO: config/bins.yaml 로딩

    def execute_callback(self, goal_handle):
        # TODO: bin_id 그라운딩 체크, compliance.py 연동, request_id 중복 실행 방지 캐시
        goal_handle.succeed()
        return PlaceInto.Result()


def main(args=None):
    rclpy.init(args=args)
    node = PlaceServer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

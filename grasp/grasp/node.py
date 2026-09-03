"""grasp 노드: world_state_raw + instance_masks를 stamp 동기화해 파지 후보를 채운 뒤 /world_state로 발행.

참조: 인터페이스_정의서.md 2.0절 (world_state 릴레이 구조)
"""
import rclpy
from rclpy.node import Node

from sort_msgs.msg import WorldState


class GraspNode(Node):
    def __init__(self):
        super().__init__('grasp_node')
        self._world_state_pub = self.create_publisher(WorldState, '/world_state', 10)
        # TODO: /perception/world_state_raw + /perception/instance_masks
        # message_filters.ApproximateTimeSynchronizer로 동기화, 전략별 파지 후보 계산 연결


def main(args=None):
    rclpy.init(args=args)
    node = GraspNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

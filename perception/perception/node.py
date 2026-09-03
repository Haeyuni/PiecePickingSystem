"""perception 노드: 물체 검출·세그멘테이션 후 world_state_raw / instance_masks 발행.

참조: 인터페이스_정의서.md 2.0절 (world_state 릴레이 구조), 3.2~3.4절 (메시지 스키마)
"""
import rclpy
from rclpy.node import Node

from sort_msgs.msg import WorldState, InstanceMasks


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        self._world_state_pub = self.create_publisher(
            WorldState, '/perception/world_state_raw', 10)
        self._instance_masks_pub = self.create_publisher(
            InstanceMasks, '/perception/instance_masks', 10)
        # TODO: RGB-D 구독(/camera/depth/image_rect_raw), 검출·세그멘테이션 파이프라인 연결


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

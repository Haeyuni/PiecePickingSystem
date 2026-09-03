#!/usr/bin/env python3
"""카메라 토픽에서 정지 프레임을 떠서 data/samples/에 저장한다 (검출 테스트용 샘플 수집).

  source /opt/ros/jazzy/setup.bash
  .venv/bin/python scripts/perception_capture.py -n 5 --interval 1.0

color는 .png, aligned depth는 .npy(uint16, mm), intrinsics는 .json으로 같은 stem에 저장한다.
cv_bridge는 쓰지 않는다 — venv의 opencv-python과 시스템 cv_bridge가 섞이면 깨지므로 직접 변환한다.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

COLOR_TOPIC = '/camera/color/image_raw'
DEPTH_TOPIC = '/camera/aligned_depth_to_color/image_raw'
INFO_TOPIC = '/camera/color/camera_info'


def image_to_numpy(msg: Image) -> np.ndarray:
    """sensor_msgs/Image -> ndarray. bgr8/rgb8/16UC1/mono8만 지원."""
    dtype, channels = {
        'bgr8': (np.uint8, 3),
        'rgb8': (np.uint8, 3),
        'mono8': (np.uint8, 1),
        '16UC1': (np.uint16, 1),
        'mono16': (np.uint16, 1),
    }[msg.encoding]
    buf = np.frombuffer(msg.data, dtype=dtype)
    arr = buf.reshape(msg.height, msg.step // np.dtype(dtype).itemsize)[:, :msg.width * channels]
    arr = arr.reshape(msg.height, msg.width, channels)
    if msg.encoding == 'rgb8':
        arr = arr[:, :, ::-1]           # 이후 처리는 전부 BGR 기준
    return arr if channels > 1 else arr[:, :, 0]


class Capture(Node):
    def __init__(self):
        super().__init__('perception_capture')
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.color = None
        self.depth = None
        self.info = None
        self.create_subscription(Image, COLOR_TOPIC, self._on_color, qos)
        self.create_subscription(Image, DEPTH_TOPIC, self._on_depth, qos)
        self.create_subscription(CameraInfo, INFO_TOPIC, self._on_info, 10)

    def _on_color(self, msg):
        self.color = image_to_numpy(msg)

    def _on_depth(self, msg):
        self.depth = image_to_numpy(msg)

    def _on_info(self, msg):
        self.info = {'fx': msg.k[0], 'fy': msg.k[4], 'cx': msg.k[2], 'cy': msg.k[5],
                     'width': msg.width, 'height': msg.height}

    def wait_for_frames(self, timeout=10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.color is not None and self.depth is not None and self.info is not None:
                return True
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--count', type=int, default=1)
    ap.add_argument('--interval', type=float, default=1.0, help='초 단위 촬영 간격')
    ap.add_argument('-o', '--out', default='data/samples')
    ap.add_argument('--prefix', default='shot')
    args = ap.parse_args()

    import cv2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = Capture()
    try:
        if not node.wait_for_frames():
            node.get_logger().error(
                f'프레임을 못 받았다. {COLOR_TOPIC} / {DEPTH_TOPIC} 가 살아있는지 확인할 것')
            return 1
        for i in range(args.count):
            deadline = time.time() + args.interval
            while time.time() < deadline:      # 최신 프레임으로 갱신
                rclpy.spin_once(node, timeout_sec=0.05)
            stem = f'{args.prefix}_{time.strftime("%H%M%S")}_{i:02d}'
            cv2.imwrite(str(out / f'{stem}.png'), node.color)
            np.save(out / f'{stem}_depth.npy', node.depth)
            (out / f'{stem}_info.json').write_text(json.dumps(node.info, indent=2))
            print(f'saved {out / stem}.png  ({node.color.shape}, depth {node.depth.dtype})')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

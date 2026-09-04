import time
from dataclasses import dataclass

import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image


@dataclass
class LiveFrame:
    rgb: np.ndarray
    depth_mm: np.ndarray
    K: np.ndarray


class LiveSceneCapture(Node):
    def __init__(self, config):
        super().__init__('grasp_test_capture')
        self._rgb = self._depth = self._K = None
        self._color_seq = self._depth_seq = 0
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, config['color_topic'], self._color, qos)
        self.create_subscription(Image, config['depth_topic'], self._depth_callback, qos)
        self.create_subscription(CameraInfo, config['info_topic'], self._info, 10)

    def _color(self, msg):
        if msg.encoding not in ('rgb8', 'bgr8'):
            return
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        self._rgb = image if msg.encoding == 'rgb8' else image[:, :, ::-1].copy()
        self._color_seq += 1

    def _depth_callback(self, msg):
        if msg.encoding not in ('16UC1', 'mono16', '32FC1'):
            return
        dtype = np.uint16 if msg.encoding in ('16UC1', 'mono16') else np.float32
        depth = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)
        self._depth = depth.astype(np.float32) if dtype is np.uint16 else depth.astype(np.float32) * 1000.0
        self._depth_seq += 1

    def _info(self, msg):
        self._K = np.asarray(msg.k, dtype=np.float32).reshape(3, 3)

    def capture(self, timeout_s=5.0):
        import rclpy
        # A stored frame may be several trials old. Each call waits for a new RGB and depth
        # message so every model trial observes the object after the operator reset.
        color_seq, depth_seq = self._color_seq, self._depth_seq
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
            if (self._color_seq > color_seq and self._depth_seq > depth_seq
                    and self._rgb is not None and self._depth is not None and self._K is not None):
                if self._rgb.shape[:2] == self._depth.shape:
                    return LiveFrame(self._rgb.copy(), self._depth.copy(), self._K.copy()), ''
                return None, 'RGB_DEPTH_SHAPE_MISMATCH'
        return None, 'CAMERA_INPUT_TIMEOUT'

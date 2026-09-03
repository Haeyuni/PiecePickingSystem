#!/usr/bin/env python3
"""라이브 카메라 스트림에 best.pt를 물려서 검출을 실시간 확인한다.

  source /opt/ros/jazzy/setup.bash
  .venv/bin/python scripts/perception_test_live.py            # 오버레이를 /perception/debug_image로 발행
  rqt_image_view /perception/debug_image                      # (다른 터미널) 화면으로 보기
  .venv/bin/python scripts/perception_test_live.py --show     # 창을 직접 띄우고 싶으면

perception 노드 본체가 아니라 모델 검증용 임시 도구다. world_state_raw는 발행하지 않고,
검출별로 클래스/신뢰도/마스크 픽셀수/카메라 좌표계 3D 중심(mm)만 로그로 찍는다.
"""
import argparse
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from perception_capture import image_to_numpy          # noqa: E402
from perception_test_image import mask_3d              # noqa: E402

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / 'model' / 'best.pt'


class LiveTest(Node):
    def __init__(self, args):
        super().__init__('perception_test_live')
        import cv2
        from ultralytics import YOLO
        self.cv2 = cv2
        self.args = args
        self.model = YOLO(args.model)
        self.get_logger().info(f'model={args.model} task={self.model.task} names={self.model.names}')

        self.depth = None
        self.info = None
        self._busy = False

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, args.depth_topic, self._on_depth, qos)
        self.create_subscription(CameraInfo, args.info_topic, self._on_info, 10)
        self.create_subscription(Image, args.color_topic, self._on_color, qos)
        self.debug_pub = self.create_publisher(Image, '/perception/debug_image', 1)

    def _on_depth(self, msg):
        self.depth = image_to_numpy(msg)

    def _on_info(self, msg):
        self.info = {'fx': msg.k[0], 'fy': msg.k[4], 'cx': msg.k[2], 'cy': msg.k[5]}

    def _on_color(self, msg):
        # 추론이 프레임레이트보다 느리므로, 처리 중에 들어온 프레임은 그냥 버린다
        if self._busy:
            return
        self._busy = True
        try:
            self._process(image_to_numpy(msg), msg.header)
        finally:
            self._busy = False

    def _process(self, bgr, header):
        a = self.args
        res = self.model.predict(bgr, conf=a.conf, imgsz=a.imgsz,
                                 device=a.device, verbose=False)[0]
        n = 0 if res.boxes is None else len(res.boxes)
        lines = []
        for i in range(n):
            cls, conf = int(res.boxes.cls[i]), float(res.boxes.conf[i])
            item = f'{res.names[cls]} {conf:.2f}'
            if res.masks is not None and self.depth is not None and self.info is not None:
                m = res.masks.data[i].cpu().numpy() > 0.5
                if m.shape != bgr.shape[:2]:
                    m = self.cv2.resize(m.astype(np.uint8), (bgr.shape[1], bgr.shape[0]),
                                        interpolation=self.cv2.INTER_NEAREST).astype(bool)
                p, ratio = mask_3d(m, self.depth, self.info)
                item += (f' valid={ratio:.2f} ' +
                         (f'cam_mm=({p[0]:.0f},{p[1]:.0f},{p[2]:.0f})' if p else 'cam_mm=None'))
            lines.append(item)
        self.get_logger().info(f'{n} det | ' + ' | '.join(lines) if n else '0 det',
                               throttle_duration_sec=a.log_period)

        vis = res.plot()
        out = Image()
        out.header = header
        out.height, out.width = vis.shape[:2]
        out.encoding = 'bgr8'
        out.is_bigendian = 0
        out.step = out.width * 3
        out.data = vis.tobytes()
        self.debug_pub.publish(out)
        if self.args.show:
            self.cv2.imshow('perception test (q: quit)', vis)
            if self.cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=str(DEFAULT_MODEL))
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--device', default=None)
    ap.add_argument('--show', action='store_true', help='cv2 창으로 직접 표시')
    ap.add_argument('--log-period', type=float, default=1.0, help='로그 throttle 주기(초)')
    ap.add_argument('--color-topic', default='/camera/color/image_raw')
    ap.add_argument('--depth-topic', default='/camera/aligned_depth_to_color/image_raw')
    ap.add_argument('--info-topic', default='/camera/color/camera_info')
    args = ap.parse_args()

    rclpy.init()
    node = LiveTest(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():          # SIGTERM으로 죽을 때는 이미 shutdown된 상태다
            rclpy.shutdown()


if __name__ == '__main__':
    main()

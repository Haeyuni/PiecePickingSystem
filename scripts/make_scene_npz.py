#!/usr/bin/env python3
"""RealSense 라이브 프레임 + model/best.pt(YOLO11-seg)로 비교용 scene_XXX.npz를 만든다.

  source /opt/ros/jazzy/setup.bash
  .venv/bin/python scripts/make_scene_npz.py --list                  # 지금 뭐가 보이는지만 확인
  .venv/bin/python scripts/make_scene_npz.py --class-name tape       # tape 하나를 담아 저장
  .venv/bin/python scripts/make_scene_npz.py --pick 1 --scene-id scene_002
  .venv/bin/python scripts/make_scene_npz.py --verify data/scenes/scene_001.npz   # ROS 없이 검사만

저장 규격 (rg2_grasp_compare/scenes/):
  rgb       uint8   (H,W,3)  RGB 순서 (OpenCV BGR 아님)
  depth_m   float32 (H,W)    미터. 무효 픽셀은 0.0
  mask      bool    (H,W)    목표 물체 하나만 True
  K         float64 (3,3)    + fx, fy, cx, cy 스칼라도 같이 넣는다
  scene_id / class_name / confidence, 그리고 참고용 depth_valid_ratio·mask_px

points_xyz는 넣지 않는다 — 비교 노트북이 depth_m + K + mask로 만든다.
cv_bridge는 쓰지 않는다 (venv opencv와 시스템 cv_bridge가 섞이면 깨진다).
"""
import argparse
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / 'model' / 'best.pt'
DEFAULT_OUT = ROOT / 'data' / 'scenes'


# ---------------------------------------------------------------- 검사

def verify(path: Path) -> int:
    """저장된 npz가 규격을 지키는지 확인하고, 노트북과 같은 방식으로 3D 점을 복원해 본다."""
    d = np.load(path, allow_pickle=False)
    print(f'{path}  keys={sorted(d.files)}')
    problems = []

    for k in ('rgb', 'depth_m', 'mask'):
        if k not in d.files:
            problems.append(f'{k} 없음')
    if problems:
        print('  FAIL:', '; '.join(problems))
        return 1

    rgb, depth, mask = d['rgb'], d['depth_m'], d['mask']
    print(f'  rgb     {rgb.shape} {rgb.dtype}')
    print(f'  depth_m {depth.shape} {depth.dtype}  min={depth.min():.3f} max={depth.max():.3f}')
    print(f'  mask    {mask.shape} {mask.dtype}  true_px={int(mask.astype(bool).sum())}')

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        problems.append('rgb가 (H,W,3)이 아니다')
    if depth.shape != rgb.shape[:2] or mask.shape != rgb.shape[:2]:
        problems.append('rgb / depth_m / mask 해상도가 다르다')
    if depth.max() > 20.0:
        problems.append(f'depth 최대 {depth.max():.1f} — mm 단위로 저장된 것 같다 (m여야 한다)')
    if not mask.astype(bool).any():
        problems.append('mask가 전부 False다')

    if 'K' in d.files:
        K = d['K']
        print(f'  K       {K.tolist()}')
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    elif all(k in d.files for k in ('fx', 'fy', 'cx', 'cy')):
        fx, fy, cx, cy = (float(d[k]) for k in ('fx', 'fy', 'cx', 'cy'))
    else:
        print('  FAIL: K도 fx/fy/cx/cy도 없다')
        return 1
    print(f'  fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}')
    for k in ('scene_id', 'class_name', 'confidence'):
        if k in d.files:
            print(f'  {k} = {d[k]}')

    # 노트북이 하는 것과 같은 역투영
    m = mask.astype(bool) & (depth > 0)
    ys, xs = np.nonzero(m)
    ratio = float(m.sum()) / max(int(mask.astype(bool).sum()), 1)
    if xs.size == 0:
        problems.append('마스크 안에 유효 depth가 하나도 없다')
    else:
        z = depth[ys, xs].astype(np.float64)
        x = (xs - cx) * z / fx
        y = (ys - cy) * z / fy
        print(f'  points  n={xs.size} depth_valid={ratio:.2f} '
              f'x[{x.min():.3f},{x.max():.3f}] y[{y.min():.3f},{y.max():.3f}] '
              f'z[{z.min():.3f},{z.max():.3f}] (m)')
        size = (x.max() - x.min(), y.max() - y.min(), z.max() - z.min())
        print(f'  bbox    {size[0]*1000:.0f} x {size[1]*1000:.0f} x {size[2]*1000:.0f} mm')
        if ratio < 0.5:
            problems.append(f'마스크 안 유효 depth 비율이 {ratio:.2f}로 낮다 — 다시 찍는 게 좋다')
        if max(size[:2]) > 0.5:
            problems.append('물체 크기가 50cm를 넘는다 — 마스크에 배경이 섞였을 수 있다')

    if problems:
        print('  ! ' + '\n  ! '.join(problems))
        return 1
    print('  OK — 업로드해도 된다')
    return 0


# ---------------------------------------------------------------- 캡처

def grab_frames(args):
    """ROS 토픽에서 color / aligned depth / intrinsics 한 세트를 받아온다."""
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from perception_capture import image_to_numpy

    class Grab(Node):
        def __init__(self):
            super().__init__('make_scene_npz')
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.color = self.depth = self.info = None
            self.create_subscription(Image, args.color_topic, self._c, qos)
            self.create_subscription(Image, args.depth_topic, self._d, qos)
            self.create_subscription(CameraInfo, args.info_topic, self._i, 10)

        def _c(self, m):
            self.color = image_to_numpy(m)

        def _d(self, m):
            self.depth = image_to_numpy(m)

        def _i(self, m):
            self.info = {'fx': m.k[0], 'fy': m.k[4], 'cx': m.k[2], 'cy': m.k[5]}

    rclpy.init()
    node = Grab()
    try:
        deadline = time.time() + args.timeout
        while time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.color is not None and node.depth is not None and node.info is not None:
                break
        else:
            raise SystemExit(f'프레임을 못 받았다 — {args.color_topic} / {args.depth_topic} 확인')
        if args.warmup > 0:                       # 자동노출·depth가 안정된 뒤의 최신 프레임을 쓴다
            print(f'{args.warmup:.1f}s 대기 후 촬영...')
            end = time.time() + args.warmup
            while time.time() < end:
                rclpy.spin_once(node, timeout_sec=0.05)
        return node.color.copy(), node.depth.copy(), dict(node.info)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def detect(bgr, args):
    """(mask_bool, class_name, confidence) 목록을 원본 해상도로 돌려준다."""
    import cv2
    from ultralytics import YOLO
    model = YOLO(args.model)
    if model.task != 'segment':
        raise SystemExit(f'{args.model} 는 task={model.task} — seg 모델이 필요하다')
    res = model.predict(bgr, conf=args.conf, imgsz=args.imgsz,
                        device=args.device, verbose=False)[0]
    h, w = bgr.shape[:2]
    out = []
    n = 0 if res.boxes is None else len(res.boxes)
    for i in range(n):
        if res.masks is None:
            continue
        m = res.masks.data[i].cpu().numpy() > 0.5
        if m.shape != (h, w):                     # 마스크는 추론 해상도 기준이라 원본으로 되돌린다
            m = cv2.resize(m.astype(np.uint8), (w, h),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        out.append((m, res.names[int(res.boxes.cls[i])], float(res.boxes.conf[i])))
    return out, res


def next_scene_id(out_dir: Path) -> str:
    used = {int(p.stem.split('_')[1]) for p in out_dir.glob('scene_*.npz')
            if p.stem.split('_')[1].isdigit()}
    return f'scene_{(max(used) + 1 if used else 1):03d}'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--verify', metavar='NPZ', help='저장된 npz만 검사하고 끝낸다 (ROS 불필요)')
    ap.add_argument('--list', action='store_true', help='검출 결과만 보고 저장하지 않는다')
    ap.add_argument('--scene-id', help='기본값은 out 디렉터리의 다음 번호')
    ap.add_argument('-o', '--out', default=str(DEFAULT_OUT))
    ap.add_argument('--class-name', help='이 클래스 중 신뢰도 최고를 목표로 삼는다')
    ap.add_argument('--pick', type=int, help='검출 목록의 인덱스를 직접 지정')
    ap.add_argument('--model', default=str(DEFAULT_MODEL))
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--device', default=None, help='0 이면 GPU, cpu 면 CPU')
    ap.add_argument('--depth-scale', type=float, default=0.001,
                    help='depth 원본 -> m 배율. ROS aligned depth는 uint16 mm라 0.001')
    ap.add_argument('--warmup', type=float, default=1.0, help='촬영 전 대기 초')
    ap.add_argument('--timeout', type=float, default=10.0)
    ap.add_argument('--no-preview', action='store_true', help='확인용 png를 만들지 않는다')
    ap.add_argument('--color-topic', default='/camera/color/image_raw')
    ap.add_argument('--depth-topic', default='/camera/aligned_depth_to_color/image_raw')
    ap.add_argument('--info-topic', default='/camera/color/camera_info')
    args = ap.parse_args()

    if args.verify:
        return verify(Path(args.verify))

    import cv2
    bgr, depth_raw, info = grab_frames(args)
    if depth_raw.shape != bgr.shape[:2]:
        raise SystemExit(f'color {bgr.shape[:2]} 와 depth {depth_raw.shape} 해상도가 다르다 — '
                         f'aligned_depth_to_color 토픽이 맞는지 확인할 것')
    print(f'frame: color {bgr.shape} depth {depth_raw.shape} {depth_raw.dtype} '
          f'fx={info["fx"]:.1f} cx={info["cx"]:.1f}')

    dets, res = detect(bgr, args)
    if not dets:
        raise SystemExit(f'검출 0개 (conf={args.conf}) — 물체를 화면 안에 두고 다시 시도')
    depth_m = depth_raw.astype(np.float32) * args.depth_scale
    for i, (m, name, conf) in enumerate(dets):
        valid = float((depth_m[m] > 0).mean()) if m.any() else 0.0
        z = depth_m[m & (depth_m > 0)]
        zc = float(np.median(z)) if z.size else float('nan')
        print(f'  [{i}] {name:<13} conf={conf:.3f} mask_px={int(m.sum()):>6} '
              f'depth_valid={valid:.2f} z_median={zc:.3f} m')
    if args.list:
        return 0

    if args.pick is not None:
        if not 0 <= args.pick < len(dets):
            raise SystemExit(f'--pick {args.pick} 은 범위 밖 (0..{len(dets)-1})')
        idx = args.pick
    else:
        cand = [i for i, (_, n, _) in enumerate(dets)
                if args.class_name is None or n == args.class_name]
        if not cand:
            raise SystemExit(f'클래스 {args.class_name} 가 안 잡혔다')
        idx = max(cand, key=lambda i: dets[i][2])
    mask, class_name, conf = dets[idx]
    print(f'선택: [{idx}] {class_name} conf={conf:.3f}')

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_id = args.scene_id or next_scene_id(out_dir)
    dst = out_dir / f'{scene_id}.npz'

    fx, fy, cx, cy = (float(info[k]) for k in ('fx', 'fy', 'cx', 'cy'))
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    mask_valid = mask & (depth_m > 0)
    np.savez_compressed(
        dst,
        rgb=np.ascontiguousarray(bgr[:, :, ::-1]),        # BGR -> RGB
        depth_m=depth_m,
        mask=mask.astype(bool),                           # 목표 물체 하나만 True
        K=K, fx=fx, fy=fy, cx=cx, cy=cy,
        scene_id=scene_id, class_name=class_name, confidence=np.float32(conf),
        depth_valid_ratio=np.float32(mask_valid.sum() / max(int(mask.sum()), 1)),
        mask_px=np.int32(mask.sum()),
    )
    print(f'saved {dst}  ({dst.stat().st_size / 1e6:.1f} MB)')

    if not args.no_preview:
        vis = res.plot()
        edge = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) - mask.astype(np.uint8)
        vis[edge > 0] = (0, 255, 255)                     # 선택한 물체 외곽선만 노란색
        prev = out_dir / f'{scene_id}_preview.png'
        cv2.imwrite(str(prev), vis)
        print(f'preview {prev}  — 노란 외곽선이 목표 물체인지 눈으로 확인할 것')

    print()
    return verify(dst)


if __name__ == '__main__':
    raise SystemExit(main())

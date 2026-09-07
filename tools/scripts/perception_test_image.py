#!/usr/bin/env python3
"""model/best.pt를 정지 이미지에 돌려서 검출 결과를 눈으로 확인한다 (ROS 불필요).

  .venv/bin/python scripts/perception_test_image.py data/samples/*.png
  .venv/bin/python scripts/perception_test_image.py data/samples --conf 0.4

같은 stem의 *_depth.npy / *_info.json이 있으면 마스크 기준 3D 중심(camera frame, mm)과
depth_valid_ratio까지 같이 뽑는다 — DetectedObject.position_base_mm 채우기 전 단계 검증용.
결과 이미지는 --out 디렉터리(기본 data/samples/pred)에 저장된다.
"""
import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / 'models' / 'best.pt'


def collect(paths):
    files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files += sorted(q for q in p.iterdir()
                            if q.suffix.lower() in {'.png', '.jpg', '.jpeg'})
        elif p.is_file():
            files.append(p)
    return files


def mask_3d(mask_bool, depth_mm, info):
    """마스크 안쪽 유효 depth의 median으로 카메라 좌표계 3D 점(mm)을 만든다."""
    ys, xs = np.nonzero(mask_bool)
    if xs.size == 0:
        return None, 0.0
    z = depth_mm[ys, xs].astype(np.float32)
    valid = z > 0
    ratio = float(valid.mean())
    if not valid.any():
        return None, ratio
    # 중심 픽셀 대신 유효 픽셀의 median depth를 쓴다 (가장자리 튀는 값에 안 흔들리게)
    zc = float(np.median(z[valid]))
    uc, vc = float(xs[valid].mean()), float(ys[valid].mean())
    x = (uc - info['cx']) * zc / info['fx']
    y = (vc - info['cy']) * zc / info['fy']
    return (x, y, zc), ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('images', nargs='+', help='이미지 파일 또는 디렉터리')
    ap.add_argument('--model', default=str(DEFAULT_MODEL))
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--device', default=None, help='0 이면 GPU, cpu 면 CPU')
    ap.add_argument('--out', default='data/samples/pred')
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    files = collect(args.images)
    if not files:
        print('입력 이미지가 없다')
        return 1

    model = YOLO(args.model)
    print(f'model={args.model} task={model.task} names={model.names}')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for f in files:
        res = model.predict(str(f), conf=args.conf, imgsz=args.imgsz,
                            device=args.device, verbose=False)[0]
        depth_p, info_p = f.with_name(f.stem + '_depth.npy'), f.with_name(f.stem + '_info.json')
        depth = np.load(depth_p) if depth_p.exists() else None
        info = json.loads(info_p.read_text()) if info_p.exists() else None

        n = 0 if res.boxes is None else len(res.boxes)
        print(f'\n{f.name}: {n} detections  ({res.speed["inference"]:.1f} ms)')
        for i in range(n):
            cls = int(res.boxes.cls[i])
            conf = float(res.boxes.conf[i])
            x1, y1, x2, y2 = res.boxes.xyxy[i].tolist()
            line = (f'  [{i}] {res.names[cls]:<13} conf={conf:.3f} '
                    f'box=({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})')
            if res.masks is not None:
                m = res.masks.data[i].cpu().numpy() > 0.5
                if m.shape != res.orig_shape:      # 마스크는 추론 해상도 기준이라 원본으로 되돌린다
                    m = cv2.resize(m.astype(np.uint8), res.orig_shape[::-1],
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
                line += f' mask_px={int(m.sum())}'
                if depth is not None and info is not None:
                    p, ratio = mask_3d(m, depth, info)
                    line += (f' depth_valid={ratio:.2f}' +
                             (f' cam_mm=({p[0]:.0f},{p[1]:.0f},{p[2]:.0f})' if p else ' cam_mm=None'))
            print(line)

        dst = out / f'{f.stem}_pred.png'
        cv2.imwrite(str(dst), res.plot())
        print(f'  -> {dst}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

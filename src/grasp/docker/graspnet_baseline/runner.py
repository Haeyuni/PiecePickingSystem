import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/opt/graspnet-baseline')
for item in (ROOT / 'models', ROOT / 'dataset', ROOT / 'utils'):
    sys.path.insert(0, str(item))

from graspnet import GraspNet, pred_decode
from graspnetAPI import GraspGroup


def main(input_path, output_path, checkpoint_path, device_name, num_points, min_width_m,
         max_width_m, max_candidates):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU를 찾지 못했습니다')
    device = torch.device(device_name)
    points = np.load(input_path, allow_pickle=False)['points_cam_m']
    if len(points) == 0:
        raise RuntimeError('입력 point cloud가 비어 있습니다')
    if not np.isfinite(points).all():
        raise RuntimeError('입력 points_cam_m에 유한하지 않은 값이 있습니다')
    print(f'GraspNet input: camera frame, meter, points={len(points)}, '
          f'z=[{points[:, 2].min():.4f}, {points[:, 2].max():.4f}]')
    rng = np.random.default_rng(0)
    indexes = rng.choice(len(points), int(num_points), replace=len(points) < int(num_points))
    net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                   cylinder_radius=.05, hmin=-.02, hmax_list=[.01, .02, .03, .04],
                   is_training=False).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(state['model_state_dict'])
    net.eval()
    input_points = torch.from_numpy(points[indexes][None].astype(np.float32)).to(device)
    with torch.no_grad():
        pred = pred_decode(net({
            'point_clouds': input_points,
            'cloud_colors': np.zeros((int(num_points), 3), dtype=np.float32),
        }))[0].detach().cpu().numpy()
    group = GraspGroup(pred)
    group = group[(group.widths >= float(min_width_m)) & (group.widths <= float(max_width_m))]
    group.nms()
    group.sort_by_score()
    candidates = []
    for grasp in group[:int(max_candidates)]:
        candidates.append({
            'score': float(grasp.score),
            'width_m': float(grasp.width),
            'rotation_matrix': grasp.rotation_matrix.tolist(),
            'translation_m': grasp.translation.tolist(),
        })
    Path(output_path).write_text(json.dumps({
        'input_frame': 'camera', 'input_unit': 'm', 'candidates': candidates,
    }), encoding='utf-8')


if __name__ == '__main__':
    try:
        if sys.argv[1] == '--check-checkpoint':
            state = torch.load(sys.argv[2], map_location='cpu')
            if 'model_state_dict' not in state:
                raise RuntimeError('model_state_dict가 없습니다')
            print('checkpoint.tar 확인 완료')
        else:
            main(*sys.argv[1:])
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise

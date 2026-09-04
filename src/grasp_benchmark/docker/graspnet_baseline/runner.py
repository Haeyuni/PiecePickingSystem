import sys
import tarfile
import time
from pathlib import Path

import gdown
import numpy as np
import torch

from common.scene_io import RG2_MAX_WIDTH_M, RG2_MIN_WIDTH_M, load_scene, pixel_from_xyz, result, write_result

ROOT = Path('/opt/graspnet-baseline')
for item in (ROOT / 'models', ROOT / 'dataset', ROOT / 'utils'):
    sys.path.insert(0, str(item))
from graspnet import GraspNet, pred_decode
from graspnetAPI import GraspGroup

CHECKPOINT_ID = '1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk'


def checkpoint():
    root = Path('/models/graspnet')
    root.mkdir(parents=True, exist_ok=True)
    files = list(root.rglob('checkpoint.tar'))
    if not files:
        archive = root / 'checkpoint-rs.tar'
        gdown.download(id=CHECKPOINT_ID, output=str(archive), quiet=False)
        with tarfile.open(archive) as opened:
            opened.extractall(root)
        files = list(root.rglob('checkpoint.tar'))
    if not files:
        raise FileNotFoundError('GraspNet checkpoint.tar를 찾지 못했습니다.')
    return files[0]


def main(path):
    scene = load_scene(path)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU를 찾지 못했습니다.')
        init_start = time.perf_counter()
        points = scene['target_points']
        rng = np.random.default_rng(0)
        index = rng.choice(len(points), 20000, replace=len(points) < 20000)
        net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4, cylinder_radius=.05,
                       hmin=-.02, hmax_list=[.01, .02, .03, .04], is_training=False).cuda()
        state = torch.load(checkpoint(), map_location='cuda')
        net.load_state_dict(state['model_state_dict'])
        net.eval()
        input_points = torch.from_numpy(points[index][None].astype(np.float32)).cuda()
        initialization_ms = round((time.perf_counter() - init_start) * 1000, 2)
        with torch.no_grad():
            net({'point_clouds': input_points, 'cloud_colors': np.zeros((20000, 3), dtype=np.float32)})
            torch.cuda.synchronize()
            start = time.perf_counter()
            pred = pred_decode(net({'point_clouds': input_points, 'cloud_colors': np.zeros((20000, 3), dtype=np.float32)}))[0].detach().cpu().numpy()
            torch.cuda.synchronize()
        inference_ms = round((time.perf_counter() - start) * 1000, 2)
        group = GraspGroup(pred)
        total = len(group)
        group = group[(group.widths >= RG2_MIN_WIDTH_M) & (group.widths <= RG2_MAX_WIDTH_M)]
        if not len(group):
            row = result(scene, 'GraspNet_baseline', 'NO_VALID_GRASP', candidate_count=total,
                         initialization_ms=initialization_ms, inference_ms=inference_ms, width_validation='REJECTED',
                         candidate_definition='raw decoder candidates before NMS', note='RG2 폭 통과 후보 없음')
        else:
            valid = len(group)
            group.nms()
            group.sort_by_score()
            best = group[0]
            x, y, z = (float(value) for value in best.translation)
            u, v = pixel_from_xyz(scene, x, y, z)
            row = result(scene, 'GraspNet_baseline', 'OK', candidate_count=total, valid_width_count=valid,
                         best_score=float(best.score), initialization_ms=initialization_ms, inference_ms=inference_ms,
                         width_m=float(best.width), x_m=x, y_m=y, z_m=z, u_px=u, v_px=v,
                         width_validation='VALID', candidate_definition='raw decoder candidates before NMS',
                         note='객체 마스크 포인트 사용; 장면 충돌·IK 미검사')
    except Exception as exc:
        row = result(scene, 'GraspNet_baseline', 'ERROR', error_message=f'{type(exc).__name__}: {exc}')
    write_result(row)


if __name__ == '__main__':
    main(sys.argv[1])

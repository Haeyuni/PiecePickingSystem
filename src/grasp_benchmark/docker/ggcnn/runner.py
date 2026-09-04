import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.feature import peak_local_max

from common.scene_io import RG2_MAX_WIDTH_M, RG2_MIN_WIDTH_M, load_scene, result, write_result

sys.path.insert(0, '/opt/ggcnn')
from models.ggcnn import GGCNN
from models.common import post_process_output

WEIGHT_URL = 'https://github.com/dougsm/ggcnn/releases/download/v0.1/ggcnn_weights_cornell.zip'


def model_path():
    model_dir = Path('/models/ggcnn')
    model_dir.mkdir(parents=True, exist_ok=True)
    archive = model_dir / 'ggcnn_weights_cornell.zip'
    if not archive.exists():
        try:
            subprocess.run(['wget', '-q', WEIGHT_URL, '-O', str(archive)], check=True)
        except (subprocess.CalledProcessError, OSError) as exc:
            archive.unlink(missing_ok=True)  # drop any partial file so the next run retries cleanly
            print(f'ERROR: GG-CNN checkpoint download failed ({WEIGHT_URL}): {exc}', file=sys.stderr)
            raise
    files = list(model_dir.rglob('ggcnn_epoch*'))
    if not files:
        try:
            shutil.unpack_archive(archive, model_dir)
        except Exception as exc:
            archive.unlink(missing_ok=True)
            print(f'ERROR: GG-CNN checkpoint archive is corrupt ({archive}): {exc}', file=sys.stderr)
            raise
        files = list(model_dir.rglob('ggcnn_epoch*'))
    if not files:
        message = f'GG-CNN 체크포인트 압축 해제 후 가중치 파일을 찾지 못했습니다: {archive}'
        print(f'ERROR: {message}', file=sys.stderr)
        raise FileNotFoundError(message)
    return next((path for path in files if 'state' in path.name.lower()), files[0])


def main(path):
    scene = load_scene(path)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU를 찾지 못했습니다.')
        init_start = time.perf_counter()
        model = GGCNN().cuda()
        loaded = torch.load(model_path(), map_location='cuda')
        state = loaded.get('model_state_dict', loaded.get('state_dict', loaded)) if isinstance(loaded, dict) else None
        if state is not None:
            model.load_state_dict(state)
        elif isinstance(loaded, torch.nn.Module):
            model = loaded.cuda()
        model.eval()
        initialization_ms = round((time.perf_counter() - init_start) * 1000, 2)
        depth = scene['depth_m'].copy()
        valid_depth = np.isfinite(depth) & (depth > 0)
        depth[~valid_depth] = np.median(depth[valid_depth])
        depth = np.clip(depth - depth[valid_depth].mean(), -1, 1).astype(np.float32)
        h, w = depth.shape
        inp = F.interpolate(torch.from_numpy(depth)[None, None].cuda(), size=(300, 300), mode='bilinear', align_corners=False)
        with torch.no_grad():
            model(inp)  # GPU warm-up is excluded from the measured inference time.
            torch.cuda.synchronize()
            start = time.perf_counter()
            q, cos, sin, width = model(inp)
            torch.cuda.synchronize()
        inference_ms = round((time.perf_counter() - start) * 1000, 2)
        q, angle, width_px = post_process_output(q, cos, sin, width)
        mask = F.interpolate(torch.from_numpy(scene['mask'].astype(np.float32))[None, None], size=(300, 300), mode='nearest').squeeze().numpy().astype(bool)
        q[~mask] = -np.inf
        peaks = peak_local_max(q, min_distance=20, threshold_abs=.2, num_peaks=20)
        fx, fy, cx, cy = scene['K'][0, 0], scene['K'][1, 1], scene['K'][0, 2], scene['K'][1, 2]
        candidates = []
        for py, px in peaks:
            v, u = min(h - 1, int((py + .5) * h / 300)), min(w - 1, int((px + .5) * w / 300))
            z = float(scene['depth_m'][v, u])
            width_m = float(width_px[py, px]) * w / 300 * z / ((fx + fy) / 2)
            if np.isfinite(z) and RG2_MIN_WIDTH_M <= width_m <= RG2_MAX_WIDTH_M:
                candidates.append((float(q[py, px]), width_m, u, v, z))
        if not candidates:
            row = result(scene, 'GG-CNN', 'NO_VALID_GRASP', candidate_count=len(peaks), initialization_ms=initialization_ms,
                         inference_ms=inference_ms, width_validation='REJECTED', candidate_definition='masked quality-map peaks before width filtering', note='마스크 내부 RG2 폭 통과 후보 없음')
        else:
            score, width_m, u, v, z = max(candidates)
            row = result(scene, 'GG-CNN', 'OK', candidate_count=len(peaks), valid_width_count=len(candidates), best_score=score,
                         initialization_ms=initialization_ms, inference_ms=inference_ms, width_m=width_m,
                         x_m=float((u - cx) * z / fx), y_m=float((v - cy) * z / fy), z_m=z, u_px=u, v_px=v,
                         width_validation='VALID', candidate_definition='masked quality-map peaks before width filtering', note='탑다운 후보; 충돌·IK 미검사')
    except Exception as exc:
        row = result(scene, 'GG-CNN', 'ERROR', error_message=f'{type(exc).__name__}: {exc}')
    write_result(row)


if __name__ == '__main__':
    main(sys.argv[1])

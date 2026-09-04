import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RG2_MIN_WIDTH_M = 0.005
RG2_MAX_WIDTH_M = 0.10


def load_scene(path):
    raw = np.load(path, allow_pickle=False)
    missing = {'rgb', 'depth_m', 'K', 'mask'} - set(raw.files)
    if missing:
        raise KeyError(f'필수 키 누락: {sorted(missing)}')
    rgb = np.asarray(raw['rgb'])
    depth = np.asarray(raw['depth_m'], dtype=np.float32)
    mask = np.asarray(raw['mask']).squeeze().astype(bool)
    K = np.asarray(raw['K'], dtype=np.float32).reshape(3, 3)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or depth.shape != rgb.shape[:2] or mask.shape != rgb.shape[:2]:
        raise ValueError(f'입력 shape 오류: rgb={rgb.shape}, depth={depth.shape}, mask={mask.shape}')
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any() or np.nanmedian(depth[valid]) > 10:
        raise ValueError('depth_m는 유효한 미터 단위 depth여야 합니다.')
    v, u = np.where(valid)
    z = depth[v, u]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    points = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z)).astype(np.float32)
    target_flags = mask[valid]
    if target_flags.sum() < 100:
        raise ValueError('마스크 내부 유효 3D 점이 100개 미만입니다.')
    return {'scene_id': str(raw['scene_id'].item()) if 'scene_id' in raw.files else Path(path).stem,
            'rgb': rgb, 'depth_m': depth, 'mask': mask, 'K': K,
            'scene_points': points, 'target_points': points[target_flags]}


def result(scene, method, status, **values):
    row = {'scene_id': scene['scene_id'], 'method': method, 'status': status,
           'candidate_count': 0, 'valid_width_count': 0, 'best_score': None,
           'initialization_ms': None, 'inference_ms': None, 'width_m': None,
           'x_m': None, 'y_m': None, 'z_m': None, 'u_px': None, 'v_px': None,
           'width_validation': 'NOT_CHECKED', 'candidate_definition': '', 'note': '',
           'error_message': '', 'tested_at': datetime.now(timezone.utc).isoformat(timespec='seconds')}
    row.update(values)
    return row


def write_result(row):
    name = f"{row['scene_id']}_{row['method'].lower().replace('-', '_')}.json"
    Path('/results', name).write_text(json.dumps(row, ensure_ascii=False, indent=2))


def pixel_from_xyz(scene, x, y, z):
    fx, fy, cx, cy = scene['K'][0, 0], scene['K'][1, 1], scene['K'][0, 2], scene['K'][1, 2]
    return int(round(x * fx / z + cx)), int(round(y * fy / z + cy))

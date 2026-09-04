import sys
import time
from pathlib import Path

import gdown
import numpy as np

from common.scene_io import load_scene, pixel_from_xyz, result, write_result

ROOT = Path('/opt/contact_graspnet')
sys.path.insert(0, str(ROOT))
# Upstream inference.py runs from this directory, so its sibling imports are available.
sys.path.insert(0, str(ROOT / 'contact_graspnet'))
MODEL_FOLDER = 'https://drive.google.com/drive/folders/1tBHKf60K8DLM5arm-Chyf7jxkzOr5zGl'


def checkpoint_dir():
    root = Path('/models/contact_graspnet')
    root.mkdir(parents=True, exist_ok=True)
    configs = list(root.rglob('config.yaml'))
    if not configs:
        try:
            gdown.download_folder(url=MODEL_FOLDER, output=str(root), quiet=False, remaining_ok=True)
        except Exception as exc:
            print(f'ERROR: Contact_GraspNet checkpoint download failed ({MODEL_FOLDER}): {exc}', file=sys.stderr)
            raise
        configs = list(root.rglob('config.yaml'))
    if not configs:
        message = 'Contact-GraspNet 공개 체크포인트를 찾지 못했습니다.'
        print(f'ERROR: {message}', file=sys.stderr)
        raise FileNotFoundError(message)
    preferred = [path.parent for path in configs if 'scene_test_2048_bs3_hor_sigma_001' in str(path.parent)]
    return preferred[0] if preferred else configs[0].parent


def unpack(predicted, scores):
    if isinstance(predicted, dict):
        keys = [key for key, value in predicted.items() if len(value)]
        if not keys:
            return np.empty((0, 4, 4)), np.empty(0)
        return np.asarray(predicted[keys[0]]), np.asarray(scores[keys[0]])
    return np.asarray(predicted), np.asarray(scores)


def main(path):
    scene = load_scene(path)
    try:
        from config_utils import load_config
        from contact_grasp_estimator import GraspEstimator
        import tensorflow.compat.v1 as tf
        tf.disable_eager_execution()
        if not tf.test.is_gpu_available(cuda_only=True):
            raise RuntimeError('CUDA GPU를 찾지 못했습니다.')
        init_start = time.perf_counter()
        checkpoint = checkpoint_dir()
        config = load_config(str(checkpoint), batch_size=1, arg_configs=[])
        estimator = GraspEstimator(config)
        estimator.build_network()
        saver = tf.train.Saver(save_relative_paths=True)
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_config.allow_soft_placement = True
        session = tf.Session(config=tf_config)
        estimator.load_weights(session, saver, str(checkpoint), mode='test')
        initialization_ms = round((time.perf_counter() - init_start) * 1000, 2)
        pc_full, segments, _ = estimator.extract_point_clouds(scene['depth_m'], scene['K'], segmap=scene['mask'].astype(np.uint8), rgb=scene['rgb'], skip_border_objects=False, z_range=[.2, 1.8])
        estimator.predict_scene_grasps(session, pc_full, pc_segments=segments, local_regions=True, filter_grasps=True, forward_passes=1)
        start = time.perf_counter()
        predicted, scores, _, _ = estimator.predict_scene_grasps(session, pc_full, pc_segments=segments, local_regions=True, filter_grasps=True, forward_passes=1)
        inference_ms = round((time.perf_counter() - start) * 1000, 2)
        grasps, scores = unpack(predicted, scores)
        if not len(grasps):
            row = result(scene, 'Contact_GraspNet', 'NO_VALID_GRASP', initialization_ms=initialization_ms,
                         inference_ms=inference_ms, width_validation='UNAVAILABLE', candidate_definition='filtered 6D pose candidates',
                         note='유효 후보 없음; 공개 출력에 RG2 폭이 없음')
        else:
            best_index = int(np.argmax(scores))
            best = grasps[best_index]
            x, y, z = (float(value) for value in best[:3, 3])
            u, v = pixel_from_xyz(scene, x, y, z)
            row = result(scene, 'Contact_GraspNet', 'OK', candidate_count=len(grasps), best_score=float(scores[best_index]),
                         initialization_ms=initialization_ms, inference_ms=inference_ms, x_m=x, y_m=y, z_m=z, u_px=u, v_px=v,
                         width_validation='UNAVAILABLE', candidate_definition='filtered 6D pose candidates',
                         note='6D 후보 생성 성공. 공개 출력에 RG2 폭이 없어 폭 통과 여부는 미판정')
        session.close()
    except Exception as exc:
        row = result(scene, 'Contact_GraspNet', 'ERROR', error_message=f'{type(exc).__name__}: {exc}')
    write_result(row)


if __name__ == '__main__':
    main(sys.argv[1])

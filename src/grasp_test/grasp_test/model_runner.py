import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from ament_index_python.packages import get_package_share_directory

METHODS = ('pca_normal', 'ggcnn', 'graspnet_baseline', 'contact_graspnet')
DISPLAY = {'pca_normal': 'PCA_Normal', 'ggcnn': 'GG-CNN', 'graspnet_baseline': 'GraspNet_baseline', 'contact_graspnet': 'Contact_GraspNet'}
CHECKPOINT_SUBDIRS = ('ggcnn', 'graspnet', 'contact_graspnet')


class ModelRunner:
    def __init__(self, checkpoints, max_inference_ms):
        self._checkpoints = Path(checkpoints)
        # Created here (not left to `docker run -v`) so the bind-mounted host dirs are
        # owned by the invoking user instead of being auto-created as root by dockerd,
        # which would block the in-container first-run checkpoint download.
        for name in CHECKPOINT_SUBDIRS:
            (self._checkpoints / name).mkdir(parents=True, exist_ok=True)
        self._max_inference_ms = float(max_inference_ms)
        self._share = Path(get_package_share_directory('grasp_benchmark'))

    def run(self, method, scene_path, output_dir, trial_id):
        log_path = output_dir / f'{trial_id}_{method}.log'
        if shutil.which('docker') is None:
            return None, 'DOCKER_UNAVAILABLE', log_path
        dockerfile = self._share / 'docker' / method / 'Dockerfile'
        image = f'piece-picking-grasp-test-{method}:latest'
        with log_path.open('w', encoding='utf-8') as log:
            build = subprocess.run(['docker', 'build', '-t', image, '-f', str(dockerfile), str(self._share)], stdout=log, stderr=subprocess.STDOUT)
            if build.returncode:
                return None, 'MODEL_BUILD_FAILED', log_path
            command = ['docker', 'run', '--rm', '-v', f'{scene_path.parent}:/scenes:ro', '-v', f'{output_dir}:/results', '-v', f'{self._checkpoints}:/models']
            if method != 'pca_normal':
                command += ['--gpus', 'all']
            command += [image, f'/scenes/{scene_path.name}']
            ran = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        result_path = output_dir / f'{scene_path.stem}_{method}.json'
        if ran.returncode or not result_path.is_file():
            return None, 'MODEL_CONTAINER_FAILED', log_path
        result = json.loads(result_path.read_text())
        if result.get('status') != 'OK':
            return result, result.get('status', 'MODEL_ERROR'), log_path
        if method == 'contact_graspnet' and float(result.get('inference_ms') or 0) > self._max_inference_ms:
            return result, 'UNUSABLE_FOR_REALTIME', log_path
        return result, '', log_path

    @staticmethod
    def write_scene(frame, target, scene_path, scene_id):
        np.savez_compressed(scene_path, rgb=frame.rgb, depth_m=frame.depth_mm / 1000.0, K=frame.K,
                            mask=target['mask'].astype(np.uint8), scene_id=scene_id)

import numpy as np
import pytest

from grasp import strategies
from grasp.strategies import graspnet_baseline
from grasp.strategies.exceptions import InferenceBusy


def test_registry_exposes_graspnet_baseline():
    assert strategies.get('graspnet_baseline') is graspnet_baseline.plan


def test_camera_meter_output_becomes_base_mm_pose():
    T_base_camera_mm = np.eye(4)
    T_base_camera_mm[:3, 3] = (100.0, -50.0, 200.0)
    T_graspnet_tcp_mm = np.eye(4)
    T_graspnet_tcp_mm[:3, 3] = (10.0, 20.0, 30.0)
    candidate = graspnet_baseline._base_candidate({
        'score': 0.8,
        'width_m': 0.04,
        'rotation_matrix': np.eye(3).tolist(),
        'translation_m': [0.1, 0.2, 0.3],
    }, T_base_camera_mm, T_graspnet_tcp_mm)

    assert candidate['strategy'] == 'graspnet_baseline'
    assert candidate['width_mm'] == 40.0
    assert candidate['pose']['position'] == {'x': 210.0, 'y': 170.0, 'z': 530.0}


def test_missing_tcp_transform_is_not_replaced_by_pca():
    with pytest.raises(RuntimeError, match='T_graspnet_tcp_mm'):
        graspnet_baseline.plan(
            np.empty((0, 3)),
            {'checkpoint_path': '/does/not/exist/checkpoint.tar'},
            context={'points_cam_mm': np.ones((80, 3)), 'T_base_camera_mm': np.eye(4)},
        )


def test_missing_checkpoint_is_not_replaced_by_pca():
    with pytest.raises(RuntimeError, match='checkpoint_path'):
        graspnet_baseline.plan(
            np.empty((0, 3)),
            {
                'checkpoint_path': '/does/not/exist/checkpoint.tar',
                'T_graspnet_tcp_mm': np.eye(4).tolist(),
            },
            context={'points_cam_mm': np.ones((80, 3)), 'T_base_camera_mm': np.eye(4)},
        )


def test_busy_inference_does_not_become_empty_candidates(monkeypatch, tmp_path):
    checkpoint = tmp_path / 'checkpoint.tar'
    checkpoint.touch()
    monkeypatch.setattr(graspnet_baseline, '_require_image', lambda image: None)
    assert graspnet_baseline._INFERENCE_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(InferenceBusy):
            graspnet_baseline.plan(
                np.empty((0, 3)),
                {
                    'checkpoint_path': str(checkpoint),
                    'T_graspnet_tcp_mm': np.eye(4).tolist(),
                },
                context={'points_cam_mm': np.ones((80, 3)), 'T_base_camera_mm': np.eye(4)},
            )
    finally:
        graspnet_baseline._INFERENCE_LOCK.release()

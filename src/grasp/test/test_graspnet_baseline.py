import numpy as np
import pytest

from grasp import strategies
from grasp.strategies import graspnet_baseline
from grasp.strategies.exceptions import InferenceBusy


def test_registry_exposes_graspnet_baseline():
    assert strategies.get('graspnet_baseline') is graspnet_baseline.plan


def _raw(rotation, translation_m, *, width_m=0.04, depth_m=0.0, score=0.8):
    return {'score': score, 'width_m': width_m, 'depth_m': depth_m,
            'rotation_matrix': np.asarray(rotation).tolist(),
            'translation_m': list(translation_m)}


def test_camera_meter_output_becomes_base_mm_pose():
    """camera/m 후보 하나가 base/mm TCP 자세로 옮겨지는지 (되잡기 없이, 순수 변환)."""
    T_base_camera_mm = np.eye(4)
    T_base_camera_mm[:3, 3] = (100.0, -50.0, 200.0)
    T_graspnet_tcp_mm = np.eye(4)
    T_graspnet_tcp_mm[:3, 3] = (10.0, 20.0, 30.0)

    # 접근축(회전 1열)이 base -Z를 향해야 기울기 필터를 통과한다.
    rotation = np.column_stack([(0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)])
    best, _ = graspnet_baseline._select_best(
        [_raw(rotation, [0.1, 0.2, 0.3])], T_base_camera_mm, T_graspnet_tcp_mm,
        15.0, 30.0, 5.0, points_base=None)

    assert best['strategy'] == 'graspnet_baseline'
    assert best['width_mm'] == pytest.approx(40.0)
    position = best['pose']['position']
    # base 위치 = T_base_camera @ (R @ t_graspnet_tcp + translation)
    expected = T_base_camera_mm[:3, 3] + rotation @ (10.0, 20.0, 30.0) + (100.0, 200.0, 300.0)
    assert (position['x'], position['y'], position['z']) == pytest.approx(tuple(expected))


def test_graspnet_depth_advances_along_the_approach_axis():
    """`depth_m`은 무는 지점까지의 거리다 — 접근축을 따라 전진해야 한다 (버리면 허공)."""
    rotation = np.column_stack([(0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)])
    common = dict(T_base_camera_mm=np.eye(4), T_graspnet_tcp_mm=np.eye(4))
    without, _ = graspnet_baseline._select_best(
        [_raw(rotation, [0.0, 0.0, 0.5], depth_m=0.0)],
        common['T_base_camera_mm'], common['T_graspnet_tcp_mm'], 15.0, 30.0, 5.0,
        points_base=None)
    with_depth, _ = graspnet_baseline._select_best(
        [_raw(rotation, [0.0, 0.0, 0.5], depth_m=0.03)],
        common['T_base_camera_mm'], common['T_graspnet_tcp_mm'], 15.0, 30.0, 5.0,
        points_base=None)
    # 접근축이 base -Z이므로 30mm 전진은 z가 30mm 낮아지는 것으로 나타난다.
    assert (without['pose']['position']['z']
            - with_depth['pose']['position']['z']) == pytest.approx(30.0)


def test_upright_filter_rejects_candidates_that_stab_from_below():
    """아래에서 위로 찌르는 자세는 abs()로 재면 0도로 통과한다 — 걸러져야 한다."""
    upward = np.column_stack([(0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)])
    best, diagnostics = graspnet_baseline._select_best(
        [_raw(upward, [0.0, 0.0, 0.5])], np.eye(4), np.eye(4), 15.0, 30.0, 5.0,
        points_base=None)
    assert best is None
    assert diagnostics['angles_deg'] == [180.0]


def _plate_cloud(half_x=40.0, half_y=25.0, thickness=8.0, n=6000, seed=0):
    """작업대 위에 놓인 납작한 판. 윗면 z=0, 아래로 thickness 만큼."""
    rng = np.random.default_rng(seed)
    return np.c_[rng.uniform(-half_x, half_x, n),
                 rng.uniform(-half_y, half_y, n),
                 rng.uniform(-thickness, 0.0, n)]


@pytest.mark.parametrize('tilt_deg', [0.0, 15.0, 30.0])
def test_refine_is_not_dragged_sideways_by_a_tilted_approach(tilt_deg):
    """**물체가 가만히 있으면 접근축을 기울여도 파지점은 제자리여야 한다.**

    예전 식은 표면을 클라우드 전체의 접근축 극값으로 잡아, 기울이면 그 극값이 먼 모서리가
    되면서 파지점이 최대 35mm 옆으로 끌려갔다(2026-09-07 실측). GraspNet은 거의 항상
    15~30도 기울어진 자세를 내므로 이 미끄러짐이 매 파지마다 작용했다.
    """
    points = _plate_cloud()
    angle = np.radians(tilt_deg)
    approach = np.array([np.sin(angle), 0.0, -np.cos(angle)])
    closing = np.cross([0.0, 1.0, 0.0], approach)
    closing /= np.linalg.norm(closing)
    other = np.cross(approach, closing)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([closing, other, approach])

    position, _ = graspnet_baseline._refine_on_cloud(T, points, grasp_depth_mm=4.0)
    # 판의 중심은 원점이다. 가로로 몇 mm 안쪽이어야 한다.
    assert np.hypot(position[0], position[1]) < 6.0, (
        f"기울기 {tilt_deg}도에서 파지점이 가로로 {np.hypot(position[0], position[1]):.1f}mm 밀렸다")
    # 깊이도 판 안쪽이어야 한다 (윗면 0, 바닥 -8).
    assert -8.0 <= position[2] <= 0.0


def test_refine_width_is_measured_across_the_finger_path():
    """개폭은 손가락이 지나는 창 안에서 재야 한다 — 마스크 전체 범위가 아니라."""
    points = _plate_cloud(half_x=40.0, half_y=25.0)
    T = np.eye(4)                       # closing=+X, other=+Y, approach=+Z
    T[:3, :3] = np.column_stack([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)])
    _, width = graspnet_baseline._refine_on_cloud(T, points, grasp_depth_mm=4.0)
    assert width == pytest.approx(80.0, abs=6.0)


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

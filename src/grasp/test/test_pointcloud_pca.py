import numpy as np

from grasp import pointcloud_utils
from grasp.strategies import heuristic_pca


def test_aligned_depth_pointcloud_produces_pca_candidate():
    mask = np.ones((10, 10), dtype=bool)
    depth_mm = np.full((10, 10), 300, dtype=np.uint16)
    points = pointcloud_utils.backproject(
        mask, depth_mm, {'fx': 100.0, 'fy': 100.0, 'cx': 4.5, 'cy': 4.5})

    candidates = heuristic_pca.plan(points, {
        'min_points': 80, 'top_surface_band_mm': 15.0, 'grasp_depth_mm': 8.0,
        'max_opening_mm': 110.0, 'min_width_mm': 5.0,
    })

    assert len(points) == 100
    assert candidates
    assert candidates[0]['strategy'] == 'heuristic_pca'

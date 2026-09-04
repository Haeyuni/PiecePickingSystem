import numpy as np

from perception import mask_utils


def test_mask_3d_rejects_mismatched_depth_grid():
    point, ratio = mask_utils.mask_3d(
        np.ones((2, 2), dtype=bool), np.ones((1, 2), dtype=np.uint16),
        {'fx': 1.0, 'fy': 1.0, 'cx': 0.0, 'cy': 0.0})

    assert point is None
    assert ratio == 0.0

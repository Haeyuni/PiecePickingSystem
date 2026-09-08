import pytest

from control.box_geometry import box_geometry_from_measurements, plan_box_place


def measured_box():
    return box_geometry_from_measurements(
        inner_corners_base_mm={
            "front_left": [0, 0, 100],
            "front_right": [100, 0, 100],
            "rear_right": [100, 80, 100],
            "rear_left": [0, 80, 100],
        },
        floor_point_base_mm=[30, 35, 20],
    )


def test_derives_center_dimensions_yaw_and_floor_from_five_measurements():
    box = measured_box()

    assert box.center_base_mm == pytest.approx((50, 40, 100))
    assert box.width_mm == pytest.approx(100)
    assert box.depth_mm == pytest.approx(80)
    assert box.yaw_deg == pytest.approx(0)
    assert box.floor_z_mm == pytest.approx(20)


def test_floor_point_xy_does_not_change_production_center():
    box = measured_box()
    other_floor_point = box_geometry_from_measurements(
        inner_corners_base_mm={
            "front_left": [0, 0, 100], "front_right": [100, 0, 100],
            "rear_right": [100, 80, 100], "rear_left": [0, 80, 100],
        },
        floor_point_base_mm=[90, 70, 20],
    )

    assert other_floor_point.center_base_mm == box.center_base_mm


def test_centers_full_footprint_and_preserves_pick_orientation():
    plan = plan_box_place(
        box=measured_box(), wall_margin_mm=5, release_clearance_mm=10,
        pickup_tcp_posx=[30, 20, 100, 12, 178, -5],
        footprint_xy=[(20, 10), (40, 10), (40, 30), (20, 30)],
        tcp_to_object_bottom_mm=35,
    )

    assert plan.target_tcp_posx == pytest.approx((50, 40, 65, 12, 178, -5))
    assert list(plan.translated_footprint) == [(40, 30), (60, 30), (60, 50), (40, 50)]


def test_rejects_footprint_that_cannot_meet_wall_margin():
    small_box = box_geometry_from_measurements(
        inner_corners_base_mm={
            "front_left": [0, 0, 50], "front_right": [40, 0, 50],
            "rear_right": [40, 40, 50], "rear_left": [0, 40, 50],
        },
        floor_point_base_mm=[20, 20, 0],
    )
    with pytest.raises(ValueError, match="does not fit"):
        plan_box_place(
            box=small_box, wall_margin_mm=6, release_clearance_mm=10,
            pickup_tcp_posx=[0, 0, 50, 0, 180, 0],
            footprint_xy=[(0, 0), (32, 0), (32, 32), (0, 32)],
            tcp_to_object_bottom_mm=20,
        )


def test_rejects_non_finite_measurement():
    with pytest.raises(ValueError, match="finite"):
        box_geometry_from_measurements(
            inner_corners_base_mm={
                "front_left": [0, 0, 100], "front_right": [100, 0, 100],
                "rear_right": [100, 100, 100], "rear_left": [0, 100, 100],
            },
            floor_point_base_mm=[50, 50, float("nan")],
        )


def test_rejects_floor_point_above_upper_corners():
    with pytest.raises(ValueError, match="below"):
        box_geometry_from_measurements(
            inner_corners_base_mm={
                "front_left": [0, 0, 100], "front_right": [100, 0, 100],
                "rear_right": [100, 100, 100], "rear_left": [0, 100, 100],
            },
            floor_point_base_mm=[50, 50, 110],
        )

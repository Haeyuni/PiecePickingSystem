import math

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


def test_keeps_pick_orientation_when_it_already_fits():
    """들어가는데도 굳이 돌리면 안 된다 — 손목을 헛돌리는 만큼 이동이 길어진다."""
    plan = plan_box_place(
        box=measured_box(), wall_margin_mm=5, release_clearance_mm=10,
        pickup_tcp_posx=[30, 20, 100, 12, 178, -5],
        footprint_xy=[(20, 10), (40, 10), (40, 30), (20, 30)],
        tcp_to_object_bottom_mm=35,
    )

    assert plan.yaw_deg == 0
    assert plan.target_tcp_posx[3:] == pytest.approx((12, 178, -5))


def test_skip_yaws_moves_on_to_the_next_fitting_angle():
    """place_server가 한 자세로는 안전 이송이 안 된다고 확인하면 그 각을 건너뛰고
    다음으로 잘 맞는 각을 받아야 한다(STEP 3, 2026-09-10) — 바구니엔 들어가지만
    갈 수 없는 자세에 계속 머물면 안 된다."""
    kwargs = dict(
        box=measured_box(), wall_margin_mm=5, release_clearance_mm=10,
        pickup_tcp_posx=[30, 20, 100, 12, 178, -5],
        footprint_xy=[(20, 10), (40, 10), (40, 30), (20, 30)],
        tcp_to_object_bottom_mm=35,
    )
    baseline = plan_box_place(**kwargs)
    assert baseline.yaw_deg == 0

    skipped = plan_box_place(**kwargs, skip_yaws={0.0})
    assert skipped.yaw_deg != 0
    # 작은 정사각형 발자국이라 다음 후보(±1도)에서도 바로 들어간다 — _yaw_candidates의
    # 탐색 순서(0, +1, -1, +2, -2, ...)를 그대로 지킨다.
    assert skipped.yaw_deg == 1.0


def test_rotates_object_that_only_fits_across_the_other_axis():
    """긴 물체가 바구니 짧은 축과 나란하면 예전엔 거절됐다 — 이제 돌려서 넣는다.

    박스 내부 100(x) x 80(y), 여유 5mm → 사용가능 90 x 70.
    물체는 85 x 30이라 x축과 나란하면 들어가고, y축과 나란하면(85 > 70) 안 들어간다.
    """
    plan = plan_box_place(
        box=measured_box(), wall_margin_mm=5, release_clearance_mm=10,
        pickup_tcp_posx=[50, 40, 100, 12, 178, -5],
        footprint_xy=[(35, 0), (65, 0), (65, 85), (35, 85)],   # y축으로 긴 물체
        tcp_to_object_bottom_mm=35,
    )

    # 각도 자체는 검색 결과라 고정하지 않는다(대각선으로 들어가는 더 작은 각도도 옳다).
    # 지켜야 할 것은 "돌렸다"와 "돌린 결과가 실제로 여유선 안"이다.
    assert plan.yaw_deg != 0
    # 손목은 그 각도만큼만 돌아간다 — ZYZ 첫 각도에 더해지고 나머지는 그대로다.
    assert plan.target_tcp_posx[3] == pytest.approx(12 + plan.yaw_deg)
    assert plan.target_tcp_posx[4:] == pytest.approx((178, -5))
    xs = [x for x, _ in plan.translated_footprint]
    ys = [y for _, y in plan.translated_footprint]
    assert min(xs) >= 5 - 1e-6 and max(xs) <= 95 + 1e-6
    assert min(ys) >= 5 - 1e-6 and max(ys) <= 75 + 1e-6


def test_rotation_moves_the_tcp_with_the_object():
    """TCP는 물체와 같은 축·같은 각으로 돌아야 한다 — 안 그러면 파지점이 어긋난다.

    TCP가 물체 무게중심에서 벗어나 있으면(여기선 +20mm) 90도 회전 뒤 그 편차도
    같이 돌아야 한다. 평행이동만 하던 예전 코드는 이걸 못 했다.
    """
    plan = plan_box_place(
        box=measured_box(), wall_margin_mm=5, release_clearance_mm=10,
        pickup_tcp_posx=[70, 40, 100, 0, 180, 0],   # 무게중심(50,40)에서 x로 +20
        footprint_xy=[(35, -2.5), (65, -2.5), (65, 82.5), (35, 82.5)],  # 무게중심 (50,40)
        tcp_to_object_bottom_mm=35,
    )

    yaw = math.radians(plan.yaw_deg)
    expected = (50 + 20 * math.cos(yaw), 40 + 20 * math.sin(yaw))
    assert plan.target_tcp_posx[:2] == pytest.approx(expected)


def test_rejection_says_how_much_it_overflowed():
    """거절할 땐 몇 mm 모자랐는지 말해야 한다 — 여유를 줄일 일인지 판단이 서야 한다."""
    with pytest.raises(ValueError, match=r"최소 \d+\.\d+mm 초과"):
        plan_box_place(
            box=measured_box(), wall_margin_mm=5, release_clearance_mm=10,
            pickup_tcp_posx=[50, 40, 100, 0, 180, 0],
            footprint_xy=[(0, 0), (200, 0), (200, 200), (0, 200)],
            tcp_to_object_bottom_mm=35,
        )


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

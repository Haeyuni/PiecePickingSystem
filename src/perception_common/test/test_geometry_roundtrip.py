"""base ↔ camera 변환 왕복 검증.

`geometry.py`가 스스로 적어 둔 대로 "조용히 틀리면 로봇이 엉뚱한 곳으로 가는 계산"이다.
재관측(D-1)은 직전 물체의 base 좌표를 카메라 평면으로 되돌려 SAM 프롬프트를 만드는데, 그
역변환이 틀리면 프롬프트가 빗나가고 물체가 통째로 사라진 것으로 처리된다 — 예외 없이,
로그도 없이.

캘리브레이션 코드와의 대조(무작위 자세 2000개, 최대 오차 5e-13mm)는 이미 되어 있다
(src/perception/README.md). 여기서는 새로 추가한 역방향이 정방향과 정확히 맞물리는지 본다.
"""
import numpy as np

from perception_common import geometry

INTRINSICS = {"fx": 600.0, "fy": 600.0, "cx": 640.0, "cy": 360.0}


def random_handeye(rng):
    matrix = np.eye(4)
    matrix[:3, :3] = geometry.posx_to_matrix(
        [0, 0, 0, *rng.uniform(-180, 180, 3)])[:3, :3]
    matrix[:3, 3] = rng.uniform(-120, 120, 3)
    return matrix


def test_camera_from_base_inverts_base_from_camera():
    rng = np.random.default_rng(20260907)
    worst = 0.0

    for _ in range(2000):
        base2gripper = geometry.posx_to_matrix(
            [*rng.uniform(-600, 600, 3), *rng.uniform(-180, 180, 3)])
        gripper2camera = random_handeye(rng)
        point_cam = rng.uniform(-500, 500, 3)

        point_base = geometry.base_from_camera(point_cam, base2gripper, gripper2camera)
        back = geometry.camera_from_base(point_base, base2gripper, gripper2camera)
        worst = max(worst, float(np.max(np.abs(np.array(back) - point_cam))))

    assert worst < 1e-6, f"왕복 오차 {worst}mm — 역변환이 정방향과 맞물리지 않는다"


def test_project_to_pixel_matches_the_unprojection_used_for_3d():
    """mask_utils.mask_3d가 쓰는 식(x=(u-cx)*z/fx)의 정방향이어야 한다."""
    u, v, z = 823.0, 194.0, 470.0
    x = (u - INTRINSICS["cx"]) * z / INTRINSICS["fx"]
    y = (v - INTRINSICS["cy"]) * z / INTRINSICS["fy"]

    pixel = geometry.project_to_pixel((x, y, z), INTRINSICS)

    assert pixel is not None
    assert abs(pixel[0] - u) < 1e-9
    assert abs(pixel[1] - v) < 1e-9


def test_principal_point_projects_to_the_center():
    assert geometry.project_to_pixel((0.0, 0.0, 300.0), INTRINSICS) == (640.0, 360.0)


def test_points_behind_the_camera_are_rejected():
    """그냥 나누면 뒤에 있는 점이 화면 안쪽으로 접혀 들어와 엉뚱한 곳을 가리킨다."""
    assert geometry.project_to_pixel((10.0, 10.0, -300.0), INTRINSICS) is None
    assert geometry.project_to_pixel((10.0, 10.0, 0.0), INTRINSICS) is None

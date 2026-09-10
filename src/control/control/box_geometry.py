"""Measured box geometry calculations. No ROS or robot driver dependencies."""
from dataclasses import dataclass
import math


CORNER_NAMES = ("front_left", "front_right", "rear_right", "rear_left")


@dataclass(frozen=True)
class BoxGeometry:
    corners_xy: tuple[tuple[float, float], ...]
    center_base_mm: tuple[float, float, float]
    width_mm: float
    depth_mm: float
    yaw_deg: float
    floor_z_mm: float


@dataclass(frozen=True)
class PlacePlan:
    target_tcp_posx: tuple[float, ...]
    translated_footprint: tuple[tuple[float, float], ...]
    box: BoxGeometry
    # 파지 자세에서 수직축(base +Z)으로 얼마나 돌려 놓는지. 0이면 파지 자세 그대로다.
    yaw_deg: float = 0.0


def _point3(value, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must be [x, y, z]")
    point = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in point):
        raise ValueError(f"{name} must contain finite values")
    return point


def box_geometry_from_measurements(*, inner_corners_base_mm,
                                   floor_point_base_mm) -> BoxGeometry:
    """Derive production geometry from four named upper corners and one floor point."""
    if not isinstance(inner_corners_base_mm, dict):
        raise ValueError("inner_corners_base_mm must contain four named corners")
    missing = [name for name in CORNER_NAMES if name not in inner_corners_base_mm]
    extra = [name for name in inner_corners_base_mm if name not in CORNER_NAMES]
    if missing or extra:
        raise ValueError(f"corner names invalid: missing={missing}, extra={extra}")

    corners = tuple(_point3(inner_corners_base_mm[name], name) for name in CORNER_NAMES)
    floor = _point3(floor_point_base_mm, "floor_point_base_mm")
    xy = tuple((point[0], point[1]) for point in corners)

    cross_products = []
    for previous, current, following in zip(xy[-1:] + xy[:-1], xy, xy[1:] + xy[:1]):
        cross_products.append(
            (current[0] - previous[0]) * (following[1] - current[1])
            - (current[1] - previous[1]) * (following[0] - current[0]))
    if any(abs(cross) <= 1e-9 for cross in cross_products):
        raise ValueError("box corners are degenerate")
    if not (all(cross > 0.0 for cross in cross_products)
            or all(cross < 0.0 for cross in cross_products)):
        raise ValueError("box corners must form a convex perimeter in named order")

    fl, fr, rr, rl = corners
    front_width = math.hypot(fr[0] - fl[0], fr[1] - fl[1])
    rear_width = math.hypot(rr[0] - rl[0], rr[1] - rl[1])
    left_depth = math.hypot(rl[0] - fl[0], rl[1] - fl[1])
    right_depth = math.hypot(rr[0] - fr[0], rr[1] - fr[1])
    width_mm = (front_width + rear_width) / 2.0
    depth_mm = (left_depth + right_depth) / 2.0

    width_dx = ((fr[0] - fl[0]) + (rr[0] - rl[0])) / 2.0
    width_dy = ((fr[1] - fl[1]) + (rr[1] - rl[1])) / 2.0
    yaw_deg = math.degrees(math.atan2(width_dy, width_dx))
    center = tuple(sum(point[axis] for point in corners) / 4.0 for axis in range(3))
    if floor[2] >= min(point[2] for point in corners):
        raise ValueError("floor point must be below all upper corners")

    return BoxGeometry(corners_xy=xy, center_base_mm=center,
                       width_mm=width_mm, depth_mm=depth_mm,
                       yaw_deg=yaw_deg, floor_z_mm=floor[2])


def _centroid(points: tuple[tuple[float, float], ...]) -> tuple[float, float]:
    area2 = 0.0
    cx = 0.0
    cy = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
        cross = x1 * y2 - x2 * y1
        area2 += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    if abs(area2) < 1e-9:
        raise ValueError("polygon area is zero")
    return cx / (3.0 * area2), cy / (3.0 * area2)


def _inside_with_margin(point, polygon, margin_mm: float) -> bool:
    return _overflow_mm(point, polygon, margin_mm) <= 0.0


def _overflow_mm(point, polygon, margin_mm: float) -> float:
    """이 점이 여유선을 **몇 mm 넘어섰는지**. 0 이하면 안쪽(여유가 그만큼 남음).

    판정(`_inside_with_margin`)과 진단이 같은 계산을 쓰게 해서, "안 들어간다"는 거절에
    항상 **얼마나** 모자란지를 붙일 수 있게 한다 — 그 숫자가 없으면 벽 여유를 조금
    줄이면 될 일인지, 아예 다른 바구니가 필요한 일인지 로그만 보고는 구분할 수 없다.
    """
    area_sign = 1.0 if sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1])
    ) > 0.0 else -1.0
    px, py = point
    worst = -math.inf
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        edge_length = math.hypot(x2 - x1, y2 - y1)
        signed_distance = area_sign * ((x2 - x1) * (py - y1) - (y2 - y1) * (px - x1))
        worst = max(worst, margin_mm - signed_distance / edge_length)
    return worst


def _footprint_overflow_mm(points, polygon, margin_mm: float) -> float:
    return max(_overflow_mm(point, polygon, margin_mm) for point in points)


def plan_box_place(*, box: BoxGeometry, wall_margin_mm: float,
                   release_clearance_mm: float, pickup_tcp_posx,
                   footprint_xy, tcp_to_object_bottom_mm: float,
                   skip_yaws: set | None = None) -> PlacePlan:
    """Center the observed footprint and require its whole boundary inside the box.

    파지 자세 그대로 들어가면 그대로 놓고, 안 들어가면 수직축으로 돌려서 들어가는 각을
    찾는다(`_fit_yaw`). 어떤 각으로도 안 들어가면 **얼마나 모자랐는지를 붙여** 거절한다.

    `skip_yaws`는 이미 시도해서 안전 이송 불가로 확인된 각들이다(STEP 3, 2026-09-10).
    호출부(place_server._execute_geometry_place)가 한 각으로 이송 자체가 불가능하면
    (바구니에는 들어가지만 그 자세로 갈 방법이 없으면) 그 각을 여기 넣고 다시 불러
    **다음으로 잘 맞는 각**을 받는다 — 바구니에 들어가는지만 보고 끝내지 않고, 갈 수
    있는 자세 중에서 고르게 된다.
    """
    footprint = tuple((float(x), float(y)) for x, y in footprint_xy)
    tcp = tuple(float(v) for v in pickup_tcp_posx)
    if len(footprint) < 3 or len(tcp) != 6:
        raise ValueError("object footprint and 6D pickup TCP are required")
    if wall_margin_mm < 0.0 or release_clearance_mm < 0.0 or tcp_to_object_bottom_mm < 0.0:
        raise ValueError("geometry distances must be non-negative")
    scalars = [float(wall_margin_mm), float(release_clearance_mm),
               float(tcp_to_object_bottom_mm), *tcp,
               *(value for point in footprint for value in point)]
    if not all(math.isfinite(value) for value in scalars):
        raise ValueError("geometry values must be finite")

    bin_center = box.center_base_mm[:2]
    footprint_center = _centroid(footprint)

    yaw_deg, translated, overflow_mm = _fit_yaw(
        footprint, footprint_center, bin_center, box.corners_xy, float(wall_margin_mm),
        skip=skip_yaws)
    if yaw_deg is None:
        raise ValueError(
            "object footprint does not fit inside measured box boundary "
            f"(어떤 각도로도 최소 {overflow_mm:.1f}mm 초과 — 바구니 내부 "
            f"{box.width_mm:.0f}x{box.depth_mm:.0f}mm, 벽 여유 {wall_margin_mm:.0f}mm)")

    # 물체를 수직축으로 yaw_deg 돌려 놓으므로 TCP도 같은 축·같은 각으로 돈다. 회전
    # 중심은 footprint 무게중심이고, 그 무게중심을 바구니 중심으로 옮긴다.
    tcp_x, tcp_y = _rotate_about(
        (tcp[0], tcp[1]), footprint_center, yaw_deg, bin_center)
    # **ZYZ에서 base +Z 회전은 첫 각도에 더하기만 하면 된다**: posx_to_matrix가
    # R = Rz(rx)·Ry(ry)·Rz(rz)로 만들므로(perception_common.geometry) 왼쪽에서 Rz(yaw)를
    # 곱하면 Rz(yaw+rx)·Ry(ry)·Rz(rz)가 되어 rx만 바뀐다. 행렬 왕복 변환이 필요 없고,
    # ry가 180도 근처(특이점)여도 파라미터가 새로 튀지 않는다.
    target = (tcp_x, tcp_y,
              box.floor_z_mm + float(tcp_to_object_bottom_mm) + float(release_clearance_mm),
              tcp[3] + yaw_deg, tcp[4], tcp[5])
    return PlacePlan(target_tcp_posx=target, translated_footprint=translated, box=box,
                     yaw_deg=yaw_deg)


def _rotate_about(point, pivot, yaw_deg: float, new_pivot):
    """`pivot` 기준으로 `yaw_deg`(도, base +Z 오른손) 돌린 뒤 `new_pivot`으로 옮긴다."""
    angle = math.radians(yaw_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    dx, dy = point[0] - pivot[0], point[1] - pivot[1]
    return (new_pivot[0] + dx * cos_a - dy * sin_a,
            new_pivot[1] + dx * sin_a + dy * cos_a)


def _fit_yaw(footprint, footprint_center, bin_center, corners_xy, wall_margin_mm,
            skip=None):
    """바구니에 들어가는 수직축 회전각을 찾는다. 반환: (각도 또는 None, 옮긴 footprint, 초과 mm).

    **0도(=파지 자세 그대로)를 가장 먼저, 그다음 작은 각도부터 본다.** 손목을 덜 돌릴수록
    이동이 짧고 도달 가능성도 높아서다 — 들어가기만 하면 되는 문제라 "가장 잘 맞는 각"을
    찾을 이유가 없다.

    **왜 회전이 필요한가**: 예전에는 평행이동만 했기 때문에, 물체가 작업대에 놓여 있던
    방향이 바구니의 짧은 축과 나란하면 그대로 거절됐다. 2026-09-09 실물에서 spray_can이
    파지 자세로는 13.9mm 초과였는데 약 120도 돌리면 3.6mm 여유로 들어갔다 — 벽 여유를
    0으로 줄여도 안 되는(3.9mm 모자란) 경우라 회전 말고는 방법이 없었다.

    회전은 손목만 돌리고 파지 자체는 그대로다(물체는 그리퍼에 잡힌 채 같이 돈다).

    `skip`에 있는 각은 건너뛴다 — 바구니에는 들어가지만 이송 자체가 불가능하다고 이미
    확인된 각들이다(`plan_box_place`의 `skip_yaws` 참조).
    """
    best_overflow = math.inf
    for yaw_deg in _yaw_candidates():
        if skip and yaw_deg in skip:
            continue
        moved = tuple(_rotate_about(point, footprint_center, yaw_deg, bin_center)
                      for point in footprint)
        overflow = _footprint_overflow_mm(moved, corners_xy, wall_margin_mm)
        if overflow <= 0.0:
            return yaw_deg, moved, overflow
        best_overflow = min(best_overflow, overflow)
    return None, (), best_overflow


def _yaw_candidates():
    """0도부터 시작해 좌우로 1도씩 벌려 나가는 순서 (0, +1, -1, +2, -2, … ±180)."""
    yield 0.0
    for step in range(1, 181):
        yield float(step)
        yield float(-step)

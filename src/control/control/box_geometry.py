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
    area_sign = 1.0 if sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1])
    ) > 0.0 else -1.0
    px, py = point
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        edge_length = math.hypot(x2 - x1, y2 - y1)
        signed_distance = area_sign * ((x2 - x1) * (py - y1) - (y2 - y1) * (px - x1))
        if signed_distance / edge_length < margin_mm:
            return False
    return True


def plan_box_place(*, box: BoxGeometry, wall_margin_mm: float,
                   release_clearance_mm: float, pickup_tcp_posx,
                   footprint_xy, tcp_to_object_bottom_mm: float) -> PlacePlan:
    """Center the observed footprint and require its whole boundary inside the box."""
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
    dx = bin_center[0] - footprint_center[0]
    dy = bin_center[1] - footprint_center[1]
    translated = tuple((x + dx, y + dy) for x, y in footprint)
    if not all(_inside_with_margin(point, box.corners_xy, wall_margin_mm)
               for point in translated):
        raise ValueError("object footprint does not fit inside measured box boundary")

    target = (tcp[0] + dx, tcp[1] + dy,
              box.floor_z_mm + float(tcp_to_object_bottom_mm) + float(release_clearance_mm),
              tcp[3], tcp[4], tcp[5])
    return PlacePlan(target_tcp_posx=target, translated_footprint=translated, box=box)

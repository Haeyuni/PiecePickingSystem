#!/usr/bin/env python3
"""같은 물체를 여러 로봇 자세에서 관측해 `position_base_mm`이 일치하는지 본다.

**무엇을 판별하는가.** base 좌표계는 로봇과 함께 움직이지 않으므로, 고정된 물체의
base 위치는 **로봇이 어디서 보든 같아야 한다**. 자세마다 다르게 나오면 카메라 좌표를
base로 옮기는 변환(`posx_to_matrix(TCP) @ T_gripper2camera`)이 틀렸다는 뜻이고, 그건
오프셋으로 덮을 수 없다 — 오프셋은 한 자세에서만 맞고 다른 자세로 가면 다시 어긋난다.

**왜 카메라 화면으로는 못 잡는가.** 디버그 오버레이는 후보 좌표를 같은 변환으로 되투영해
그린다. 변환 오차가 물체 위치와 후보 위치에 똑같이 실리므로 화면에서는 상쇄되어 완벽하게
맞아 보인다. "화면은 맞는데 실물은 빗나간다"가 바로 이 오차의 서명이다.

**쓰는 법.** 물체를 작업대에 두고 **움직이지 않은 채**, 이 스크립트를 켜고 로봇을
2~4자세로 옮긴다(높이·좌우·기울기를 바꿀수록 좋다). TCP가 충분히 움직이면 자동으로
한 표본을 기록하고, 끝나면 물체별 편차를 낸다. 로봇 조작은 사람이 한다 — 이 스크립트는
관측만 한다.

    python3 check_handeye_consistency.py --samples 4

**읽는 법.**
  편차가 몇 mm  → 캘리브레이션 정상. 파지 실패 원인은 다른 데 있다(TCP 정의, 깊이 편향 등).
  편차가 수십 mm → hand-eye 확정. 재캘리브레이션이 필요하고, 오프셋 튜닝은 헛수고다.
"""
import argparse
import math

import numpy as np
import rclpy
from rclpy.node import Node

from dsr_msgs2.srv import GetCurrentPosx
from sort_msgs.msg import WorldState

# perception/grasp가 좌표를 만들 때 쓰는 것과 **같은** ZYZ 해석이어야 한다.
from perception_common.geometry import posx_to_matrix

POSX_SERVICE = "/dsr01/dsr_controller2/aux_control/get_current_posx"


def rotation_diff_deg(a: list[float], b: list[float]) -> float:
    ra = posx_to_matrix(a)[:3, :3]
    rb = posx_to_matrix(b)[:3, :3]
    cos = max(-1.0, min(1.0, (float((ra.T @ rb).trace()) - 1.0) / 2.0))
    return math.degrees(math.acos(cos))


def solve_handeye_translation(samples: list[dict], name: str
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """고정된 물체를 여러 자세에서 본 결과로 **hand-eye 이동 성분의 보정량**을 푼다.

    현재 보고값은 `reported_i = R_i (R_he p_i + t_he) + t_i` 다. `t_he`가 툴 좌표계에서
    δ만큼 틀렸다면 옳은 값은 `reported_i + R_i δ` 이고, 물체가 고정돼 있으므로 그것이
    자세와 무관하게 같은 점 P여야 한다:

        R_i δ - P = -reported_i

    자세마다 식 3개, 미지수 6개(δ, P) — **회전이 서로 다른** 자세가 2개 이상이면 풀린다.
    회전이 같으면 R_i가 전부 같아 δ와 P가 분리되지 않으므로 호출부가 회전차를 먼저 본다.

    **한계.** `position_base_mm`은 그 시점에 **보이는 면**의 중심이라, 카메라가 크게
    돌면 물체 자체의 보이는 부분이 달라져 P가 실제로 조금 움직인다. 즉 여기서 나온 δ에는
    그 시점 의존성이 섞인다 — 잔차가 그 크기를 알려주므로 함께 읽어야 한다. 작고 납작한
    물체를 쓰고 기울기 변화를 20~40도로 제한하면 이 영향이 작아진다.
    """
    rows, rhs = [], []
    for s in samples:
        rotation = posx_to_matrix(s["tcp"])[:3, :3]
        rows.append(np.hstack([rotation, -np.eye(3)]))
        rhs.append(-np.array(s["objects"][name]))
    solution, *_ = np.linalg.lstsq(np.vstack(rows), np.hstack(rhs), rcond=None)
    delta, point = solution[:3], solution[3:]
    residuals = np.array([
        np.linalg.norm(posx_to_matrix(s["tcp"])[:3, :3] @ delta
                       + np.array(s["objects"][name]) - point)
        for s in samples])
    return delta, point, residuals


class Collector(Node):
    def __init__(self, samples: int, move_mm: float, move_deg: float):
        super().__init__("handeye_consistency")
        self._target = samples
        self._move_mm = move_mm
        self._move_deg = move_deg
        self._poses: list[list[float]] = []
        self.samples: list[dict] = []
        self._posx = self.create_client(GetCurrentPosx, POSX_SERVICE)
        self.create_subscription(WorldState, "/world_state", self._on_world, 10)
        self.get_logger().info(
            f"물체를 고정한 채 로봇을 옮겨 주세요 — TCP가 {move_mm:.0f}mm 또는 "
            f"{move_deg:.0f}도 이상 움직일 때마다 한 표본씩 기록합니다 (목표 {samples}개)")

    def done(self) -> bool:
        return len(self.samples) >= self._target

    def _tcp(self) -> list[float] | None:
        if not self._posx.wait_for_service(timeout_sec=1.0):
            return None
        future = self._posx.call_async(GetCurrentPosx.Request(ref=0))
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        result = future.result()
        if result is None or not result.success or not result.task_pos_info:
            return None
        return [float(v) for v in result.task_pos_info[0].data[:6]]

    def _is_new_pose(self, pose: list[float]) -> bool:
        for old in self._poses:
            moved = math.dist(pose[:3], old[:3])
            turned = max(abs(pose[i] - old[i]) for i in range(3, 6))
            if moved < self._move_mm and turned < self._move_deg:
                return False
        return True

    def _on_world(self, msg: WorldState) -> None:
        if self.done():
            return
        objects = {o.class_name: (o.position_base_mm.x, o.position_base_mm.y,
                                  o.position_base_mm.z)
                   for o in msg.objects if o.graspable}
        if not objects:
            return
        pose = self._tcp()
        if pose is None or not self._is_new_pose(pose):
            return
        self._poses.append(pose)
        self.samples.append({"tcp": pose, "objects": objects})
        self.get_logger().info(
            f"표본 {len(self.samples)}/{self._target} 기록 — "
            f"TCP=({pose[0]:.0f}, {pose[1]:.0f}, {pose[2]:.0f}) 물체 {len(objects)}개")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--move-mm", type=float, default=60.0,
                    help="이만큼 움직여야 새 자세로 본다")
    ap.add_argument("--move-deg", type=float, default=10.0)
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    rclpy.init()
    node = Collector(args.samples, args.move_mm, args.move_deg)
    deadline = node.get_clock().now().nanoseconds * 1e-9 + args.timeout
    try:
        while rclpy.ok() and not node.done():
            rclpy.spin_once(node, timeout_sec=0.5)
            if node.get_clock().now().nanoseconds * 1e-9 > deadline:
                print(f"\n타임아웃 — 표본 {len(node.samples)}개만 모았다")
                break
    finally:
        samples = node.samples
        node.destroy_node()
        rclpy.shutdown()

    if len(samples) < 2:
        raise SystemExit("표본이 2개 미만이라 비교할 수 없다. 로봇을 더 크게 움직일 것")

    print(f"\n표본 {len(samples)}개\n")
    for i, s in enumerate(samples):
        t = s["tcp"]
        print(f"  자세{i}: TCP=({t[0]:7.1f}, {t[1]:7.1f}, {t[2]:7.1f}) "
              f"ZYZ=({t[3]:6.1f}, {t[4]:6.1f}, {t[5]:6.1f})")

    shared = set(samples[0]["objects"])
    for s in samples[1:]:
        shared &= set(s["objects"])
    if not shared:
        raise SystemExit("\n모든 자세에서 함께 보인 물체가 없다. 시야에 계속 들어오게 배치할 것")

    print(f"\n{'물체':38s} {'X 편차':>9} {'Y 편차':>9} {'Z 편차':>9} {'합':>9}")
    print("-" * 78)
    worst = 0.0
    for name in sorted(shared):
        points = np.array([s["objects"][name] for s in samples])
        spread = points.max(axis=0) - points.min(axis=0)
        total = float(np.linalg.norm(spread))
        worst = max(worst, total)
        print(f"{name[:36]:38s} {spread[0]:8.1f}mm {spread[1]:8.1f}mm "
              f"{spread[2]:8.1f}mm {total:8.1f}mm")
        for i, p in enumerate(points):
            print(f"    자세{i}: ({p[0]:8.1f}, {p[1]:8.1f}, {p[2]:8.1f})")

    turn = max((rotation_diff_deg(a["tcp"], b["tcp"])
                for a in samples for b in samples), default=0.0)
    print(f"\n자세 간 최대 회전차 = {turn:.1f}도")
    if turn >= 10.0:
        print("\n--- hand-eye 이동 성분 보정량 (툴 좌표계 mm) ---")
        print("고정된 물체가 자세와 무관하게 같은 자리로 보이려면 얼마를 더해야 하는지 "
              "역산한 값이다.")
        for name in sorted(shared):
            delta, point, residuals = solve_handeye_translation(samples, name)
            print(f"  {name[:34]:36s} δ=({delta[0]:+7.1f}, {delta[1]:+7.1f}, {delta[2]:+7.1f}) "
                  f"잔차 최대 {residuals.max():5.1f}mm")
        print("  읽는 법: δ가 물체마다 비슷하고 잔차가 작으면 **hand-eye 이동값이 그만큼 "
              "틀렸다**는 뜻이다.")
        print("           data/calibration/T_gripper2camera.npy 의 [:3,3]에 δ를 더하면 된다.")
        print("           δ가 물체마다 제각각이거나 잔차가 크면 이동이 아니라 회전 오차이므로")
        print("           handeye_calibration.py로 다시 풀어야 한다.")
    else:
        print("회전차가 10도 미만이라 이동 성분 보정량은 풀 수 없다 "
              "(자세를 기울여 가며 다시 재라).")

    print("\n" + "=" * 78)
    if worst < 10.0:
        print(f"편차 최대 {worst:.1f}mm — 변환은 자세에 무관하게 일관적이다.")
        print("파지가 빗나간다면 hand-eye가 아니라 다른 원인이다 (TCP 정의, depth 편향 등).")
    elif worst < 30.0:
        print(f"편차 최대 {worst:.1f}mm — 경계선. depth 잡음일 수도, 캘리브레이션 오차일 수도 있다.")
        print("자세를 더 크게 벌려(높이·각도) 다시 재볼 것.")
    else:
        print(f"편차 최대 {worst:.1f}mm — **hand-eye 변환이 틀렸다.**")
        print("오프셋 튜닝으로는 못 고친다(한 자세에서만 맞는다). 재캘리브레이션이 필요하다:")
        print("  tools/calibration/handeye_calibration.py")
    print("=" * 78)


if __name__ == "__main__":
    main()

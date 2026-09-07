#!/usr/bin/env python3
"""GraspNet 후보 위치와 perception 실측 물체 위치의 차이를 재서
`graspnet_baseline.T_graspnet_tcp_mm`에 넣을 이동(translation)을 구한다.

**왜 필요한가.** GraspNet은 자기 gripper frame으로 파지 자세를 내놓는데, 그 원점은 손가락
사이 접촉점 근처다. 우리가 로봇에 보내는 것은 RG2의 TCP 자세이므로 그 둘 사이의 고정
변환(`T_graspnet_tcp_mm`)이 필요하다. 회전(축 맞바꿈)은 grasp_params.yaml에 들어가 있지만
**이동은 아직 0**이다 — 2026-09-05에 208mm를 넣었더니 후보가 물체에서 216mm 벗어나
되돌려 놓은 상태이고, 그때 "잔여 오차를 재서 채운다"가 미완으로 남았다. 이 스크립트가 그
측정이다.

**측정 원리.**
    T_base_tcp = M @ T_gn_tcp,        M = T_base_camera @ T_camera_graspnet
    T_gn_tcp = [R_swap | t]           (지금 t = 0)
현재 후보 위치는 M의 원점(M_p)이고, 우리가 원하는 것은 그 TCP가 물체 위치 O에 오는 것이다.
    M_R @ t = O - M_p = d      →      t = M_R^T @ d = R_swap @ R_cand^T @ d
(R_cand = 후보의 base 기준 회전 = M_R @ R_swap 이므로 M_R^T = R_swap @ R_cand^T)

`t`는 **GraspNet gripper frame으로 표현한 TCP 원점**이다. GraspNet 축은 X=접근,
Y=닫힘, Z=나머지이므로(grasp_params.yaml 주석), 물리적으로는 X 성분이 지배적이어야 한다 —
TCP가 접촉점보다 접근축 뒤쪽에 있기 때문이다. Y/Z가 크게 나오면 회전 매핑(R_swap)부터
의심해야 한다.

**한계.** 기준으로 삼는 `position_base_mm`은 마스크 안쪽 median depth로 만든 점, 즉
"보이는 윗면의 중심"이지 이상적인 파지점이 아니다. 200mm 규모의 오차를 잡는 데는 충분하지만
마지막 몇 mm는 이 값으로 확정하지 말 것.

쓰는 법(grasp 컨테이너 안에서):
    python3 measure_graspnet_offset.py --topic /world_state_probe --samples 20
"""
import argparse
import math

import numpy as np
import rclpy
import yaml
from rclpy.node import Node

from sort_msgs.msg import WorldState

DEFAULT_CONFIG = "/ros2_ws/install/grasp/share/grasp/config/grasp_params.yaml"


def quaternion_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0.0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class Collector(Node):
    def __init__(self, topic: str, samples: int, r_swap: np.ndarray, top_only: bool):
        super().__init__("graspnet_offset_probe_listener")
        self._target = samples
        self._r_swap = r_swap
        self._top_only = top_only
        self.rows: list[dict] = []
        self._seen_stamps: set[tuple[int, int]] = set()
        self.create_subscription(WorldState, topic, self._on_world, 10)
        self.get_logger().info(f"{topic} 구독 — 후보 {samples}개 모을 때까지 대기")

    def done(self) -> bool:
        return len(self.rows) >= self._target

    def _on_world(self, msg: WorldState) -> None:
        key = (msg.stamp.sec, msg.stamp.nanosec)
        if key in self._seen_stamps:
            return
        self._seen_stamps.add(key)
        for obj in msg.objects:
            if not obj.grasp_candidates:
                continue
            o = np.array([obj.position_base_mm.x, obj.position_base_mm.y,
                          obj.position_base_mm.z], dtype=float)
            for rank, cand in enumerate(obj.grasp_candidates):
                if self._top_only and rank > 0:
                    continue
                p = cand.pose.position
                q = cand.pose.orientation
                c = np.array([p.x, p.y, p.z], dtype=float)
                r_cand = quaternion_to_matrix(q.x, q.y, q.z, q.w)
                d = o - c                              # base 기준, 후보 → 물체
                t = self._r_swap @ r_cand.T @ d        # GraspNet gripper frame
                self.rows.append({
                    "object_id": obj.object_id, "class_name": obj.class_name,
                    "rank": rank, "score": cand.score,
                    "dist_mm": float(np.linalg.norm(d)), "t": t,
                    "strategy": cand.strategy,
                })
        if self.rows:
            self.get_logger().info(f"수집 {len(self.rows)}/{self._target}",
                                   throttle_duration_sec=3.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/world_state_probe")
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--exclude-class", action="append", default=[],
                    help="이 클래스는 통계에서 뺀다(부분 문자열 일치). 여러 번 쓸 수 있다. "
                         "형상·검출 품질 때문에 다른 물체와 계통이 다른 물체를 골라내는 용도")
    ap.add_argument("--all-ranks", action="store_true",
                    help="기본은 1순위 후보만 본다. 모든 후보를 쓰려면 지정")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    t_gn_tcp = np.asarray((cfg.get("graspnet_baseline") or {}).get("T_graspnet_tcp_mm"),
                          dtype=float)
    if t_gn_tcp.shape != (4, 4):
        raise SystemExit("설정에서 T_graspnet_tcp_mm(4x4)을 읽지 못했습니다")
    current_t = t_gn_tcp[:3, 3]
    r_swap = t_gn_tcp[:3, :3]
    if not np.allclose(current_t, 0.0):
        print(f"주의: 현재 T_graspnet_tcp_mm에 이동이 이미 들어 있다 {current_t} — "
              "아래 결과는 '거기서 더 더할 값'이다.")

    rclpy.init()
    node = Collector(args.topic, args.samples, r_swap, not args.all_ranks)
    deadline = node.get_clock().now().nanoseconds * 1e-9 + args.timeout
    try:
        while rclpy.ok() and not node.done():
            rclpy.spin_once(node, timeout_sec=0.5)
            if node.get_clock().now().nanoseconds * 1e-9 > deadline:
                print(f"타임아웃 — {len(node.rows)}개만 모았다")
                break
    finally:
        rows = node.rows
        node.destroy_node()
        rclpy.shutdown()

    if not rows:
        raise SystemExit("후보를 하나도 못 받았다. probe 노드와 전략 설정을 확인할 것")

    if args.exclude_class:
        kept = [r for r in rows
                if not any(x in r["class_name"] for x in args.exclude_class)]
        print(f"제외: {', '.join(args.exclude_class)} → {len(rows)} - {len(rows) - len(kept)} "
              f"= {len(kept)}개 사용")
        rows = kept
        if not rows:
            raise SystemExit("제외하고 나니 표본이 없다")

    strategies = {r["strategy"] for r in rows}
    print(f"\n표본 {len(rows)}개 (전략 {', '.join(sorted(strategies))})")
    if strategies != {"graspnet_baseline"}:
        print("경고: graspnet_baseline이 아닌 후보가 섞여 있다 — probe의 strategy 설정 확인")

    ts = np.array([r["t"] for r in rows])
    dists = np.array([r["dist_mm"] for r in rows])
    median = np.median(ts, axis=0)
    print(f"후보-물체 거리(mm): 중앙값 {np.median(dists):.1f}, "
          f"범위 [{dists.min():.1f}, {dists.max():.1f}]")
    print("\nGraspNet gripper frame 기준 TCP 오프셋 t (mm)   [X=접근, Y=닫힘, Z=나머지]")
    for i, axis in enumerate("XYZ"):
        col = ts[:, i]
        print(f"  {axis}: 중앙값 {np.median(col):+8.1f}   평균 {col.mean():+8.1f}   "
              f"표준편차 {col.std():6.1f}   범위 [{col.min():+.1f}, {col.max():+.1f}]")

    # 물체별로 갈리는지가 핵심이다 — 고정 변환이면 클래스와 무관하게 같은 값이어야 한다.
    print("\n물체별 중앙값 (클래스명 포함 — 어느 물체가 튀는지 보려는 것):")
    by_object: dict[str, str] = {}
    for r in rows:
        by_object[r["object_id"]] = r["class_name"]
    for oid in sorted(by_object):
        sub = np.array([r["t"] for r in rows if r["object_id"] == oid])
        dsub = np.array([r["dist_mm"] for r in rows if r["object_id"] == oid])
        m = np.median(sub, axis=0)
        print(f"  {oid:8s} {by_object[oid][:34]:36s} n={len(sub):3d}  "
              f"t=({m[0]:+7.1f}, {m[1]:+7.1f}, {m[2]:+7.1f})  거리 {np.median(dsub):5.1f}")

    final = current_t + median
    print("\n" + "=" * 66)
    print("grasp_params.yaml의 T_graspnet_tcp_mm에 넣을 값 (마지막 열이 이동):")
    for i in range(3):
        print(f"    - [{r_swap[i,0]:.1f}, {r_swap[i,1]:.1f}, {r_swap[i,2]:.1f}, {final[i]:.1f}]")
    print("    - [0.0, 0.0, 0.0, 1.0]")
    print("=" * 66)
    print("표준편차가 크면(수십 mm) 회전 매핑이나 관측 품질부터 의심할 것 — "
          "고정 변환이라면 물체·자세와 무관하게 같은 값이 나와야 한다.")


if __name__ == "__main__":
    main()

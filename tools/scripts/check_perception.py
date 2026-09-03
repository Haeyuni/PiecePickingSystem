#!/usr/bin/env python3
"""perception 노드가 인터페이스 계약을 지키는지 확인한다.

실행 중인 `perception_node`의 발행을 일정 시간 받아서 검사한다 — 모델 성능(mAP)이 아니라
**계약**을 본다. 검출이 맞는지는 사람이 `/perception/debug_image`로 보고, 여기서 보는 것은
grasp·planner가 이 메시지를 받았을 때 성립해야 하는 것들이다.

확인 항목 (인터페이스_정의서.md 2.0·3.2~3.4절):
- 두 토픽이 **같은 stamp**로 나온다 — grasp가 그 값으로 마스크와 물체를 짝짓는다
- `grasp_candidates`는 비어 있다 (채우는 것은 grasp의 일)
- `frame_id`가 base이고 좌표가 작업반경 안이다
- 마스크는 mono8 0/255이고 해상도가 컬러 이미지와 같다
- 마스크의 object_id가 WorldState의 물체와 대응한다
- `graspable=false`인 물체가 `needs_reobserve`에 담긴다
- 속성이 objects.yaml/DB와 일치한다
- object_id가 프레임 사이에 유지된다

사용법: (워크스페이스 source 후) python3 scripts/check_perception.py [--seconds 15]
"""
import argparse
import pathlib
import sys

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from sort_msgs.msg import InstanceMasks, WorldState

ROOT = pathlib.Path(__file__).resolve().parents[1]
OBJECTS_YAML = ROOT / "perception" / "config" / "objects.yaml"
WORKSPACE_RADIUS_MM = 900.0      # M0609 (planner/src/validator.py와 같은 값)


class Collector(Node):
    def __init__(self):
        super().__init__("check_perception")
        self.worlds: list[WorldState] = []
        self.masks: list[InstanceMasks] = []
        self.color_shape = None
        self.create_subscription(WorldState, "/perception/world_state_raw",
                                 lambda m: self.worlds.append(m), 10)
        self.create_subscription(InstanceMasks, "/perception/instance_masks",
                                 lambda m: self.masks.append(m), 10)
        self.create_subscription(CameraInfo, "/camera/color/camera_info",
                                 self._on_info, 10)

    def _on_info(self, msg: CameraInfo) -> None:
        self.color_shape = (msg.height, msg.width)


def stamp_key(stamp) -> tuple[int, int]:
    return (stamp.sec, stamp.nanosec)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=15.0)
    args = ap.parse_args()

    seed = yaml.safe_load(OBJECTS_YAML.read_text(encoding="utf-8")) or {}
    seed_objects = seed.get("objects") or {}

    rclpy.init()
    node = Collector()
    deadline = node.get_clock().now().nanoseconds + args.seconds * 1e9
    while node.get_clock().now().nanoseconds < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)

    failures: list[str] = []

    def expect(label, actual, wanted):
        ok = actual == wanted
        print(f"{'OK ' if ok else 'NG '} {label}: {actual} (기대 {wanted})")
        if not ok:
            failures.append(label)

    def check(label, ok, detail=""):
        print(f"{'OK ' if ok else 'NG '} {label}{(' — ' + detail) if detail else ''}")
        if not ok:
            failures.append(label)

    print(f"{args.seconds:.0f}초 수신: world_state_raw {len(node.worlds)}건, "
          f"instance_masks {len(node.masks)}건\n")
    if not node.worlds:
        print("world_state_raw를 한 건도 받지 못했다 — perception_node가 떠 있는지 확인하세요.")
        node.destroy_node(); rclpy.shutdown()
        return 1

    # --- 두 토픽의 stamp 짝 --------------------------------------------------
    world_stamps = {stamp_key(w.stamp) for w in node.worlds}
    mask_stamps = {stamp_key(m.stamp) for m in node.masks}
    # 두 토픽은 같은 호출에서 연달아 나가므로, 수신 창의 처음·끝 한 프레임은 짝이 잘릴 수
    # 있다. 그것은 계약 위반이 아니라 샘플링 경계다. 검사할 성질은 두 가지다:
    #   (1) 짝 없는 마스크가 없다 — 마스크만 있고 물체가 없으면 grasp가 짝지을 대상이 없다
    #   (2) 짝 없는 world는 경계에서만 나온다
    orphan_masks = mask_stamps - world_stamps
    unmatched_worlds = world_stamps - mask_stamps
    boundary = {min(world_stamps), max(world_stamps)} if world_stamps else set()
    check("두 토픽의 stamp가 짝을 이룬다",
          bool(world_stamps) and not orphan_masks and unmatched_worlds <= boundary,
          f"짝 없는 마스크 {len(orphan_masks)}건, "
          f"짝 없는 world {len(unmatched_worlds)}건(경계 제외 "
          f"{len(unmatched_worlds - boundary)}건)")

    masks_by_stamp = {stamp_key(m.stamp): m for m in node.masks}
    latest = node.worlds[-1]

    expect("frame_id", latest.frame_id, "base")
    expect("schema_version", latest.schema_version, "1.0.0")
    check("trace_id가 비어 있지 않다", bool(latest.trace_id), latest.trace_id)

    # --- 물체별 계약 ---------------------------------------------------------
    all_objects = [o for w in node.worlds for o in w.objects]
    check("물체를 하나 이상 검출했다", bool(all_objects),
          f"{len(all_objects)}건 (0이면 카메라 앞에 물체를 두고 다시 실행)")

    check("grasp_candidates가 비어 있다",
          all(len(o.grasp_candidates) == 0 for o in all_objects),
          "채우는 것은 grasp의 일이다 (인터페이스_정의서 2.0절)")

    reachable = [o for o in all_objects if o.graspable]
    check("파지 가능 물체의 좌표가 작업반경 안이다",
          all(np.linalg.norm([o.position_base_mm.x, o.position_base_mm.y,
                              o.position_base_mm.z]) <= WORKSPACE_RADIUS_MM
              for o in reachable),
          f"{len(reachable)}건 검사")

    bad_reason = [o.object_id for o in all_objects
                  if o.graspable and o.not_graspable_reason]
    check("graspable=true면 사유가 비어 있다", not bad_reason, str(bad_reason))

    for world in node.worlds:
        expected = sorted(o.object_id for o in world.objects if not o.graspable)
        if sorted(world.needs_reobserve) != expected:
            failures.append("needs_reobserve 불일치")
            print(f"NG  needs_reobserve 불일치: {sorted(world.needs_reobserve)} != {expected}")
            break
    else:
        print("OK  graspable=false 물체가 needs_reobserve에 담긴다")

    # --- 속성이 seed와 일치하는가 --------------------------------------------
    mismatched = []
    for o in all_objects:
        s = seed_objects.get(o.class_name)
        if s is None:
            if not o.needs_confirmation:
                mismatched.append(f"{o.class_name}: seed에 없는데 needs_confirmation=false")
            continue
        if o.name_ko != (s.get("name_ko") or "") or o.profile != s.get("profile"):
            mismatched.append(f"{o.class_name}: name_ko/profile이 seed와 다름 "
                              f"({o.name_ko}/{o.profile} vs {s.get('name_ko')}/{s.get('profile')})")
    check("속성이 objects.yaml과 일치한다", not mismatched, "; ".join(sorted(set(mismatched))))

    # --- 마스크 ---------------------------------------------------------------
    mask_problems = []
    for world in node.worlds:
        masks = masks_by_stamp.get(stamp_key(world.stamp))
        if masks is None:
            continue
        world_ids = {o.object_id for o in world.objects}
        if not set(masks.object_ids) <= world_ids:
            mask_problems.append(f"마스크에만 있는 object_id {set(masks.object_ids) - world_ids}")
        if len(masks.object_ids) != len(masks.masks):
            mask_problems.append("object_ids와 masks 길이가 다르다")
        for image in masks.masks:
            if image.encoding != "mono8":
                mask_problems.append(f"인코딩이 {image.encoding} (mono8이어야 함)")
                continue
            values = set(np.unique(np.frombuffer(image.data, dtype=np.uint8)))
            if not values <= {0, 255}:
                mask_problems.append(f"0/255 외의 값 {sorted(values)[:5]}")
            if node.color_shape and (image.height, image.width) != node.color_shape:
                mask_problems.append(
                    f"해상도 {image.height}x{image.width} != 컬러 {node.color_shape}")
    check("마스크가 mono8 0/255이고 컬러와 해상도가 같다", not mask_problems,
          "; ".join(sorted(set(mask_problems))[:3]))

    # --- object_id 유지 -------------------------------------------------------
    per_frame = [{o.object_id for o in w.objects} for w in node.worlds if w.objects]
    if len(per_frame) >= 2:
        common = set.intersection(*per_frame)
        check("object_id가 프레임 사이에 유지된다", bool(common),
              f"{len(per_frame)}프레임 내내 유지된 id {sorted(common)}")
    else:
        print("--  object_id 유지: 물체가 있는 프레임이 2개 미만이라 건너뜀")

    node.destroy_node()
    rclpy.shutdown()

    print()
    for f in sorted(set(failures)):
        print(f"실패 — {f}")
    print("perception 계약 통과" if not failures else f"{len(set(failures))}건 실패")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

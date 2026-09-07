#!/usr/bin/env python3
"""로봇을 자로 써서 perception의 물체 위치가 실제와 맞는지 잰다.

**왜 이게 필요한가.** 파지점이 클라우드 위에 있는지, 로봇이 명령대로 갔는지는 전부
**같은 변환**(`posx_to_matrix(TCP) @ T_gripper2camera`)을 지난 값끼리의 비교라, 그 변환이
틀려도 서로 완벽하게 맞아 보인다. 화면 오버레이도 마찬가지다. 그 변환의 진위는 **변환을
거치지 않은 값**과 대볼 때만 드러나고, 이 셀에서 그런 값은 로봇 자신의 `get_current_posx`뿐이다.

**사용법.**
  1) 물체 하나를 고르고, 로봇을 조그(jog)해서 **그리퍼 손끝을 그 물체 위에 정확히** 둔다.
     - 물체를 집는 자세 그대로, 손끝이 물체 표면에 막 닿는 높이가 좋다.
     - 조그는 사람이 한다. 이 스크립트는 읽기만 하고 아무 명령도 보내지 않는다.
  2) 그 자세에서 이 스크립트를 실행한다.
  3) 출력의 "차이"가 곧 **perception 좌표의 실제 오차**다.

    python3 touch_check.py --object editb          # 클래스명 일부로 지정

**읽는 법.**
  차이가 몇 mm      → 변환 정상. 파지 실패 원인은 다른 데 있다.
  차이가 수십 mm    → perception 좌표가 그만큼 틀렸다. hand-eye 재캘리브레이션이 답이고,
                      오프셋으로 덮으면 이 자세에서만 맞는다.
  **주의**: TCP가 손끝이 아니라 플랜지에 정의돼 있으면 그 차이만큼 Z가 통째로 어긋난다.
  이 검사는 그 경우도 같이 드러낸다(Z만 크게 어긋나고 XY는 작게 나온다).
"""
import argparse
import time

import rclpy
from rclpy.node import Node

from dsr_msgs2.srv import GetCurrentPosx
from sort_msgs.msg import WorldState


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="", help="클래스명 일부(생략하면 전부 표시)")
    args = ap.parse_args()

    rclpy.init()
    node = Node("touch_check")
    state = {}
    node.create_subscription(WorldState, "/world_state",
                             lambda m: state.__setitem__("world", m), 10)
    posx = node.create_client(GetCurrentPosx,
                              "/dsr01/dsr_controller2/aux_control/get_current_posx")

    t0 = time.monotonic()
    while "world" not in state and time.monotonic() - t0 < 20:
        rclpy.spin_once(node, timeout_sec=0.3)
    if "world" not in state:
        raise SystemExit("/world_state를 못 받았다 — perception이 도는지 확인할 것")

    if not posx.wait_for_service(timeout_sec=5.0):
        raise SystemExit("get_current_posx 서비스가 없다 — 로봇 드라이버 확인")
    future = posx.call_async(GetCurrentPosx.Request(ref=0))
    rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
    result = future.result()
    if result is None or not result.success:
        raise SystemExit("get_current_posx 응답 없음")
    tcp = [float(v) for v in result.task_pos_info[0].data[:6]]

    objects = [o for o in state["world"].objects
               if not args.object or args.object.lower() in o.class_name.lower()]
    node.destroy_node()
    rclpy.shutdown()

    print(f"\n로봇 TCP (get_current_posx, 변환 없음)")
    print(f"  위치 = ({tcp[0]:8.1f}, {tcp[1]:8.1f}, {tcp[2]:8.1f}) mm")
    print(f"  자세 = ZYZ({tcp[3]:7.1f}, {tcp[4]:7.1f}, {tcp[5]:7.1f}) deg\n")

    if not objects:
        raise SystemExit(f"'{args.object}'에 해당하는 물체가 없다")

    print(f"{'물체':34s} {'perception 위치':>26s} {'차이 (perception - TCP)':>28s}")
    print("-" * 92)
    for o in objects:
        p = o.position_base_mm
        dx, dy, dz = p.x - tcp[0], p.y - tcp[1], p.z - tcp[2]
        total = (dx * dx + dy * dy + dz * dz) ** 0.5
        print(f"{o.class_name[:32]:34s} ({p.x:7.1f}, {p.y:7.1f}, {p.z:7.1f}) "
              f"({dx:+7.1f}, {dy:+7.1f}, {dz:+7.1f}) |{total:6.1f}mm")

    print("\n손끝을 올려둔 그 물체의 행을 보세요. XY 차이가 실제 좌표 오차이고,")
    print("Z 차이에는 'TCP가 손끝이 아닌 만큼'이 함께 들어갑니다.")


if __name__ == "__main__":
    main()

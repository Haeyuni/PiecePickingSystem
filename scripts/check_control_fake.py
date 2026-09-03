#!/usr/bin/env python3
"""control 액션 서버 계약 확인 (개발계획.md B4 완료 기준).

`FAKE_ROBOT=1`로 뜬 control에 액션을 직접 보낸다. web을 거치지 않는 이유는 여기서 보려는
것이 **control이 지키기로 한 계약**이기 때문이다 — 중복 방지·목적지 그라운딩·취소는
로봇 유무와 무관하게 성립해야 하고, 실물로 바꿀 때 새로 검증할 것은 모션뿐이어야 한다.

확인 항목:
- 같은 request_id로 두 번 보내면 두 번째는 재실행하지 않고 이전 결과를 돌려준다
- bins.yaml에 없는 bin_id는 실패로 거부된다 (planner 검증기와 별개로 control도 확인)
- 진행 중인 home은 취소된다 (Home.action이 서비스가 아니라 액션인 이유)
- 액션이 도는 동안 robot_state가 busy로, 끝나면 idle로 돌아온다

사용법: (워크스페이스 source 후) python3 scripts/check_control_fake.py
"""
import sys
import time
import uuid

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from sort_msgs.action import Home, PlaceInto, Pick
from sort_msgs.msg import RobotState

SCHEMA_VERSION = "1.0.0"
TIMEOUT_S = 20.0


class Probe(Node):
    def __init__(self):
        super().__init__("check_control_fake")
        self.pick = ActionClient(self, Pick, "pick")
        self.place = ActionClient(self, PlaceInto, "place_into")
        self.home = ActionClient(self, Home, "home")
        self.modes = []
        self.create_subscription(RobotState, "/control/robot_state", self._on_state, 10)

    def _on_state(self, msg):
        if not self.modes or self.modes[-1] != msg.mode:
            self.modes.append(msg.mode)


def spin_until(executor, predicate, timeout_s=TIMEOUT_S):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and not predicate():
        executor.spin_once(timeout_sec=0.05)
    return predicate()


def send(executor, client, goal, timeout_s=TIMEOUT_S):
    """goal을 보내고 결과를 받는다. (result, 소요시간초)"""
    if not client.wait_for_server(timeout_sec=5.0):
        raise RuntimeError(f"{client._action_name} 액션 서버에 연결하지 못했습니다")
    started = time.monotonic()
    send_future = client.send_goal_async(goal)
    spin_until(executor, send_future.done, timeout_s)
    handle = send_future.result()
    result_future = handle.get_result_async()
    spin_until(executor, result_future.done, timeout_s)
    return result_future.result().result, time.monotonic() - started


def pick_goal(request_id):
    from geometry_msgs.msg import Pose

    g = Pick.Goal()
    g.schema_version = SCHEMA_VERSION
    g.trace_id = "tr-check-control"
    g.request_id = request_id
    g.object_id = "obj_001"
    g.profile = "fragile"
    g.grasp_pose = Pose()
    g.grasp_pose.position.x = 412.5   # mm (인터페이스_정의서 1.1절)
    g.grasp_pose.position.y = -38.0
    g.grasp_pose.position.z = 62.0
    g.grasp_pose.orientation.w = 1.0
    g.max_retries = 1
    return g


def main() -> int:
    rclpy.init()
    node = Probe()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    failures = []

    def expect(label, actual, wanted):
        ok = actual == wanted
        print(f"{'OK ' if ok else 'NG '} {label}: {actual} (기대 {wanted})")
        if not ok:
            failures.append(label)

    try:
        # --- 1) request_id 중복 방지 ------------------------------------------
        request_id = f"rq-dup-{uuid.uuid4().hex[:8]}"
        first, first_s = send(executor, node.pick, pick_goal(request_id))
        expect("최초 pick 성공", first.success, True)

        second, second_s = send(executor, node.pick, pick_goal(request_id))
        expect("중복 pick 성공 응답", second.success, True)
        # 재실행했다면 phase 지연만큼(4 x 0.4s) 다시 걸린다. 캐시 적중이면 즉시 돌아온다.
        expect("중복 pick이 재실행되지 않음", second_s < first_s / 2, True)
        print(f"    (최초 {first_s:.2f}s → 중복 {second_s:.2f}s)")

        # --- 2) bin_id 그라운딩 ------------------------------------------------
        g = PlaceInto.Goal()
        g.schema_version = SCHEMA_VERSION
        g.trace_id = "tr-check-control"
        g.request_id = f"rq-bin-{uuid.uuid4().hex[:8]}"
        g.object_id = "obj_001"
        g.profile = "normal"
        g.bin_id = "없는_박스"
        g.max_retries = 1
        result, _ = send(executor, node.place, g)
        expect("미등록 bin_id 거부", result.success, False)
        expect("거부 사유", result.failure_reason, PlaceInto.Result.REASON_UNREACHABLE)

        # 등록된 bin_id는 통과해야 한다 — 위 실패가 그라운딩 때문이지 place가 통째로
        # 고장난 것이 아님을 같이 보여야 검사로서 의미가 있다.
        g.bin_id = "left_box"
        g.request_id = f"rq-bin-ok-{uuid.uuid4().hex[:8]}"
        result, _ = send(executor, node.place, g)
        expect("등록된 bin_id 통과", result.success, True)

        # --- 3) home 취소 -------------------------------------------------------
        hg = Home.Goal()
        hg.schema_version = SCHEMA_VERSION
        hg.request_id = f"rq-home-{uuid.uuid4().hex[:8]}"
        node.home.wait_for_server(timeout_sec=5.0)
        send_future = node.home.send_goal_async(hg)
        spin_until(executor, send_future.done)
        handle = send_future.result()
        cancel_future = handle.cancel_goal_async()
        spin_until(executor, cancel_future.done)
        expect("home 취소 수락", len(cancel_future.result().goals_canceling) > 0, True)

        # --- 4) robot_state 전이 -----------------------------------------------
        spin_until(executor, lambda: node.modes and node.modes[-1] == RobotState.MODE_IDLE, 5.0)
        transitions = node.modes
        print(f"    관측된 mode 전이: {' → '.join(transitions)}")
        expect("busy를 거쳐 idle로 복귀", 
               RobotState.MODE_BUSY in transitions and transitions[-1] == RobotState.MODE_IDLE,
               True)

    finally:
        node.destroy_node()
        rclpy.shutdown()

    print()
    for f in failures:
        print(f"실패 — {f}")
    print("control 계약 통과" if not failures else f"{len(failures)}건 실패")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

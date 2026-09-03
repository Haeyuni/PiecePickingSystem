"""pick 액션 서버. compliance(토크 판정) AND visual_verification 통과 시에만 success=true.

참조: 인터페이스_정의서.md 4.1절 (Pick.action)

`FAKE_ROBOT=1`이면 실제 모션 없이 phase를 순서대로 발행하고 성공을 반환한다(개발계획 B4).

실패 주입: `FAKE_FAIL_OBJECT`에 object_id를 넣으면 그 물체의 pick이 `grasp_failed`로
실패한다. 재계획(FR-16)은 실패가 나야만 지나가는 경로인데, fake 모드는 정의상 늘
성공하므로 그대로는 그 경로를 밟을 방법이 없다. mock 어댑터의 `MOCK_FAIL_OBJECT`와
같은 이유로 같은 자리에 둔다 — 두 경로가 같은 시나리오로 검증되어야 교체가 안전하다.
**fake와 실물이 공유하는 것**: request_id 중복 방지, robot_state 전이, 실패 사유 분류.
이 계약들은 로봇 유무와 무관하므로 fake 경로에서도 그대로 지켜져야 한다 — 그래야 실물로
바꿀 때 새로 검증할 것이 순응제어·모션뿐이다.
"""
import os
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from sort_msgs.action import Pick

from .request_cache import RequestCache
from .robot_state_publisher import is_fake_robot, store

SCHEMA_VERSION = "1.0.0"

# fake 모드에서 각 phase에 머무는 시간. 실물에서는 모션 시간이 이를 대체한다.
FAKE_PHASE_DURATION_S = 0.4

PHASES = [
    Pick.Feedback.PHASE_APPROACHING,
    Pick.Feedback.PHASE_CONTACT_DETECTED,
    Pick.Feedback.PHASE_LIFTING,
    Pick.Feedback.PHASE_VERIFYING,
]


class PickServer(Node):
    def __init__(self):
        super().__init__('pick_server')
        self._cache = RequestCache()
        self._action_server = ActionServer(
            self, Pick, 'pick', self.execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=ReentrantCallbackGroup(),
        )
        self.get_logger().info(f"pick 액션 서버 준비 ({'fake' if is_fake_robot() else '실물'} 모드)")

    def _goal_callback(self, goal_request):
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        # 정지는 busy일 때 가장 필요한 동작이라 언제나 받아들인다 (웹_인터페이스_정의서 2.6절)
        self.get_logger().warning("pick 취소 요청 수신")
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        goal = goal_handle.request

        # 재전송된 요청이면 다시 움직이지 않고 이전 결과를 그대로 돌려준다.
        # 필드만으로는 중복이 막히지 않는다 (인터페이스_정의서 1절).
        cached = self._cache.get(goal.request_id)
        if cached is not None:
            self.get_logger().warning(
                f"중복 request_id={goal.request_id} — 재실행하지 않고 이전 결과 반환")
            goal_handle.succeed()
            return cached

        self.get_logger().info(
            f"pick 시작 object={goal.object_id} profile={goal.profile} "
            f"request={goal.request_id}")
        store.set_busy("pick")
        started = time.monotonic()

        try:
            for phase in PHASES:
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    return self._result(False, Pick.Result.REASON_NO_CONTACT, started)

                feedback = Pick.Feedback()
                feedback.phase = phase
                goal_handle.publish_feedback(feedback)

                if is_fake_robot():
                    time.sleep(FAKE_PHASE_DURATION_S)
                else:
                    # TODO(실물): compliance.py(접촉 감지·토크 판정)와 MoveIt2 모션 연결.
                    # 좌표는 mm로 들어오므로 units.pose_mm_to_m()으로 환산해 넘긴다.
                    raise NotImplementedError("실물 pick은 compliance/MoveIt2 연결 후 활성화한다")

            if self._injected_failure(goal.object_id):
                self.get_logger().warning(
                    f"실패 주입 (FAKE_FAIL_OBJECT={goal.object_id}) — grasp_failed 반환")
                result = self._result(False, Pick.Result.REASON_GRASP_FAILED, started)
                self._cache.put(goal.request_id, result)
                goal_handle.succeed()
                return result

            store.set_gripper(width_mm=42.0, closed=True)
            result = self._result(True, Pick.Result.REASON_NONE, started,
                                  visual_passed=True, torque=[0.4, 1.9, 2.6, 2.4])
            self._cache.put(goal.request_id, result)
            goal_handle.succeed()
            return result

        except Exception as e:
            self.get_logger().error(f"pick 실패: {e}")
            store.set_error()
            goal_handle.abort()
            return self._result(False, Pick.Result.REASON_GRASP_FAILED, started)
        finally:
            if store.snapshot()["mode"] != "error":
                store.set_idle()

    @staticmethod
    def _injected_failure(object_id: str) -> bool:
        return bool(is_fake_robot() and os.environ.get("FAKE_FAIL_OBJECT") == object_id)

    def _result(self, success, reason, started, visual_passed=False, torque=None):
        result = Pick.Result()
        result.success = success
        result.failure_reason = reason
        result.retries_used = 0
        result.cycle_time_ms = (time.monotonic() - started) * 1000
        result.torque_trace_summary = torque or []
        result.visual_verification_passed = visual_passed
        result.visual_verification_note = "" if visual_passed else "미검증"
        return result


def main(args=None):
    rclpy.init(args=args)
    node = PickServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

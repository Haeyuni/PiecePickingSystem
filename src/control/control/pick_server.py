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
import pathlib
import time

import rclpy
import yaml
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from control_msgs.action import GripperCommand
from dsr_msgs2.action import MovelH2r
from sort_msgs.action import Pick

from . import dsr_motion
from .config_paths import skill_params_path
from .request_cache import RequestCache
from .robot_state_publisher import is_fake_robot, store

SCHEMA_VERSION = "1.0.0"


def load_skill_params(path: pathlib.Path | None = None) -> dict:
    path = path or skill_params_path()
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

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
        callbacks = ReentrantCallbackGroup()
        self._action_server = ActionServer(
            self, Pick, 'pick', self.execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=callbacks,
        )

        params = load_skill_params()
        self._profiles = params.get("profiles") or {}
        motion = params.get("motion") or {}
        self._approach_height_mm = float(motion.get("approach_height_mm", 80.0))
        self._linear_vel_mm_s = float(motion.get("linear_vel_mm_s", 30.0))
        self._linear_acc_mm_s2 = float(motion.get("linear_acc_mm_s2", 30.0))
        self._rot_vel_deg_s = float(motion.get("rot_vel_deg_s", 20.0))
        self._rot_acc_deg_s2 = float(motion.get("rot_acc_deg_s2", 20.0))
        self._movel_client = ActionClient(self, MovelH2r, dsr_motion.MOVEL_ACTION,
                                          callback_group=callbacks)
        self._gripper_client = ActionClient(self, GripperCommand, dsr_motion.GRIPPER_ACTION,
                                            callback_group=callbacks)

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
            if is_fake_robot():
                for phase in PHASES:
                    if goal_handle.is_cancel_requested:
                        goal_handle.canceled()
                        return self._result(False, Pick.Result.REASON_NO_CONTACT, started)
                    self._publish_phase(goal_handle, phase)
                    time.sleep(FAKE_PHASE_DURATION_S)
                width_mm, visual_passed, torque = 42.0, True, [0.4, 1.9, 2.6, 2.4]
            else:
                width_mm = self._pick_real(goal_handle, goal)
                if width_mm is None:
                    goal_handle.canceled()
                    return self._result(False, Pick.Result.REASON_NO_CONTACT, started)
                visual_passed, torque = False, []

            if self._injected_failure(goal.object_id):
                self.get_logger().warning(
                    f"실패 주입 (FAKE_FAIL_OBJECT={goal.object_id}) — grasp_failed 반환")
                result = self._result(False, Pick.Result.REASON_GRASP_FAILED, started)
                self._cache.put(goal.request_id, result)
                goal_handle.succeed()
                return result

            store.set_gripper(width_mm=width_mm, closed=True)
            result = self._result(True, Pick.Result.REASON_NONE, started,
                                  visual_passed=visual_passed, torque=torque)
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
    def _publish_phase(goal_handle, phase) -> None:
        feedback = Pick.Feedback()
        feedback.phase = phase
        goal_handle.publish_feedback(feedback)

    def _pick_real(self, goal_handle, goal) -> float | None:
        """위치제어만으로 실물 pick을 수행한다 (1단계 — compliance.py/visual_verification.py가
        아직 빈 스텁이라 접촉감지·파지확인 없이 grasp_pose를 그대로 믿고 움직인다).

        grasp_pose 바로 위(approach_height_mm)에서 한 번 멈췄다 내려가 그리퍼를 힘 제한
        (profile의 max_grip_force_n)으로 닫고 다시 들어올린다. 실패하면 RuntimeError,
        취소되면 None — dsr_motion의 각 호출이 취소 시 False를 돌려주므로 여기서
        `goal_handle.is_cancel_requested`로 두 경우를 구분한다(취소는 오류가 아니다).
        """
        from . import units

        target_posx = dsr_motion.pose_mm_to_posx(goal.grasp_pose)
        approach_posx = list(target_posx)
        approach_posx[2] += self._approach_height_mm
        max_force = float((self._profiles.get(goal.profile) or {}).get("max_grip_force_n", 20.0))

        def move(pos):
            return dsr_motion.move_linear(self._movel_client, pos, goal_handle,
                                          self._linear_vel_mm_s, self._linear_acc_mm_s2,
                                          self._rot_vel_deg_s, self._rot_acc_deg_s2)

        self._publish_phase(goal_handle, Pick.Feedback.PHASE_APPROACHING)
        if not move(approach_posx):
            if goal_handle.is_cancel_requested:
                return None
            raise RuntimeError("접근 위치로 이동 실패")
        if not move(target_posx):
            if goal_handle.is_cancel_requested:
                return None
            raise RuntimeError("파지 위치로 이동 실패")

        self._publish_phase(goal_handle, Pick.Feedback.PHASE_CONTACT_DETECTED)
        ok, width_m = dsr_motion.move_gripper(self._gripper_client, 0.0, max_force, goal_handle)
        if not ok:
            if goal_handle.is_cancel_requested:
                return None
            raise RuntimeError("그리퍼 닫기 실패")

        self._publish_phase(goal_handle, Pick.Feedback.PHASE_LIFTING)
        if not move(approach_posx):
            if goal_handle.is_cancel_requested:
                return None
            raise RuntimeError("들어올리기 실패")

        self._publish_phase(goal_handle, Pick.Feedback.PHASE_VERIFYING)
        self.get_logger().warning(
            "위치제어만으로 pick 완료 — compliance/visual_verification 미구현이라 "
            "실제 파지 여부는 확인되지 않았다")
        return units.m_to_mm(width_m)

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

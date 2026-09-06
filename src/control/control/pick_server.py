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

from dsr_msgs2.action import MovelH2r
from dsr_msgs2.srv import GetCurrentPosx
from onrobot_rg_msgs.srv import GripperPose, SetCommand
from sensor_msgs.msg import JointState
from sort_msgs.action import Pick

from . import dsr_motion
from .config_paths import skill_params_path
from .request_cache import RequestCache
from .robot_state_publisher import is_fake_robot, store

SCHEMA_VERSION = "1.0.0"


class _LiftFailedError(RuntimeError):
    """그리퍼가 이미 물체를 문 뒤(닫기 완료) 들어올리기 단계에서 난 실패.

    일반 RuntimeError와 구분하는 이유: 예전엔 이 경우도 다른 모든 예외와 똑같이
    grasp_failed로 보고했다 — 파지는 성공했는데 "파지 실패"로 나와 헷갈리게 했다
    (2026-09-04 실물로 확인). `width_mm`을 들고 있어 execute_callback이 실제로
    뭔가를 물었는지 판단해 사유를 구분해 보고할 수 있게 한다.
    """

    def __init__(self, message: str, width_mm: float | None):
        super().__init__(message)
        self.width_mm = width_mm


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
        motion = params.get("motion") or {}
        self._approach_height_mm = float(motion.get("approach_height_mm", 80.0))
        self._linear_vel_mm_s = float(motion.get("linear_vel_mm_s", 30.0))
        self._linear_acc_mm_s2 = float(motion.get("linear_acc_mm_s2", 30.0))
        self._rot_vel_deg_s = float(motion.get("rot_vel_deg_s", 20.0))
        self._rot_acc_deg_s2 = float(motion.get("rot_acc_deg_s2", 20.0))
        self._gripper_open_m = float(motion.get("gripper_open_m", 0.110))
        # grasp_pose.z까지만 내려가면 실물에서 접근이 부족한 사례가 있었다 — 하강 목표를
        # 이만큼 더 낮춘다 (skill_params.yaml 주석 참조).
        self._pick_depth_extra_mm = float(motion.get("pick_depth_extra_mm", 0.0))
        gripper = params.get("gripper") or {}
        # 이 미만이면 "닫혔지만 아무것도 안 물렸다"로 본다. compliance/visual_verification이
        # 아직 없어 이게 유일한 파지 확인 수단이다 — grasp_test/robot_executor.py의
        # 같은 판정(min_grip_width_mm)과 같은 기본값을 쓴다.
        self._min_grip_width_mm = float(gripper.get("min_grip_width_mm", 5.0))
        # GraspCandidate.gripper_width_mm(전략이 예측한 필요 개폭)에 더하는 여유 — 이만큼만
        # 더 열면 접근 중 손가락이 물체를 스치지 않으면서도 다 열 때보다 빨리 닫히고,
        # 클러터에서 옆 물체를 안 건드린다. 모르면(0 이하) 기존처럼 최대로 연다.
        self._gripper_width_margin_mm = float(gripper.get("width_margin_mm", 15.0))
        self._movel_client = ActionClient(self, MovelH2r, dsr_motion.MOVEL_ACTION,
                                          callback_group=callbacks)
        self._posx_client = self.create_client(
            GetCurrentPosx, dsr_motion.GET_CURRENT_POSX_SERVICE, callback_group=callbacks)
        self._gripper_cmd_client = self.create_client(
            SetCommand, dsr_motion.GRIPPER_COMMAND_SERVICE, callback_group=callbacks)
        self._gripper_pose_client = self.create_client(
            GripperPose, "/onrobot/pose", callback_group=callbacks)
        self._gripper_joint_angle: float | None = None
        self.create_subscription(JointState, dsr_motion.GRIPPER_JOINT_STATES_TOPIC,
                                 self._on_gripper_state, 5, callback_group=callbacks)

        self.get_logger().info(f"pick 액션 서버 준비 ({'fake' if is_fake_robot() else '실물'} 모드)")

    def _on_gripper_state(self, msg: JointState) -> None:
        if msg.position:
            self._gripper_joint_angle = msg.position[0]

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
                if width_mm <= self._min_grip_width_mm:
                    # 그리퍼가 끝까지 닫혔다 — 물체 없이 빈 채로 닫힌 것과 구분이 안 되던
                    # 부분. 예전엔 여기서 모션이 끝까지 돌았다는 이유만으로 success=True를
                    # 돌려줘서, 실제로는 아무것도 못 집었는데도 place_into가 그대로 이어져
                    # 빈 그리퍼로 목적지까지 이동하는 사고로 이어졌다.
                    self.get_logger().warning(
                        f"pick 실패: 그리퍼가 빈 채로 닫힘 (width={width_mm:.1f}mm <= "
                        f"min_grip_width_mm={self._min_grip_width_mm:.1f}mm)")
                    store.set_gripper(width_mm=width_mm, closed=True)
                    result = self._result(False, Pick.Result.REASON_GRASP_FAILED, started)
                    self._cache.put(goal.request_id, result)
                    goal_handle.succeed()
                    return result
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

        except _LiftFailedError as e:
            # 그리퍼는 이미 뭔가를 물고 있다 — grasp_failed로 보고하면 "못 집었다"로
            # 읽혀 planner가 같은 물체를 다시 집으려 들 수 있다(실제로는 물체가 붙잡힌
            # 채 공중에 멈춰 있다). robot_state에도 실제 개폭을 반영해 둔다.
            width_display = f"{e.width_mm:.1f}mm" if e.width_mm is not None else "측정 실패"
            self.get_logger().error(
                f"pick 실패(파지는 성공, 들어올리기부터 실패): {e} (width={width_display})")
            # width_mm이 None(측정 실패)이면 0.0은 순전히 화면 표시용 대체값이다 — 아래
            # goal_handle.abort()가 REASON_UNREACHABLE로 보고하는 판정 자체는 width 값과
            # 무관하다(그리퍼 상태를 "불확실"로 다루는 쪽은 orchestrator, 이 값이 아니다).
            store.set_gripper(width_mm=e.width_mm if e.width_mm is not None else 0.0, closed=True)
            store.set_error()
            goal_handle.abort()
            return self._result(False, Pick.Result.REASON_UNREACHABLE, started)
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

    def _open_width_m(self, gripper_width_mm: float) -> float:
        """파지 전 그리퍼를 열 목표 개폭(m). GraspCandidate.gripper_width_mm에
        여유(width_margin_mm)만 더해서 연다 — 전략이 폭을 못 냈으면(0 이하)
        예전처럼 최대로 열어서 안전한 쪽으로 대체한다."""
        if gripper_width_mm <= 0.0:
            return self._gripper_open_m
        max_open_mm = self._gripper_open_m * 1000.0
        target_mm = min(gripper_width_mm + self._gripper_width_margin_mm, max_open_mm)
        return target_mm / 1000.0

    def _pick_real(self, goal_handle, goal) -> float | None:
        """위치제어만으로 실물 pick을 수행한다 (1단계 — compliance.py/visual_verification.py가
        아직 빈 스텁이라 접촉감지·파지확인 없이 grasp_pose를 그대로 믿고 움직인다).

        grasp_pose 바로 위(approach_height_mm)에서 한 번 멈췄다 내려가 그리퍼를 닫고
        다시 들어올린다. 힘(N)은 profile별 max_grip_force_n을 정확히 넣지 못한다 —
        `/onrobot/sendCommand`가 문자 명령이라 서버 기본값(보수적인 축, dsr_motion.py
        참조)을 그대로 쓴다. 실패하면 RuntimeError, 취소되면 None — dsr_motion의 각
        호출이 취소 시 False/None을 돌려주므로 여기서 `goal_handle.is_cancel_requested`로
        두 경우를 구분한다(취소는 오류가 아니다).
        """
        target_posx = dsr_motion.pose_mm_to_posx(goal.grasp_pose)
        target_posx[2] -= self._pick_depth_extra_mm
        approach_posx = list(target_posx)
        approach_posx[2] += self._approach_height_mm
        target_xyz = target_posx[:3]
        approach_xyz = approach_posx[:3]

        def move(pos):
            return dsr_motion.move_linear(self._movel_client, pos, goal_handle,
                                          self._linear_vel_mm_s, self._linear_acc_mm_s2,
                                          self._rot_vel_deg_s, self._rot_acc_deg_s2,
                                          posx_client=self._posx_client, logger=self.get_logger())

        def next_target(xyz, last_pose):
            """`xyz`로 위치만 바꾸고 회전은 유지한다 — 연속 이동에서 계산값을 재사용하면
            안 되는 이유는 dsr_motion.py 모듈 docstring(ZYZ 특이점) 참조.

            회전값은 가능하면 get_current_posx로 정지 후 새로 읽은 값을 쓴다. action
            feedback의 마지막 샘플(`last_pose`)은 스트리밍 도중 값이라, 컨트롤러가 정지
            후 재정규화하는 ZYZ 표현과 파라미터가 다를 수 있다(같은 문서, "물리적으로는
            같은 방향인데 파라미터값이 다를 수 있다" 참조). 그 값을 다음 이동에 그대로
            쓰면 "제자리에서 수직으로" 대신 두 자세 사이를 새로 보간해 대각선/최단거리로
            움직이는 문제가 있었다(2026-09-05 실물 확인).

            movel 직후 aux_control이 10~20초씩 무응답일 수 있다는 걸 이미 알고 있으므로
            (get_current_posx 참조), 여기서 재시도까지 하며 오래 기다리는 건 거의 항상
            헛수고다 — next_target은 정의상 매번 move() 직후, 즉 그 무응답 구간 한복판에서
            불린다. 그래서 짧게 한 번만 찔러보고(재시도 없음), 안 되면 바로 feedback 값으로
            대체한다(2026-09-06, 스텝 전환 지연 조사 — 재시도 포함 최대 2.5초였던 걸
            0.5초로 줄임. pick+place 한 사이클에 이 호출이 6~7번 있어 누적 효과가 크다)."""
            settled = dsr_motion.get_current_posx(self._posx_client, goal_handle,
                                                  timeout_s=0.5, retries=1)
            if settled is not None:
                return [xyz[0], xyz[1], xyz[2], settled[3], settled[4], settled[5]]
            if last_pose is None:
                return None
            return [xyz[0], xyz[1], xyz[2], last_pose[3], last_pose[4], last_pose[5]]

        # 그리퍼를 먼저 연다. 예전엔 여기서 열지 않고 바로 닫기만 했다 — 이전 사이클에서
        # 그리퍼가 닫힌 채 남아 있으면(파지 실패 후, 또는 place_into가 도중에 멈춘 경우)
        # 다음 pick이 이미 닫힌 그리퍼로 "닫기"만 반복해 애초에 아무것도 못 무는 문제가 있었다.
        open_command = dsr_motion.gripper_width_command(self._open_width_m(goal.gripper_width_mm))
        if not dsr_motion.send_gripper_command(self._gripper_cmd_client, open_command):
            if goal_handle.is_cancel_requested:
                return None
            raise RuntimeError("그리퍼 열기 명령 전송 실패")
        if dsr_motion.wait_gripper_settled(lambda: self._gripper_joint_angle, goal_handle) is None:
            if goal_handle.is_cancel_requested:
                return None
            raise RuntimeError("그리퍼가 열리는 동안 응답이 없다")

        self._publish_phase(goal_handle, Pick.Feedback.PHASE_APPROACHING)
        approach_ok, approach_pose = move(approach_posx)
        if not approach_ok:
            if goal_handle.is_cancel_requested:
                return None
            raise RuntimeError("접근 위치로 이동 실패")

        descend_posx = next_target(target_xyz, approach_pose)
        if descend_posx is None:
            if goal_handle.is_cancel_requested:
                return None
            raise RuntimeError("현재 자세를 읽지 못했다")
        descend_ok, descend_pose = move(descend_posx)
        if not descend_ok:
            if goal_handle.is_cancel_requested:
                return None
            raise RuntimeError("파지 위치로 이동 실패")

        self._publish_phase(goal_handle, Pick.Feedback.PHASE_CONTACT_DETECTED)
        if not dsr_motion.send_gripper_command(self._gripper_cmd_client, "c"):
            if goal_handle.is_cancel_requested:
                return None
            raise RuntimeError("그리퍼 닫기 명령 전송 실패")
        final_angle = dsr_motion.wait_gripper_settled(
            lambda: self._gripper_joint_angle, goal_handle)
        if final_angle is None:
            if goal_handle.is_cancel_requested:
                return None
            raise RuntimeError("그리퍼가 닫히는 동안 응답이 없다")

        # 그리퍼가 실제로 닫힌 개폭 — 들어올리기가 이후에 실패해도 "물었는지 여부"는
        # 이미 정해져 있다. 이 뒤의 실패는 grasp_failed가 아니라 별도로 구분해 보고한다.
        width_mm = dsr_motion.gripper_width_mm(self._gripper_pose_client, final_angle)
        if width_mm is None:
            # /onrobot/pose가 응답하지 않았다 — 그리퍼는 이미 닫혔는데(wait_gripper_settled
            # 통과) 실제로 얼마나 닫혔는지를 몰라 "확실히 빈 채로 닫혔다"와 구분할 수
            # 없다. 여기서 0.0으로 대체하면 곧바로 execute_callback의 min_grip_width_mm
            # 비교가 무조건 "빈 그리퍼"로 떨어져, 실제로는 뭔가를 물고 있을 수도 있는
            # 상황을 확정적 실패로 오판한다(2026-09-05, gripper_width_mm 조사에서 확인).
            # 이미 그리퍼를 문 뒤의 실패이므로 _LiftFailedError로 올려 "불확실" 취급이
            # 되게 한다(REASON_UNREACHABLE — orchestrator가 이 경우 자동 재계획 안 함).
            if goal_handle.is_cancel_requested:
                return None
            raise _LiftFailedError("그리퍼 개폭 측정 실패 — 닫기는 완료됐으나 파지 여부 불확실",
                                   width_mm)

        self._publish_phase(goal_handle, Pick.Feedback.PHASE_LIFTING)
        lift_posx = next_target(approach_xyz, descend_pose)
        if lift_posx is None:
            if goal_handle.is_cancel_requested:
                return None
            raise _LiftFailedError("현재 자세를 읽지 못했다", width_mm)
        lift_ok, _ = move(lift_posx)
        if not lift_ok:
            if goal_handle.is_cancel_requested:
                return None
            raise _LiftFailedError("들어올리기 실패", width_mm)

        self._publish_phase(goal_handle, Pick.Feedback.PHASE_VERIFYING)
        self.get_logger().warning(
            "위치제어만으로 pick 완료 — compliance/visual_verification 미구현이라 "
            "실제 파지 여부는 확인되지 않았다")
        return width_mm

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

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
import math
import os
import pathlib
import threading
import time

import rclpy
import yaml
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from dsr_msgs2.action import MovelH2r
from dsr_msgs2.srv import Fkin, GetCurrentPosx, GetCurrentSolutionSpace, Ikin
from onrobot_rg_msgs.msg import OnRobotRGInput
from onrobot_rg_msgs.srv import GripperPose, SetCommand
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from sort_msgs.action import Pick
from sort_msgs.msg import GraspCandidate

from . import dsr_motion, grasp_selection
from .config_paths import skill_params_path
from .request_cache import RequestCache
from .robot_state_publisher import is_fake_robot, store

SCHEMA_VERSION = "1.0.0"

# 실행할 후보로 고른 것을 알리는 토픽. grasp 노드가 구독해 디버그 오버레이에서 그 후보
# 하나만 다른 색으로 그린다 — 화면에 숫자를 늘리지 않고 "무엇이 뽑혔는지"만 보여준다.
# 최신 하나만 의미가 있고 늦게 붙은 구독자도 그것을 받아야 하므로 transient local이다.
SELECTED_GRASP_TOPIC = "/control/selected_grasp"


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


class _NeverCancelled:
    """`wait_gripper_settled`에 넘길 더미 goal_handle — 취소할 goal이 없는 문맥
    (_startup_gripper_reset)에서 쓴다. 그 함수는 이 속성 하나만 읽는다."""
    is_cancel_requested = False


class _Canceled(Exception):
    """내부 전용 — 취소 요청을 헬퍼에서 호출부까지 올리기 위한 신호.

    `_pick_real`은 취소를 None 반환으로 알리는데, 위 중첩 헬퍼는 반환값으로 그걸
    구분할 수 없어(정상 완료도 None) 예외로 올린다. 밖으로 새면 execute_callback의
    포괄 except가 "실패"로 잡아버리므로 반드시 _pick_real 안에서 잡는다.
    """


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
        # 파지 후보 지점보다 **물체 안쪽으로** 더 파고드는 양 (skill_params.yaml 주석 참조).
        # TCP 원점과 손끝의 차이는 여기가 아니라 tool.grasp_center_offset_mm이 담당한다 —
        # 둘을 한 값으로 겸하면 어느 쪽을 튜닝하는지 알 수 없어진다(그 혼동이 40mm/45mm
        # 같은 근거 없는 값들의 출처였다).
        self._pick_depth_extra_mm = float(motion.get("pick_depth_extra_mm", 0.0))
        # TCP to modeled pad/reference (tool mm), not measured physical finger geometry.
        self._grasp_center_offset_mm = dsr_motion.grasp_center_offset_mm(params)
        gripper = params.get("gripper") or {}
        # 이 미만이면 "닫혔지만 아무것도 안 물렸다"로 본다. compliance/visual_verification이
        # 아직 없어, /onrobot/status의 Grip detected가 없을 때는 이게 유일한 파지 확인 수단이다.
        self._min_grip_width_mm = float(gripper.get("min_grip_width_mm", 5.0))
        # GraspCandidate.gripper_width_mm(전략이 예측한 필요 개폭)에 더하는 여유 — 이만큼만
        # 더 열면 접근 중 손가락이 물체를 스치지 않으면서도 다 열 때보다 빨리 닫히고,
        # 클러터에서 옆 물체를 안 건드린다. 모르면(0 이하) 기존처럼 최대로 연다.
        self._gripper_width_margin_mm = float(gripper.get("width_margin_mm", 15.0))
        # 닫을 때 목표 개폭 = 물체폭 x 이 비율. 물체에 실제로 힘이 걸려야 Grip detected가
        # 의미를 갖는다(위 닫기 주석 참조).
        # **고정 mm가 아니라 비율인 이유**: 고정 10mm를 빼면 110mm 물체는 91%(거의 안
        # 조임), 15mm 물체는 33%(과하게 조임)가 되어 물체 크기에 따라 조임 정도가 널뛴다.
        self._grip_close_ratio = float(gripper.get("grip_close_ratio", 0.8))
        # 프로필별 파지력(N). 2026-09-07 이전에는 이 값이 로봇까지 전달될 방법이 없어
        # 선언만 되어 있었다(dsr_motion.gripper_width_command 주석 참조).
        self._profile_force_n = {
            name: float((spec or {}).get("max_grip_force_n", 0.0) or 0.0)
            for name, spec in (params.get("profiles") or {}).items()
        }
        # 열 때 쓰는 힘. 닫기에서 낮은 힘(fragile 5N)을 설정한 뒤 그대로 두면 다음 열기가
        # 그 힘으로 나가므로, 열기에는 항상 넉넉한 힘을 명시해 상태 의존을 없앤다.
        self._gripper_open_force_n = float(gripper.get("open_force_n", 40.0))
        # 후보 선택 파라미터. 개폭 관련 값은 위에서 이미 읽은 실행 기준을 그대로 넘긴다 —
        # 검사 기준과 실행 기준이 갈라지면 "통과시켜 놓고 다르게 동작하는" 후보가 생긴다.
        self._selection = self._selection_params(params)
        # 관절 한계는 SelectionParams가 아니라 IK 질의(dsr_motion.verify_ik)가 쓴다.
        # 설정에 없으면 URDF에서 읽은 M0609 값을 그대로 쓴다.
        limits = (params.get("grasp_selection") or {}).get("joint_limits_deg")
        self._joint_limits_deg = (tuple(float(v) for v in limits)
                                  if isinstance(limits, (list, tuple)) and len(limits) == 6
                                  else dsr_motion.JOINT_LIMITS_DEG)
        self._movel_client = ActionClient(self, MovelH2r, dsr_motion.MOVEL_ACTION,
                                          callback_group=callbacks)
        self._posx_client = self.create_client(
            GetCurrentPosx, dsr_motion.GET_CURRENT_POSX_SERVICE, callback_group=callbacks)
        # 후보의 접근·파지 자세가 실제로 풀리는지 확인한다. place_server가 놓는 높이를
        # 정할 때 쓰던 것과 같은 서비스다(place_server._reachable) — 그 판정 로직은
        # dsr_motion.verify_ik로 합쳐 두 서버가 같은 규칙을 쓴다.
        self._ikin_client = self.create_client(
            Ikin, dsr_motion.IKIN_SERVICE, callback_group=callbacks)
        self._fkin_client = self.create_client(
            Fkin, dsr_motion.FKIN_SERVICE, callback_group=callbacks)
        self._solution_space_client = self.create_client(
            GetCurrentSolutionSpace, dsr_motion.GET_CURRENT_SOLUTION_SPACE_SERVICE,
            callback_group=callbacks)
        self._selected_pub = self.create_publisher(
            GraspCandidate, SELECTED_GRASP_TOPIC,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self._gripper_cmd_client = self.create_client(
            SetCommand, dsr_motion.GRIPPER_COMMAND_SERVICE, callback_group=callbacks)
        self._gripper_pose_client = self.create_client(
            GripperPose, "/onrobot/pose", callback_group=callbacks)
        self._gripper_joint_angle: float | None = None
        self.create_subscription(JointState, dsr_motion.GRIPPER_JOINT_STATES_TOPIC,
                                 self._on_gripper_state, 5, callback_group=callbacks)
        # RG2 컨트롤러의 상태 비트필드. bit1(Grip detected)이 **개폭과 무관하게** 물체를
        # 물었는지 알려준다 — 변형체를 제대로 물어도 0mm까지 닫히는 문제(아래 파지 판정
        # 주석 참조)를 개폭 추정 없이 해결한다. (time, gsta)로 들고 있어 신선도를 본다.
        self._gripper_status: tuple[float, int] | None = None
        self.create_subscription(OnRobotRGInput, dsr_motion.GRIPPER_STATUS_TOPIC,
                                 self._on_gripper_status, 5, callback_group=callbacks)

        self.get_logger().info(f"pick 액션 서버 준비 ({'fake' if is_fake_robot() else '실물'} 모드)")

        # 기동할 때 그리퍼를 알려진 상태(최대 개방)로 되돌린다. 별도 스레드인 이유는
        # 로봇 드라이버가 아직 안 떠 있을 수 있어서다 — 여기서 기다리면 control 노드
        # 전체 기동이 막힌다.
        if not is_fake_robot():
            threading.Thread(target=self._startup_gripper_reset, daemon=True).start()

    def _startup_gripper_reset(self, wait_driver_s: float = 60.0) -> None:
        """control 기동 시 그리퍼를 최대 개방으로 되돌린다.

        **왜 종료가 아니라 시작에 두는가.** 직전 실행이 비정상으로 끝났거나(예외, 컨테이너
        재시작) 사람이 Stop을 눌렀으면 그 실행의 정리 코드는 아예 안 돌았을 수 있다. 정리를
        "끝낼 때"에 걸어두면 그 경우를 못 막는다. 반대로 "시작할 때" 알려진 상태를 만들면
        직전이 어떻게 끝났든 상관없이 같은 상태에서 출발한다 — pick이 매번 그리퍼를 먼저
        여는 것(_pick_real)도 같은 이유이고, 이건 그 범위를 프로세스 기동까지 넓힌 것이다.

        조건 없이 연다. 물체를 물고 있었다면 그 자리에 떨어지는데, 물고 있는지 여부는
        지금 알 수 없다 — RG2의 'Grip detected'(gsta 비트1)를 OnRobotRGControllerServer가
        토픽으로 안 내보내서 개폭만으로는 "물고 있음"과 "빈 채로 중간 개폭에 멈춤"이
        구분되지 않는다. 그 판별과 사용자 알림은 나중에 보완한다(드라이버가 gsta를
        발행하도록 고치는 것이 선결).
        """
        deadline = time.monotonic() + wait_driver_s
        while time.monotonic() < deadline and self._gripper_joint_angle is None:
            time.sleep(0.5)
        if self._gripper_joint_angle is None:
            self.get_logger().warning(
                f"기동 그리퍼 열기 건너뜀 — {wait_driver_s:.0f}초 동안 "
                f"{dsr_motion.GRIPPER_JOINT_STATES_TOPIC}가 오지 않았다(로봇 드라이버 미기동?). "
                "그리퍼는 직전 상태 그대로다.")
            return

        command = dsr_motion.gripper_width_command(
            self._gripper_open_m, self._gripper_open_force_n)
        if not dsr_motion.send_gripper_command(self._gripper_cmd_client, command):
            self.get_logger().warning(
                "기동 그리퍼 열기 실패 — 명령을 못 보냈다(그리퍼 드라이버 미기동?). "
                "그리퍼는 직전 상태 그대로다.")
            return
        # wait_gripper_settled는 취소를 볼 goal_handle을 요구한다 — 기동 시점에는 취소할
        # goal이 없으므로 "취소 안 됨"만 답하는 더미를 넘긴다.
        if dsr_motion.wait_gripper_settled(
                lambda: self._gripper_joint_angle, _NeverCancelled()) is None:
            self.get_logger().warning("기동 그리퍼 열기: 여는 동안 응답이 없다(타임아웃)")
            return
        self.get_logger().info(
            f"기동 그리퍼 열기 완료 — 최대 개방({self._gripper_open_m * 1000:.0f}mm)")

    def _on_gripper_status(self, msg: OnRobotRGInput) -> None:
        self._gripper_status = (time.monotonic(), int(msg.gsta))

    def _grip_detected(self, max_age_s: float = 2.0) -> bool | None:
        """RG2가 보고하는 'Grip detected'(gsta bit1). 모르면 None.

        None을 돌려주는 경우를 호출부가 반드시 구분해야 한다 — 드라이버가 이 토픽을
        발행하지 않는 구버전이거나(2026-09-07 이전) 값이 낡았으면 "안 물었다"가 아니라
        "모른다"이고, 그때는 예전 개폭 판정으로 떨어져야 한다.
        """
        if self._gripper_status is None:
            return None
        stamp, gsta = self._gripper_status
        if time.monotonic() - stamp > max_age_s:
            return None
        return bool(gsta & 0x02)

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
            f"request={goal.request_id} 후보={len(goal.grasp_candidates) or 1}개")
        store.set_busy("pick")
        started = time.monotonic()
        # 아래 except 절들이 읽는다 — _select_candidate 자체가 예외를 내면 아직 미할당이다.
        selected = None

        try:
            # **움직이기 전에 후보를 고른다.** 여기서 전부 탈락하면 로봇은 한 번도 움직이지
            # 않았으므로 그리퍼도 팔도 그대로다 — 모션 도중 실패(unreachable/grasp_failed)와
            # 구분해서 보고해야 orchestrator가 그리퍼 상태를 의심할 필요가 없다.
            selected, evaluations, wrist_pose = self._select_candidate(goal, goal_handle)
            if selected is None:
                reasons = "; ".join(
                    f"{e.candidate.candidate_id or '#%d' % e.candidate.rank}={e.status}"
                    f"({e.rejection_reason})" for e in evaluations) or "후보 없음"
                self.get_logger().warning(
                    f"pick 중단: 실행 가능한 파지 후보가 없다 — 로봇을 움직이지 않았다. {reasons}")
                result = self._result(False, Pick.Result.REASON_NO_FEASIBLE_GRASP, started)
                self._cache.put(goal.request_id, result)
                goal_handle.succeed()
                return result

            if is_fake_robot():
                for phase in PHASES:
                    if goal_handle.is_cancel_requested:
                        goal_handle.canceled()
                        return self._result(False, Pick.Result.REASON_NO_CONTACT, started)
                    self._publish_phase(goal_handle, phase)
                    time.sleep(FAKE_PHASE_DURATION_S)
                width_mm, visual_passed, torque = 42.0, True, [0.4, 1.9, 2.6, 2.4]
            else:
                pick_result = self._pick_real(goal_handle, goal, selected, wrist_pose)
                if pick_result is None:          # 취소 (_pick_real의 모든 취소 경로가 None)
                    goal_handle.canceled()
                    return self._result(False, Pick.Result.REASON_NO_CONTACT, started)
                width_mm, close_target_mm = pick_result
                # --- 파지 성공 판정 ---------------------------------------
                # **1순위: RG2 컨트롤러의 Grip detected 비트.** 개폭과 무관하게 물었는지를
                # 알려주는 확정 신호다. 드라이버가 /onrobot/status를 발행할 때만 쓸 수 있다
                # (2026-09-07 추가 — 그 이전 빌드로 돌고 있으면 None이 온다).
                #
                # **2순위(폴백): 개폭 추정.** "끝까지 닫혔으면 빈 것"이라는 가정인데
                # **변형체에서 정반대로 틀린다** — 치약처럼 눌리는 물체는 제대로 물어도
                # 0mm까지 닫힌다(2026-09-07 실물: depth로 쥐고 있음을 확인했는데 개폭은
                # 0.0mm였다). 그 오판정이 재계획 → 새 pick → 그리퍼 열기로 이어져 쥐고
                # 있던 물체를 그 자리에 떨어뜨리는 사고까지 갔다. 확정 신호가 없을 때만 쓴다.
                grip = self._grip_detected()
                if grip is not None and close_target_mm <= 0.0:
                    # 완전 닫기로 닫은 경우다 — 빈 그리퍼도 grip=True가 되므로 이 신호를
                    # 쓰면 안 된다. 개폭 판정으로 떨어뜨린다(그것도 변형체에 약하지만,
                    # 확실히 틀리는 신호보다는 낫다).
                    grip = None
                if grip is None:
                    grasped = width_mm > self._min_grip_width_mm
                    basis = (f"개폭 추정 (width={width_mm:.1f}mm vs "
                             f"min_grip_width_mm={self._min_grip_width_mm:.1f}mm) "
                             "— /onrobot/status 미수신, 드라이버 재기동 필요")
                else:
                    grasped = grip
                    basis = f"Grip detected={grip} (width={width_mm:.1f}mm)"
                self.get_logger().info(f"파지 판정: {'성공' if grasped else '실패'} | {basis}")
                if not grasped:
                    # 실제로는 아무것도 못 집었는데 success를 돌려주면 place_into가 그대로
                    # 이어져 빈 그리퍼로 목적지까지 가는 사고가 된다.
                    self.get_logger().warning(f"pick 실패: 물체를 물지 못했다 | {basis}")
                    store.set_gripper(width_mm=width_mm, closed=True)
                    result = self._result(False, Pick.Result.REASON_GRASP_FAILED, started,
                                          candidate_id=selected.candidate.candidate_id)
                    self._cache.put(goal.request_id, result)
                    goal_handle.succeed()
                    return result
                visual_passed, torque = False, []

            if self._injected_failure(goal.object_id):
                self.get_logger().warning(
                    f"실패 주입 (FAKE_FAIL_OBJECT={goal.object_id}) — grasp_failed 반환")
                result = self._result(False, Pick.Result.REASON_GRASP_FAILED, started,
                                      candidate_id=selected.candidate.candidate_id)
                self._cache.put(goal.request_id, result)
                goal_handle.succeed()
                return result

            store.set_gripper(width_mm=width_mm, closed=True)
            result = self._result(True, Pick.Result.REASON_NONE, started,
                                  visual_passed=visual_passed, torque=torque,
                                  candidate_id=selected.candidate.candidate_id)
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
            return self._result(False, Pick.Result.REASON_UNREACHABLE, started,
                                candidate_id=self._selected_id(selected))
        except Exception as e:
            self.get_logger().error(f"pick 실패: {e}")
            store.set_error()
            goal_handle.abort()
            return self._result(False, Pick.Result.REASON_GRASP_FAILED, started,
                                candidate_id=self._selected_id(selected))
        finally:
            if store.snapshot()["mode"] != "error":
                store.set_idle()

    @staticmethod
    def _selected_id(selected) -> str:
        """예외 처리에서 쓰는 안전한 candidate_id — 아직 못 골랐으면 빈 문자열."""
        return selected.candidate.candidate_id if selected is not None else ""

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

    def _selection_params(self, params: dict) -> grasp_selection.SelectionParams:
        """skill_params.yaml의 `grasp_selection` 블록 → SelectionParams.

        개폭 기준 세 값은 설정에서 다시 읽지 않고 **실행이 쓰는 값을 그대로 넘긴다** —
        같은 뜻의 값이 두 군데서 따로 읽히면 한쪽만 바꿨을 때 조용히 갈라진다.
        """
        spec = params.get("grasp_selection") or {}
        defaults = grasp_selection.SelectionParams()
        weights = dict(defaults.weights)
        for name, value in (spec.get("weights") or {}).items():
            if name not in weights:
                self.get_logger().warning(
                    f"grasp_selection.weights에 모르는 항 '{name}' — 무시한다 "
                    f"(쓸 수 있는 항: {', '.join(sorted(weights))})")
                continue
            weights[name] = float(value)
        floor = spec.get("min_tcp_z_mm", defaults.min_tcp_z_mm)
        return grasp_selection.SelectionParams(
            max_opening_mm=self._gripper_open_m * 1000.0,
            min_grip_width_mm=self._min_grip_width_mm,
            width_margin_mm=self._gripper_width_margin_mm,
            support_tolerance_mm=float(
                spec.get("support_tolerance_mm", defaults.support_tolerance_mm)),
            min_tcp_z_mm=None if floor is None else float(floor),
            max_evaluated=int(spec.get("max_evaluated", defaults.max_evaluated)),
            comfort_min_width_mm=float(
                spec.get("comfort_min_width_mm", defaults.comfort_min_width_mm)),
            center_scale_floor_mm=float(
                spec.get("center_scale_floor_mm", defaults.center_scale_floor_mm)),
            joint_margin_ref_deg=float(
                spec.get("joint_margin_ref_deg", defaults.joint_margin_ref_deg)),
            weights=weights,
        )

    @staticmethod
    def _candidates_from_goal(goal) -> list:
        """goal → 검사할 후보 목록(점수 내림차순, planner가 준 순서 그대로).

        `grasp_candidates`가 비어 있으면 `grasp_pose` 하나로 만든다 — 예전 planner나
        goal을 직접 쏘는 도구가 그대로 동작해야 한다. 그 경우에도 개폭·IK·안전 검사는
        똑같이 지나간다(그게 이 작업이 막으려는 실패 모드다).
        """
        raw = list(getattr(goal, "grasp_candidates", None) or [])
        if not raw:
            return [grasp_selection.Candidate(
                candidate_id="", rank=0, pose=goal.grasp_pose,
                # 후보가 하나뿐이라 점수는 순위에 영향이 없다.
                score=1.0, gripper_width_mm=float(goal.gripper_width_mm))]
        return [grasp_selection.Candidate(
            candidate_id=candidate.candidate_id, rank=rank, pose=candidate.pose,
            score=float(candidate.score),
            gripper_width_mm=float(candidate.gripper_width_mm),
            strategy=candidate.strategy)
            for rank, candidate in enumerate(raw)]

    @staticmethod
    def _object_context(goal) -> grasp_selection.ObjectContext:
        """goal에 실려 온 물체 정보 → 랭킹용 컨텍스트.

        중심 좌표가 정확히 (0,0,0)이면 "안 실렸다"로 본다 — base 원점은 로봇 바닥이라
        거기에 물체가 있을 수 없고, 예전 planner가 보낸 goal에는 이 필드가 아예 없다.
        """
        center = getattr(goal, "object_center_mm", None)
        center_mm = None
        if center is not None and any(abs(float(v)) > 1e-6
                                      for v in (center.x, center.y, center.z)):
            center_mm = (float(center.x), float(center.y), float(center.z))
        return grasp_selection.ObjectContext(
            object_id=goal.object_id,
            center_mm=center_mm,
            height_mm=float(getattr(goal, "object_height_mm", 0.0) or 0.0),
            depth_valid_ratio=float(getattr(goal, "depth_valid_ratio", 0.0) or 0.0),
        )

    def _select_candidate(self, goal, goal_handle):
        """실행할 후보 하나를 고른다. **로봇은 전혀 움직이지 않는다.**

        `(선택된 Evaluation 또는 None, 평가 목록, 읽어둔 손목 자세)`를 돌려준다.
        손목 자세를 함께 돌려주는 이유는 `_pick_real`이 같은 값을 다시 읽지 않게 하기
        위해서다 — movel 직후 get_current_posx가 10~20초 무응답인 구간이 있어
        (dsr_motion.get_current_posx) 같은 값을 두 번 묻는 것은 그냥 손해다.
        """
        candidates = self._candidates_from_goal(goal)
        obj = self._object_context(goal)
        fake = is_fake_robot()
        wrist_pose = None if fake else dsr_motion.get_current_posx(
            self._posx_client, goal_handle, timeout_s=1.0, retries=2)
        current_zyz = wrist_pose[3:] if wrist_pose else None
        # ikin은 solution space마다 다른 해를 낸다. movel은 현재 space를 유지하므로
        # "지금 상태에서 갈 수 있는가"를 물으려면 현재 space로 물어야 한다.
        sol_space = 2 if fake else dsr_motion.current_solution_space(
            self._solution_space_client)

        def geometry_of(pose):
            target_posx, approach_posx, axis = dsr_motion.plan_pick_posx(
                pose, self._grasp_center_offset_mm, self._approach_height_mm,
                self._pick_depth_extra_mm, current_zyz_deg=current_zyz)
            return grasp_selection.PickGeometry(
                target_posx=target_posx, approach_posx=approach_posx, approach_axis=axis,
                pad_reference_mm=dsr_motion.grasp_center_from_posx(
                    target_posx, self._grasp_center_offset_mm))

        def ik_verdict_of(posx):
            return dsr_motion.verify_ik(self._ikin_client, self._fkin_client, posx,
                                        sol_space=sol_space, limits=self._joint_limits_deg)

        evaluations = grasp_selection.evaluate_candidates(
            candidates, obj, self._selection, geometry_of, ik_verdict_of)
        selected = grasp_selection.select(evaluations)

        # 후보별 판정은 **로그에만** 남긴다(웹에는 이 숫자들을 내보내지 않는다).
        for evaluation in evaluations:
            self.get_logger().info(
                f"[후보검사] object={goal.object_id} {grasp_selection.log_line(evaluation)}")
        tally = grasp_selection.counts(evaluations)
        unknown_ik = sum(1 for e in evaluations
                         if e.approach_ik is None and e.status != grasp_selection.STATUS_WIDTH_INVALID)
        self.get_logger().info(
            f"[후보선택] object={goal.object_id} request_id={goal.request_id} "
            f"입력={len(candidates)} 검사={len(evaluations)} "
            f"(상한 max_evaluated={self._selection.max_evaluated}) "
            f"width탈락={tally[grasp_selection.STATUS_WIDTH_INVALID]} "
            f"ik탈락={tally[grasp_selection.STATUS_IK_FAILED]} "
            f"joint탈락={tally[grasp_selection.STATUS_JOINT_LIMIT]} "
            f"safety탈락={tally[grasp_selection.STATUS_SAFETY_INVALID]} "
            f"valid={tally[grasp_selection.STATUS_VALID] + tally[grasp_selection.STATUS_SELECTED]} "
            + (f"selected={selected.candidate.candidate_id or '(단일 grasp_pose)'} "
               f"total={selected.total_score:.3f}" if selected else "selected=없음"))
        if unknown_ik:
            self.get_logger().warning(
                f"[후보선택] IK 확인 실패 {unknown_ik}건 — ikin/fkin 서비스가 응답하지 않아 "
                "도달 가능성을 검증하지 못한 채 통과시켰다(로봇 드라이버 미기동?)")
        self._publish_selected(selected)
        return selected, evaluations, wrist_pose

    def _publish_selected(self, selected) -> None:
        """고른 후보를 발행한다. 못 골랐으면 빈 candidate_id로 발행해 이전 선택을 지운다 —
        안 그러면 grasp 오버레이가 지난 사이클의 후보를 계속 강조한다."""
        msg = GraspCandidate()
        if selected is not None:
            msg.pose = selected.candidate.pose
            msg.score = float(selected.candidate.score)
            msg.gripper_width_mm = float(selected.candidate.gripper_width_mm)
            msg.candidate_id = selected.candidate.candidate_id
            msg.strategy = selected.candidate.strategy
        self._selected_pub.publish(msg)

    def _pick_real(self, goal_handle, goal, selected, wrist_pose) -> tuple[float, float] | None:
        """위치제어만으로 실물 pick을 수행한다 (1단계 — compliance.py/visual_verification.py가
        아직 빈 스텁이라 접촉감지·파지확인 없이 grasp_pose를 그대로 믿고 움직인다).

        grasp_pose 바로 위(approach_height_mm)에서 한 번 멈췄다 내려가 그리퍼를 닫고
        다시 들어올린다. 힘(N)은 profile별 max_grip_force_n을 정확히 넣지 못한다 —
        `/onrobot/sendCommand`가 문자 명령이라 서버 기본값(보수적인 축, dsr_motion.py
        참조)을 그대로 쓴다. 실패하면 RuntimeError, 취소되면 None — dsr_motion의 각
        호출이 취소 시 False/None을 돌려주므로 여기서 `goal_handle.is_cancel_requested`로
        두 경우를 구분한다(취소는 오류가 아니다).
        """
        # **후보 선택(_select_candidate)이 이미 정한 것을 그대로 실행한다.** 여기서 좌표를
        # 다시 계산하면 검사한 자세와 실행하는 자세가 갈라져, IK를 통과시킨 근거가
        # 사라진다. 손목 자세(wrist_pose)도 그때 읽은 것을 넘겨받는다.
        #
        # 그 계산이 하는 일(dsr_motion.plan_pick_posx):
        # - grasp_pose targets a modeled pad/reference, not the TCP or a measured finger.
        #   Rotate the configured offset with the selected wrist orientation before subtracting.
        # - **같은 파지의 두 표현 중 손목에 가까운 쪽을 고른다.** 평행 그리퍼는 접근축 둘레
        #   180도 뒤집어도 물리적으로 같은 파지인데(접근축 동일, 닫힘축만 부호 반대), 전략이
        #   내는 닫힘축 부호는 임의다. 먼 쪽을 그대로 명령해 손목이 172도를 헛돌고 접근에만
        #   14초를 쓴 적이 있다(2026-09-07 실물, obj_130 — 뒤집었으면 7.6도면 됐다).
        #   현재 자세를 못 읽으면 원래 자세를 그대로 쓴다(모르면 바꾸지 않는다).
        grasp_pose = selected.candidate.pose
        candidate_width_mm = float(selected.candidate.gripper_width_mm)
        geometry = selected.geometry
        grasp_point = [float(grasp_pose.position.x),
                       float(grasp_pose.position.y),
                       float(grasp_pose.position.z)]
        target_posx = list(geometry.target_posx)
        # Exact rotated offset used by the command, including nonzero tool X/Y.
        # target_posx에는 하강분(pick_depth_extra_mm)이 이미 들어 있으므로 빼고 견준다.
        rotated_offset = [grasp_point[i]
                          - (target_posx[i] - self._pick_depth_extra_mm * geometry.approach_axis[i])
                          for i in range(3)]

        # **접근 여유와 하강 깊이는 base Z가 아니라 파지 자세의 접근축을 따라 준다.**
        #
        # 2026-09-07 실물 조사: GraspNet 후보는 수직에서 37도 기울어져 있었는데, 예전
        # 코드는 두 오프셋을 `posx[2] ± 값`으로 **base Z에만** 더했다. 기울어진 자세에서
        # 그렇게 하면 오프셋이 접근축을 벗어나 수평 성분이 생긴다 — 실측 후보
        # (접근축 [0.193, 0.570, -0.798]) 기준으로:
        #   하강 40mm : 올바름 (+7.7, +22.8, -31.9) vs 예전 (0, 0, -40)
        #               → 끝점이 수평으로 24.1mm 어긋남 (거의 전부 -Y)
        #   접근 80mm : 올바름 (-15.4, -45.6, +63.9) vs 예전 (0, 0, +80)
        #               → 시작점이 수평으로 48.2mm 어긋남 (거의 전부 +Y)
        # 그래서 그리퍼가 물체 옆(+Y)에서 수직으로 내려와 반대쪽(-Y)으로 지나가며
        # 물체를 밀어냈다. "중심이 아니라 가장자리로 내려가며 밀어내는" 증상의 정체다.
        #
        # heuristic_pca는 항상 수직 파지(접근축 = (0,0,-1))라 이 버그가 드러나지 않았다 —
        # 그 경우 아래 계산은 예전 식과 **정확히 같은 값**이 된다(접근축 z 성분만 남는다).
        # 즉 이 수정은 PCA 동작을 바꾸지 않고 기울어진 파지만 바로잡는다.
        approach_axis = list(geometry.approach_axis)
        target_xyz = target_posx[:3]
        approach_posx = list(geometry.approach_posx)
        approach_xyz = approach_posx[:3]

        tilt_deg = math.degrees(math.acos(min(1.0, abs(approach_axis[2]))))
        # Model prediction only; extra deliberately shifts it from the fitted pose.
        reference_xyz = dsr_motion.grasp_center_from_posx(
            [*target_xyz, *target_posx[3:]], self._grasp_center_offset_mm)
        offset = self._grasp_center_offset_mm
        self.get_logger().info(
            f"pick 좌표 계산 request_id={goal.request_id} object={goal.object_id} "
            f"candidate={selected.candidate.candidate_id or '(단일 grasp_pose)'} "
            f"fitted_pose_xyz_mm=({grasp_point[0]:.3f}, {grasp_point[1]:.3f}, {grasp_point[2]:.3f}) "
            f"fitted_pose_quaternion_xyzw=({grasp_pose.orientation.x:.6f}, "
            f"{grasp_pose.orientation.y:.6f}, {grasp_pose.orientation.z:.6f}, "
            f"{grasp_pose.orientation.w:.6f}) "
            f"| 파지점(base,mm)=({grasp_point[0]:.1f}, {grasp_point[1]:.1f}, {grasp_point[2]:.1f})"
            f" ZYZ=({target_posx[3]:.1f}, {target_posx[4]:.1f}, {target_posx[5]:.1f})deg"
            f" | 접근축(base)=({approach_axis[0]:.3f}, {approach_axis[1]:.3f}, {approach_axis[2]:.3f})"
            f" 수직대비 {tilt_deg:.1f}deg"
            f" | modeled_reference_offset_tool_mm=({offset[0]:.3f}, {offset[1]:.3f}, {offset[2]:.3f})"
            + (f" | 손목회전 {dsr_motion.rotation_diff_deg(wrist_pose[3:], target_posx[3:]):.1f}deg"
               if wrist_pose else " | 손목회전 미상")
            + f" | 하강 {self._pick_depth_extra_mm:.1f}mm + 접근 {self._approach_height_mm:.1f}mm"
            " (둘 다 접근축 기준)"
            f" | 명령 TCP=({target_xyz[0]:.1f}, {target_xyz[1]:.1f}, {target_xyz[2]:.1f})"
            f" | predicted_modeled_reference_xyz_mm=({reference_xyz[0]:.3f}, {reference_xyz[1]:.3f}, {reference_xyz[2]:.3f})"
            f" | 접근 TCP=({approach_xyz[0]:.1f}, {approach_xyz[1]:.1f}, {approach_xyz[2]:.1f})")
        self.get_logger().warning(
            f"pick reference model request_id={goal.request_id} object={goal.object_id}: "
            "configured offset is not independently validated physical finger geometry; "
            "5.3mm calibration was empirical: 89.1-(81.8+2.0)=5.3mm. "
            "RG2 opening changes reach; reconstructed pad/reference is not actual physical finger. "
            f"planned_open_width_mm={self._open_width_m(candidate_width_mm) * 1000.0:.3f}")

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
        #
        # **최대로 한 번 열었다가 목표 폭으로 좁힌다**(2026-09-06 요청). 목표 폭으로
        # 곧바로 가지 않는 이유는 출발 개폭이 매번 다르기 때문이다 — 직전 사이클이
        # 어디서 끝났느냐에 따라 목표 폭보다 좁은 데서 벌리며 올 수도, 넓은 데서 좁히며
        # 올 수도 있다. 항상 최대에서 좁히는 방향으로 통일하면 도달 개폭이 방향에 따라
        # 달라지지 않고(기구 유격은 접근 방향에 따라 남는 쪽이 다르다), 매 pick이 같은
        # 알려진 상태에서 출발한다.
        def open_gripper_to(width_m: float, what: str) -> None:
            command = dsr_motion.gripper_width_command(
                width_m, self._gripper_open_force_n)
            if not dsr_motion.send_gripper_command(self._gripper_cmd_client, command):
                if goal_handle.is_cancel_requested:
                    raise _Canceled
                raise RuntimeError(f"그리퍼 {what} 명령 전송 실패")
            if dsr_motion.wait_gripper_settled(
                    lambda: self._gripper_joint_angle, goal_handle) is None:
                if goal_handle.is_cancel_requested:
                    raise _Canceled
                raise RuntimeError(f"그리퍼가 {what} 동안 응답이 없다")

        open_width_m = self._open_width_m(candidate_width_mm)
        try:
            open_gripper_to(self._gripper_open_m, "최대로 여는")
            # 목표가 이미 최대면 같은 명령을 두 번 보낼 뿐이라 건너뛴다.
            if open_width_m < self._gripper_open_m:
                self.get_logger().info(
                    f"그리퍼 최대 개방({self._gripper_open_m * 1000:.0f}mm) 후 "
                    f"목표 개폭 {open_width_m * 1000:.1f}mm로 좁힘 "
                    f"(물체폭 {candidate_width_mm:.1f} + 여유 {self._gripper_width_margin_mm:.1f}mm)")
                open_gripper_to(open_width_m, "목표 개폭으로 좁히는")
            elif candidate_width_mm <= 0.0:
                self.get_logger().info(
                    f"그리퍼 최대 개방({self._gripper_open_m * 1000:.0f}mm) — "
                    "파지 후보가 물체폭을 못 내서 최대 개폭을 그대로 쓴다")
            else:
                # 폭은 왔지만 여유를 더하면 최대 개폭을 넘는 경우다. 위 메시지("폭을 못 냈다")를
                # 그대로 쓰면 폭 전달이 끊긴 것으로 오해한다 — 실제로 그렇게 오해한 적이 있다
                # (2026-09-07, GraspNet 후보 폭 86.4mm + 여유 30mm = 116.4mm > 110mm).
                self.get_logger().info(
                    f"그리퍼 최대 개방({self._gripper_open_m * 1000:.0f}mm) — 물체폭 "
                    f"{candidate_width_mm:.1f} + 여유 {self._gripper_width_margin_mm:.1f}mm가 "
                    "최대 개폭을 넘어 최대로 연다")
        except _Canceled:
            return None

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
        predicted_reference = dsr_motion.grasp_center_from_posx(
            descend_posx, self._grasp_center_offset_mm)
        self.get_logger().info(
            f"pick depth equation request_id={goal.request_id} object={goal.object_id} units=mm "
            "final_commanded_tcp_z = fitted_pose_z - rotated_offset_base_z + extra_mm * approach_axis_z: "
            f"{descend_posx[2]:.6f} = {grasp_point[2]:.6f} - ({rotated_offset[2]:.6f}) "
            f"+ {self._pick_depth_extra_mm:.6f} * ({approach_axis[2]:.6f}) "
            f"| rotated_offset_base_xyz=({rotated_offset[0]:.6f}, {rotated_offset[1]:.6f}, {rotated_offset[2]:.6f}) "
            f"| final_commanded_tcp_z={descend_posx[2]:.6f} "
            f"predicted_modeled_reference_z={predicted_reference[2]:.6f} "
            f"= {descend_posx[2]:.6f} + ({predicted_reference[2] - descend_posx[2]:.6f}) "
            "(final descent orientation; modeled pad/reference, not physical finger)")
        descend_ok, descend_pose = move(descend_posx)

        # Log measured TCP before failure exit; small error does not validate support clearance.
        reached = dsr_motion.get_current_posx(self._posx_client, goal_handle,
                                              timeout_s=1.0, retries=2)
        if reached is None:
            reached = descend_pose
            source = "feedback"
        else:
            source = "get_current_posx"
        if reached is not None:
            reference = dsr_motion.grasp_center_from_posx(reached, self._grasp_center_offset_mm)
            self.get_logger().info(
                f"[모션오차] request_id={goal.request_id} object={goal.object_id} descent_ok={descend_ok} "
                f"final_commanded_tcp_z={descend_posx[2]:.6f} actual_tcp_z={reached[2]:.6f} "
                f"dz={reached[2] - descend_posx[2]:+.6f} units=mm "
                f"명령TCP=({descend_posx[0]:.1f}, {descend_posx[1]:.1f}, {descend_posx[2]:.1f}) "
                f"실제TCP=({reached[0]:.1f}, {reached[1]:.1f}, {reached[2]:.1f}) [{source}] "
                f"d=({reached[0] - descend_posx[0]:+.1f}, {reached[1] - descend_posx[1]:+.1f}, "
                f"{reached[2] - descend_posx[2]:+.1f})mm")
            self.get_logger().info(
                f"[modeled reference residual] request_id={goal.request_id} object={goal.object_id} "
                f"파지점=({grasp_point[0]:.1f}, {grasp_point[1]:.1f}, {grasp_point[2]:.1f}) "
                f"reconstructed_pad_reference_xyz_mm=({reference[0]:.1f}, {reference[1]:.1f}, {reference[2]:.1f}) "
                "(not actual physical finger) "
                f"d=({reference[0] - grasp_point[0]:+.1f}, {reference[1] - grasp_point[1]:+.1f}, "
                f"{reference[2] - grasp_point[2]:+.1f})mm "
                f"(하강 {self._pick_depth_extra_mm:.1f}mm 포함) "
                f"| 물체폭={candidate_width_mm:.1f}mm")
        else:
            self.get_logger().warning(
                f"[모션오차] request_id={goal.request_id} object={goal.object_id} descent_ok={descend_ok} "
                f"final_commanded_tcp_z={descend_posx[2]:.6f} actual_tcp_z=unavailable dz=unavailable "
                "units=mm source=unavailable")

        if not descend_ok:
            if goal_handle.is_cancel_requested:
                return None
            raise RuntimeError("파지 위치로 이동 실패")

        self._publish_phase(goal_handle, Pick.Feedback.PHASE_CONTACT_DETECTED)

        # **목표 개폭으로 닫는다 — 'c'(완전 닫기)로 닫으면 파지 판정이 성립하지 않는다.**
        # 'c'는 손가락이 맞닿을 때까지 민다. 그러면 빈 그리퍼에서도 손가락끼리 부딪혀
        # 힘 한계에 도달하고, 컨트롤러는 그 저항을 파지로 읽는다 — 실제로 2026-09-07에
        # 아무것도 안 잡았는데 `Grip detected=True (width=0.0mm)`가 나왔다.
        #
        # 물체 폭보다 조금 좁은 값을 목표로 주면 판정이 성립한다:
        #   물체 있음 → 물체에서 막혀 힘이 걸림 → grip=True
        #   물체 없음 → 목표 개폭까지 저항 없이 닫힘 → grip=False
        # **폭 미상('c' 완전 닫기) 경로는 여기까지 오지 않는다.** 후보 검사
        # (grasp_selection.check_width)가 width<=0을 실행 전에 탈락시키고 다음 후보로
        # 넘어가기 때문이다 — 그 경로가 바로 "빈 그리퍼인데 Grip detected=True"로
        # 성공 판정이 나던 통로였다(2026-09-07 실물). 여기 도달했다면 검사와 실행이
        # 갈라진 것이므로, 조용히 'c'로 떨어지지 말고 소리내어 실패해야 한다.
        if candidate_width_mm <= 0.0:
            raise RuntimeError(
                f"개폭 미상 후보가 실행까지 왔다 (candidate="
                f"{selected.candidate.candidate_id or '(단일 grasp_pose)'}) — "
                "grasp_selection.check_width와 실행 경로가 어긋났다")
        close_target_mm = max(candidate_width_mm * self._grip_close_ratio,
                              self._min_grip_width_mm)
        force_n = self._profile_force_n.get(goal.profile) or None
        close_command = dsr_motion.gripper_width_command(
            close_target_mm / 1000.0, force_n)
        self.get_logger().info(
            f"그리퍼 닫기: 목표 {close_target_mm:.1f}mm "
            f"(물체폭 {candidate_width_mm:.1f}mm x {self._grip_close_ratio:.2f}) "
            + (f"힘 {force_n:.1f}N (profile={goal.profile})" if force_n
               else f"힘 미지정 (profile={goal.profile}에 max_grip_force_n 없음)"))
        if not dsr_motion.send_gripper_command(self._gripper_cmd_client, close_command):
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
        # **반드시 2-튜플로 돌려준다.** execute_callback이
        # `width_mm, close_target_mm = pick_result`로 푼다 — close_target_mm은 파지 판정에
        # 필요하다('c'로 완전히 닫은 경우 Grip detected 비트를 믿으면 안 되는데, 그 구분이
        # close_target_mm <= 0 이다).
        # 2026-09-07 실물: 여기서 float 하나만 돌려주고 있어 **모든 실물 pick이** 닫기 직후
        # `cannot unpack non-iterable float object`로 실패했다. 호출부만 2개로 바꾸고
        # 반환문을 안 고친 것이었고, 타입 힌트(tuple[float, float])도 이미 2개였다.
        return width_mm, close_target_mm

    @staticmethod
    def _injected_failure(object_id: str) -> bool:
        return bool(is_fake_robot() and os.environ.get("FAKE_FAIL_OBJECT") == object_id)

    def _result(self, success, reason, started, visual_passed=False, torque=None,
                candidate_id: str = ""):
        result = Pick.Result()
        result.success = success
        result.failure_reason = reason
        # 어느 후보를 실제로 실행했는지. 못 고른 경우와 grasp_pose 하나만 받은 경우는 빈 값이다.
        result.selected_candidate_id = candidate_id
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

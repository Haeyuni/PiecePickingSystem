"""pick 액션 서버. 실물 성공 판정은 RG2 'Grip detected' 비트로 한다(아래 _pick_real 주석
참조). compliance.py는 진단용 torque_trace_summary만 채운다 — 판정에는 안 쓴다.

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

from dsr_msgs2.action import MovejH2r, MovelH2r
from dsr_msgs2.srv import Fkin, GetCurrentPosx, GetCurrentSolutionSpace, GetExternalTorque, Ikin
from onrobot_rg_msgs.msg import OnRobotRGInput
from onrobot_rg_msgs.srv import GripperPose, SetCommand
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from sort_msgs.action import Pick
from sort_msgs.msg import SelectedGrasp

from . import compliance, dsr_motion, grasp_selection, motion_feasibility
from .config_paths import skill_params_path
from .request_cache import RequestCache
from .robot_state_publisher import is_fake_robot, store

SCHEMA_VERSION = "1.0.0"

# 실행할 후보로 고른 것을 알리는 토픽. grasp 노드가 구독해 디버그 오버레이에서 그 후보
# 하나만 다른 색으로 그린다 — 화면에 숫자를 늘리지 않고 "무엇이 뽑혔는지"만 보여준다.
# 최신 하나만 의미가 있고 늦게 붙은 구독자도 그것을 받아야 하므로 transient local이다.
SELECTED_GRASP_TOPIC = "/control/selected_grasp"


def _posx_key(posx) -> tuple:
    """IK 결과를 기하(posx)별로 기억할 때 쓰는 키. 0.1mm/0.1도까지만 본다 — 같은 값을
    실수 비교로 놓치지 않으면서 서로 다른 자세를 뭉치지도 않는 해상도다."""
    return tuple(round(float(v), 1) for v in list(posx)[:6])


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
        # 이 미만이면 "닫혔지만 아무것도 안 물렸다"로 본다. /onrobot/status의 Grip
        # detected가 없을 때는 이게 유일한 파지 확인 수단이다.
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
        # grip_level별 파지력(N). 2026-09-07 이전에는 이 값이 로봇까지 전달될 방법이 없어
        # 선언만 되어 있었다(dsr_motion.gripper_width_command 주석 참조).
        # 단계 1..5 → 힘 40..20N. 없는/0 단계는 None으로 남겨 _close_grip에서 5로 처리한다.
        self._grip_level_force_n = {
            int(level): float((spec or {}).get("max_grip_force_n", 0.0) or 0.0)
            for level, spec in (params.get("grip_levels") or {}).items()
        }
        # 열 때 쓰는 힘. 닫기에서 낮은 힘(fragile 5N)을 설정한 뒤 그대로 두면 다음 열기가
        # 그 힘으로 나가므로, 열기에는 항상 넉넉한 힘을 명시해 상태 의존을 없앤다.
        self._gripper_open_force_n = float(gripper.get("open_force_n", 40.0))
        # 후보 선택 파라미터. 개폭 관련 값은 위에서 이미 읽은 실행 기준을 그대로 넘긴다 —
        # 검사 기준과 실행 기준이 갈라지면 "통과시켜 놓고 다르게 동작하는" 후보가 생긴다.
        self._selection = self._selection_params(params)
        # STEP 1 실물 비교에서 legacy/enhanced가 같은 후보를 골랐고 contact_support 전달도
        # 확인했다(2026-09-10). production은 enhanced 하나만 사용한다. 비교용 runtime
        # 분기는 STEP 3에서 제거했다.
        # --- 모션 실행 가능성 (STEP 2) ---
        spec = params.get("grasp_selection") or {}
        # **기본 켜짐.** 켜면 후보마다 (자세 표현 × solution space) 조합과 구간 경로까지
        # 확인해 "현재 관절 자세로는 못 가는" 후보를 살려낸다 — robot-aware selection의
        # 본체다. skill_params.yaml도 true로 두고 있으므로 기본값을 그와 맞춘다.
        #
        # 2026-09-10에 이 값을 잠시 껐던 이유(pick 이동이 8초 → 17초)는 **후보 선택이
        # 아니라 control 컨테이너 상태였고, 재시작으로 해소됐다.** 검사가 느리게 만든
        # 것이 아니었으므로 기본-off로 둘 근거가 없다.
        self._motion_feasibility_enabled = bool(
            spec.get("motion_feasibility_enabled", True))
        # 경로 중간에서 관절이 이보다 튀면 그 구간에서 configuration이 갈린 것으로 본다.
        self._max_joint_jump_deg = float(spec.get(
            "max_joint_jump_deg", motion_feasibility.DEFAULT_MAX_JOINT_JUMP_DEG))
        # 다른 solution space로 바꾸려면 관절 이동(movej)이 필요한데, 그건 직선 경로가
        # 아니라 팔이 크게 휘돈다. **이 값 자체는 더 이상 후보를 버리지 않는다** — 판정은
        # motion_feasibility.config_switch_feasible이 전환 구간의 FK를 실제로 찍어 명백한
        # 충돌만 보고, 이 값을 넘으면 참고 경고만 남긴다. place가 STEP 3에서 먼저 겪은
        # 것과 같은 이유다: 각도 크기만으로 거르면 solution space를 바꾸는 데 정상적으로
        # 필요한 폭(실물 168~257도)까지 막혀 멀쩡한 후보가 전부 사라진다.
        self._max_config_switch_jump_deg = float(
            spec.get("max_config_switch_jump_deg", 90.0))
        # "현재 → 접근" 구간이 직선으로 막혔을 때 **우회**에 쓸 여유 높이. 두 끝점 중 높은
        # 쪽에서 이만큼 더 올라갔다가 건너간다. 직선이 되면 쓰이지 않는다 — 우회는 후보를
        # 버리는 대신 돌아가는 수단이지, 매번 높이 올라가라는 뜻이 아니다.
        self._transit_clearance_mm = float(spec.get("transit_clearance_mm", 50.0))
        # configuration을 바꿔야 할 때 쓰는 관절 이동 속도. home과 같은 값을 기본으로 쓴다 —
        # 이 이동은 직선이 아니라 팔이 휘도는 이동이라 빠르게 할 이유가 없다.
        home = params.get("home") or {}
        self._joint_vel_deg_s = float(spec.get("config_switch_vel_deg_s",
                                               home.get("vel_deg_s", 20.0)))
        self._joint_acc_deg_s2 = float(spec.get("config_switch_acc_deg_s2",
                                                home.get("acc_deg_s2", 20.0)))
        # `_select_candidate`가 채운다(검사 때 찾은 IK 해 / 그때의 로봇 상태 / 실행 계획).
        # `_ik_group`은 후보 기하(접근+파지)를 같은 solution space에서 함께 푼 결과다.
        self._ik_choice: dict = {}
        self._ik_group: dict = {}
        self._current_sol_space: int | None = None
        self._current_posj: list | None = None
        self._motion_plan: dict | None = None
        # 관절 한계는 SelectionParams가 아니라 IK 질의(dsr_motion.verify_ik)가 쓴다.
        # 설정에 없으면 URDF에서 읽은 M0609 값을 그대로 쓴다.
        limits = (params.get("grasp_selection") or {}).get("joint_limits_deg")
        self._joint_limits_deg = (tuple(float(v) for v in limits)
                                  if isinstance(limits, (list, tuple)) and len(limits) == 6
                                  else dsr_motion.JOINT_LIMITS_DEG)
        self._movel_client = ActionClient(self, MovelH2r, dsr_motion.MOVEL_ACTION,
                                          callback_group=callbacks)
        # 다른 solution space에서만 풀리는 후보를 실행할 때만 쓴다(STEP 2) — movel은
        # configuration을 바꾸지 못하므로 그 자세의 관절해로 직접 이동해야 한다.
        self._movej_client = ActionClient(self, MovejH2r, dsr_motion.MOVEJ_ACTION,
                                          callback_group=callbacks)
        # 컨트롤러가 "이 목표는 못 간다"고 내는 알람을 지켜본다 — 없으면 movel이
        # goal을 accept한 채 아무것도 안 하는 경우가 60초 타임아웃을 다 채운다
        # (dsr_motion.MotionErrorMonitor 참조).
        self._motion_errors = dsr_motion.MotionErrorMonitor(self, callbacks)
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
            SelectedGrasp, SELECTED_GRASP_TOPIC,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self._gripper_cmd_client = self.create_client(
            SetCommand, dsr_motion.GRIPPER_COMMAND_SERVICE, callback_group=callbacks)
        self._gripper_pose_client = self.create_client(
            GripperPose, "/onrobot/pose", callback_group=callbacks)
        # compliance.py 진단(torque_trace_summary)용 — 판정에는 안 쓴다(compliance.py 참조).
        self._torque_client = self.create_client(
            GetExternalTorque, dsr_motion.GET_EXTERNAL_TORQUE_SERVICE, callback_group=callbacks)
        self._gripper_joint_angle: float | None = None
        self.create_subscription(JointState, dsr_motion.GRIPPER_JOINT_STATES_TOPIC,
                                 self._on_gripper_state, 5, callback_group=callbacks)
        # 팔 관절각. get_current_posx(aux_control)가 무응답일 때 fkin으로 손목 자세를
        # 복원하는 데 쓴다 — 그게 없으면 뒤집기 보정이 꺼져 172도를 헛돈다(_wrist_posx).
        self._arm_joint_deg: tuple[float, ...] | None = None
        self._arm_joint_stamp: float | None = None
        self.create_subscription(JointState, dsr_motion.ARM_JOINT_STATES_TOPIC,
                                 self._on_arm_state, 5, callback_group=callbacks)
        # RG2 컨트롤러의 상태 비트필드. bit1(Grip detected)이 **개폭과 무관하게** 물체를
        # 물었는지 알려준다 — 변형체를 제대로 물어도 0mm까지 닫히는 문제(아래 파지 판정
        # 주석 참조)를 개폭 추정 없이 해결한다. (time, gsta)로 들고 있어 신선도를 본다.
        self._gripper_status: tuple[float, int, float] | None = None
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
                lambda: self._gripper_joint_angle, _NeverCancelled(),
                get_status=lambda: self._gripper_status) is None:
            self.get_logger().warning("기동 그리퍼 열기: 여는 동안 응답이 없다(타임아웃)")
            return
        self.get_logger().info(
            f"기동 그리퍼 열기 완료 — 최대 개방({self._gripper_open_m * 1000:.0f}mm)")

    def _on_gripper_status(self, msg: OnRobotRGInput) -> None:
        self._gripper_status = (time.monotonic(), int(msg.gsta), float(msg.gwdf) / 10.0)

    def _grip_detected(self, max_age_s: float = 2.0) -> bool | None:
        """RG2가 보고하는 'Grip detected'(gsta bit1). 모르면 None.

        None을 돌려주는 경우를 호출부가 반드시 구분해야 한다 — 드라이버가 이 토픽을
        발행하지 않는 구버전이거나(2026-09-07 이전) 값이 낡았으면 "안 물었다"가 아니라
        "모른다"이고, 그때는 예전 개폭 판정으로 떨어져야 한다.
        """
        if self._gripper_status is None:
            return None
        stamp, gsta, _ = self._gripper_status
        if time.monotonic() - stamp > max_age_s:
            return None
        return bool(gsta & 0x02)

    def _gripper_evidence(self, since: float | None = None, max_age_s: float = 2.0):
        if self._gripper_status is None:
            return None
        stamp, gsta, status_width_mm = self._gripper_status
        if time.monotonic() - stamp > max_age_s or (since is not None and stamp < since):
            return None
        return {"stamp": stamp, "gsta": gsta, "grip": bool(gsta & 0x02),
                "status_width_mm": status_width_mm}

    def _on_gripper_state(self, msg: JointState) -> None:
        if msg.position:
            self._gripper_joint_angle = msg.position[0]

    def _on_arm_state(self, msg: JointState) -> None:
        """팔 관절각(라디안 → 도). **이름으로 골라낸다** — 순서를 가정하면 드라이버가
        순서를 바꿨을 때 조용히 엉뚱한 자세를 만든다."""
        index = {name: i for i, name in enumerate(msg.name)}
        try:
            self._arm_joint_deg = tuple(
                math.degrees(float(msg.position[index[name]]))
                for name in dsr_motion.ARM_JOINT_NAMES)
            self._arm_joint_stamp = time.monotonic()
        except (KeyError, IndexError, TypeError, ValueError):
            return

    def _wrist_posx(self, goal_handle):
        """지금 TCP posx. aux_control이 무응답이면 fkin으로 복원한다.

        **이 값이 None이면 뒤집기 보정이 꺼진다.** 평행 그리퍼는 닫힘축 부호가 뒤집혀도
        같은 파지라 두 표현 중 가까운 쪽을 골라야 하는데(dsr_motion.
        nearest_equivalent_grasp_rotation), 현재 자세를 모르면 그 선택을 못 해 전략이 낸
        표현을 그대로 명령한다 — 2026-09-07 실물에서 172.4도를 헛돌아 접근에만 14초가
        걸린 그 경로다. pick은 movel 직후(= aux_control 무응답 구간)에 시작하므로 특히
        **재시도에서** 이 폴백이 없으면 그 문제가 되살아난다.
        """
        posx = dsr_motion.get_current_posx(self._posx_client, goal_handle,
                                           timeout_s=1.0, retries=2)
        if posx is not None:
            return posx
        if goal_handle.is_cancel_requested:
            return None
        joints = self._arm_joint_deg
        if (self._arm_joint_stamp is None
                or time.monotonic() - self._arm_joint_stamp > 2.0):
            joints = None
        posx = dsr_motion.posx_via_fkin(self._fkin_client, joints)
        if posx is None:
            self.get_logger().warning(
                "손목 자세를 못 읽었다 (get_current_posx 무응답 + fkin 폴백도 실패) — "
                "뒤집기 보정과 접근 적합도 항이 빠진 채 진행한다")
        else:
            self.get_logger().info(
                "get_current_posx 무응답 — /dsr01/joint_states + fkin으로 손목 자세를 "
                f"복원했다 (TCP={[round(v, 1) for v in posx[:3]]})")
        return posx

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
            f"pick 시작 object={goal.object_id} grip_level={goal.grip_level} "
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
                width_mm, torque = 42.0, [0.4, 1.9, 2.6, 2.4]
                executed_tcp = list(selected.geometry.target_posx)
            else:
                pick_result = self._pick_real(goal_handle, goal, selected, wrist_pose)
                if pick_result is None:          # 취소 (_pick_real의 모든 취소 경로가 None)
                    goal_handle.canceled()
                    return self._result(False, Pick.Result.REASON_NO_CONTACT, started)
                width_mm, close_target_mm, executed_tcp, evidence = pick_result
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
                pre, post, lifted = (evidence.get(name) for name in ("pre", "post", "lift"))
                if pre is None or post is None or lifted is None or pre["grip"]:
                    basis = (f"pre={pre} post={post} lift={lifted} "
                             f"pose_width_close={width_mm:.1f}mm "
                             f"pose_width_lift={evidence.get('lift_width_mm')}")
                    self.get_logger().error(
                        f"[PICK_VERIFY] 상태 불확실 — 자동 재계획 금지 | {basis}")
                    store.set_gripper(width_mm=width_mm, closed=True)
                    store.set_error()
                    result = self._result(False, Pick.Result.REASON_UNREACHABLE, started,
                                          candidate_id=selected.candidate.candidate_id)
                    self._cache.put(goal.request_id, result)
                    goal_handle.abort()
                    return result
                grasped = bool(pre and post and lifted
                               and not pre["grip"] and post["grip"] and lifted["grip"])
                basis = (f"pre={pre} post={post} lift={lifted} "
                         f"pose_width_close={width_mm:.1f}mm "
                         f"pose_width_lift={evidence.get('lift_width_mm')}")
                self.get_logger().info(
                    f"[PICK_VERIFY] {'성공' if grasped else '실패'} | {basis}")
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
                # torque는 compliance.py가 모은 진단값이다 — 판정에는 안 쓴다.
                torque = evidence.get("torque_trace_summary", [])

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
                                   torque=torque,
                                   candidate_id=selected.candidate.candidate_id,
                                   source_observation_id=goal.source_observation_id,
                                   executed_tcp=executed_tcp)
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
            # 접근 적합도(2.5차). 세 각도는 grasp_params.yaml의 같은 이름 값들과 짝이다 —
            # zero는 grasp의 hard 상한과 같아야 "그 위는 애초에 안 온다"가 성립한다.
            approach_angle_prefer_deg=float(
                spec.get("approach_angle_prefer_deg", defaults.approach_angle_prefer_deg)),
            approach_angle_soft_max_deg=float(
                spec.get("approach_angle_soft_max_deg", defaults.approach_angle_soft_max_deg)),
            approach_angle_zero_deg=float(
                spec.get("approach_angle_zero_deg", defaults.approach_angle_zero_deg)),
            approach_travel_ref_mm=float(
                spec.get("approach_travel_ref_mm", defaults.approach_travel_ref_mm)),
            approach_rotation_ref_deg=float(
                spec.get("approach_rotation_ref_deg", defaults.approach_rotation_ref_deg)),
            comfort_joint_delta_ref_deg=float(
                spec.get("comfort_joint_delta_ref_deg",
                         defaults.comfort_joint_delta_ref_deg)),
            weights=weights,
        )

    def _candidates_from_goal(self, goal) -> list:
        """goal → 검사할 후보 목록(점수 내림차순, planner가 준 순서 그대로).

        `grasp_candidates`가 비어 있으면 `grasp_pose` 하나로 만든다 — 예전 planner나
        goal을 직접 쏘는 도구가 그대로 동작해야 한다. 그 경우에도 개폭·IK·안전 검사는
        똑같이 지나간다(그게 이 작업이 막으려는 실패 모드다).

        **다른 물체의 후보는 여기서 버린다.** `candidate_id`는 "<object_id>#<순위>"라
        어느 물체에서 나왔는지가 값 안에 들어 있다(grasp/node.py). 지금까지 control은
        그걸 확인하지 않고 goal에 실려 온 것을 전부 검사했는데, 그러면 planner/orchestrator
        어딘가에서 목록이 섞였을 때 **점수가 높다는 이유로 명령과 무관한 물체를 집는**
        경로가 열린다. 실제로 섞이는 경로를 아직 못 찾았더라도, 그 사고는 조용히 일어나고
        결과가 물리적이라 여기서 값싸게 막아 둔다(2026-09-08).
        """
        raw = list(getattr(goal, "grasp_candidates", None) or [])
        if not raw:
            return [grasp_selection.Candidate(
                candidate_id="", rank=0, pose=goal.grasp_pose,
                # 후보가 하나뿐이라 점수는 순위에 영향이 없다.
                score=1.0, gripper_width_mm=float(goal.gripper_width_mm))]

        target = str(goal.object_id or "")
        kept, foreign = [], []
        for candidate in raw:
            owner = str(candidate.candidate_id or "").split("#")[0]
            # candidate_id가 비어 있으면 출처를 모른다 — 예전 planner가 채워 보내던
            # 형식이라 버리지 않고 통과시킨다(그 경우 goal 자체가 이 물체의 것이다).
            if owner and target and owner != target:
                foreign.append(candidate.candidate_id)
                continue
            kept.append(candidate)
        if foreign:
            self.get_logger().error(
                f"[대상불일치] object={target} goal에 다른 물체의 후보 {len(foreign)}개가 "
                f"실려 왔다 — 버린다: {foreign[:5]}"
                + (" ..." if len(foreign) > 5 else ""))
        return [grasp_selection.Candidate(
            candidate_id=candidate.candidate_id, rank=rank, pose=candidate.pose,
            score=float(candidate.score),
            gripper_width_mm=float(candidate.gripper_width_mm),
            strategy=candidate.strategy,
            # -1(미상)과 0.0(지지 없음)은 뜻이 다르다. 구 planner/도구가 보낸 goal에는
            # 필드 자체가 없을 수 있어 getattr 기본값도 -1이다.
            contact_support_score=float(
                getattr(candidate, "contact_support_score", -1.0)))
            for rank, candidate in enumerate(kept)]

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

    def _pad_floor_obstacle(self, obj):
        """경로가 **지지면(작업대) 아래로 손끝을 끌고 가는지** 보는 가벼운 충돌 검사기.

        보는 것은 TCP가 아니라 **손끝 예상 높이**다 — TCP는 패드보다 위에 있어서 TCP만
        보면 이미 작업대를 쓸고 있는 경로도 통과한다. 기준과 허용치는 끝점 검사
        (grasp_selection.check_min_safety)와 **같은 값**을 쓴다: 끝점에서 통과한 자세가
        경로 검사에서 다른 이유로 떨어지면 안 되기 때문이다.

        지지면을 모르면(관측에 물체 높이가 없으면) 검사기를 아예 만들지 않는다 — 모르는
        것을 위험하다고 바꾸지 않는다(motion_feasibility 모듈 docstring).
        """
        support_z = obj.support_z_mm
        if support_z is None:
            return None
        floor = float(support_z) - float(self._selection.support_tolerance_mm)

        def obstacle(posx) -> str:
            pad_z = dsr_motion.grasp_center_from_posx(
                list(posx), self._grasp_center_offset_mm)[2]
            if float(pad_z) < floor:
                return (f"경로 지점의 손끝 예상 z {float(pad_z):.1f}mm가 지지면 "
                        f"{float(support_z):.1f}mm 아래(허용 "
                        f"{self._selection.support_tolerance_mm:.1f}mm)")
            return ""

        return obstacle

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
        wrist_pose = None if fake else self._wrist_posx(goal_handle)
        current_zyz = wrist_pose[3:] if wrist_pose else None
        # ikin은 solution space마다 다른 해를 낸다. movel은 현재 space를 유지하므로
        # "지금 상태에서 갈 수 있는가"를 물으려면 현재 space로 물어야 한다.
        sol_space = 2 if fake else dsr_motion.current_solution_space(
            self._solution_space_client)

        def geometry_of(pose):
            target_posx, approach_posx, axis = dsr_motion.plan_pick_posx(
                pose, self._grasp_center_offset_mm, self._approach_height_mm,
                self._pick_depth_extra_mm, current_zyz_deg=current_zyz)
            geometry = grasp_selection.PickGeometry(
                target_posx=target_posx, approach_posx=approach_posx, approach_axis=axis,
                pad_reference_mm=dsr_motion.grasp_center_from_posx(
                    target_posx, self._grasp_center_offset_mm))
            if self._motion_feasibility_enabled:
                self._resolve_ik_group(geometry, ik_query, spaces)
            return geometry

        # (자세 표현 × solution space) 조합을 순서대로 물어 **첫 성공**을 쓴다 (STEP 2).
        # 예전에는 현재 space에서 한 번 물어보고 실패하면 그 후보를 버렸는데, 그 실패의
        # 뜻은 대개 "이 파지가 불가능하다"가 아니라 "이 관절 자세로는 못 간다"였다.
        # 찾은 해는 기하(posx)별로 기억해 뒀다가 실행이 같은 것을 쓴다 — 검사와 실행이
        # 갈라지면 IK를 통과시킨 근거가 사라진다.
        spaces = motion_feasibility.solution_space_order(sol_space)
        self._ik_choice = {}
        self._ik_group = {}
        self._current_sol_space = sol_space
        # 지금 관절값 — 현재 자세를 현재 space로 ikin한 해다. configuration을 바꿔야 할 때
        # "얼마나 움직여야 하나"를 재는 기준이고, 이걸 위해 새 서비스를 만들지 않는다.
        self._current_posj = None
        if wrist_pose is not None and self._motion_feasibility_enabled:
            current_verdict = dsr_motion.verify_ik(
                self._ikin_client, self._fkin_client, wrist_pose,
                sol_space=sol_space, limits=self._joint_limits_deg)
            if getattr(current_verdict, "known", False):
                self._current_posj = getattr(current_verdict, "posj", None)

        # 한 번의 후보 선택 안에서 같은 (posx, space)를 여러 번 묻게 된다(짝 탐색 →
        # 구간 검사 → 우회 경로). 로봇이 그동안 안 움직이므로 답은 같다 — 캐시로 묶어
        # ikin 왕복(실측 약 31ms)을 줄인다.
        ik_query = motion_feasibility.ik_cache(
            lambda posx, space: dsr_motion.verify_ik(
                self._ikin_client, self._fkin_client, posx,
                sol_space=space, limits=self._joint_limits_deg))

        def ik_verdict_of(posx):
            if not self._motion_feasibility_enabled:
                # STEP 2 이전과 동일 — 현재 solution space에서 한 번만 묻는다.
                return ik_query(posx, sol_space)
            key = _posx_key(posx)
            found = self._ik_choice.get(key)
            if found is not None:
                # `geometry_of`가 접근+파지를 **한 space에서 함께** 푼 결과다. 여기서
                # 지점별로 다시 찾으면 둘이 서로 다른 space로 갈라져 실행할 수 없게 된다.
                return found.verdict
            found = motion_feasibility.find_ik(
                self._wrist_representations(posx, current_zyz), ik_query, spaces)
            if found is not None:
                self._ik_choice[key] = found
                return found.verdict
            # 아무 조합도 안 됐다 — 현재 space·원래 표현의 판정을 그대로 돌려준다.
            # (무응답이면 known=False라 grasp_selection이 "확인 못 함"으로 다룬다.)
            return ik_query(posx, spaces[0])

        evaluations = grasp_selection.evaluate_candidates(
            candidates, obj, self._selection, geometry_of, ik_verdict_of,
            # 접근 적합도(2.5차): 지금 자세에서 접근 지점까지 얼마나 움직여야 하는지.
            # wrist_pose는 위에서 이미 읽은 값이라 서비스를 다시 부르지 않는다.
            # 못 읽었으면(None) 이 항만 빠지고 나머지 판정은 그대로 돈다.
            current_posx=wrist_pose,
            # ZYZ는 ry가 180도 근처면 같은 방향이 다른 (rx,rz)로 나온다 — 성분 차로
            # 재면 안 되므로 회전행렬 각도차를 쓰는 함수를 넘긴다(dsr_motion 참조).
            rotation_diff=dsr_motion.rotation_diff_deg)
        selected, _, _ = grasp_selection.select(evaluations, mode="enhanced")
        # **여기서 끝내지 않는다 (STEP 2).** 고른 후보가 끝점 IK는 통과했어도 실제 구간이
        # 이어지지 않을 수 있다 — 그러면 그 후보를 건너뛰고 다음 순위 후보를 본다.
        # 다만 그 검사로 **후보가 사라지지는 않는다**: 전부 걸리면 랭킹 1위를 그대로
        # 실행한다(_first_feasible ③).
        motion_plan = None
        if selected is not None and self._motion_feasibility_enabled:
            selected, motion_plan = self._first_feasible(
                evaluations, selected, goal.object_id, wrist_pose, ik_query,
                self._pad_floor_obstacle(obj))
        self._motion_plan = motion_plan

        # execution pool이 여기 상한보다 크면 뒤쪽 후보가 **조용히** 잘린다 — grasp가
        # 다양성으로 고른 선택지가 사라지므로 드러내 알린다(STEP 1 §12).
        if len(candidates) > self._selection.max_evaluated:
            self.get_logger().warning(
                f"[후보선택] object={goal.object_id} 받은 후보 {len(candidates)}개가 "
                f"max_evaluated={self._selection.max_evaluated}보다 많다 — 뒤쪽 "
                f"{len(candidates) - self._selection.max_evaluated}개는 평가하지 않는다 "
                "(grasp_params.yaml execution_pool_size와 맞출 것)")

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
            f"geom탈락={tally[grasp_selection.STATUS_GEOMETRY_INVALID]} "
            f"ik탈락={tally[grasp_selection.STATUS_IK_FAILED]} "
            f"joint탈락={tally[grasp_selection.STATUS_JOINT_LIMIT]} "
            f"safety탈락={tally[grasp_selection.STATUS_SAFETY_INVALID]} "
            f"valid={tally[grasp_selection.STATUS_VALID] + tally[grasp_selection.STATUS_SELECTED]} "
            + (f"mode=enhanced "
               f"selected={selected.candidate.candidate_id or '(단일 grasp_pose)'} "
               f"legacy={selected.legacy_score:.3f} "
               f"enhanced={selected.total_score:.3f}" if selected else "selected=없음"))
        if unknown_ik:
            self.get_logger().warning(
                f"[후보선택] IK 확인 실패 {unknown_ik}건 — ikin/fkin 서비스가 응답하지 않아 "
                "도달 가능성을 검증하지 못한 채 통과시켰다(로봇 드라이버 미기동?)")
        self._publish_selected(selected, goal.trace_id, goal.source_observation_id)
        return selected, evaluations, wrist_pose

    def _first_feasible(self, evaluations, selected, object_id: str, wrist_pose,
                        ik_query, obstacle):
        """랭킹 순서대로 보면서 **모션까지 실제로 가능한 첫 후보**를 고른다 (STEP 2).

        끝점 IK만 통과하고 구간이 안 이어지는 후보가 있다 — 예전에는 그걸 그대로 실행해
        movel이 오류 없이 안 움직이거나 도중에 실패했다. 여기서 걸러 다음 후보로 넘어간다.
        **후보가 하나 안 된다고 pick 전체를 실패시키지 않는다**(STEP 1 계약 그대로).

        **검사는 후보를 버리는 수단이 아니라 고르는 수단이다 (2026-09-10 보완).** 세 단계로
        누그러뜨린다 — ① 중간 보간 지점까지 깨끗한 후보, ② 끝점만 확실한 후보(place가 먼저
        쓴 2패스와 같은 완화), ③ 그래도 없으면 **랭킹 1위를 그대로 실행한다.** ③이 있는
        이유는 이 검사가 직선 보간과 강제 space라는 가정 위에 서 있어서, 그것 때문에 로봇이
        아예 안 움직이는 쪽이 더 나쁜 결과이기 때문이다(motion_feasibility 모듈 docstring
        "애매하면 막지 않는다").

        반환: `(선택된 Evaluation, 실행 계획)`. 후보가 아예 없을 때만 `(None, None)`이다.
        """
        ranked = sorted(
            [e for e in evaluations
             if e.status in (grasp_selection.STATUS_VALID, grasp_selection.STATUS_SELECTED)],
            key=lambda e: (-(e.total_score or 0.0), e.candidate.rank))
        if not ranked:
            return None, None

        for strict in (True, False):
            for evaluation in ranked:
                plan, reason = self._motion_plan_for(
                    evaluation, wrist_pose, ik_query, obstacle, strict)
                if plan is not None:
                    self._mark_selected(selected, evaluation)
                    self._log_plan(object_id, evaluation, plan, reason, strict)
                    return evaluation, plan
                if strict:
                    self.get_logger().info(
                        f"[모션검사] object={object_id} "
                        f"{evaluation.candidate.candidate_id or '(단일 grasp_pose)'} "
                        f"깨끗한 경로 없음 — {reason}")

        # ③ 마지막 안전장치. 여기 오는 후보들은 **끝점 IK를 이미 통과했다**(끝점이 안 되는
        # 후보는 grasp_selection이 앞에서 탈락시킨다). 남은 것은 경로 판정의 의심뿐이라,
        # 로봇을 세우는 대신 실행해 보고 실패하면 모션 단계가 보고하게 한다.
        evaluation, plan = self._fallback_plan(ranked)
        self._mark_selected(selected, evaluation)
        self.get_logger().warning(
            f"[모션검사] object={object_id} 모든 후보가 경로 검사에서 걸렸다 — 랭킹 1위 "
            f"{evaluation.candidate.candidate_id or '(단일 grasp_pose)'}를 그대로 실행한다 "
            "(검사는 고르는 수단이지 막는 수단이 아니다. 실패하면 모션 단계가 보고한다)")
        return evaluation, plan

    @staticmethod
    def _mark_selected(selected, evaluation) -> None:
        if selected is not None and evaluation is not selected:
            selected.status = grasp_selection.STATUS_VALID
            evaluation.status = grasp_selection.STATUS_SELECTED

    def _log_plan(self, object_id: str, evaluation, plan: dict, reason: str,
                  strict: bool) -> None:
        if reason:
            self.get_logger().info(f"[모션검사] object={object_id} {reason}")
        if not strict:
            self.get_logger().info(
                f"[모션검사] object={object_id} "
                f"{evaluation.candidate.candidate_id or '(단일 grasp_pose)'} 채택 — 끝점은 "
                "확실하고 중간 보간 지점만 안 풀렸다(직선 보간 가정이라 실패로 치지 않는다)")
        if plan.get("waypoints"):
            self.get_logger().warning(
                f"[모션검사] object={object_id} 접근까지 직선이 막혀 우회한다 "
                f"(경로 '{plan['route']}', 경유 {len(plan['waypoints'])}곳)")
        if plan.get("switch_posj") is not None:
            self.get_logger().warning(
                f"[모션검사] object={object_id} "
                f"{evaluation.candidate.candidate_id}는 현재 관절 자세로는 못 가고 "
                f"solution space {plan['sol_space']}에서만 풀린다 — 접근 자세로 "
                "관절 이동한 뒤 내려간다")

    def _fallback_plan(self, ranked):
        """검사를 통과한 후보가 하나도 없을 때 **그래도 실행할** 후보와 계획.

        접근과 파지가 같은 solution space에서 풀린 후보를 우선한다 — 그게 없으면 하강
        도중에 configuration이 갈려 실행 자체가 성립하지 않기 때문이다. 그마저 없으면
        랭킹 1위를 예전(STEP 2 이전) 경로 그대로 실행한다.
        """
        for evaluation in ranked:
            group = self._ik_group.get(_posx_key(evaluation.geometry.target_posx))
            if group is None:
                continue
            approach_option, grasp_option = group.options[0], group.options[1]
            switch_posj = (list(approach_option.posj)
                           if group.sol_space != self._current_sol_space
                           and approach_option.posj else None)
            return evaluation, {
                "sol_space": group.sol_space, "switch_posj": switch_posj,
                "approach_posx": list(approach_option.posx),
                "target_posx": list(grasp_option.posx),
                "representation": group.representation,
                "route": "direct", "waypoints": []}
        evaluation = ranked[0]
        return evaluation, {
            "sol_space": None, "switch_posj": None,
            "approach_posx": list(evaluation.geometry.approach_posx),
            "target_posx": list(evaluation.geometry.target_posx),
            "representation": "original", "route": "direct", "waypoints": []}

    @staticmethod
    def _wrist_representations(posx, current_zyz):
        """같은 물리적 파지를 나타내는 TCP 자세 표현들 — 선호 순서대로.

        평행 그리퍼는 접근축 둘레 180도를 뒤집어도 같은 파지다(닫힘축만 부호가 반대).
        원래 표현으로 IK가 안 풀려도 뒤집은 표현은 풀리는 경우가 있으므로 둘 다 후보로
        둔다. 앞쪽이 선호 — `plan_pick_posx`가 이미 현재 손목에 가까운 쪽을 골라 두었으니
        원래 표현이 앞이다(헛도는 180도 회전을 다시 만들지 않는다).
        """
        flipped = dsr_motion.flipped_wrist_posx(posx)
        options = [(list(posx), "original")]
        if flipped is not None:
            options.append((list(flipped), "equivalent_wrist"))
        return options

    def _resolve_ik_group(self, geometry, ik_query, spaces):
        """후보의 접근점과 파지점을 **같은 solution space에서 함께** 푼다 (2026-09-10).

        지점마다 따로 풀면 접근이 space 3, 파지가 space 5에서 먼저 풀리는 일이 생기는데
        movel은 한 이동 안에서 space를 못 바꾸므로 그 조합은 실행할 수 없다. 예전에는 그걸
        나중에 발견해 **후보를 통째로 버렸다** — 정작 둘 다 되는 space가 있는데도 그랬다.
        여기서 짝으로 찾아 두면 그 탈락이 사라진다.

        찾은 해는 기하(posx)별로 기억해 뒀다가 랭킹·구간검사·실행이 같은 것을 쓴다.
        """
        approach, target = list(geometry.approach_posx), list(geometry.target_posx)
        groups = [("original", [approach, target])]
        flipped = [dsr_motion.flipped_wrist_posx(approach),
                   dsr_motion.flipped_wrist_posx(target)]
        if all(value is not None for value in flipped):
            groups.append(("equivalent_wrist", flipped))
        group = motion_feasibility.find_ik_group(groups, ik_query, spaces)
        if group is None:
            return None
        # 실행이 쓰는 좌표는 group.options[*].posx다(뒤집은 표현이면 그 좌표). 조회는
        # 원래 기하 좌표로 들어오므로 **양쪽 키 모두** 같은 해를 가리키게 해 둔다.
        self._ik_group[_posx_key(target)] = group
        self._ik_choice[_posx_key(approach)] = group.options[0]
        self._ik_choice[_posx_key(target)] = group.options[1]
        for option in group.options:
            self._ik_choice.setdefault(_posx_key(option.posx), option)
        return group

    def _motion_plan_for(self, evaluation, wrist_pose, ik_query, obstacle,
                         strict: bool = True):
        """고른 후보를 실제로 실행할 수 있는지 — 끝점이 아니라 **구간**으로 확인한다.

        검사 구간: 현재 → 접근, 접근 → 파지, 파지 → 후퇴(= 접근으로 되돌아감).
        각 구간을 몇 지점으로 나눠 IK·관절 연속성·명백한 충돌을 본다(path_feasible).

        **현재 → 접근이 직선으로 막히면 우회 경유점을 넣어 본다** (2026-09-10 보완).
        그 구간이 막혔다는 것은 대개 "이 파지가 불가능하다"가 아니라 "지금 자리에서 곧장
        가면 걸린다"는 뜻이라, place가 이미 쓰고 있는 경로 생성기(motion_feasibility.routes)
        를 같은 방식으로 쓴다. 접근 → 파지 하강은 우회하지 않는다 — 그 구간은 접근축을 따라
        곧게 내려가야 파지가 성립한다.

        반환: `(계획 dict 또는 None, 사유)`.
        """
        geometry = evaluation.geometry
        approach_option = self._ik_choice.get(_posx_key(geometry.approach_posx))
        grasp_option = self._ik_choice.get(_posx_key(geometry.target_posx))
        if approach_option is None or grasp_option is None:
            # IK를 못 물어본 경우(ikin 무응답)다 — 여기서 막지 않는다. 예전과 같은 경로로
            # 그대로 실행하고, 실패하면 모션 단계가 보고한다.
            return {"sol_space": None, "switch_posj": None,
                    "approach_posx": list(geometry.approach_posx),
                    "target_posx": list(geometry.target_posx),
                    "route": "direct", "waypoints": []}, "IK 확인 못 함 — 기존 경로로 진행"

        space = approach_option.sol_space
        if grasp_option.sol_space != space:
            # 짝 탐색(`_resolve_ik_group`)이 8개 space를 전부 보고도 둘을 같은 space에서
            # 풀지 못했다는 뜻이다. 이건 검사의 보수성이 아니라 실제 제약이다 — 하강
            # 도중 configuration이 갈리면 그 movel은 성립하지 않는다.
            return None, (f"접근(space {space})과 파지(space {grasp_option.sol_space})를 "
                          "동시에 푸는 solution space가 8개 중 없다")

        for name, start, end in (("접근→파지", approach_option.posx, grasp_option.posx),
                                 ("파지→후퇴", grasp_option.posx, approach_option.posx)):
            check = motion_feasibility.path_feasible(
                start, end, ik_query, space, obstacle=obstacle,
                max_joint_jump_deg=self._max_joint_jump_deg, strict=strict)
            if not check.feasible:
                return None, f"{name} 구간 실행 불가 — {check.reason}"

        switch_posj = None
        if space != self._current_sol_space:
            # 다른 configuration에서만 되는 후보다. movel은 space를 못 바꾸므로 접근
            # 자세의 관절해로 **관절 이동**해서 그 configuration으로 들어간 뒤 내려간다.
            reason = self._config_switch_reason(approach_option, obstacle)
            if reason:
                return None, reason
            switch_posj = list(approach_option.posj)

        route, waypoints, reason = self._approach_route(
            wrist_pose, approach_option, space, ik_query, obstacle, strict,
            switching=switch_posj is not None)
        if route is None:
            return None, reason
        return {"sol_space": space, "switch_posj": switch_posj,
                "approach_posx": list(approach_option.posx),
                "target_posx": list(grasp_option.posx),
                "representation": approach_option.representation,
                "route": route, "waypoints": waypoints}, ""

    def _config_switch_reason(self, approach_option, obstacle) -> str:
        """configuration 전환(movej)을 막아야 할 이유. 없으면 빈 문자열.

        **관절 이동량으로 막지 않는다** (2026-09-10, place가 STEP 3에서 먼저 겪은 것).
        각도 크기는 그 전환이 위험하다는 근거가 아니다 — solution space를 바꾸려면 팔·팔꿈치·
        손목 중 하나가 크게 도는 것이 정상이고, 각도로 거르면 그 정상적인 전환까지 전부
        막혀 후보가 사라진다. 실제로 지나는 지점의 FK를 찍어 **명백한 충돌만** 본다.
        """
        if not approach_option.posj:
            # 현재 관절값이나 목표 관절값을 모른다 — 모르는 것을 위험하다고 하지 않는다.
            return ""
        if self._current_posj is None:
            self.get_logger().info(
                "현재 관절값을 몰라 configuration 전환 폭을 재지 못한다 — 막지 않고 진행한다")
            return ""
        check = motion_feasibility.config_switch_feasible(
            self._current_posj, approach_option.posj,
            lambda joint: dsr_motion.posx_via_fkin(self._fkin_client, joint),
            obstacle=obstacle)
        if not check.feasible:
            return f"configuration 전환 경로가 명백한 충돌 — {check.reason}"
        jump = motion_feasibility.joint_jump_deg(self._current_posj, approach_option.posj)
        if jump is not None and jump > self._max_config_switch_jump_deg:
            self.get_logger().info(
                f"configuration 전환에 관절이 {jump:.0f}도 움직인다 "
                f"(참고 한계 {self._max_config_switch_jump_deg:.0f}도) — 전환 경로에 "
                "명백한 충돌은 없어 그대로 진행한다")
        return ""

    def _approach_route(self, wrist_pose, approach_option, space, ik_query, obstacle,
                        strict: bool, switching: bool):
        """"현재 → 접근" 구간의 경로. 직선이 되면 직선, 막히면 우회 경유점을 찾는다.

        반환: `(경로 이름 또는 None, 경유점 목록, 사유)`. 경유점 목록은 접근점 자체를
        포함하지 않는다 — 실행은 경유점을 지난 뒤 원래대로 접근점으로 이동한다.
        """
        if wrist_pose is None or switching:
            # 손목 자세를 모르거나(fake) configuration을 바꿔 들어가는 경우다. 후자는
            # 관절 이동이 시작점을 통째로 바꾸므로 여기서 직선 여부를 따질 의미가 없다.
            return "direct", [], ""
        transit_z = max(float(wrist_pose[2]),
                        float(approach_option.posx[2])) + self._transit_clearance_mm
        first_reason = ""
        for name, waypoints in motion_feasibility.routes(
                list(wrist_pose), list(approach_option.posx), transit_z):
            start = list(wrist_pose)
            blocked = ""
            for waypoint in waypoints:
                check = motion_feasibility.path_feasible(
                    start, waypoint, ik_query, space, obstacle=obstacle,
                    max_joint_jump_deg=self._max_joint_jump_deg, strict=strict)
                if not check.feasible:
                    blocked = check.reason
                    if check.endpoint_failure and name == "direct":
                        # 접근점 자체에 해가 없다는 뜻이면 우회로도 못 푼다.
                        return None, [], f"현재→접근 구간 실행 불가 — {check.reason}"
                    break
                start = list(waypoint)
            if not blocked:
                return name, [list(w) for w in waypoints[:-1]], ""
            first_reason = first_reason or blocked
        return None, [], f"현재→접근 구간이 직선·우회 모두 불가 — {first_reason}"

    def _publish_selected(self, selected, trace_id: str, observation_id: str) -> None:
        """고른 후보를 발행한다. 못 골랐으면 빈 candidate_id로 발행해 이전 선택을 지운다 —
        안 그러면 grasp 오버레이가 지난 사이클의 후보를 계속 강조한다."""
        msg = SelectedGrasp()
        msg.schema_version = SCHEMA_VERSION
        msg.trace_id = trace_id
        msg.observation_id = observation_id
        if selected is not None:
            msg.candidate.pose = selected.candidate.pose
            msg.candidate.score = float(selected.candidate.score)
            msg.candidate.gripper_width_mm = float(selected.candidate.gripper_width_mm)
            msg.candidate.candidate_id = selected.candidate.candidate_id
            msg.candidate.strategy = selected.candidate.strategy
        self._selected_pub.publish(msg)

    def _pick_real(self, goal_handle, goal, selected, wrist_pose) -> tuple[float, float] | None:
        """위치제어만으로 실물 pick을 수행한다 (grasp_pose를 그대로 믿고 움직인다.
        compliance.py는 외부토크를 진단용으로만 기록한다 — 접근을 멈추거나 파지를 판정하지
        않는다, 판정은 RG2 'Grip detected' 비트가 한다).

        grasp_pose 바로 위(approach_height_mm)에서 한 번 멈췄다 내려가 그리퍼를 닫고
        다시 들어올린다. 힘(N)은 grip_level별 max_grip_force_n으로 정확히 넣지 못한다 —
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
        # 모션 검사(STEP 2)가 다른 자세 표현으로만 IK가 풀린다고 판단했으면 **그 표현을
        # 그대로 실행한다** — 검사한 것과 다른 좌표를 명령하면 IK를 통과시킨 근거가 사라진다.
        plan = getattr(self, "_motion_plan", None) or {}
        if plan.get("approach_posx"):
            approach_posx = list(plan["approach_posx"])
            target_posx = list(plan["target_posx"])
            target_xyz = target_posx[:3]
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
                                          posx_client=self._posx_client, logger=self.get_logger(),
                                          error_monitor=self._motion_errors)

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
                    lambda: self._gripper_joint_angle, goal_handle,
                    get_status=lambda: self._gripper_status) is None:
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

        # 진단용 외부토크 기록(compliance.py) — 판정에는 안 쓴다. 서비스가 무응답이면
        # sample()이 조용히 건너뛰므로 이 사이클 전체를 막지 않는다.
        torque_trace = compliance.TorqueTrace()

        self._publish_phase(goal_handle, Pick.Feedback.PHASE_APPROACHING)
        # 다른 solution space에서만 풀리는 후보다 — movel은 configuration을 못 바꾸므로
        # 접근 자세의 관절해로 **관절 이동**해 그 configuration으로 들어간다(STEP 2).
        # 이동량은 검사에서 이미 상한(max_config_switch_jump_deg) 안임을 확인했다.
        if plan.get("switch_posj"):
            self.get_logger().warning(
                f"관절 이동으로 solution space {plan['sol_space']} 진입 "
                f"(목표 관절 {[round(v, 1) for v in plan['switch_posj']]})")
            if not dsr_motion.move_joint(self._movej_client, plan["switch_posj"],
                                         goal_handle, self._joint_vel_deg_s,
                                         self._joint_acc_deg_s2,
                                         logger=self.get_logger()):
                if goal_handle.is_cancel_requested:
                    return None
                raise RuntimeError("다른 solution space로 관절 이동 실패")
        # 직선으로 가면 걸리는 것이 확인돼 우회 경유점이 잡힌 경우다 (2026-09-10).
        # **후보를 버리는 대신 돌아간다** — 경유점은 검사에서 이미 IK·연속성·지지면을
        # 통과한 지점들이고, 직선이 되는 평소에는 이 목록이 비어 있어 예전과 같다.
        waypoint_pose = None
        for index, waypoint in enumerate(plan.get("waypoints") or []):
            previous = (list(plan["waypoints"][index - 1]) if index
                        else (list(wrist_pose) if wrist_pose else None))
            target = list(waypoint)
            if (waypoint_pose is not None and previous is not None
                    and list(waypoint)[3:6] == previous[3:6]):
                # 회전이 그대로인 구간은 **실제로 도달한 회전값**을 쓴다 — 계산값을 이어
                # 쓰면 ZYZ 표현 차이로 제자리 대신 대각선으로 보간된다(dsr_motion docstring).
                target = next_target(list(waypoint)[:3], waypoint_pose)
                if target is None:
                    if goal_handle.is_cancel_requested:
                        return None
                    raise RuntimeError("현재 자세를 읽지 못했다")
            self.get_logger().info(
                f"접근 우회 경유점 {index + 1}/{len(plan['waypoints'])} "
                f"({plan.get('route')}): ({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f})")
            waypoint_ok, waypoint_pose = move(target)
            if not waypoint_ok:
                if goal_handle.is_cancel_requested:
                    return None
                raise RuntimeError("접근 우회 경유점으로 이동 실패")

        approach_ok, approach_pose = move(approach_posx)
        if not approach_ok:
            if goal_handle.is_cancel_requested:
                return None
            raise RuntimeError("접근 위치로 이동 실패")
        torque_trace.sample(self._torque_client, "approach")

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
        if reached is None:
            raise RuntimeError("파지 위치의 실제 TCP를 읽지 못해 snapshot을 만들 수 없다")

        self._publish_phase(goal_handle, Pick.Feedback.PHASE_CONTACT_DETECTED)
        torque_trace.sample(self._torque_client, "contact")

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
        # grip_level이 0(미상)이면 가장 약하게(5)로 처리 — min_readme/FR-05b.
        force_n = self._grip_level_force_n.get(goal.grip_level or 5) or None
        close_command = dsr_motion.gripper_width_command(
            close_target_mm / 1000.0, force_n)
        self.get_logger().info(
            f"그리퍼 닫기: 목표 {close_target_mm:.1f}mm "
            f"(물체폭 {candidate_width_mm:.1f}mm x {self._grip_close_ratio:.2f}) "
            + (f"힘 {force_n:.1f}N (grip_level={goal.grip_level})" if force_n
               else f"힘 미지정 (grip_level={goal.grip_level}에 max_grip_force_n 없음)"))
        pre_evidence = self._gripper_evidence()
        close_sent = time.monotonic()
        if not dsr_motion.send_gripper_command(self._gripper_cmd_client, close_command):
            if goal_handle.is_cancel_requested:
                return None
            raise RuntimeError("그리퍼 닫기 명령 전송 실패")
        final_angle = dsr_motion.wait_gripper_settled(
            lambda: self._gripper_joint_angle, goal_handle,
            get_status=lambda: self._gripper_status)
        if final_angle is None:
            if goal_handle.is_cancel_requested:
                return None
            raise RuntimeError("그리퍼가 닫히는 동안 응답이 없다")
        torque_trace.sample(self._torque_client, "close")

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
        post_evidence = self._gripper_evidence(since=close_sent)
        closure_tcp = self._wrist_posx(goal_handle)
        if closure_tcp is None:
            raise _LiftFailedError("close 후 실제 TCP를 읽지 못해 snapshot을 만들 수 없다", width_mm)

        self._publish_phase(goal_handle, Pick.Feedback.PHASE_LIFTING)
        lift_posx = next_target(approach_xyz, descend_pose)
        if lift_posx is None:
            if goal_handle.is_cancel_requested:
                return None
            raise _LiftFailedError("현재 자세를 읽지 못했다", width_mm)
        lift_started = time.monotonic()
        lift_ok, _ = move(lift_posx)
        if not lift_ok:
            if goal_handle.is_cancel_requested:
                return None
            raise _LiftFailedError("들어올리기 실패", width_mm)
        torque_trace.sample(self._torque_client, "lift")
        lift_evidence = self._gripper_evidence(since=lift_started)
        lift_width_mm = dsr_motion.gripper_width_mm(
            self._gripper_pose_client, self._gripper_joint_angle)

        self._publish_phase(goal_handle, Pick.Feedback.PHASE_VERIFYING)
        if torque_trace.summary():
            self.get_logger().info(
                f"[COMPLIANCE] request_id={goal.request_id} 외부토크(Nm) {torque_trace.log_line()} "
                "— 진단용, 파지 판정에는 안 씀(Grip detected 비트가 판정한다)")
        # **반드시 2-튜플로 돌려준다.** execute_callback이
        # `width_mm, close_target_mm = pick_result`로 푼다 — close_target_mm은 파지 판정에
        # 필요하다('c'로 완전히 닫은 경우 Grip detected 비트를 믿으면 안 되는데, 그 구분이
        # close_target_mm <= 0 이다).
        # 2026-09-07 실물: 여기서 float 하나만 돌려주고 있어 **모든 실물 pick이** 닫기 직후
        # `cannot unpack non-iterable float object`로 실패했다. 호출부만 2개로 바꾸고
        # 반환문을 안 고친 것이었고, 타입 힌트(tuple[float, float])도 이미 2개였다.
        return width_mm, close_target_mm, list(closure_tcp[:6]), {
            "pre": pre_evidence,
            "post": post_evidence,
            "lift": lift_evidence,
            "lift_width_mm": lift_width_mm,
            "torque_trace_summary": torque_trace.summary(),
        }

    @staticmethod
    def _injected_failure(object_id: str) -> bool:
        return bool(is_fake_robot() and os.environ.get("FAKE_FAIL_OBJECT") == object_id)

    def _result(self, success, reason, started, torque=None,
                 candidate_id: str = "", source_observation_id: str = "",
                 executed_tcp=None):
        result = Pick.Result()
        result.success = success
        result.failure_reason = reason
        # 어느 후보를 실제로 실행했는지. 못 고른 경우와 grasp_pose 하나만 받은 경우는 빈 값이다.
        result.selected_candidate_id = candidate_id
        result.source_observation_id = source_observation_id
        result.has_executed_tcp_posx = executed_tcp is not None
        if executed_tcp is not None:
            result.executed_tcp_posx = [float(v) for v in executed_tcp]
        result.retries_used = 0
        result.cycle_time_ms = (time.monotonic() - started) * 1000
        result.torque_trace_summary = torque or []
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

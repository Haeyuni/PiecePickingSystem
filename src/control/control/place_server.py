"""place_into 액션 서버. bin_id → bins.yaml 좌표 변환.

참조: 인터페이스_정의서.md 4.2절 (PlaceInto.action)

`bin_id` 그라운딩을 여기서도 한 번 더 한다. planner의 검증기가 이미 확인하지만(FR-11),
control은 planner를 거치지 않는 호출(재전송, 수동 테스트, 향후 다른 클라이언트)도 받는다.
좌표를 실제로 아는 쪽에서 마지막으로 확인하는 것이 옳다.
"""
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
from dsr_msgs2.srv import (Fkin, GetCurrentPosx, GetCurrentSolutionSpace,
                           GetToolForce, Ikin)
from onrobot_rg_msgs.msg import OnRobotRGInput
from onrobot_rg_msgs.srv import SetCommand
from sensor_msgs.msg import JointState
from sort_msgs.action import PlaceInto

from . import box_geometry, compliance, dsr_motion, motion_feasibility
from .config_paths import skill_params_path
from .request_cache import RequestCache
from .robot_state_publisher import is_fake_robot, store

SCHEMA_VERSION = "1.0.0"


class _SafeTransitUnreachable(RuntimeError):
    """안전고도를 유지한 채로는 바구니 상공까지 갈 방법이 없다 (STEP 2, 2026-09-10).

    일반 실패와 구분한다 — 이건 "이동하다 실패"가 아니라 **"그 높이를 지키면서 갈 수 있는
    관절 configuration·경로가 아예 없다"**이고, 높이를 낮춰 억지로 성공시키지 않기로 한
    결과다. orchestrator/사람이 배치나 목적지를 바꿔야 풀린다.
    """
FAKE_PHASE_DURATION_S = 0.4

PHASES = [
    PlaceInto.Feedback.PHASE_MOVING,
    PlaceInto.Feedback.PHASE_INSERTING,
    PlaceInto.Feedback.PHASE_RELEASING,
    PlaceInto.Feedback.PHASE_VERIFYING,
]

def bins_yaml_path() -> pathlib.Path:
    """bins.yaml 위치.

    colcon으로 설치하면 `share/control/config/`에 놓이고, 소스에서 직접 실행하면
    패키지 옆 `config/`에 있다. 설치 경로를 먼저 보고 없으면 소스로 떨어진다 —
    `__file__` 기준 상대경로만 쓰면 설치본에서 조용히 빈 목적지 목록이 된다.
    """
    override = os.environ.get("BINS_YAML")
    if override:
        return pathlib.Path(override)

    try:
        from ament_index_python.packages import get_package_share_directory

        installed = pathlib.Path(get_package_share_directory("control")) / "config" / "bins.yaml"
        if installed.is_file():
            return installed
    except Exception:
        pass

    return pathlib.Path(__file__).resolve().parents[1] / "config" / "bins.yaml"


def load_bins(path: pathlib.Path | None = None) -> dict:
    """bins.yaml의 목적지 정의. 좌표 단위는 mm (인터페이스_정의서 1.1절)."""
    path = path or bins_yaml_path()
    with path.open(encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("bins") or {}


def load_skill_params(path: pathlib.Path | None = None) -> dict:
    path = path or skill_params_path()
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_motion_params(path: pathlib.Path | None = None) -> dict:
    return load_skill_params(path).get("motion") or {}


class PlaceServer(Node):
    def __init__(self):
        super().__init__('place_server')
        self._cache = RequestCache()
        try:
            self._bins = load_bins()
        except FileNotFoundError:
            self.get_logger().error(f"bins.yaml을 찾을 수 없습니다: {bins_yaml_path()}")
            self._bins = {}

        callbacks = ReentrantCallbackGroup()
        self._action_server = ActionServer(
            self, PlaceInto, 'place_into', self.execute_callback,
            goal_callback=lambda _: GoalResponse.ACCEPT,
            cancel_callback=self._cancel_callback,
            callback_group=callbacks,
        )

        motion = load_motion_params()
        self._approach_height_mm = float(motion.get("approach_height_mm", 80.0))
        # place만 pick과 다른 속도를 쓸 수 있도록 place_* 키를 우선 보고, 없으면 공용값으로
        # 떨어진다 — pick_server.py는 이 place_* 키를 보지 않으므로 pick 속도는 그대로다.
        self._linear_vel_mm_s = float(motion.get("place_linear_vel_mm_s",
                                                 motion.get("linear_vel_mm_s", 30.0)))
        self._linear_acc_mm_s2 = float(motion.get("place_linear_acc_mm_s2",
                                                  motion.get("linear_acc_mm_s2", 30.0)))
        self._rot_vel_deg_s = float(motion.get("place_rot_vel_deg_s",
                                               motion.get("rot_vel_deg_s", 20.0)))
        self._rot_acc_deg_s2 = float(motion.get("place_rot_acc_deg_s2",
                                                motion.get("rot_acc_deg_s2", 20.0)))
        self._gripper_open_m = float(motion.get("gripper_open_m", 0.110))
        # bin_pose는 빈 그리퍼로 티칭한 높이라, 물체를 든 채로 그 z까지 그대로 내려가면
        # 물체(또는 그리퍼)가 바구니 바닥/기존 내용물에 먼저 닿는다 — 실물에서 충돌로
        # 안전모드에 들어간 사고가 있었다. 그 여유만큼 목표 z를 올려서 내려간다.
        self._release_clearance_mm = float(motion.get("release_clearance_mm", 30.0))
        # TCP 원점 → 손가락이 실제로 만나는 지점 (툴 좌표계 mm). pick과 **같은 값**을 써야
        # 한다 — orchestrator가 넘기는 object_bottom_offset_mm이 grasp_pose(=파지점) 기준
        # 값이라, 그걸 TCP 기준으로 옮기려면 이 오프셋이 필요하다. 아래 release_z 참조.
        self._grasp_center_offset_mm = dsr_motion.grasp_center_offset_mm(load_skill_params())
        self._place_safe_clearance_mm = float(motion.get("place_safe_clearance_mm", 100.0))
        # place_into 성공 뒤 orchestrator가 곧바로 home(관절이동)을 부른다(web/orchestrator.py).
        # 물러난 높이(approach 지점)에서 바로 관절이동을 시작하면 바구니 테두리 바로 위라
        # 여유가 적어, 그만큼 더 올라간 뒤에 home으로 넘어가도록 한다.
        self._home_rise_mm = float(motion.get("home_rise_mm", 200.0))
        # STEP 2 이송 경로. 위치와 회전을 한 movel에 같이 크게 바꾸면 로봇이 아예 움직이지
        # 않은 적이 있어(아래 ②/③ 분리 주석), direct는 회전량이 이 값 이하일 때만 쓴다.
        self._direct_rotation_limit_deg = float(
            motion.get("place_direct_rotation_limit_deg", 20.0))
        # 이송에서 실제로 실행해 볼 (경로 x solution space) 조합 수. 실패 한 번이 최대
        # 60초라 무한정 돌 수 없다 — 사전 검사를 통과한 조합만 시도하므로 보통 1회로 끝난다.
        self._transit_attempts = int(motion.get("place_transit_attempts", 3))
        self._max_config_switch_jump_deg = float(
            motion.get("place_max_config_switch_jump_deg", 90.0))
        # box geometry 모드에서 한 자세(yaw)가 바구니엔 들어가지만 안전 이송 자체가
        # 8개 solution space 어디로도 안 되면, 그 각을 skip_yaws에 넣고 box_geometry가
        # 다음으로 잘 맞는 각을 고르게 해 다시 계획한다(STEP 3, 2026-09-10 실물 —
        # 파지 자세 그대로는 全 space에서 이송 불가였던 사례). legacy_taught_pose는
        # 시도할 다른 자세가 없으므로 이 값과 무관하게 항상 1회다.
        self._yaw_retry_attempts = int(motion.get("place_yaw_retry_attempts", 4))
        home = load_skill_params().get("home") or {}
        self._joint_vel_deg_s = float(
            motion.get("place_config_switch_vel_deg_s", home.get("vel_deg_s", 20.0)))
        self._joint_acc_deg_s2 = float(
            motion.get("place_config_switch_acc_deg_s2", home.get("acc_deg_s2", 20.0)))
        self._joint_limits_deg = self._JOINT_LIMITS_DEG

        # 순응 하강(compliance.py). 목표까지 **한 번의 movel로** 내려가면서 ContactWatch가
        # 힘을 보고, 임계값을 넘으면 그 자리에서 goal을 취소해 멈춘다(2026-09-09 구조 변경 —
        # 예전에는 place_descent_step_mm로 쪼개 스텝 사이에만 확인했다).
        params = load_skill_params()
        compliance_cfg = params.get("compliance") or {}
        self._place_descent_enabled = bool(compliance_cfg.get("place_descent_enabled", False))
        # 이 구간만 느리게 간다. **오버슈트가 속도에 그대로 비례하기 때문이다** — 접촉을
        # 알아채고 멈추기까지 (확인 주기 + 취소 왕복 + 감속)만큼은 더 내려간다. place의
        # 공용 속도(place_linear_vel_mm_s, 90mm/s)로 내려가면 확인 주기 0.1초만으로도
        # 9mm를 더 눌러 버린다.
        self._place_descent_vel_mm_s = float(
            compliance_cfg.get("place_descent_vel_mm_s", 30.0))
        self._place_descent_poll_s = float(compliance_cfg.get("place_descent_poll_s", 0.1))
        # 접촉 임계값(N) — place_into는 물체별 grip_level을 받지 않으므로(PlaceInto.action
        # 참조) profile/grip_level별로 나누지 않고 고정값 하나만 쓴다. 0/미설정이면
        # 순응 하강을 건너뛴다(임계값 없이 "접촉했다"고 판단할 기준이 없다).
        self._place_contact_threshold_n = float(compliance_cfg.get("contact_threshold_n", 0.0) or 0.0)
        self._movel_client = ActionClient(self, MovelH2r, dsr_motion.MOVEL_ACTION,
                                          callback_group=callbacks)
        self._movej_client = ActionClient(self, MovejH2r, dsr_motion.MOVEJ_ACTION,
                                          callback_group=callbacks)
        # 컨트롤러가 "이 목표는 못 간다"고 내는 알람을 지켜본다 — 없으면 movel이
        # goal을 accept한 채 아무것도 안 하는 경우가 60초 타임아웃을 다 채운다
        # (dsr_motion.MotionErrorMonitor 참조).
        self._motion_errors = dsr_motion.MotionErrorMonitor(self, callbacks)
        self._ikin_client = self.create_client(
            Ikin, "/dsr01/dsr_controller2/motion/ikin", callback_group=callbacks)
        # STEP 2: 안전고도를 고정한 채 다른 관절 configuration을 찾으려면 ikin 왕복 검증
        # (fkin)과 현재 solution space가 필요하다.
        self._fkin_client = self.create_client(
            Fkin, dsr_motion.FKIN_SERVICE, callback_group=callbacks)
        self._solution_space_client = self.create_client(
            GetCurrentSolutionSpace, dsr_motion.GET_CURRENT_SOLUTION_SPACE_SERVICE,
            callback_group=callbacks)
        self._posx_client = self.create_client(
            GetCurrentPosx, dsr_motion.GET_CURRENT_POSX_SERVICE, callback_group=callbacks)
        self._gripper_cmd_client = self.create_client(
            SetCommand, dsr_motion.GRIPPER_COMMAND_SERVICE, callback_group=callbacks)
        # 순응 하강(compliance.ContactWatch)용 — place_descent_enabled가 꺼져 있으면
        # 이 클라이언트는 만들어지되 쓰이지 않는다.
        self._tool_force_client = self.create_client(
            GetToolForce, dsr_motion.GET_TOOL_FORCE_SERVICE, callback_group=callbacks)
        self._gripper_joint_angle: float | None = None
        self.create_subscription(JointState, dsr_motion.GRIPPER_JOINT_STATES_TOPIC,
                                 self._on_gripper_state, 5, callback_group=callbacks)
        # RG2가 보고하는 동작 상태. wait_gripper_settled가 gsta bit0(Busy)로 "다 열렸다"를
        # 판단한다 — 관절각 정지로 추측하면 명령 시작 전 정지 구간을 완료로 오독한다
        # (dsr_motion.wait_gripper_settled 참조, 2026-09-08 실물). 여기서 안 넘기면
        # place의 열기도 pick의 닫기와 같은 조기 종료를 겪는다: 아직 덜 열린 채로
        # release가 끝났다고 보고 물러나면 물체가 손가락에 끌려간다.
        self._gripper_status: tuple[float, int, float] | None = None
        self.create_subscription(OnRobotRGInput, dsr_motion.GRIPPER_STATUS_TOPIC,
                                 self._on_gripper_status, 5, callback_group=callbacks)

        self.get_logger().info(
            f"place_into 액션 서버 준비 (목적지 {list(self._bins)}, "
            f"{'fake' if is_fake_robot() else '실물'} 모드)")

    def _on_gripper_state(self, msg: JointState) -> None:
        if msg.position:
            self._gripper_joint_angle = msg.position[0]

    def _on_gripper_status(self, msg: OnRobotRGInput) -> None:
        self._gripper_status = (time.monotonic(), int(msg.gsta), float(msg.gwdf) / 10.0)

    def _cancel_callback(self, goal_handle):
        self.get_logger().warning("place_into 취소 요청 수신")
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        goal = goal_handle.request

        cached = self._cache.get(goal.request_id)
        if cached is not None:
            self.get_logger().warning(
                f"중복 request_id={goal.request_id} — 재실행하지 않고 이전 결과 반환")
            goal_handle.succeed()
            return cached

        if goal.bin_id not in self._bins:
            self.get_logger().error(f"등록되지 않은 bin_id: {goal.bin_id}")
            goal_handle.abort()
            return self._result(False, PlaceInto.Result.REASON_UNREACHABLE, time.monotonic())

        bin_spec = self._bins[goal.bin_id]
        bin_pose = bin_spec.get("pose", {})
        # 놓는 높이 여유는 목적지마다 다를 수 있다 — bins.yaml 의 해당 목적지에
        # release_clearance_mm 이 있으면 그것을, 없으면 skill_params.yaml 의 공용값을 쓴다.
        # 바구니 깊이·기존 내용물 높이가 목적지마다 다르므로 공용값 하나로 맞추면
        # 한쪽은 너무 높아 떨어뜨리고 다른 쪽은 너무 낮아 부딪힌다.
        release_clearance_mm = float(
            bin_spec.get("release_clearance_mm", self._release_clearance_mm))
        # **물체가 TCP 아래로 내려와 있는 만큼을 따로 더한다.** bin_pose는 빈 그리퍼로
        # 티칭한 높이라, 물체를 든 채 그 높이까지 내려가면 물체가 바구니 바닥에 먼저 닿는다.
        # 그러면 로봇은 목표 z에 도달하지 못한 채 계속 밀어붙이다 안전모드로 들어간다
        # (2026-09-06 실물: 목표 213.4mm인데 217.9mm에서 멈춘 뒤 알람 → heartbeat 유실
        # → 권한 회수로 파이프라인 전체가 멈췄다).
        #
        # 이 값은 web/orchestrator가 perception의 실측 물체 높이(DetectedObject.height_mm)와
        # 실제로 쓴 파지 z로 계산해 goal에 실어 보낸다. 0이면 미상 — 그때만 예전처럼
        # release_clearance_mm 하나로 버틴다(하위호환).
        #
        # 이렇게 나누면 release_clearance_mm이 "물체 높이 짐작"이 아니라 **순수한 안전
        # 여유**가 된다 — 물체가 커지면 자동으로 더 높은 데서 놓고, 여유값은 그대로 둔다.
        geometry_mode = "inner_corners_base_mm" in bin_spec
        footprint = None
        if geometry_mode:
            footprint = [(point.x, point.y) for point in goal.object_footprint_base_mm.points]
            if (not goal.has_pick_snapshot or "floor_point_base_mm" not in bin_spec
                    or "wall_margin_mm" not in bin_spec):
                self.get_logger().error(
                    "box geometry 설정 또는 frozen PickSnapshot이 불완전해 place를 중단한다")
                goal_handle.abort()
                return self._result(False, PlaceInto.Result.REASON_PLACE_FAILED, time.monotonic())

        store.set_busy("place_into")
        started = time.monotonic()
        # 바구니엔 들어가지만 그 자세로는 안전 이송이 8개 solution space 어디로도 안 된
        # 것으로 확인된 yaw들. box_geometry가 다음 후보를 고를 때 이 각들을 건너뛴다
        # (STEP 3, 2026-09-10 — legacy_taught_pose는 시도할 다른 자세가 없어 1회뿐이다).
        tried_yaws: set = set()
        max_yaw_attempts = self._yaw_retry_attempts if geometry_mode else 1

        try:
            for yaw_attempt in range(1, max_yaw_attempts + 1):
                plan = None
                if geometry_mode:
                    try:
                        box = box_geometry.box_geometry_from_measurements(
                            inner_corners_base_mm=bin_spec["inner_corners_base_mm"],
                            floor_point_base_mm=bin_spec["floor_point_base_mm"],
                        )
                        plan = box_geometry.plan_box_place(
                            box=box,
                            wall_margin_mm=float(bin_spec["wall_margin_mm"]),
                            release_clearance_mm=release_clearance_mm,
                            pickup_tcp_posx=list(goal.pickup_tcp_posx),
                            footprint_xy=footprint,
                            tcp_to_object_bottom_mm=float(goal.tcp_to_object_bottom_mm),
                            skip_yaws=tried_yaws,
                        )
                    except (KeyError, TypeError, ValueError) as exc:
                        self.get_logger().error(f"box geometry place 거부: {exc}")
                        goal_handle.abort()
                        return self._result(False, PlaceInto.Result.REASON_PLACE_FAILED, started)
                    target_posx = list(plan.target_tcp_posx)
                    release_z = target_posx[2]
                    detail = (f"geometry floor {box.floor_z_mm:.1f} + TCP→물체바닥 "
                              f"{goal.tcp_to_object_bottom_mm:.1f}mm + 여유 {release_clearance_mm:.1f}mm")
                    self.get_logger().info(
                        f"[PlacePlan] mode=box_geometry 자세시도={yaw_attempt}/{max_yaw_attempts} "
                        f"obs={goal.source_observation_id} "
                        f"object={goal.object_id} bin={goal.bin_id} target="
                        f"{[round(v, 2) for v in target_posx]} footprint_points={len(footprint)} "
                        f"center={tuple(round(v, 2) for v in box.center_base_mm)} "
                        f"width={box.width_mm:.1f} depth={box.depth_mm:.1f} yaw={box.yaw_deg:.1f}deg "
                        # 파지 자세 그대로 넣는지, 돌려서 넣는지 — 돌린 경우 손목이 그만큼 더
                        # 움직이므로(안전고도에서의 제자리 회전) 사고 분석 때 구분이 필요하다.
                        f"place_rotation={plan.yaw_deg:+.1f}deg")
                else:
                    bottom_offset_mm = max(
                        0.0, float(getattr(goal, "object_bottom_offset_mm", 0.0) or 0.0))
                    # Legacy pose의 물체 높이 보정은 파지점 기준이므로 TCP↔modeled reference를
                    # 변환한다. Geometry mode는 실제 pickup TCP snapshot을 써서 이 모델값을 쓰지 않는다.
                    tcp_to_finger_z_mm = 0.0
                    if bottom_offset_mm > 0.0:
                        tcp_to_finger_z_mm = dsr_motion.grasp_center_from_posx(
                            dsr_motion.bin_pose_to_posx(bin_pose), self._grasp_center_offset_mm)[2] \
                            - float(bin_pose.get("z", 0.0))
                    release_z = (float(bin_pose.get("z", 0.0)) + bottom_offset_mm
                                 + release_clearance_mm - tcp_to_finger_z_mm)
                    target_posx = dsr_motion.bin_pose_to_posx(bin_pose)
                    target_posx[2] = release_z
                    if bottom_offset_mm > 0.0:
                        detail = (f"물체높이보정 {bottom_offset_mm:.1f}mm + 여유 "
                                  f"{release_clearance_mm:.1f}mm - TCP↔손끝 "
                                  f"{tcp_to_finger_z_mm:+.1f}mm")
                    else:
                        detail = (f"여유 {release_clearance_mm:.1f}mm "
                                  "(물체높이 미상 — 고정 여유만 사용)")
                    self.get_logger().info(
                        f"[PlacePlan] mode=legacy_taught_pose object={goal.object_id} bin={goal.bin_id} "
                        "reason=inner_corners_base_mm 미실측")

                self.get_logger().info(
                    f"place_into 시작 object={goal.object_id} bin={goal.bin_id} "
                    f"target=({target_posx[0]:.1f}, {target_posx[1]:.1f}, {target_posx[2]:.1f})mm "
                    f"놓는높이={release_z:.1f}mm ({detail})")

                try:
                    if is_fake_robot():
                        for phase in PHASES:
                            if goal_handle.is_cancel_requested:
                                goal_handle.canceled()
                                return self._result(False, PlaceInto.Result.REASON_NO_CONTACT, started)
                            self._publish_phase(goal_handle, phase)
                            time.sleep(FAKE_PHASE_DURATION_S)
                    else:
                        if self._place_real(goal_handle, target_posx) is None:
                            goal_handle.canceled()
                            return self._result(False, PlaceInto.Result.REASON_NO_CONTACT, started)

                    store.set_gripper(width_mm=0.0, closed=False)
                    result = self._result(True, PlaceInto.Result.REASON_NONE, started)
                    self._cache.put(goal.request_id, result)
                    goal_handle.succeed()
                    return result

                except _SafeTransitUnreachable as e:
                    if geometry_mode and yaw_attempt < max_yaw_attempts:
                        # 물체는 아직 그리퍼에 안전하게 들려 있다(놓기 전 실패라서다) —
                        # 바구니에 들어가는 다음 자세로 처음부터 다시 계획한다. `_place_real`이
                        # 매번 그 시점의 실제 자세를 다시 읽으므로(get_current_posx) 이 실패로
                        # 팔이 어디에 멈춰 있었든 그 자리를 새 시작점으로 쓴다.
                        tried_yaws.add(plan.yaw_deg)
                        self.get_logger().warning(
                            f"이 자세(yaw={plan.yaw_deg:+.1f}deg)로는 안전 이송 불가 — 바구니에 "
                            f"들어가는 다음 자세로 다시 계획한다 ({yaw_attempt}/{max_yaw_attempts}): {e}")
                        continue
                    # **높이를 낮춰 억지로 성공시키지 않는다** (STEP 2 §6/§10). 물체를 든 채
                    # 작업대·테두리를 넘으라고 둔 높이를 IK 사정으로 깎으면 그 높이를 둔 이유가
                    # 사라진다. 시도할 다른 자세도 더 없다.
                    self.get_logger().error(f"place_into 실패(안전 이송 불가): {e}")
                    store.set_error()
                    goal_handle.abort()
                    return self._result(
                        False, PlaceInto.Result.REASON_SAFE_TRANSIT_UNREACHABLE, started)
                except Exception as e:
                    self.get_logger().error(f"place_into 실패: {e}")
                    store.set_error()
                    goal_handle.abort()
                    return self._result(False, PlaceInto.Result.REASON_PLACE_FAILED, started)
            # 여기 도달하면 안 된다 — 마지막 시도(yaw_attempt == max_yaw_attempts)는 항상 위에서
            # 반환한다. max_yaw_attempts가 0 이하로 설정되는 잘못된 config만 대비한 방어값이다.
            return self._result(
                False, PlaceInto.Result.REASON_SAFE_TRANSIT_UNREACHABLE, started)
        finally:
            if store.snapshot()["mode"] != "error":
                store.set_idle()

    # M0609 관절 한계(도). robot_description에서 읽은 값 — J3만 ±150이고 나머지는 ±360.
    _JOINT_LIMITS_DEG = (360.0, 360.0, 150.0, 360.0, 360.0, 360.0)

    def _reachable_rise_z(self, xy, rot, desired_z: float, floor_z: float) -> float:
        """**놓기를 마친 뒤** home으로 넘어가기 전 상승 높이. 못 올라가면 낮춘다.

        **안전 이송고도(transit_z)와 혼동하면 안 된다.** 저쪽은 물체를 든 채 작업대·테두리를
        넘는 높이라 IK 사정으로 낮추지 않는다(STEP 2 §6). 여기는 이미 물체를 놓은 뒤라
        낮춰도 물체가 걸릴 일이 없고, 목적은 movej로 넘어갈 때의 여유 확보뿐이다.

        각 높이를 solution space 0~7로 확인한다 — 현재 configuration에서 못 가도 다른
        configuration이면 되는 경우가 있어서다. 무응답이면 그대로 진행한다(모르면 막지 않음).
        """
        spaces = motion_feasibility.solution_space_order(
            dsr_motion.current_solution_space(self._solution_space_client))
        z = float(desired_z)
        while z > floor_z:
            found = motion_feasibility.find_ik(
                [([xy[0], xy[1], z, *rot], "original")],
                lambda posx, space: dsr_motion.verify_ik(
                    self._ikin_client, self._fkin_client, posx,
                    sol_space=space, limits=self._joint_limits_deg),
                spaces)
            if found is not None:
                break
            z = max(floor_z, z - 20.0)
            self.get_logger().info(f"home 전 상승이 팔 범위 밖 — {z:.0f}mm로 낮춘다")
        return z

    def _transit_plan(self, current_pose, target_posx, transit_z: float,
                      skip: set | None = None):
        """**고정된 안전고도에서** 바구니 상공까지 갈 방법을 찾는다 (STEP 2, 2026-09-10).

        예전에는 IK가 안 풀리면 안전고도를 20mm씩 낮춰 가며 재시도했다. 그 높이는 이송 중
        물체가 작업대·바구니 테두리를 넘어가라고 둔 값이라, IK 사정으로 깎으면 그 높이를
        둔 이유 자체가 사라진다. **이제 높이는 고정하고 대신 두 가지를 바꿔 가며 찾는다.**

        1. **solution space** — 현재 것부터, 그다음 드라이버가 지원하는 나머지(0~7).
           같은 [x, y, safe_z, 자세]를 다른 관절 configuration으로 가는 것이다.
        2. **경로** — direct 우선, 안 되면 상승→회전→이동 / 상승→이동→회전.

        **손목을 뒤집지 않는다.** pick 뒤에는 물체를 들고 있어서, RG2가 대칭이라는 이유로
        TCP 자세를 180도 뒤집으면 물체의 방향이 함께 바뀐다 — 같은 작업이 아니다
        (dsr_motion.flipped_wrist_posx 주석 참조).

        반환: 실행 계획 dict 또는 None(= 어떤 조합으로도 불가). 다른 space를 쓸 때는
        Safe-Z의 동일 TCP 지점에서 configuration을 바꿀 관절해와 시점도 함께 반환한다.
        """
        current_space = dsr_motion.current_solution_space(self._solution_space_client)
        spaces = motion_feasibility.solution_space_order(current_space)
        rotation_span = motion_feasibility.rotation_span_deg(
            list(current_pose)[3:6], list(target_posx)[3:6], dsr_motion.rotation_diff_deg)
        # 이송 구간에서 작업대 아래로 내려가는 경로는 명백한 충돌이다. 바닥을 모르면
        # (측정 없음) 이 검사는 아예 만들어지지 않는다 — 모르는 것을 막지는 않는다.
        obstacle = motion_feasibility.support_plane_obstacle(
            min(float(current_pose[2]), float(target_posx[2])))

        def ik_query(posx, space):
            return dsr_motion.verify_ik(self._ikin_client, self._fkin_client, posx,
                                        sol_space=space, limits=self._joint_limits_deg)

        # **두 번 훑는다 (2026-09-10 실물).** 1차는 중간 보간 지점까지 전부 깨끗한 경로만
        # 받고, 아무것도 없으면 2차에서 **끝점만** 확실한 경로를 받는다.
        #
        # 그날 place는 경로 3개 x space 8개를 모두 "경로 중간 지점의 IK 실패"로 버리고
        # 로봇이 아예 움직이지 않았다 — 사람이 보기엔 그냥 멈춰 선 것이다. 중간 지점은
        # 직선 보간 가정 위에서 space까지 강제해 물어본 값이라 그렇게까지 믿을 것이
        # 못 되고(motion_feasibility.path_feasible의 strict 설명), 릴리스 동작은 애초에
        # 끝점만 봤다. 그러니 중간 지점 판정은 **더 나은 경로를 고르는 선호**로 쓰고,
        # 없다고 해서 움직이지 못하게 만들지는 않는다. 그래도 못 가면 그때가 진짜
        # safe_transit_unreachable이고, 실행에서 거부당하면 호출부가 다음 조합으로 넘어간다.
        for strict in (True, False):
            plan = self._transit_search(current_pose, target_posx, transit_z, spaces,
                                        rotation_span, obstacle, ik_query, skip, strict)
            if plan is not None:
                return plan
        return None

    def _transit_search(self, current_pose, target_posx, transit_z, spaces,
                        rotation_span, obstacle, ik_query, skip, strict: bool):
        """`_transit_plan`의 한 번 훑기. strict의 뜻은 path_feasible 설명 참조."""
        current_space = spaces[0]
        for name, waypoints in motion_feasibility.routes(current_pose, target_posx, transit_z):
            if name == "direct":
                # **direct는 이미 안전고도 위에 있을 때만 쓴다 (2026-09-10 실물에서 발견).**
                # 그 아래에서 direct로 가면 출발부터 대각선으로 움직여 **안전고도 아래에서
                # 수평 이동이 시작된다** — 물체를 든 채 작업대·테두리를 넘으라고 둔 높이인데
                # 그 구간을 건너뛰는 셈이라, 높이를 고정한 의미가 사라진다.
                # (실물 로그: 현재 378.8 → 안전고도 428.8인데 direct가 채택돼 대각선으로
                #  갔다. 회전량이 0이라 아래 회전 가드만으로는 못 걸렀다.)
                if float(current_pose[2]) < float(transit_z) - 1.0:
                    self.get_logger().info(
                        f"place 이송 경로 'direct' 제외 — 현재 {float(current_pose[2]):.1f}mm가 "
                        f"안전고도 {float(transit_z):.1f}mm보다 낮아 대각선 이동이 그 아래를 지난다")
                    continue
                # **위치와 회전을 한 movel에 같이 크게 바꾸지 않는다.** 그렇게 묶었더니
                # 로봇이 아예 움직이지 않은 적이 있다(이 파일 상단 ②/③ 분리 주석).
                if rotation_span is None or rotation_span > self._direct_rotation_limit_deg:
                    continue
            for space in spaces:
                if skip and (name, space) in skip:
                    continue   # 이미 실행에서 실패한 조합 — 같은 것을 또 시도하지 않는다
                start = list(current_pose)
                infeasible = ""
                soft = 0
                switch_before = None
                switch_posj = None
                path_start = 0
                if space != current_space:
                    # movel은 현재 solution space를 유지한다. 다른 space의 IK가 풀린다는
                    # 사실만 확인하고 같은 posx를 명령하면 실제 로봇은 그 space로 가지 않는다.
                    # 먼저 현재 configuration으로 Safe-Z의 동일 TCP 지점까지 올라간 뒤,
                    # 그 안전 지점에서 검증된 관절해로 configuration을 바꾼다.
                    if (not waypoints or float(waypoints[0][2]) < float(transit_z) - 1.0
                            or list(waypoints[0])[:2] != list(current_pose)[:2]
                            or list(waypoints[0])[3:6] != list(current_pose)[3:6]):
                        infeasible = "다른 configuration으로 바꿀 Safe-Z 수직 상승 지점이 없다"
                    else:
                        rise_check = motion_feasibility.path_feasible(
                            start, waypoints[0], ik_query, current_space,
                            obstacle=obstacle, strict=strict)
                        soft += rise_check.soft_failures
                        if not rise_check.feasible:
                            infeasible = f"configuration 전환 전 상승 불가 — {rise_check.reason}"
                        else:
                            before = ik_query(waypoints[0], current_space)
                            after = ik_query(waypoints[0], space)
                            jump = motion_feasibility.joint_jump_deg(
                                getattr(before, "posj", None), getattr(after, "posj", None))
                            if not getattr(after, "ok", False):
                                infeasible = "Safe-Z 전환 지점의 다른 configuration IK 실패"
                            else:
                                # **관절 이동량이 아니라 실제로 지나는 지점으로 판단한다**
                                # (STEP 3, 2026-09-10 — motion_feasibility.config_switch_feasible
                                # 참조). 각도 크기만으로 거르면 solution space 전환에 정상적으로
                                # 필요한 큰 폭(그날 실측 168~257도)까지 함께 막힌다.
                                switch_check = motion_feasibility.config_switch_feasible(
                                    getattr(before, "posj", None), getattr(after, "posj", None),
                                    lambda joint: dsr_motion.posx_via_fkin(
                                        self._fkin_client, joint),
                                    obstacle=obstacle)
                                soft += switch_check.soft_failures
                                if not switch_check.feasible:
                                    infeasible = (f"configuration 전환 경로가 명백한 충돌 — "
                                                  f"{switch_check.reason}")
                                else:
                                    if jump is not None and jump > self._max_config_switch_jump_deg:
                                        self.get_logger().info(
                                            f"configuration 전환에 관절이 {jump:.0f}도 움직인다 "
                                            f"(참고 한계 {self._max_config_switch_jump_deg:.0f}도) — "
                                            "전환 경로에 명백한 충돌은 없어 그대로 진행한다")
                                    switch_before = 1
                                    switch_posj = list(after.posj)
                                    start = list(waypoints[0])
                                    path_start = 1
                for waypoint in waypoints[path_start:]:
                    if infeasible:
                        break
                    check = motion_feasibility.path_feasible(
                        start, waypoint, ik_query, space, obstacle=obstacle, strict=strict)
                    soft += check.soft_failures
                    if not check.feasible:
                        infeasible = check.reason
                        break
                    start = list(waypoint)
                if not infeasible:
                    self.get_logger().info(
                        f"place 이송 경로 '{name}' 채택 (solution space {space}, "
                        f"안전고도 {transit_z:.1f}mm 고정, 회전량 "
                        f"{'미상' if rotation_span is None else f'{rotation_span:.0f}deg'})"
                        + ("" if soft == 0 else
                           f" — 끝점은 확실하지만 중간 보간 지점 {soft}곳의 IK가 안 풀렸다"
                           "(직선 보간 가정이라 실패로 치지 않는다. 실행에서 거부되면 "
                           "다음 경로로 넘어간다)"))
                    return {
                        "name": name,
                        "waypoints": [list(w) for w in waypoints],
                        "sol_space": space,
                        "switch_before": switch_before,
                        "switch_posj": switch_posj,
                    }
                self.get_logger().info(
                    f"place 이송 경로 '{name}' space {space} 불가 — {infeasible}"
                    + ("" if strict else " (끝점 기준)"))
        return None

    @staticmethod
    def _publish_phase(goal_handle, phase) -> None:
        feedback = PlaceInto.Feedback()
        feedback.phase = phase
        goal_handle.publish_feedback(feedback)

    def _place_real(self, goal_handle, target_posx) -> bool | None:
        """계획된 TCP posx로 위치제어 place를 수행한다.

        geometry mode에서는 frozen pick orientation과 box 계산 좌표이고, legacy mode에서는
        기존 taught bin pose에 높이 보정을 더한 좌표다.
        성공 True, 취소 None, 그 외 실패는 RuntimeError.

        순응 하강 여부는 self._place_descent_enabled/self._place_contact_threshold_n
        (skill_params.yaml compliance 블록의 고정값)이 정한다 — 켜져 있고 임계값이
        설정돼 있을 때만 마지막 하강(바구니 접근 높이 → 목표 높이)을 **한 번의 movel로
        내려가면서** GetToolForce로 접촉을 감시하고, 넘으면 그 자리에서 멈춘다
        (descend_compliant). 나머지 구간(안전고도 이동, 물러나기)은 그대로 위치제어다 —
        접촉 위험이 있는 구간은 바구니 안으로 들어가는 그 한 구간뿐이다.
        """
        target_posx = list(target_posx)
        approach_posx = list(target_posx)
        approach_posx[2] += self._approach_height_mm
        target_xyz = target_posx[:3]
        approach_xyz = approach_posx[:3]

        def move(pos, vel_mm_s=None, stop_when=None, stop_poll_s=0.1):
            """`vel_mm_s`를 주면 그 이동만 다른 속도로 간다(순응 하강만 느리게 쓴다).
            `stop_when`은 이동 도중 로봇을 세우는 훅이다 — dsr_motion.move_linear 참조."""
            return dsr_motion.move_linear(self._movel_client, pos, goal_handle,
                                          vel_mm_s if vel_mm_s is not None
                                          else self._linear_vel_mm_s,
                                          self._linear_acc_mm_s2,
                                          self._rot_vel_deg_s, self._rot_acc_deg_s2,
                                          posx_client=self._posx_client, logger=self.get_logger(),
                                          error_monitor=self._motion_errors,
                                          stop_when=stop_when, stop_poll_s=stop_poll_s)

        def next_target(xyz, last_pose):
            """`xyz`로 위치만 바꾸고 회전은 유지한다 (dsr_motion.py 모듈 docstring —
            ZYZ 특이점 참조).

            회전값은 가능하면 get_current_posx로 정지 후 새로 읽은 값을 쓴다. action
            feedback의 마지막 샘플(`last_pose`)은 스트리밍 도중 값이라, 컨트롤러가 정지
            후 재정규화하는 ZYZ 표현과 파라미터가 다를 수 있다(같은 문서, "물리적으로는
            같은 방향인데 파라미터값이 다를 수 있다" 참조). 그 값을 다음 이동에 그대로
            쓰면 "제자리에서 수직으로" 대신 두 자세 사이를 새로 보간해 대각선/최단거리로
            움직이는 문제가 있었다(2026-09-05 실물 확인 — retreat이 특히 두드러졌다).

            movel 직후 aux_control이 10~20초씩 무응답일 수 있다는 걸 이미 알고 있으므로
            (get_current_posx 참조), 여기서 재시도까지 하며 오래 기다리는 건 거의 항상
            헛수고다 — next_target은 정의상 매번 move() 직후, 즉 그 무응답 구간 한복판에서
            불린다. 그래서 짧게 한 번만 찔러보고(재시도 없음), 안 되면 바로 feedback 값으로
            대체한다(2026-09-06, 스텝 전환 지연 조사 — 재시도 포함 최대 2.5초였던 걸
            0.5초로 줄임. place 한 번에 이 호출이 여러 번 있어 누적 효과가 크다)."""
            settled = dsr_motion.get_current_posx(self._posx_client, goal_handle,
                                                  timeout_s=0.5, retries=1)
            if settled is not None:
                return [xyz[0], xyz[1], xyz[2], settled[3], settled[4], settled[5]]
            if last_pose is None:
                return None
            return [xyz[0], xyz[1], xyz[2], last_pose[3], last_pose[4], last_pose[5]]

        def descend_compliant(xy, start_pose, target_z, threshold_n):
            """`start_pose`의 z에서 `target_z`까지 **한 번의 movel로 쭉 내려가면서**
            GetToolForce로 접촉을 감시한다. 임계값을 넘으면 그 자리에서 movel을 취소해
            멈추고 더 내려가지 않는다 — 깨지기 쉬운 물체를 계산된 목표까지 위치제어로
            밀어붙이지 않기 위해서다(compliance.py 모듈 docstring 참조).

            **2026-09-09에 스텝 방식을 걷어냈다.** 예전에는 5mm씩 쪼개 movel을 여러 번
            보내고 스텝 사이에서만 힘을 봤다 — 멈출 수 있는 지점이 스텝 경계뿐이라
            사실상 5mm 단위 위치제어였고, 80mm 하강에 movel이 16번 나가 그만큼 느렸다.
            지금은 `stop_when`으로 이동 도중에 멈춘다(dsr_motion.call_action_blocking).

            **오버슈트는 남는다.** 접촉을 알아채고 로봇이 실제로 서기까지
            (확인 주기 + 취소 왕복 + 감속)만큼은 더 내려간다 — 그래서 이 구간만
            `place_descent_vel_mm_s`로 느리게 간다. 스텝 방식에도 같은 한계가 있었고
            (스텝 경계까지는 무조건 내려갔다) 그쪽이 더 컸다.

            반환: (ok, 최종 pose, 접촉으로 조기 정지했는지). ok=False면 호출부가 다른
            move() 실패와 똑같이 취소/RuntimeError로 처리한다 — 취소는 `move()`가 내부에서
            이미 감지하므로 여기서 또 확인할 필요는 없다(그 결과를 그대로 돌려준다).
            """
            if start_pose[2] - target_z <= 0.0:
                # 이미 목표 높이거나 더 낮다 — 내려갈 구간이 없으니 감시할 것도 없다.
                return (*move([xy[0], xy[1], target_z,
                              start_pose[3], start_pose[4], start_pose[5]]), False)

            waypoint = next_target([xy[0], xy[1], target_z], start_pose)
            if waypoint is None:
                return False, None, False

            watch = compliance.ContactWatch(self._tool_force_client, threshold_n)
            ok, pose = move(waypoint, vel_mm_s=self._place_descent_vel_mm_s,
                            stop_when=watch, stop_poll_s=self._place_descent_poll_s)
            # **감시가 실제로 돌았는지를 매번 남긴다.** GetToolForce가 무응답이면 접촉은
            # 영영 안 잡히는데 하강 자체는 성공으로 끝나므로, 이 줄이 없으면 "순응 하강을
            # 켜 뒀다"와 "순응 하강이 사실상 꺼져 있었다"를 로그로 구분할 수 없다.
            self.get_logger().info(
                f"[COMPLIANCE] 하강 감시 ({start_pose[2]:.1f} → {target_z:.1f}mm, "
                f"{self._place_descent_vel_mm_s:.0f}mm/s): {watch.summary()}")
            if watch.triggered and pose is not None:
                self.get_logger().warning(
                    f"[COMPLIANCE] place 하강 접촉 감지 — z={pose[2]:.1f}mm에서 멈춤 "
                    f"(목표 {target_z:.1f}mm) — 그 아래로는 안 내려간다")
            return ok, pose, watch.triggered

        # place_into 접근은 **수직·수평만** 쓴다: ①제자리 수직 상승 ②제자리 회전
        # ③안전고도 수평 이동 ④바구니 바로 위에서 수직 하강.
        #
        # 한 번의 movel로 바구니 접근 지점까지 바로 가면 XY와 Z가 같이 변하는 최단 직선이
        # 되어 바구니 테두리·작업대·다른 물체를 비스듬히 스친다. 위처럼 쪼개면 매 구간이
        # 축 하나만 움직이므로 어디를 지나는지가 눈에 보이고 충돌 여지가 없다.
        #
        # ②(회전)를 ③(수평 이동)과 합치지 않는 이유: ry가 180도 근처(ZYZ 특이점)라 방금
        # 도달한 회전과 바구니용으로 계산한 회전이 물리적으로 같은 방향이어도 파라미터가
        # 크게 다를 수 있는데(모듈 docstring 참조), 그 회전 변화를 큰 위치 변화와 한
        # movel에 묶었더니 로봇이 아예 움직이지 않은 적이 있다. 회전은 안전고도에서
        # 제자리로 끝내고, 이동 구간은 순수 병진만 남긴다 — 코드 전체가 지키는
        # "한 movel에 하나만 크게 바꾼다" 규칙(next_target 참조)과 같다.
        #
        # 안전고도(transit_z)는 바구니 XY 위에서 실제로 IK가 풀리는 높이여야 한다.
        # 2026-09-06 ikin 실측: 이 셀의 left_box XY 기준 z=500mm까지 해가 존재한다.
        self._publish_phase(goal_handle, PlaceInto.Feedback.PHASE_MOVING)
        # 이 호출은 next_target()과 달리 대체 경로(feedback 폴백)가 없다 — place_into
        # 시작 직후 딱 한 번, 안전고도 계산에 반드시 필요한 현재 자세를 구하는 자리라서다.
        # 그런데 여기 호출 시점은 정의상 pick의 들어올리기 movel이 막 끝난 직후, 즉
        # aux_control이 10~20초씩 무응답이라고 문서화된 그 구간 한복판이다. 실제로
        # 재시도 budget을 2.5초로 짧게 뒀더니 place_into가 시작하자마자 거의 매번
        # "현재 자세를 읽지 못했다"로 실패했다(2026-09-06 실물 확인 — 파지는 성공했는데
        # place가 즉시 실패). 대체 경로가 없는 이 호출만은 get_current_posx의 기본값
        # (최대 약 48초, 그 무응답 구간을 넉넉히 버틴다)을 그대로 쓴다.
        current_pose = dsr_motion.get_current_posx(self._posx_client, goal_handle)
        if current_pose is None:
            if goal_handle.is_cancel_requested:
                return None
            raise RuntimeError("현재 자세를 읽지 못했다")
        # 안전고도: 현재 높이와 바구니 접근 높이 중 높은 쪽 + 여유. 예전엔 여기서
        # ikin 서비스로 이 높이가 실제로 풀리는지 매번 확인했는데, 그 함수가 서비스
        # 무응답 시 True(가능하다고 가정)를 돌려주도록 되어 있어 정작 필요한 순간엔
        # 아무것도 걸러주지 못하면서 한 번에 최대 15초를 썼다(2026-09-06 실물 로그).
        # 도달 가능 범위는 위 주석대로 셀을 바꿀 때 ikin으로 한 번 재보면 된다.
        transit_z = max(current_pose[2], approach_xyz[2]) + self._place_safe_clearance_mm

        # **그 높이에서 바구니 상공까지 실제로 팔이 닿는지 확인하고, 안 되면 낮춘다.**
        # 2026-09-07 실물: TCP(GripperDA_v1, 208mm)를 복구하자 같은 목표점이 도달 불가가
        # 됐다 — TCP가 208mm 앞으로 나가면서 플랜지가 그만큼 더 뻗어야 하기 때문이다
        # (704mm → 839mm, M0609 최대 900mm). movel은 오류를 내지 않고 그냥 움직이지 않아
        # 60초 타임아웃으로 끝났다. 314행의 "z=500mm까지 해가 존재한다"는 2026-09-06
        # 측정은 **TCP가 없던 상태**의 값이라 더 이상 유효하지 않다.
        self.get_logger().info(
            f"place 안전고도 = {transit_z:.1f}mm **고정** (바구니 접근 {approach_xyz[2]:.1f}, "
            f"현재 {current_pose[2]:.1f}, 여유 {self._place_safe_clearance_mm:.0f}) — "
            "IK가 안 풀려도 낮추지 않는다(STEP 2)")

        # **높이를 낮추는 대신 관절 configuration과 경로를 바꿔 가며 찾는다.**
        over_target = [approach_xyz[0], approach_xyz[1], transit_z,
                       approach_posx[3], approach_posx[4], approach_posx[5]]

        # **실행 중 실패하면 다른 경로/자세로 넘어간다 (2026-09-10 실물에서 발견).**
        #
        # ikin이 멀쩡한 해를 줘도 컨트롤러가 실제로는 그 목표를 거부하고 **조용히 안
        # 움직이는** 경우가 있다(이 파일 위쪽 2026-09-07 21:08 기록). 사전 검사만으로
        # 경로 하나를 정해 놓고 실행에서 실패하면 그대로 끝나 버려서, 애써 만든 다른
        # 경로·다른 solution space가 쓰이지 못했다 — 실물에서 로봇이 멈춰 선 채 아무
        # 변화가 없던 것이 이것이다.
        #
        # **여기서도 높이는 낮추지 않는다.** 바꾸는 것은 경로와 관절 configuration뿐이다.
        # 시도 횟수를 묶어 두는 이유: 실패 한 번이 최대 60초(컨트롤러가 goal을 받고도
        # 안 움직이면 타임아웃까지 간다)라 조합을 전부 돌면 사람이 기다릴 수 없다.
        traverse_pose = None
        attempted: set = set()
        start_pose = list(current_pose)
        for attempt in range(1, self._transit_attempts + 1):
            plan = self._transit_plan(start_pose, over_target, transit_z, skip=attempted)
            if plan is None:
                raise _SafeTransitUnreachable(
                    f"안전고도 {transit_z:.1f}mm에서 바구니 상공까지 갈 방법이 없다 — "
                    f"경로·solution space 조합을 {len(attempted)}개 시도했다"
                    if attempted else
                    f"안전고도 {transit_z:.1f}mm에서 바구니 상공까지 갈 방법이 없다 — "
                    "solution space 0~7과 direct/우회 경로를 모두 확인했다")
            route_name = plan["name"]
            waypoints = plan["waypoints"]
            transit_space = plan["sol_space"]
            attempted.add((route_name, transit_space))

            failed_at = None
            traverse_pose = None
            for index, waypoint in enumerate(waypoints, start=1):
                if plan["switch_before"] == index - 1:
                    self.get_logger().warning(
                        f"Safe-Z에서 solution space {transit_space}로 configuration 전환 "
                        f"(관절 이동 최대 {self._max_config_switch_jump_deg:.0f}도 제한)")
                    switched = dsr_motion.move_joint(
                        self._movej_client, plan["switch_posj"], goal_handle,
                        self._joint_vel_deg_s, self._joint_acc_deg_s2,
                        logger=self.get_logger())
                    if not switched:
                        if goal_handle.is_cancel_requested:
                            return None
                        failed_at = (index, "configuration 전환 실패")
                        break
                # 회전값은 앞 이동이 실제로 도달한 값을 쓴다 — ZYZ는 같은 방향이 다른
                # 파라미터로 표현될 수 있어(모듈 docstring) 계산값을 그대로 이어 쓰면
                # 로봇이 제자리 대신 엉뚱하게 보간한다.
                target = waypoint
                if traverse_pose is not None and list(waypoint)[3:6] == list(
                        waypoints[index - 2])[3:6]:
                    target = next_target(list(waypoint)[:3], traverse_pose)
                    if target is None:
                        if goal_handle.is_cancel_requested:
                            return None
                        raise RuntimeError("현재 자세를 읽지 못했다")
                move_started = time.monotonic()
                step_ok, step_pose = move(target)
                if not step_ok:
                    if goal_handle.is_cancel_requested:
                        return None
                    failed_at = (index, self._motion_errors.since(move_started))
                    break
                traverse_pose = step_pose

            if failed_at is None:
                break

            index, alarm = failed_at
            self.get_logger().warning(
                f"이송 경로 '{route_name}'(space {transit_space}) "
                f"{index}/{len(waypoints)}번째 이동 실패"
                + (f" — {alarm}" if alarm else " — 알람 없음(무응답/타임아웃)")
                + f". 안전고도 {transit_z:.1f}mm는 그대로 두고 다른 경로/자세를 찾는다 "
                  f"({attempt}/{self._transit_attempts})")
            if attempt >= self._transit_attempts:
                raise _SafeTransitUnreachable(
                    f"안전고도 {transit_z:.1f}mm를 유지한 채 {attempt}개 경로/자세를 "
                    "실제로 시도했지만 전부 실패했다")
            # 부분 이동이 있었을 수 있다 — 다음 후보는 **지금 있는 자리**에서 다시 짠다.
            resumed = dsr_motion.get_current_posx(self._posx_client, goal_handle)
            if resumed is None:
                if goal_handle.is_cancel_requested:
                    return None
                raise RuntimeError("현재 자세를 읽지 못했다")
            start_pose = list(resumed)

        # ④ 수직 하강 (바구니 바로 위 → 접근 지점)
        descend_posx = next_target(approach_xyz, traverse_pose)
        if descend_posx is None:
            if goal_handle.is_cancel_requested:
                return None
            raise RuntimeError("현재 자세를 읽지 못했다")
        approach_ok, approach_pose = move(descend_posx)
        if not approach_ok:
            if goal_handle.is_cancel_requested:
                return None
            raise RuntimeError("접근 위치로 이동 실패")

        self._publish_phase(goal_handle, PlaceInto.Feedback.PHASE_INSERTING)
        if self._place_descent_enabled and self._place_contact_threshold_n > 0.0:
            insert_ok, insert_pose, contact_stopped = descend_compliant(
                approach_xyz, approach_pose, target_xyz[2],
                self._place_contact_threshold_n)
        else:
            insert_posx = next_target(target_xyz, approach_pose)
            if insert_posx is None:
                if goal_handle.is_cancel_requested:
                    return None
                raise RuntimeError("현재 자세를 읽지 못했다")
            insert_ok, insert_pose = move(insert_posx)
            contact_stopped = False
        if not insert_ok:
            if goal_handle.is_cancel_requested:
                return None
            raise RuntimeError("배치 위치로 이동 실패")
        if contact_stopped:
            self.get_logger().info(
                "[COMPLIANCE] 목표 높이 전에 접촉으로 멈췄다 — 이 높이에서 그대로 놓는다")

        self._publish_phase(goal_handle, PlaceInto.Feedback.PHASE_RELEASING)
        width_command = dsr_motion.gripper_width_command(self._gripper_open_m)
        if not dsr_motion.send_gripper_command(self._gripper_cmd_client, width_command):
            if goal_handle.is_cancel_requested:
                return None
            raise RuntimeError("그리퍼 열기 명령 전송 실패")
        if dsr_motion.wait_gripper_settled(
                lambda: self._gripper_joint_angle, goal_handle,
                get_status=lambda: self._gripper_status) is None:
            if goal_handle.is_cancel_requested:
                return None
            raise RuntimeError("그리퍼가 열리는 동안 응답이 없다")

        self._publish_phase(goal_handle, PlaceInto.Feedback.PHASE_VERIFYING)
        retreat_posx = next_target(approach_xyz, insert_pose)
        if retreat_posx is None:
            if goal_handle.is_cancel_requested:
                return None
            raise RuntimeError("현재 자세를 읽지 못했다")
        retreat_ok, retreat_pose = move(retreat_posx)
        if not retreat_ok:
            if goal_handle.is_cancel_requested:
                return None
            raise RuntimeError("물러나기 실패")

        # orchestrator가 place_into 성공 직후 home으로 자동 복귀시킨다(web/orchestrator.py
        # _execute_steps). 물러난 높이(approach_xyz)에서 바로 관절이동(movej)으로 넘어가면
        # 바구니 테두리 바로 위에서 팔이 방향을 크게 트는 셈이라 여유가 적다 — 그만큼 더
        # 올라간 뒤에 home을 부르도록 여기서 한 번 더 상승한다.
        # **여기도 도달 가능한 높이인지 확인한다.** 바구니는 팔 범위 가장자리라 위로 얼마
        # 못 올라간다 — 2026-09-07 실물: 293.8 + 200 = 493.8을 명령해 movel이 움직이지
        # 않은 채 60초 타임아웃 → "home 이동 전 상승 실패"로 place 전체가 실패했다.
        # (놓기까지는 정상으로 끝난 뒤였다.)
        #
        # **이 높이는 안전 이송고도(transit_z)와 다른 것이다.** 이송고도는 물체를 **든 채**
        # 작업대·테두리를 넘어가라고 둔 값이라 IK 때문에 낮추지 않는다(STEP 2). 반면 여기는
        # 이미 물체를 놓은 뒤 home(관절이동)으로 넘어가기 전의 여유이므로, 못 올라가면
        # 낮춰도 물체가 걸릴 일이 없다 — 그래서 여기서만 단계적으로 낮춰 본다.
        rise_home_z = self._reachable_rise_z(
            approach_xyz, retreat_pose[3:] if retreat_pose else approach_posx[3:],
            approach_xyz[2] + self._home_rise_mm, approach_xyz[2])
        rise_home_posx = next_target(
            [approach_xyz[0], approach_xyz[1], rise_home_z], retreat_pose)
        if rise_home_posx is None:
            if goal_handle.is_cancel_requested:
                return None
            raise RuntimeError("현재 자세를 읽지 못했다")
        rise_home_ok, _ = move(rise_home_posx)
        if not rise_home_ok:
            if goal_handle.is_cancel_requested:
                return None
            raise RuntimeError("home 이동 전 상승 실패")

        self.get_logger().info(
            "place_into 완료 — 순응 하강은 " +
            ("켜짐" if self._place_descent_enabled else "꺼짐(기본값)") +
            " — 꺼져 있으면 위치제어만으로 목표까지 내려간다")
        return True

    def _result(self, success, reason, started):
        result = PlaceInto.Result()
        result.success = success
        result.failure_reason = reason
        result.retries_used = 0
        result.cycle_time_ms = (time.monotonic() - started) * 1000
        return result


def main(args=None):
    rclpy.init(args=args)
    node = PlaceServer()
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

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

from dsr_msgs2.action import MovelH2r
from dsr_msgs2.srv import GetCurrentPosx, Ikin
from onrobot_rg_msgs.srv import SetCommand
from sensor_msgs.msg import JointState
from sort_msgs.action import PlaceInto

from . import dsr_motion
from .config_paths import skill_params_path
from .request_cache import RequestCache
from .robot_state_publisher import is_fake_robot, store

SCHEMA_VERSION = "1.0.0"
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
        self._movel_client = ActionClient(self, MovelH2r, dsr_motion.MOVEL_ACTION,
                                          callback_group=callbacks)
        self._ikin_client = self.create_client(
            Ikin, "/dsr01/dsr_controller2/motion/ikin", callback_group=callbacks)
        self._posx_client = self.create_client(
            GetCurrentPosx, dsr_motion.GET_CURRENT_POSX_SERVICE, callback_group=callbacks)
        self._gripper_cmd_client = self.create_client(
            SetCommand, dsr_motion.GRIPPER_COMMAND_SERVICE, callback_group=callbacks)
        self._gripper_joint_angle: float | None = None
        self.create_subscription(JointState, dsr_motion.GRIPPER_JOINT_STATES_TOPIC,
                                 self._on_gripper_state, 5, callback_group=callbacks)

        self.get_logger().info(
            f"place_into 액션 서버 준비 (목적지 {list(self._bins)}, "
            f"{'fake' if is_fake_robot() else '실물'} 모드)")

    def _on_gripper_state(self, msg: JointState) -> None:
        if msg.position:
            self._gripper_joint_angle = msg.position[0]

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

        bin_pose = self._bins[goal.bin_id].get("pose", {})
        # 놓는 높이 여유는 목적지마다 다를 수 있다 — bins.yaml 의 해당 목적지에
        # release_clearance_mm 이 있으면 그것을, 없으면 skill_params.yaml 의 공용값을 쓴다.
        # 바구니 깊이·기존 내용물 높이가 목적지마다 다르므로 공용값 하나로 맞추면
        # 한쪽은 너무 높아 떨어뜨리고 다른 쪽은 너무 낮아 부딪힌다.
        release_clearance_mm = float(
            self._bins[goal.bin_id].get("release_clearance_mm", self._release_clearance_mm))
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
        bottom_offset_mm = max(0.0, float(getattr(goal, "object_bottom_offset_mm", 0.0) or 0.0))
        # **orchestrator의 값은 파지점(grasp_pose) 기준이지 TCP 기준이 아니다.**
        # 그것은 `grasp_pose.z - (물체 윗면 z - 물체 높이)`, 즉 "손가락이 만나는 지점에서
        # 물체 바닥까지"다. 여기서 명령하는 것은 TCP이고 손가락은 TCP에서
        # tool.grasp_center_offset_mm만큼 떨어져 있으므로, 그만큼을 빼야 물체 바닥이
        # 실제로 원하는 높이에 온다 (pick과 place가 같은 "파지점" 규약을 쓰게 하는 것).
        #
        # 이 보정이 없던 동안 place는 필요보다 약 18mm **높은** 데서 놓았다 — 안전한
        # 방향이라 문제가 드러나지 않았지만, release_clearance_mm이 문서대로
        # "순수한 안전 여유"가 되려면 여기서 정확히 맞춰야 한다. 이 변경으로 실제 여유는
        # 48mm에서 설정값(30mm) 그대로가 된다.
        #
        # **물체 높이를 모르면(0) 이 변환을 하지 않는다.** 그 경우 바꿀 대상인 "파지점 기준
        # 값" 자체가 없어서 빼 봐야 안전 여유만 18mm 깎일 뿐이다. 미상 경로는 예전 동작을
        # 그대로 둔다.
        tcp_to_finger_z_mm = 0.0
        if bottom_offset_mm > 0.0:
            tcp_to_finger_z_mm = dsr_motion.grasp_center_from_posx(
                dsr_motion.bin_pose_to_posx(bin_pose), self._grasp_center_offset_mm)[2] \
                - float(bin_pose.get("z", 0.0))
        release_z = (float(bin_pose.get("z", 0.0)) + bottom_offset_mm
                     + release_clearance_mm - tcp_to_finger_z_mm)
        if bottom_offset_mm > 0.0:
            detail = (f"물체높이보정 {bottom_offset_mm:.1f}mm + 여유 {release_clearance_mm:.1f}mm"
                      f" - TCP↔손끝 {tcp_to_finger_z_mm:+.1f}mm")
        else:
            detail = f"여유 {release_clearance_mm:.1f}mm (물체높이 미상 — 고정 여유만 사용)"
        self.get_logger().info(
            f"place_into 시작 object={goal.object_id} bin={goal.bin_id} "
            f"target=({bin_pose.get('x')}, {bin_pose.get('y')}, {bin_pose.get('z')})mm "
            f"놓는높이={release_z:.1f}mm ({detail})")
        store.set_busy("place_into")
        started = time.monotonic()

        try:
            if is_fake_robot():
                for phase in PHASES:
                    if goal_handle.is_cancel_requested:
                        goal_handle.canceled()
                        return self._result(False, PlaceInto.Result.REASON_NO_CONTACT, started)
                    self._publish_phase(goal_handle, phase)
                    time.sleep(FAKE_PHASE_DURATION_S)
            else:
                if self._place_real(goal_handle, bin_pose,
                                    release_z - float(bin_pose.get("z", 0.0))) is None:
                    goal_handle.canceled()
                    return self._result(False, PlaceInto.Result.REASON_NO_CONTACT, started)

            store.set_gripper(width_mm=0.0, closed=False)
            result = self._result(True, PlaceInto.Result.REASON_NONE, started)
            self._cache.put(goal.request_id, result)
            goal_handle.succeed()
            return result

        except Exception as e:
            self.get_logger().error(f"place_into 실패: {e}")
            store.set_error()
            goal_handle.abort()
            return self._result(False, PlaceInto.Result.REASON_PLACE_FAILED, started)
        finally:
            if store.snapshot()["mode"] != "error":
                store.set_idle()

    # M0609 관절 한계(도). robot_description에서 읽은 값 — J3만 ±150이고 나머지는 ±360.
    _JOINT_LIMITS_DEG = (360.0, 360.0, 150.0, 360.0, 360.0, 360.0)

    def _reachable(self, posx) -> bool | None:
        """`posx`에 실제로 도달할 수 있는지. True/False, 판단 불가면 None.

        **`ikin`의 `success`를 믿으면 안 된다.** 이 드라이버는 해가 없어도 `success=True`를
        돌려주고 관절값만 터무니없이 낸다 — 2026-09-07 실측: 바구니 상공 목표에 대해
        `success=True`인데 conv_posj가 `[-1674, 317, -1445, -180, 493, -1854]`도였다.
        그래서 **관절값이 한계 안인지 직접 본다.**

        무응답이면 None을 돌려 호출부가 "판단 못 함"으로 다루게 한다. 예전 구현은 이 경우
        True(가능하다고 가정)를 돌려줘서 정작 필요한 순간에 아무것도 못 걸렀다.
        """
        if not self._ikin_client.service_is_ready():
            return None
        request = Ikin.Request()
        request.pos = [float(v) for v in posx[:6]]
        request.sol_space = 2      # 이 셀의 통상 solution space (get_current_posx data[6])
        request.ref = 0
        future = self._ikin_client.call_async(request)
        finished = threading.Event()
        future.add_done_callback(lambda _f: finished.set())
        if not finished.wait(timeout=2.0):
            return None
        result = future.result()
        if result is None or not result.success:
            return False
        return all(abs(float(q)) <= limit
                   for q, limit in zip(result.conv_posj, self._JOINT_LIMITS_DEG))

    def _reachable_z(self, xy, rot, desired_z: float, floor_z: float, what: str) -> float:
        """`xy`/`rot`에서 실제로 팔이 닿는 가장 높은 z를 찾는다(desired_z에서 20mm씩 낮춤).

        도달 불가한 높이를 그대로 명령하면 movel이 **오류 없이 그냥 안 움직이고** 60초
        타임아웃으로 끝난다(2026-09-07 실물). 그래서 보내기 전에 걸러야 한다.
        """
        z = float(desired_z)
        while z > floor_z:
            verdict = self._reachable([xy[0], xy[1], z, *rot])
            if verdict is None:
                self.get_logger().warning(
                    f"{what} {z:.0f}mm 도달 가능성 확인 실패(ikin 무응답) — 그대로 진행")
                break
            if verdict:
                break
            z = max(floor_z, z - 20.0)
            self.get_logger().info(f"{what}가 팔 범위 밖 — {z:.0f}mm로 낮춘다")
        return z

    @staticmethod
    def _publish_phase(goal_handle, phase) -> None:
        feedback = PlaceInto.Feedback()
        feedback.phase = phase
        goal_handle.publish_feedback(feedback)

    def _place_real(self, goal_handle, bin_pose: dict,
                    release_offset_mm: float) -> bool | None:
        """위치제어만으로 실물 place_into를 수행한다 (pick_server._pick_real과 같은 1단계
        제약 — compliance/visual_verification 없이 bins.yaml 좌표를 그대로 믿는다).

        bin_pose 바로 위(approach_height_mm)에서 한 번 멈췄다, bin_pose.z + release_offset_mm
        까지만 내려가 그리퍼를 열고 다시 들어올린다 — bin_pose 그 자체(z)까지 내려가지 않는다.
        `release_offset_mm`은 "물체가 TCP 아래로 내려온 양 + 목적지별 안전 여유"의 합이다
        (execute_callback 참조) — 호출부가 계산해 넘긴다.
        성공 True, 취소 None, 그 외 실패는 RuntimeError.
        """
        target_posx = dsr_motion.bin_pose_to_posx(bin_pose)
        target_posx[2] += release_offset_mm
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
        transit_z = self._reachable_z(approach_xyz, approach_posx[3:], transit_z,
                                      max(current_pose[2], approach_xyz[2]), "안전고도")
        self.get_logger().info(
            f"place 안전고도 = {transit_z:.1f}mm (바구니 접근 {approach_xyz[2]:.1f}, "
            f"현재 {current_pose[2]:.1f}, 여유 {self._place_safe_clearance_mm:.0f})")

        # ① 수직 상승 (XY·회전 고정)
        rise_ok, _ = move([current_pose[0], current_pose[1], transit_z,
                           current_pose[3], current_pose[4], current_pose[5]])
        if not rise_ok:
            if goal_handle.is_cancel_requested:
                return None
            raise RuntimeError("안전 높이 상승 실패")

        # ② 제자리 회전 (위치 고정, 바구니 배치 자세로)
        rotate_ok, rotate_pose = move([current_pose[0], current_pose[1], transit_z,
                                       approach_posx[3], approach_posx[4], approach_posx[5]])
        if not rotate_ok:
            if goal_handle.is_cancel_requested:
                return None
            raise RuntimeError("바구니 배치 자세로 회전 실패")

        # ③ 수평 이동 (Z·회전 고정 — 회전은 ②에서 실제로 도달한 값을 그대로 쓴다)
        traverse_posx = next_target([approach_xyz[0], approach_xyz[1], transit_z], rotate_pose)
        if traverse_posx is None:
            if goal_handle.is_cancel_requested:
                return None
            raise RuntimeError("현재 자세를 읽지 못했다")
        traverse_ok, traverse_pose = move(traverse_posx)
        if not traverse_ok:
            if goal_handle.is_cancel_requested:
                return None
            raise RuntimeError("바구니 상공으로 수평 이동 실패")

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
        insert_posx = next_target(target_xyz, approach_pose)
        if insert_posx is None:
            if goal_handle.is_cancel_requested:
                return None
            raise RuntimeError("현재 자세를 읽지 못했다")
        insert_ok, insert_pose = move(insert_posx)
        if not insert_ok:
            if goal_handle.is_cancel_requested:
                return None
            raise RuntimeError("배치 위치로 이동 실패")

        self._publish_phase(goal_handle, PlaceInto.Feedback.PHASE_RELEASING)
        width_command = dsr_motion.gripper_width_command(self._gripper_open_m)
        if not dsr_motion.send_gripper_command(self._gripper_cmd_client, width_command):
            if goal_handle.is_cancel_requested:
                return None
            raise RuntimeError("그리퍼 열기 명령 전송 실패")
        if dsr_motion.wait_gripper_settled(lambda: self._gripper_joint_angle, goal_handle) is None:
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
        rise_home_z = self._reachable_z(
            approach_xyz, retreat_pose[3:] if retreat_pose else approach_posx[3:],
            approach_xyz[2] + self._home_rise_mm, approach_xyz[2], "home 전 상승")
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

        self.get_logger().warning(
            "위치제어만으로 place_into 완료 — compliance/visual_verification 미구현이라 "
            "실제 배치 여부는 확인되지 않았다")
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

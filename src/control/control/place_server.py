"""place_into 액션 서버. bin_id → bins.yaml 좌표 변환.

참조: 인터페이스_정의서.md 4.2절 (PlaceInto.action)

`bin_id` 그라운딩을 여기서도 한 번 더 한다. planner의 검증기가 이미 확인하지만(FR-11),
control은 planner를 거치지 않는 호출(재전송, 수동 테스트, 향후 다른 클라이언트)도 받는다.
좌표를 실제로 아는 쪽에서 마지막으로 확인하는 것이 옳다.
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


def load_motion_params(path: pathlib.Path | None = None) -> dict:
    path = path or skill_params_path()
    with path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config.get("motion") or {}


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
        self._linear_vel_mm_s = float(motion.get("linear_vel_mm_s", 30.0))
        self._linear_acc_mm_s2 = float(motion.get("linear_acc_mm_s2", 30.0))
        self._rot_vel_deg_s = float(motion.get("rot_vel_deg_s", 20.0))
        self._rot_acc_deg_s2 = float(motion.get("rot_acc_deg_s2", 20.0))
        self._gripper_open_m = float(motion.get("gripper_open_m", 0.110))
        self._movel_client = ActionClient(self, MovelH2r, dsr_motion.MOVEL_ACTION,
                                          callback_group=callbacks)
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
        self.get_logger().info(
            f"place_into 시작 object={goal.object_id} bin={goal.bin_id} "
            f"target=({bin_pose.get('x')}, {bin_pose.get('y')}, {bin_pose.get('z')})mm")
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
                if self._place_real(goal_handle, bin_pose) is None:
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

    @staticmethod
    def _publish_phase(goal_handle, phase) -> None:
        feedback = PlaceInto.Feedback()
        feedback.phase = phase
        goal_handle.publish_feedback(feedback)

    def _place_real(self, goal_handle, bin_pose: dict) -> bool | None:
        """위치제어만으로 실물 place_into를 수행한다 (pick_server._pick_real과 같은 1단계
        제약 — compliance/visual_verification 없이 bins.yaml 좌표를 그대로 믿는다).

        bin_pose 바로 위(approach_height_mm)에서 한 번 멈췄다 내려가 그리퍼를 열고
        다시 들어올린다. 성공 True, 취소 None, 그 외 실패는 RuntimeError.
        """
        target_posx = dsr_motion.bin_pose_to_posx(bin_pose)
        approach_posx = list(target_posx)
        approach_posx[2] += self._approach_height_mm
        target_xyz = target_posx[:3]
        approach_xyz = approach_posx[:3]

        def move(pos):
            return dsr_motion.move_linear(self._movel_client, pos, goal_handle,
                                          self._linear_vel_mm_s, self._linear_acc_mm_s2,
                                          self._rot_vel_deg_s, self._rot_acc_deg_s2,
                                          posx_client=self._posx_client)

        def posx_at(xyz):
            """`xyz`로 위치만 바꾸고 회전은 방금 도달한 실제 자세를 그대로 쓴다
            (dsr_motion.py 모듈 docstring — ZYZ 특이점 참조)."""
            current = dsr_motion.get_current_posx(self._posx_client, goal_handle)
            if current is None:
                return None
            return [xyz[0], xyz[1], xyz[2], current[3], current[4], current[5]]

        self._publish_phase(goal_handle, PlaceInto.Feedback.PHASE_MOVING)
        if not move(approach_posx):
            if goal_handle.is_cancel_requested:
                return None
            raise RuntimeError("접근 위치로 이동 실패")

        self._publish_phase(goal_handle, PlaceInto.Feedback.PHASE_INSERTING)
        insert_posx = posx_at(target_xyz)
        if insert_posx is None:
            if goal_handle.is_cancel_requested:
                return None
            raise RuntimeError("현재 자세를 읽지 못했다")
        if not move(insert_posx):
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
        retreat_posx = posx_at(approach_xyz)
        if retreat_posx is None:
            if goal_handle.is_cancel_requested:
                return None
            raise RuntimeError("현재 자세를 읽지 못했다")
        if not move(retreat_posx):
            if goal_handle.is_cancel_requested:
                return None
            raise RuntimeError("물러나기 실패")

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

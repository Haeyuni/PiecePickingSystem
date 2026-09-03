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
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from sort_msgs.action import PlaceInto

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


class PlaceServer(Node):
    def __init__(self):
        super().__init__('place_server')
        self._cache = RequestCache()
        try:
            self._bins = load_bins()
        except FileNotFoundError:
            self.get_logger().error(f"bins.yaml을 찾을 수 없습니다: {bins_yaml_path()}")
            self._bins = {}

        self._action_server = ActionServer(
            self, PlaceInto, 'place_into', self.execute_callback,
            goal_callback=lambda _: GoalResponse.ACCEPT,
            cancel_callback=self._cancel_callback,
            callback_group=ReentrantCallbackGroup(),
        )
        self.get_logger().info(
            f"place_into 액션 서버 준비 (목적지 {list(self._bins)}, "
            f"{'fake' if is_fake_robot() else '실물'} 모드)")

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
            for phase in PHASES:
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    return self._result(False, PlaceInto.Result.REASON_NO_CONTACT, started)

                feedback = PlaceInto.Feedback()
                feedback.phase = phase
                goal_handle.publish_feedback(feedback)

                if is_fake_robot():
                    time.sleep(FAKE_PHASE_DURATION_S)
                else:
                    # TODO(실물): bins.yaml 좌표(mm) → units.mm_to_m() → MoveIt2 모션,
                    # 놓기 후 compliance.py로 배치 성공 판정
                    raise NotImplementedError("실물 place_into는 MoveIt2 연결 후 활성화한다")

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

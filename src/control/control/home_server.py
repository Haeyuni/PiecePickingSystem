"""home 액션 서버. 홈 자세 복귀.

참조: 인터페이스_정의서.md 4.3절 (Home.action), 웹_인터페이스_정의서.md 2.6절

pick/place_into와 달리 대상 물체도 목적지도 없다. 그래도 액션인 이유는 `Home.action`
주석에 적어둔 대로 **정지로 취소할 수 있어야 하기 때문**이다. 그 요구가 없었다면 서비스가
맞다 — 인터페이스 모양이 요구에서 나온 것이지 관례에서 나온 것이 아니다.

홈 자세는 `bins.yaml`이 아니라 `skill_params.yaml`에 둔다. 목적지(bin)가 아니라
제어 파라미터이기 때문이다.

**실물 이동은 `dsr_msgs2/action/MovejH2r`(관절각, deg)를 그대로 부른다.** 좌표가 아니라
관절각이라 `units.py`의 mm 환산 대상이 아니다 — 회전을 rad로 통일하는 규칙(인터페이스_정의서
1.1절)도 base 좌표계 pose 얘기이지 로봇 자체 관절각 표현과는 별개다(get_current_posx가
ZYZ 오일러를 도(deg) 단위로 돌려주는 것과 같은 이유, perception_common/geometry.py 참조).
"""
import pathlib
import time

import rclpy
import yaml
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from dsr_msgs2.action import MovejH2r
from sort_msgs.action import Home

from . import dsr_motion
from .config_paths import skill_params_path
from .request_cache import RequestCache
from .robot_state_publisher import is_fake_robot, store

SCHEMA_VERSION = "1.0.0"
FAKE_PHASE_DURATION_S = 0.6


def load_home_params(path: pathlib.Path | None = None) -> dict:
    path = path or skill_params_path()
    with path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config.get("home") or {}


class HomeServer(Node):
    def __init__(self):
        super().__init__('home_server')
        self._cache = RequestCache()
        callbacks = ReentrantCallbackGroup()
        self._action_server = ActionServer(
            self, Home, 'home', self.execute_callback,
            goal_callback=lambda _: GoalResponse.ACCEPT,
            cancel_callback=self._cancel_callback,
            callback_group=callbacks,
        )

        home = load_home_params()
        self._joint_deg = list(home.get("joint_deg") or [0.0] * 6)
        self._vel_deg_s = float(home.get("vel_deg_s", 20.0))
        self._acc_deg_s2 = float(home.get("acc_deg_s2", 20.0))
        self._movej_client = ActionClient(self, MovejH2r, dsr_motion.MOVEJ_ACTION,
                                          callback_group=callbacks)

        self.get_logger().info(
            f"home 액션 서버 준비 ({'fake' if is_fake_robot() else '실물'} 모드, "
            f"관절각={self._joint_deg} deg)")

    def _cancel_callback(self, goal_handle):
        self.get_logger().warning("home 취소 요청 수신")
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        goal = goal_handle.request

        cached = self._cache.get(goal.request_id)
        if cached is not None:
            self.get_logger().warning(
                f"중복 request_id={goal.request_id} — 재실행하지 않고 이전 결과 반환")
            goal_handle.succeed()
            return cached

        self.get_logger().info(f"home 시작 request={goal.request_id}")
        # 홈이동 중에도 mode는 busy다. web의 409 차단 규칙(2.1절)이 그 사이 들어오는
        # 명령을 막아야 하는데, current_skill이 "none"이면 화면에 무엇 때문에 막혔는지
        # 표시할 수 없다.
        store.set_busy("home")
        started = time.monotonic()

        try:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return self._result(False, Home.Result.REASON_NONE, started)

            feedback = Home.Feedback()
            feedback.phase = Home.Feedback.PHASE_MOVING
            goal_handle.publish_feedback(feedback)

            if is_fake_robot():
                time.sleep(FAKE_PHASE_DURATION_S)
            elif not self._move_home_real(goal_handle):
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    return self._result(False, Home.Result.REASON_NONE, started)
                raise RuntimeError("movej_h2r 실패")

            result = self._result(True, Home.Result.REASON_NONE, started)
            self._cache.put(goal.request_id, result)
            goal_handle.succeed()
            return result

        except Exception as e:
            self.get_logger().error(f"home 실패: {e}")
            store.set_error()
            goal_handle.abort()
            return self._result(False, Home.Result.REASON_UNREACHABLE, started)
        finally:
            if store.snapshot()["mode"] != "error":
                store.set_idle()

    def _move_home_real(self, goal_handle) -> bool:
        """MovejH2r로 실제 관절 이동 (블로킹 대기·취소 전파는 dsr_motion.move_joint 참조)."""
        return dsr_motion.move_joint(self._movej_client, self._joint_deg, goal_handle,
                                     self._vel_deg_s, self._acc_deg_s2)

    def _result(self, success, reason, started):
        result = Home.Result()
        result.success = success
        result.failure_reason = reason
        result.cycle_time_ms = (time.monotonic() - started) * 1000
        return result


def main(args=None):
    rclpy.init(args=args)
    node = HomeServer()
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

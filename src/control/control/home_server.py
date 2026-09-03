"""home 액션 서버. 홈 자세 복귀.

참조: 인터페이스_정의서.md 4.3절 (Home.action), 웹_인터페이스_정의서.md 2.6절

pick/place_into와 달리 대상 물체도 목적지도 없다. 그래도 액션인 이유는 `Home.action`
주석에 적어둔 대로 **정지로 취소할 수 있어야 하기 때문**이다. 그 요구가 없었다면 서비스가
맞다 — 인터페이스 모양이 요구에서 나온 것이지 관례에서 나온 것이 아니다.

홈 자세는 `bins.yaml`이 아니라 `skill_params.yaml`에 둔다. 목적지(bin)가 아니라
제어 파라미터이기 때문이다.
"""
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from sort_msgs.action import Home

from .request_cache import RequestCache
from .robot_state_publisher import is_fake_robot, store

SCHEMA_VERSION = "1.0.0"
FAKE_PHASE_DURATION_S = 0.6


class HomeServer(Node):
    def __init__(self):
        super().__init__('home_server')
        self._cache = RequestCache()
        self._action_server = ActionServer(
            self, Home, 'home', self.execute_callback,
            goal_callback=lambda _: GoalResponse.ACCEPT,
            cancel_callback=self._cancel_callback,
            callback_group=ReentrantCallbackGroup(),
        )
        self.get_logger().info(
            f"home 액션 서버 준비 ({'fake' if is_fake_robot() else '실물'} 모드)")

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
            else:
                # TODO(실물): skill_params.yaml의 홈 관절각으로 MoveIt2 movej.
                # 좌표가 아니라 관절각이라 units.py 환산 대상이 아니다(회전은 rad로 통일).
                raise NotImplementedError("실물 home은 MoveIt2 연결 후 활성화한다")

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

"""/control/safety_events 발행 (비상정지·충돌·제한 위반).

참조: 인터페이스_정의서.md 3.6절 — 검증기(FR-12)의 게이트 조건으로 사용됨

**감지 소스 두 가지**(둘 다 doosan-robot2 드라이버, dsr_msgs2). 이 저장소엔 dsr_msgs2가
마운트로만 존재해(src/control/docker-entrypoint.sh 참조) 필드는 doosan-robot2 GitHub
소스(dsr_msgs2/srv/GetRobotState.srv, dsr_msgs2/msg/RobotError.msg, jazzy 브랜치,
2026-09-08 확인)로만 검증했다 — 실물로는 확인하지 못했다.

1. **`system/get_robot_state` 주기 폴링.** STATE_EMERGENCY_STOP(6)이면 e_stop,
   STATE_SAFE_STOP(5)이면 collision으로 본다 — 컨트롤러가 안전정지의 정확한 원인(충돌
   감지 vs 안전 입력)을 구분해 주지 않아, "사람이 개입해야 풀리는 정지"라는 공통점으로
   collision에 묶는다(dsr_motion.ROBOT_STATE_* 주석 참조). STATE_SAFE_OFF류는 일부러
   뺐다 — 드라이브 인가 전(정상 기동 초기)에도 거치는 상태라, 안전 이벤트로 잡으면
   정상 기동이 막힌다.
2. **`/dsr01/error`(RobotError) 구독.** group이 SAFETY_CONTROLLER(5)인 알람만
   limit_violation으로 본다. MOTION(2) 그룹은 "이 자세로 못 간다" 같은 계획 거부라
   안전 이벤트가 아니다 — dsr_motion.MotionErrorMonitor가 이미 그 그룹을 pick/place
   실행 실패로 다루고 있다(같은 토픽을 두 번 구독한다).

**같은 조건이 유지돼도 매 폴링/알람마다 다시 쏘지 않는다** — robot_state 기반은 상태가
바뀔 때(진입)만, 알람 기반은 같은 (group, code)가 반복되면 한 번만 발행한다. 알람은
로그 스트림이라 "해소됨"을 알려주는 신호가 따로 없어 그 종류는 지우지 않는다(프로세스
생애 동안 유지) — robot_state 기반은 벗어나면 clear한다.

FAKE_ROBOT=1이면 감지 소스가 없어 폴링/구독을 하지 않는다.
"""
import threading

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from sort_msgs.msg import SafetyEvent

from . import dsr_motion
from .robot_state_publisher import is_fake_robot, store

SCHEMA_VERSION = "1.0.0"
POLL_PERIOD_S = 0.5


class SafetyMonitorNode(Node):
    def __init__(self):
        super().__init__('safety_monitor')
        self._pub = self.create_publisher(SafetyEvent, '/control/safety_events', 10)
        self._lock = threading.Lock()
        self._last_state_event: str | None = None       # robot_state 기반 — 마지막으로 쏜 종류
        self._last_error_key: tuple[int, int] | None = None  # 알람 기반 — 마지막 (group, code)

        if is_fake_robot():
            self.get_logger().info("safety_monitor: FAKE_ROBOT=1 — 감지 소스 없이 대기만 한다")
            return

        from dsr_msgs2.msg import RobotError
        from dsr_msgs2.srv import GetRobotState

        callbacks = ReentrantCallbackGroup()
        self._state_client = self.create_client(
            GetRobotState, dsr_motion.GET_ROBOT_STATE_SERVICE, callback_group=callbacks)
        self.create_timer(POLL_PERIOD_S, self._poll_robot_state, callback_group=callbacks)
        self.create_subscription(RobotError, dsr_motion.MOTION_ERROR_TOPIC,
                                 self._on_error, 10, callback_group=callbacks)
        self.get_logger().info(
            f"safety_monitor: {dsr_motion.GET_ROBOT_STATE_SERVICE} 폴링 + "
            f"{dsr_motion.MOTION_ERROR_TOPIC} 구독 시작 (주기 {POLL_PERIOD_S}s)")

    def _poll_robot_state(self) -> None:
        state = dsr_motion.get_robot_state(self._state_client, timeout_s=POLL_PERIOD_S)
        if state is None:
            return  # 서비스 미기동/무응답 — 판단할 근거가 없다. "안전"으로 단정하지 않는다.
        if state == dsr_motion.ROBOT_STATE_EMERGENCY_STOP:
            self._enter_state_event(
                SafetyEvent.TYPE_ESTOP, SafetyEvent.SEVERITY_CRITICAL,
                f"컨트롤러 robot_state={state} (STATE_EMERGENCY_STOP)")
        elif state == dsr_motion.ROBOT_STATE_SAFE_STOP:
            self._enter_state_event(
                SafetyEvent.TYPE_COLLISION, SafetyEvent.SEVERITY_CRITICAL,
                f"컨트롤러 robot_state={state} (STATE_SAFE_STOP — 충돌 감지 또는 안전 입력, "
                "컨트롤러가 원인을 구분해 주지 않는다)")
        else:
            self._clear_state_event()

    def _enter_state_event(self, event_type: str, severity: str, description: str) -> None:
        with self._lock:
            if self._last_state_event == event_type:
                return  # 이미 같은 종류를 쏜 채로 유지 중 — 반복 발행하지 않는다
            self._last_state_event = event_type
        store.set_estopped()
        self._publish(event_type, severity, description, requires_manual_reset=True)

    def _clear_state_event(self) -> None:
        with self._lock:
            if self._last_state_event is None:
                return
            self._last_state_event = None
        store.clear_estopped()

    def _on_error(self, msg) -> None:
        if int(msg.group) != dsr_motion.ERROR_GROUP_SAFETY_CONTROLLER:
            return
        key = (int(msg.group), int(msg.code))
        with self._lock:
            if self._last_error_key == key:
                return
            self._last_error_key = key
        critical = int(msg.level) >= dsr_motion.ERROR_LEVEL_ERROR
        severity = SafetyEvent.SEVERITY_CRITICAL if critical else SafetyEvent.SEVERITY_WARNING
        self._publish(
            SafetyEvent.TYPE_LIMIT_VIOLATION, severity,
            f"안전 컨트롤러 알람 code={msg.code} {(msg.msg1 or '').strip()}",
            requires_manual_reset=critical)

    def _publish(self, event_type: str, severity: str, description: str,
                 requires_manual_reset: bool) -> None:
        msg = SafetyEvent()
        msg.schema_version = SCHEMA_VERSION
        msg.stamp = self.get_clock().now().to_msg()
        msg.event_type = event_type
        msg.severity = severity
        msg.description = description
        msg.requires_manual_reset = requires_manual_reset
        msg.trace_id = ""
        msg.aborted_request_id = ""
        self._pub.publish(msg)
        self.get_logger().warning(f"[SAFETY] {event_type}/{severity}: {description}")


def main(args=None):
    rclpy.init(args=args)
    node = SafetyMonitorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

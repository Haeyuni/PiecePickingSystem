"""컨트롤러 "도달 불가" 알람을 이동 실패로 읽는 경로 — ROS2 없이 검증한다.

배경(2026-09-07 21:08 실물): place_into가 바구니 상공(334.7, -479.2, 321.6)으로 가려다
컨트롤러가 `[ERR] Pose(...) is NOT REACHABLE`(alarm 1206)을 냈는데, **movel은 goal을
accept한 채 아무것도 하지 않았다.** 실패 응답이 없으니 호출부는 60초 타임아웃을 다 채웠고,
그게 세 번 반복돼 3분 뒤 place_failed로 끝났다. 사람 눈에는 "place를 하러 안 간다"로 보인다.

`ikin` 사전 검사로는 못 막는다 — 2026-09-08 실측: 바로 그 거부된 자세를 물으면
`success=True, conv_posj=[-55.6, 32.4, 63.8, 0.05, 83.9, -55.3]`으로 관절 한계 안의
멀쩡한 해가 나온다. 그래서 **실제로 보내보고 알람을 듣는** 이 경로가 유일한 판별이다.

여기서 지키려는 계약 두 개:

1. 알람은 **그 goal을 보낸 뒤에 온 것만** 센다. 토픽에는 지난 알람도 흘러다니므로,
   시각을 안 보면 예전 알람 하나로 이후 모든 이동이 실패한다.
2. 안전고도를 낮춰 재시도하는 것은 **도달 불가 알람일 때뿐이다.** 취소·무응답·허용오차
   초과까지 높이 탓으로 돌리면 원인을 가린 채 같은 실패를 반복한다.
"""
import ast
import pathlib
import sys
import threading
import time
import unittest
from types import SimpleNamespace as NS

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))                    # src/control

CONTROL = pathlib.Path(__file__).parents[1] / "control"


def _load(source: pathlib.Path, name: str, kind, namespace: dict):
    """`source`에서 최상위 정의 하나만 떼어 `namespace`에 넣는다 (rclpy/dsr_msgs2 미import)."""
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, kind) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name]


def _method(source: pathlib.Path, cls_name: str, method: str, namespace: dict):
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[method]


class FakeNode:
    """create_subscription만 받아 콜백을 붙잡아 둔다."""

    def __init__(self):
        self.callback = None
        self.topic = None

    def create_subscription(self, _msg_type, topic, callback, _depth, callback_group=None):
        self.topic = topic
        self.callback = callback
        return NS()


def _monitor():
    """dsr_msgs2를 stub해 MotionErrorMonitor 하나를 만든다. (모니터, 가짜노드)."""
    import types

    dsr = types.ModuleType("dsr_msgs2")
    msg = types.ModuleType("dsr_msgs2.msg")
    msg.RobotError = type("RobotError", (), {})
    dsr.msg = msg
    sys.modules.setdefault("dsr_msgs2", dsr)
    sys.modules.setdefault("dsr_msgs2.msg", msg)

    ns = {"threading": threading, "time": time}
    source = CONTROL / "dsr_motion.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for name in ("ERROR_GROUP_MOTION", "ALARM_NOT_REACHABLE", "BLOCKING_MOTION_ALARMS",
                 "MOTION_ERROR_TOPIC"):
        node = next(n for n in tree.body
                    if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), ns)
    cls = _load(source, "MotionErrorMonitor", ast.ClassDef, ns)
    node = FakeNode()
    return cls(node), node


def alarm(code=1206, group=2, msg1="[ERR] Pose(...) is NOT REACHABLE"):
    return NS(code=code, group=group, msg1=msg1)


class MotionErrorMonitorTest(unittest.TestCase):

    def test_subscribes_to_controller_error_topic(self):
        _, node = _monitor()
        self.assertEqual(node.topic, "/dsr01/error")

    def test_not_reachable_alarm_after_send_is_reported(self):
        monitor, node = _monitor()
        sent_at = time.monotonic()
        node.callback(alarm())
        reason = monitor.since(sent_at)
        self.assertIsNotNone(reason)
        self.assertIn("1206", reason)
        self.assertIn("NOT REACHABLE", reason)

    def test_alarm_from_before_this_goal_is_ignored(self):
        """지난 이동의 알람이 다음 이동을 실패시키면 안 된다."""
        monitor, node = _monitor()
        node.callback(alarm())
        time.sleep(0.01)
        self.assertIsNone(monitor.since(time.monotonic()))

    def test_non_motion_group_ignored(self):
        monitor, node = _monitor()
        sent_at = time.monotonic()
        node.callback(alarm(group=1))          # SYSTEM
        self.assertIsNone(monitor.since(sent_at))

    def test_unrelated_motion_alarm_ignored(self):
        """이동을 막지 않는 알람까지 실패로 읽으면 정상 이동이 끊긴다."""
        monitor, node = _monitor()
        sent_at = time.monotonic()
        node.callback(alarm(code=9999))
        self.assertIsNone(monitor.since(sent_at))

    def test_no_alarm_means_none(self):
        monitor, _ = _monitor()
        self.assertIsNone(monitor.since(time.monotonic()))


class RetryLowerTest(unittest.TestCase):
    """place_server._retry_lower — 안전고도를 낮춰 다시 해 볼지 판단."""

    def setUp(self):
        self.logs = []
        self.retry_lower = _method(CONTROL / "place_server.py", "PlaceServer",
                                   "_retry_lower", {})

    def _server(self, reason):
        return NS(_motion_errors=NS(since=lambda _t: reason),
                  get_logger=lambda: NS(info=self.logs.append,
                                        warning=self.logs.append,
                                        error=self.logs.append))

    def test_lowers_on_not_reachable_alarm(self):
        ok = self.retry_lower(self._server("컨트롤러 알람 code=1206 NOT REACHABLE"),
                              321.6, 263.8, time.monotonic(), "수평 이동")
        self.assertTrue(ok)

    def test_does_not_lower_without_alarm(self):
        """알람이 없으면 다른 원인이다 — 높이를 깎아 원인을 가리면 안 된다."""
        ok = self.retry_lower(self._server(None), 321.6, 263.8, time.monotonic(), "수평 이동")
        self.assertFalse(ok)

    def test_stops_at_floor(self):
        """접근 높이 아래로는 못 내려간다 — 그 밑은 바구니 테두리다."""
        ok = self.retry_lower(self._server("컨트롤러 알람 code=1206 NOT REACHABLE"),
                              263.8, 263.8, time.monotonic(), "수평 이동")
        self.assertFalse(ok)
        self.assertTrue(any("더 낮출 수 없다" in line for line in self.logs))


if __name__ == "__main__":
    unittest.main()

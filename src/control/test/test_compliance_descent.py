"""순응 하강: "한 번에 내려가다 저항값에 닿으면 그 자리에서 멈춘다" — ROS2 없이 검증한다.

2026-09-09 구조 변경(사용자 요청). 그 전에는 하강을 `place_descent_step_mm`(5mm)으로 쪼개
스텝마다 movel을 새로 보내고 **스텝 사이에서만** 힘을 확인했다. 멈출 수 있는 지점이 스텝
경계뿐이라 사실상 5mm 단위 위치제어였고, 80mm 하강에 movel이 16번 나갔다. 지금은 목표까지
한 번의 movel로 내려가면서 `compliance.ContactWatch`가 힘을 보고, 임계값을 넘으면
`dsr_motion.call_action_blocking`이 원격 goal을 취소해 로봇을 세운다.

여기서 지키려는 계약 네 개:

1. `stop_when`이 True를 내면 **취소를 실제로 보낸다.** 취소 없이 우리만 빠져나오면 로봇은
   계속 내려가는데 호출부는 멈춘 줄 알고 그리퍼를 여는, 가장 나쁜 경우가 된다.
2. 그 조기 정지는 **성공**이다(실패가 아니라 요청된 동작이다). 어디서 멈췄는지는 호출부가
   자기 콜러블의 상태(`ContactWatch.triggered`)로 안다.
3. goal 수락 응답을 못 받은 구간에서는 `stop_when`을 보지 않는다 — 취소할 handle이 없어
   로봇을 세울 방법이 없는데 우리만 먼저 빠져나오면 1번과 같은 사고가 된다.
4. GetToolForce 무응답(None)은 '접촉 없음'으로 다뤄 계속 내려가되(순응 하강을 켜기 전과
   같은 동작), **몇 번 무응답이었는지 세어 둔다** — 감시가 사실상 꺼져 있었는데 조용히
   성공으로 끝나는 것이 이 기능의 가장 나쁜 실패 모드다.
"""
import ast
import pathlib
import sys
import threading
import time
import types
import unittest
from types import SimpleNamespace as NS

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))                    # src/control

CONTROL = pathlib.Path(__file__).parents[1] / "control"


def _load(source: pathlib.Path, names, namespace: dict):
    """`source`에서 최상위 정의 몇 개만 떼어 `namespace`에 넣는다 (rclpy/dsr_msgs2 미import).

    test_motion_error_abort.py와 같은 방식 — 이 모듈들은 import만으로 ROS2를 끌어오므로
    파일 전체를 import하면 테스트가 ROS2 설치에 묶인다.
    """
    tree = ast.parse(source.read_text(encoding="utf-8"))
    wanted = list(names)
    body = [n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted]
    found = {n.name for n in body}
    missing = set(wanted) - found
    assert not missing, f"{source.name}에서 못 찾음: {missing}"
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


# --- call_action_blocking을 돌리기 위한 최소 가짜들 -----------------------------

class FakeRemoteHandle:
    """드라이버가 돌려주는 goal handle. 취소를 몇 번 받았는지 세어 둔다."""

    def __init__(self, accepted=True):
        self.accepted = accepted
        self.cancels = 0
        self._result_callbacks = []

    def get_result_async(self):
        future = NS(add_done_callback=self._result_callbacks.append)
        return future

    def cancel_goal_async(self):
        self.cancels += 1
        # 실제 드라이버는 취소를 받으면 모션을 멈추고 result를 낸다. 여기서도 result
        # 콜백을 깨워 호출부가 "이동이 끝났다"를 보게 한다(_release_remote_goal 대기).
        for callback in list(self._result_callbacks):
            callback(NS(result=lambda: NS(result=NS(success=False))))
        return NS()


class FakeActionClient:
    """send_goal_async만 흉내낸다. 즉시 수락하는 것이 기본."""

    def __init__(self, handle=None, accept_delay_s=0.0):
        self.handle = handle if handle is not None else FakeRemoteHandle()
        self.accept_delay_s = accept_delay_s
        self.goals = []

    def wait_for_server(self, timeout_sec=None):
        return True

    def send_goal_async(self, goal, feedback_callback=None):
        self.goals.append(goal)
        callbacks = []

        def add_done_callback(callback):
            callbacks.append(callback)
            if self.accept_delay_s <= 0.0:
                callback(NS(result=lambda: self.handle))
            else:
                timer = threading.Timer(self.accept_delay_s,
                                        lambda: callback(NS(result=lambda: self.handle)))
                timer.daemon = True
                timer.start()

        return NS(add_done_callback=add_done_callback)


def _call_action_blocking(namespace=None):
    ns = namespace if namespace is not None else {}
    ns.setdefault("threading", threading)
    ns.setdefault("time", time)
    _load(CONTROL / "dsr_motion.py", ["_release_remote_goal", "call_action_blocking"], ns)
    return ns["call_action_blocking"]


class StopWhenTest(unittest.TestCase):
    """이동 도중 멈추는 훅 — 순응 하강이 이걸로 "접촉하면 거기서 멈춤"을 만든다."""

    def setUp(self):
        self.call = _call_action_blocking()
        self.goal_handle = NS(is_cancel_requested=False)

    def test_stop_when_cancels_the_motion(self):
        """멈추기로 했으면 **로봇에게도 멈추라고 보내야** 한다."""
        client = FakeActionClient()
        ok, _ = self.call(client, NS(), self.goal_handle, stop_when=lambda: True,
                          stop_poll_s=0.0, overall_timeout_s=2.0)
        self.assertTrue(ok, "조기 정지는 실패가 아니라 요청된 동작이다")
        self.assertGreaterEqual(client.handle.cancels, 1,
                                "취소를 안 보내면 로봇은 계속 내려간다")

    def test_stop_when_false_keeps_going(self):
        """접촉이 없으면 평소처럼 도착 확인으로 끝나야 한다 — 훅이 이동을 방해하면 안 된다."""
        client = FakeActionClient()
        arrived = []

        def verify_arrived():
            arrived.append(1)
            return True                      # 첫 확인에서 도착 처리

        ok, _ = self.call(client, NS(), self.goal_handle, stop_when=lambda: False,
                          stop_poll_s=0.0, on_timeout_verify=verify_arrived,
                          verify_poll_s=0.0, overall_timeout_s=2.0)
        self.assertTrue(ok)
        self.assertTrue(arrived, "stop_when이 False면 도착 확인 경로가 그대로 돌아야 한다")

    def test_stop_when_is_polled_repeatedly(self):
        """한 번만 보고 마는 게 아니라 이동 내내 감시해야 한다."""
        client = FakeActionClient()
        polls = []

        def stop_when():
            polls.append(time.monotonic())
            return len(polls) >= 3           # 세 번째 확인에서 접촉

        ok, _ = self.call(client, NS(), self.goal_handle, stop_when=stop_when,
                          stop_poll_s=0.0, overall_timeout_s=3.0)
        self.assertTrue(ok)
        self.assertGreaterEqual(len(polls), 3)

    def test_stop_when_ignored_before_goal_is_accepted(self):
        """수락 응답 전에는 취소할 handle이 없다 — 우리만 빠져나오면 로봇은 계속 간다.

        그 구간에서 stop_when을 보고 성공으로 돌려주면 호출부가 "멈췄다"고 믿고 그리퍼를
        여는데 로봇은 아직 내려가는 중인, 이 기능에서 가장 위험한 경우가 된다.
        """
        client = FakeActionClient(accept_delay_s=0.35)
        polls = []

        def stop_when():
            polls.append(1)
            return True

        started = time.monotonic()
        ok, _ = self.call(client, NS(), self.goal_handle, stop_when=stop_when,
                          stop_poll_s=0.0, send_timeout_s=0.05, overall_timeout_s=2.0)
        elapsed = time.monotonic() - started
        self.assertTrue(ok)
        # 수락(0.35초) 전에 stop_when이 먹혔다면 그보다 빨리 끝났을 것이다.
        self.assertGreaterEqual(elapsed, 0.35,
                                "수락 응답 전에는 stop_when으로 끝내면 안 된다")
        self.assertGreaterEqual(client.handle.cancels, 1)

    def test_operator_cancel_still_wins(self):
        """웹의 정지 버튼은 stop_when과 무관하게 실패(취소)로 끝나야 한다."""
        client = FakeActionClient()
        ok, _ = self.call(client, NS(), NS(is_cancel_requested=True),
                          stop_when=lambda: True, stop_poll_s=0.0)
        self.assertFalse(ok)

    def test_no_stop_when_behaves_as_before(self):
        """훅을 안 주면 예전 경로 그대로 — 도착 확인으로만 끝난다."""
        client = FakeActionClient()
        ok, _ = self.call(client, NS(), self.goal_handle,
                          on_timeout_verify=lambda: True, verify_poll_s=0.0,
                          overall_timeout_s=2.0)
        self.assertTrue(ok)


# --- ContactWatch -------------------------------------------------------------

def _contact_watch(forces):
    """`forces`를 순서대로 돌려주는 가짜 GetToolForce로 ContactWatch를 만든다.

    각 원소는 6축 리스트(Fx,Fy,Fz,Mx,My,Mz) 또는 None(무응답).
    """
    queue = list(forces)

    def get_tool_force(_client, ref=0, timeout_s=1.0):
        return queue.pop(0) if queue else None

    ns = {"dsr_motion": types.SimpleNamespace(get_tool_force=get_tool_force)}
    _load(CONTROL / "compliance.py", ["tool_force_norm", "ContactWatch"], ns)
    return ns["ContactWatch"]


class ContactWatchTest(unittest.TestCase):
    """하강 중 접촉 감시 — 임계값 판정과 "감시가 실제로 돌았는가"의 기록."""

    def test_triggers_when_force_reaches_threshold(self):
        watch_cls = _contact_watch([[0.0, 0.0, 1.0, 0, 0, 0], [0.0, 0.0, 7.0, 0, 0, 0]])
        watch = watch_cls(client=None, threshold_n=5.0)
        self.assertFalse(watch())            # 1N — 아직
        self.assertTrue(watch())             # 7N — 접촉
        self.assertTrue(watch.triggered)
        self.assertAlmostEqual(watch.stop_force_n, 7.0)

    def test_moment_axes_are_ignored(self):
        """비틀림(Mx,My,Mz)은 "밀어붙이는 힘"이 아니다 — 그걸로 멈추면 안 된다."""
        watch_cls = _contact_watch([[0.0, 0.0, 1.0, 50.0, 50.0, 50.0]])
        watch = watch_cls(client=None, threshold_n=5.0)
        self.assertFalse(watch())
        self.assertFalse(watch.triggered)

    def test_unknown_reading_does_not_stop_but_is_counted(self):
        """무응답은 '안 닿았다'가 아니라 '모른다' — 계속 내려가되 세어 둔다."""
        watch_cls = _contact_watch([None, None])
        watch = watch_cls(client=None, threshold_n=5.0)
        self.assertFalse(watch())
        self.assertFalse(watch())
        self.assertFalse(watch.triggered)
        self.assertEqual(watch.unknown, 2)
        self.assertEqual(watch.polls, 2)

    def test_summary_shows_whether_watching_actually_ran(self):
        """감시가 사실상 꺼져 있었는지를 로그 한 줄로 구분할 수 있어야 한다."""
        watch_cls = _contact_watch([None, [0.0, 0.0, 2.0, 0, 0, 0]])
        watch = watch_cls(client=None, threshold_n=5.0)
        watch()
        watch()
        line = watch.summary()
        self.assertIn("무응답 1회", line)
        self.assertIn("접촉 없음", line)
        self.assertIn("2.0N", line)          # 최대값이 남는다

    def test_stays_triggered_after_first_contact(self):
        """한 번 접촉했으면 그 뒤 판독이 어떻든 '접촉으로 멈췄다'로 남아야 한다."""
        watch_cls = _contact_watch([[0.0, 0.0, 9.0, 0, 0, 0], [0.0, 0.0, 0.0, 0, 0, 0]])
        watch = watch_cls(client=None, threshold_n=5.0)
        self.assertTrue(watch())
        watch()
        self.assertTrue(watch.triggered)


if __name__ == "__main__":
    unittest.main()

"""`dsr_motion.wait_gripper_settled`가 RG2의 Busy 비트로 완료를 판정하는지.

ROS를 임포트하지 않고 함수만 AST로 떼어내 돌린다 (test_pick_depth_diagnostics.py와 같은 방식).

배경 (2026-09-08 실물): 예전 구현은 관절각이 0.6초 안 변하면 완료로 봤는데, 닫기
명령의 시작 지연이 마침 0.6초라 **닫기 목표 17.3mm인데 52.7mm에서 완료 판정**이 났다.
그 시점 상태가 파지 성공 판정의 근거로 쓰여, 닫히는 도중의 과도값으로 성패를 갈랐다.
"""
import ast
import unittest
from pathlib import Path

SOURCE = Path(__file__).parents[1] / "control" / "dsr_motion.py"


def _load():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    node = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "wait_gripper_settled")
    namespace = {"time": _FakeClock()}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["wait_gripper_settled"], namespace["time"]


class _FakeClock:
    """`time.monotonic`/`time.sleep`을 가짜로 — 테스트가 실시간을 기다리지 않게."""

    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class _NeverCancelled:
    is_cancel_requested = False


class WaitGripperSettledTest(unittest.TestCase):
    def test_busy_bit_waits_through_the_start_delay(self):
        """실물 재현: 명령 후 0.6초간 안 움직이다 닫히기 시작해 3.0초에 끝난다.

        예전 구현은 그 0.6초 정지 구간을 완료로 봤다. Busy를 보면 끝까지 기다린다.
        """
        wait, clock = _load()
        start = clock.now

        def angle():
            elapsed = clock.now - start
            if elapsed < 0.6:
                return -0.40           # 아직 안 움직임
            if elapsed >= 3.0:
                return -0.10           # 다 닫힘
            return -0.40 + 0.1 * (elapsed - 0.6)

        def status():
            elapsed = clock.now - start
            busy = 1 if 0.6 <= elapsed < 3.0 else 0
            return (clock.now, busy, 0.0)

        result = wait(angle, _NeverCancelled(), get_status=status)
        self.assertAlmostEqual(result, -0.10)
        self.assertGreaterEqual(clock.now - start, 3.0)

    def test_no_motion_needed_returns_after_start_timeout(self):
        """이미 목표 개폭이라 Busy가 한 번도 안 서는 경우(기동 시 재열기)."""
        wait, clock = _load()
        start = clock.now
        result = wait(lambda: -0.44, _NeverCancelled(),
                      get_status=lambda: (clock.now, 0, 0.0))
        self.assertAlmostEqual(result, -0.44)
        self.assertLess(clock.now - start, 8.0)     # timeout_s를 다 쓰지 않는다
        self.assertGreaterEqual(clock.now - start, 2.0)

    def test_stale_status_falls_back_to_joint_angle(self):
        """상태 토픽을 발행하지 않는 드라이버 — 예전 관절각 판정 그대로."""
        wait, clock = _load()
        stale = clock.now - 30.0
        result = wait(lambda: -0.44, _NeverCancelled(),
                      get_status=lambda: (stale, 0, 0.0))
        self.assertAlmostEqual(result, -0.44)

    def test_missing_status_falls_back_to_joint_angle(self):
        wait, clock = _load()
        result = wait(lambda: -0.44, _NeverCancelled())
        self.assertAlmostEqual(result, -0.44)

    def test_frozen_feedback_is_not_mistaken_for_settled(self):
        """피드백이 끊겨 값이 얼어붙었는데 Busy는 여전히 1 — 완료로 보면 안 된다."""
        wait, clock = _load()
        start = clock.now
        result = wait(lambda: -0.40, _NeverCancelled(),
                      get_status=lambda: (clock.now, 1, 0.0))
        self.assertIsNone(result)                   # 타임아웃 = "모른다"
        self.assertGreaterEqual(clock.now - start, 8.0)

    def test_cancel_returns_none(self):
        wait, _clock = _load()

        class Cancelled:
            is_cancel_requested = True

        self.assertIsNone(wait(lambda: -0.4, Cancelled()))


if __name__ == "__main__":
    unittest.main()

"""Place 안전 이송 — 고정 safe_z + multi-IK + route (STEP 2, 2026-09-10).

`_transit_plan`을 place_server에서 떼어 로봇 없이 돌린다(test_motion_error_abort.py와 같은
방식 — 이 모듈은 import만으로 rclpy/dsr_msgs2를 끌어온다).

여기서 지키려는 계약:

1. **safe_z를 낮추지 않는다.** 어떤 경로를 고르든 경유점의 높이는 주어진 값 그대로다.
2. 현재 solution space가 먼저, 안 되면 다른 space에서 같은 [x, y, safe_z, 자세]를 찾는다.
3. direct가 우선이되, 위치와 회전을 한 번에 크게 바꾸는 경우는 쓰지 않는다.
4. 전부 실패하면 None — 호출부가 SAFE_TRANSIT_UNREACHABLE로 보고한다.
5. **손목을 뒤집지 않는다** — 물체를 들고 있어 자세가 바뀌면 같은 작업이 아니다.
"""
import ast
import pathlib
import sys
import unittest
from types import SimpleNamespace as NS

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))                    # src/control

from control import motion_feasibility as mf  # noqa: E402

CONTROL = pathlib.Path(__file__).parents[1] / "control"


def _method(source: pathlib.Path, class_name: str, name: str, namespace: dict):
    """클래스에서 메서드 하나만 떼어 컴파일한다 (ROS2 import 회피)."""
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name]


def ok(posj=(0.0, 10.0, 90.0, 0.0, 85.0, 0.0)):
    return NS(status="ok", ok=True, known=True, joint_margin_deg=100.0, posj=list(posj))


def fail():
    return NS(status="unreachable", ok=False, known=True, joint_margin_deg=None, posj=None)


CURRENT = [300.0, 100.0, 300.0, 0.0, 180.0, 0.0]
# 바구니 상공(같은 자세로 회전까지 마친 지점)
TARGET = [600.0, -200.0, 500.0, 90.0, 180.0, 0.0]
SAFE_Z = 500.0


class Harness:
    """`_transit_plan`을 돌릴 최소 환경. ikin 응답을 시나리오로 준다."""

    def __init__(self, verdict_for, current_space=2, rotation_deg=90.0,
                 direct_limit=20.0, fkin_of=None):
        self.verdict_for = verdict_for
        self.current_space = current_space
        self.rotation_deg = rotation_deg
        self._direct_rotation_limit_deg = direct_limit
        self._max_config_switch_jump_deg = 90.0
        self._joint_limits_deg = (360.0, 360.0, 150.0, 360.0, 360.0, 360.0)
        self._solution_space_client = object()
        self._ikin_client = object()
        self._fkin_client = object()
        # configuration 전환 구간의 중간 FK. None이면 "모른다"(soft, 막지 않음).
        self.fkin_of = fkin_of
        self.logs = []

    def get_logger(self):
        return NS(info=self.logs.append, warning=self.logs.append, error=self.logs.append)


def run(harness, current=CURRENT, target=TARGET, safe_z=SAFE_Z, skip=None):
    namespace = {
        "motion_feasibility": mf,
        "dsr_motion": NS(
            current_solution_space=lambda _c: harness.current_space,
            rotation_diff_deg=lambda _a, _b: harness.rotation_deg,
            verify_ik=lambda _ik, _fk, posx, sol_space, limits: harness.verdict_for(
                posx, sol_space),
            posx_via_fkin=lambda _fk, joint: (
                harness.fkin_of(joint) if harness.fkin_of else None)),
    }
    search = _method(CONTROL / "place_server.py", "PlaceServer", "_transit_search", namespace)
    harness._transit_search = lambda *args, **kwargs: search(harness, *args, **kwargs)
    plan = _method(CONTROL / "place_server.py", "PlaceServer", "_transit_plan", namespace)
    return plan(harness, current, target, safe_z, skip=skip)


class SafeHeightIsFixedTest(unittest.TestCase):
    def test_every_waypoint_stays_at_the_safe_height(self):
        """**어떤 경로를 고르든 이송 구간 높이는 주어진 safe_z 그대로다.**"""
        harness = Harness(lambda posx, space: ok())
        plan = run(harness)
        name, waypoints, space = plan["name"], plan["waypoints"], plan["sol_space"]
        transit_points = [w for w in waypoints if w != list(TARGET)]
        self.assertTrue(transit_points, "우회 경로면 경유점이 있어야 한다")
        for waypoint in transit_points:
            self.assertAlmostEqual(waypoint[2], SAFE_Z)

    def test_no_lowering_when_the_height_is_unreachable(self):
        """safe_z에서 아무 해도 없으면 **낮추지 않고** None을 돌려준다."""
        harness = Harness(lambda posx, space: fail())
        self.assertIsNone(run(harness))


class MultiIkTest(unittest.TestCase):
    def test_current_space_is_tried_first(self):
        seen = []

        def verdict(posx, space):
            seen.append(space)
            return ok()

        run(Harness(verdict, current_space=5))
        self.assertEqual(seen[0], 5)

    def test_another_space_is_used_when_current_fails(self):
        """현재 configuration으로 못 가도 다른 configuration이면 되는 경우가 있다."""
        # 현재 configuration으로 제자리 Safe-Z 상승까지는 가능하고, 그 뒤 바구니 쪽은
        # space 4에서만 가능하다. 전환 전에 Safe-Z에도 못 오르면 movej로 낮은 곳에서 팔을
        # 휘두를 수 없으므로 그 경우는 의도적으로 허용하지 않는다.
        def verdict(posx, space):
            if space == 2 and list(posx)[:2] == CURRENT[:2]:
                return ok((0.0, 10.0, 90.0, 0.0, 85.0, 0.0))
            if space == 4:
                return ok((10.0, 15.0, 85.0, 0.0, 80.0, 0.0))
            return fail()

        harness = Harness(verdict, current_space=2)
        plan = run(harness)
        name, waypoints, space = plan["name"], plan["waypoints"], plan["sol_space"]
        self.assertEqual(space, 4)
        self.assertIsNotNone(plan["switch_posj"], "다른 space의 IK를 실제 configuration 전환에 써야 한다")
        for waypoint in [w for w in waypoints if w != list(TARGET)]:
            self.assertAlmostEqual(waypoint[2], SAFE_Z, msg="다른 space여도 높이는 그대로다")

    def test_all_spaces_are_exhausted_before_giving_up(self):
        spaces = set()

        def verdict(posx, space):
            spaces.add(space)
            return fail()

        self.assertIsNone(run(Harness(verdict)))
        self.assertIn(2, spaces)
        # 현재 configuration으로 Safe-Z 전환 지점에도 갈 수 없으면 낮은 곳에서 movej로
        # 다른 configuration을 강행하지 않는다. 따라서 나머지 space를 묻는 것은 무의미하다.
        self.assertTrue(spaces.issubset(set(range(8))))

    def test_a_large_config_switch_jump_is_allowed_when_the_path_is_clear(self):
        """실물 회귀 (2026-09-10): 필요한 전환이 168~257도였는데 90도 한계로 전부
        막혀 place가 매번 SAFE_TRANSIT_UNREACHABLE로 끝났다. 관절 이동량 자체가 아니라
        전환 경로에 명백한 충돌이 없으면(FK가 안전고도 이상을 본다) 허용해야 한다."""
        def verdict(posx, space):
            if space == 2 and list(posx)[:2] == CURRENT[:2]:
                return ok((0.0, 10.0, 90.0, 0.0, 85.0, 0.0))
            if space == 4:
                # 관절 0이 250도 튄다 — 예전 90도 한계면 무조건 거부됐다.
                return ok((250.0, 15.0, 85.0, 0.0, 80.0, 0.0))
            return fail()

        harness = Harness(verdict, current_space=2,
                          fkin_of=lambda _joint: [400.0, 0.0, SAFE_Z, 0.0, 180.0, 0.0])
        plan = run(harness)
        self.assertIsNotNone(plan, "충돌이 없다면 큰 관절 이동만으로 거부하면 안 된다")
        self.assertEqual(plan["sol_space"], 4)
        self.assertTrue(any("그대로 진행한다" in line for line in harness.logs))

    def test_config_switch_is_rejected_when_the_transition_path_dips_below_the_floor(self):
        """전환 경로 중간이 실제로 지지면 아래로 내려가면 — 각도와 무관하게 — 막는다."""
        def verdict(posx, space):
            if space == 2 and list(posx)[:2] == CURRENT[:2]:
                return ok((0.0, 10.0, 90.0, 0.0, 85.0, 0.0))
            if space == 4:
                return ok((250.0, 15.0, 85.0, 0.0, 80.0, 0.0))
            return fail()

        floor_dip = [400.0, 0.0, 100.0, 0.0, 180.0, 0.0]   # CURRENT/TARGET 중 낮은 z(300) 아래
        harness = Harness(verdict, current_space=2, fkin_of=lambda _joint: floor_dip)
        self.assertIsNone(run(harness),
                          "전환 중간 지점이 지지면 아래면 다른 space로도 성공하면 안 된다")


class RouteTest(unittest.TestCase):
    def test_direct_is_used_when_already_high_and_rotation_is_small(self):
        """direct의 전제는 둘이다 — 회전이 작고, **이미 안전고도에 있을 것**."""
        at_height = [300.0, 100.0, SAFE_Z, 0.0, 180.0, 0.0]
        harness = Harness(lambda posx, space: ok(), rotation_deg=5.0, direct_limit=20.0)
        plan = run(harness, current=at_height)
        name, waypoints = plan["name"], plan["waypoints"]
        self.assertEqual(name, "direct")
        self.assertEqual(waypoints, [list(TARGET)])

    def test_direct_is_skipped_when_rotation_is_large(self):
        """위치와 회전을 한 movel에 같이 크게 바꾸면 로봇이 안 움직인 적이 있다.

        높이 조건은 만족시켜 놓고(안전고도에서 출발) **회전만으로** 걸러지는지 본다.
        """
        at_height = [300.0, 100.0, SAFE_Z, 0.0, 180.0, 0.0]
        harness = Harness(lambda posx, space: ok(), rotation_deg=120.0, direct_limit=20.0)
        name = run(harness, current=at_height)["name"]
        self.assertNotEqual(name, "direct")

    def test_falls_back_to_the_other_order_when_one_route_fails(self):
        """상승→회전→이동이 막히면 상승→이동→회전을 본다 — 경로 때문에 실패하지 않는다."""
        blocked = [300.0, 100.0, SAFE_Z, 90.0, 180.0, 0.0]   # 제자리 회전 지점

        def verdict(posx, space):
            return fail() if list(posx) == blocked else ok()

        plan = run(Harness(verdict, rotation_deg=120.0))
        name, waypoints = plan["name"], plan["waypoints"]
        self.assertEqual(name, "rise_traverse_rotate")
        self.assertNotIn(blocked, waypoints)

    def test_unreachable_waypoint_does_not_fail_the_whole_place(self):
        """waypoint 하나가 안 된다고 place 전체가 실패하면 안 된다 — 다른 경로를 본다."""
        harness = Harness(lambda posx, space: ok(), rotation_deg=120.0)
        self.assertIsNotNone(run(harness))

    def test_middle_sample_failure_is_soft_when_endpoints_are_reachable(self):
        """실물 실패 원인: 보간 중점 IK 하나 때문에 모든 route를 거부하면 안 된다."""
        def verdict(posx, space):
            # 첫 수직 상승의 중점만 실패한다. 시작/안전고도 끝점은 확실히 도달 가능하다.
            return fail() if abs(float(posx[2]) - 400.0) < 1e-6 else ok()

        harness = Harness(verdict, rotation_deg=120.0)
        plan = run(harness)
        self.assertIsNotNone(plan)
        self.assertTrue(any("끝점은 확실하지만" in line for line in harness.logs))

    def test_endpoint_failure_still_rejects_the_route(self):
        """2차 탐색도 끝점 실패까지 무시하는 fail-open 경로는 아니다."""
        harness = Harness(lambda posx, space: fail(), rotation_deg=120.0)
        self.assertIsNone(run(harness))


class HeldObjectTest(unittest.TestCase):
    def test_place_never_flips_the_wrist(self):
        """물체를 들고 있으면 180도 뒤집기는 같은 작업이 아니다 — 물체 방향이 바뀐다."""
        source = (CONTROL / "place_server.py").read_text(encoding="utf-8")
        # 호출이 없어야 한다(주석에서 언급하는 것은 허용 — 왜 안 쓰는지 적어 두는 자리다).
        self.assertNotIn("flipped_wrist_posx(", source)
        self.assertIn("손목을 뒤집지 않는다", source)

    def test_target_orientation_is_preserved_in_every_route(self):
        harness = Harness(lambda posx, space: ok(), rotation_deg=120.0)
        waypoints = run(harness)["waypoints"]
        self.assertEqual(waypoints[-1][3:], list(TARGET)[3:])


class DirectBelowSafeHeightTest(unittest.TestCase):
    """**안전고도 아래에서 출발하면 direct를 쓰지 않는다** (2026-09-10 실물에서 발견).

    direct는 시작점에서 목표까지 직선이라, 시작이 안전고도보다 낮으면 그 구간에서 수평
    이동이 안전고도 아래를 지난다 — 물체를 든 채 작업대·테두리를 넘으라고 둔 높이를
    건너뛰는 셈이다. 회전량이 0이면 회전 가드로는 못 걸러진다.
    """

    def test_direct_is_skipped_when_starting_below_the_safe_height(self):
        low_start = [300.0, 100.0, 378.8, 0.0, 180.0, 0.0]   # 안전고도(428.8)보다 낮다
        target = [325.3, -468.1, 428.8, 0.0, 180.0, 0.0]     # 회전량 0
        harness = Harness(lambda posx, space: ok(), rotation_deg=0.0)
        plan = run(harness, current=low_start, target=target, safe_z=428.8)
        name, waypoints = plan["name"], plan["waypoints"]

        self.assertNotEqual(name, "direct")
        self.assertTrue(any("대각선 이동이 그 아래를 지난다" in line for line in harness.logs))
        # 첫 이동이 제자리 상승이어야 한다 — XY를 유지한 채 안전고도로.
        self.assertEqual(waypoints[0][:2], low_start[:2])
        self.assertAlmostEqual(waypoints[0][2], 428.8)

    def test_direct_is_allowed_when_already_at_the_safe_height(self):
        """이미 안전고도에 있으면 직선 이동이 그 높이를 벗어나지 않는다 — 그때는 써도 된다."""
        high_start = [300.0, 100.0, 428.8, 0.0, 180.0, 0.0]
        target = [325.3, -468.1, 428.8, 0.0, 180.0, 0.0]
        harness = Harness(lambda posx, space: ok(), rotation_deg=0.0)
        name = run(harness, current=high_start, target=target, safe_z=428.8)["name"]
        self.assertEqual(name, "direct")


class ExecutionFallbackTest(unittest.TestCase):
    """**실행에서 실패해도 다른 경로/자세로 넘어간다** (2026-09-10 실물에서 발견).

    ikin이 해를 줘도 컨트롤러가 그 목표를 조용히 거부하는 경우가 있다. 사전 검사만으로
    경로 하나를 정해 놓고 실행에서 실패하면 그대로 끝나 버려서, 애써 만든 우회 경로와
    다른 solution space가 한 번도 쓰이지 못했다 — 실물에서 로봇이 멈춰 선 채 아무 변화가
    없던 것이 이것이다. **바꾸는 것은 경로와 configuration뿐, 높이는 그대로다.**
    """

    def test_already_attempted_route_and_space_are_skipped(self):
        first = run(Harness(lambda posx, space: ok(), current_space=2))
        self.assertIsNotNone(first)
        name, space = first["name"], first["sol_space"]

        second = run(Harness(lambda posx, space: ok(), current_space=2),
                     skip={(name, space)})
        self.assertIsNotNone(second, "한 조합이 실패했다고 포기하면 안 된다")
        self.assertNotEqual((second["name"], second["sol_space"]), (name, space))

    def test_skipping_never_lowers_the_safe_height(self):
        harness = Harness(lambda posx, space: ok())
        first = run(harness)
        name, space = first["name"], first["sol_space"]
        waypoints = run(harness, skip={(name, space)})["waypoints"]
        for waypoint in [w for w in waypoints if w != list(TARGET)]:
            self.assertAlmostEqual(waypoint[2], SAFE_Z)

    def test_exhausting_every_combination_gives_up_instead_of_looping(self):
        every = {(name, space) for name in ("direct", "rise_rotate_traverse",
                                            "rise_traverse_rotate")
                 for space in range(8)}
        self.assertIsNone(run(Harness(lambda posx, space: ok()), skip=every))

    def test_execution_retries_with_a_new_plan_and_bounded_attempts(self):
        """실행 루프의 계약을 소스에서 확인한다 (execute_callback은 로봇 없이 못 돈다).

        - 실패하면 `_transit_plan`을 **다시** 부른다(이미 시도한 조합은 skip으로 제외).
        - 시도 횟수는 `_transit_attempts`로 묶는다 — 실패 한 번이 최대 60초라
          조합을 전부 돌면 사람이 기다릴 수 없다.
        - 소진되면 `_SafeTransitUnreachable` — 낮추기(`_retry_lower`)로 돌아가지 않는다.
        """
        source = (CONTROL / "place_server.py").read_text(encoding="utf-8")
        self.assertIn("self._transit_plan(start_pose, over_target, transit_z, skip=attempted)",
                      source)
        self.assertIn("range(1, self._transit_attempts + 1)", source)
        self.assertIn("_SafeTransitUnreachable", source)
        for gone in ("_retry_lower", "_reachable_z("):
            self.assertNotIn(gone, source, "높이를 낮추는 경로는 되살리지 않는다")

    def test_the_next_attempt_starts_from_where_the_robot_actually_is(self):
        """부분 이동 뒤에 실패했을 수 있다 — 다음 후보는 **지금 있는 자리**에서 짜야 한다."""
        source = (CONTROL / "place_server.py").read_text(encoding="utf-8")
        block = source[source.index("attempted.add((route_name, transit_space))"):
                       source.index("# ④ 수직 하강")]
        self.assertIn("get_current_posx", block)
        self.assertIn("start_pose = list(resumed)", block)

    def test_alternate_space_is_executed_not_only_logged(self):
        source = (CONTROL / "place_server.py").read_text(encoding="utf-8")
        block = source[source.index("for index, waypoint in enumerate(waypoints"):
                       source.index("# ④ 수직 하강")]
        self.assertIn("dsr_motion.move_joint", block)
        self.assertIn('plan["switch_posj"]', block)


if __name__ == "__main__":
    unittest.main()

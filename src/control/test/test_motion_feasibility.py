"""모션 실행 가능성 규칙 — 로봇도 ROS2도 없이 (STEP 2, 2026-09-10).

여기서 지키려는 계약:

1. **현재 solution space를 먼저 본다.** 그다음에야 드라이버가 지원하는 나머지(0~7)를 본다.
2. **첫 IK 실패로 끝내지 않는다.** 다른 자세 표현·다른 space에 해가 있으면 그걸 쓴다.
3. **끝점만 보지 않는다.** 중간 지점의 IK와 관절 연속성까지 본다.
4. **direct가 항상 먼저다.** 우회 경로는 direct가 안 될 때만, 그리고 안전 높이를 낮추지
   않은 채로만 쓴다.
5. **모르면 막지 않는다.** ikin 무응답도, 정보 없는 충돌 검사도 후보를 죽이지 않는다.
"""
import pathlib
import sys
import unittest
from types import SimpleNamespace as NS

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))                    # src/control

from control import motion_feasibility as mf  # noqa: E402


def ok(posj=(0.0, 10.0, 90.0, 0.0, 85.0, 0.0)):
    return NS(status="ok", ok=True, known=True, joint_margin_deg=100.0, posj=list(posj))


def fail():
    return NS(status="unreachable", ok=False, known=True, joint_margin_deg=None, posj=None)


def unknown():
    return NS(status="unknown", ok=False, known=False, joint_margin_deg=None, posj=None)


def posx(x=400.0, y=0.0, z=300.0, rot=(0.0, 180.0, 0.0)):
    return [x, y, z, *rot]


class SolutionSpaceOrderTest(unittest.TestCase):
    def test_driver_range_is_zero_to_seven(self):
        """dsr_msgs2/Ikin.srv의 `sol_space : 0 ~ 7`을 그대로 쓴다 — 추측 범위가 아니다."""
        self.assertEqual(mf.SOLUTION_SPACES, tuple(range(8)))

    def test_current_space_comes_first(self):
        order = mf.solution_space_order(2)
        self.assertEqual(order[0], 2)
        self.assertEqual(sorted(order), list(range(8)))

    def test_unknown_current_space_still_checks_all(self):
        self.assertEqual(mf.solution_space_order(None), list(range(8)))


class FindIkTest(unittest.TestCase):
    def test_first_space_failure_moves_to_the_next_space(self):
        seen = []

        def ik(posx_value, space):
            seen.append(space)
            return ok() if space == 5 else fail()

        found = mf.find_ik([(posx(), "original")], ik, mf.solution_space_order(2))
        self.assertIsNotNone(found)
        self.assertEqual(found.sol_space, 5)
        self.assertEqual(seen[0], 2, "현재 space를 먼저 물어봐야 한다")

    def test_equivalent_wrist_is_tried_when_the_original_fails(self):
        original, flipped = posx(), posx(rot=(180.0, 180.0, 0.0))

        def ik(posx_value, space):
            return ok() if list(posx_value) == flipped else fail()

        found = mf.find_ik([(original, "original"), (flipped, "equivalent_wrist")],
                           ik, mf.solution_space_order(2))
        self.assertIsNotNone(found)
        self.assertEqual(found.representation, "equivalent_wrist")

    def test_preferred_representation_wins_when_both_work(self):
        found = mf.find_ik([(posx(), "original"), (posx(), "equivalent_wrist")],
                           lambda p, s: ok(), mf.solution_space_order(2))
        self.assertEqual(found.representation, "original")
        self.assertEqual(found.sol_space, 2, "현재 space에서 되면 거기서 끝난다")

    def test_no_solution_anywhere_returns_none(self):
        self.assertIsNone(mf.find_ik([(posx(), "original")], lambda p, s: fail(),
                                     mf.solution_space_order(2)))

    def test_unknown_is_not_treated_as_success(self):
        """무응답을 성공으로 치면 실행에서 그대로 실패한다."""
        self.assertIsNone(mf.find_ik([(posx(), "original")], lambda p, s: unknown(),
                                     mf.solution_space_order(2)))


class PathSamplesTest(unittest.TestCase):
    def test_endpoints_are_included_and_position_is_interpolated(self):
        samples = mf.path_samples(posx(z=300.0), posx(z=400.0), count=3)
        self.assertEqual([s[2] for s in samples], [300.0, 350.0, 400.0])

    def test_rotation_is_held_at_the_target_not_interpolated(self):
        """ZYZ 중간값 보간은 ry가 180도 근처에서 엉뚱한 자세를 만든다 — 끝점 자세를 쓴다."""
        samples = mf.path_samples(posx(rot=(0.0, 180.0, 0.0)),
                                  posx(rot=(90.0, 170.0, 30.0)), count=4)
        for sample in samples:
            self.assertEqual(sample[3:], [90.0, 170.0, 30.0])


class PathFeasibleTest(unittest.TestCase):
    def test_endpoint_ok_but_middle_unreachable_is_infeasible(self):
        """끝점만 보면 통과하는 경로를 중간에서 걸러낸다 — 이게 이 검사의 존재 이유다."""
        def ik(posx_value, space):
            return fail() if abs(posx_value[2] - 350.0) < 1e-6 else ok()

        check = mf.path_feasible(posx(z=300.0), posx(z=400.0), ik, 2)
        self.assertFalse(check.feasible)
        self.assertIn("IK 실패", check.reason)

    def test_large_joint_jump_between_samples_is_infeasible(self):
        poses = [ok((0.0, 10, 90, 0, 85, 0)), ok((150.0, 10, 90, 0, 85, 0)),
                 ok((150.0, 10, 90, 0, 85, 0))]

        def ik(posx_value, space):
            return poses.pop(0) if poses else ok()

        check = mf.path_feasible(posx(z=300.0), posx(z=400.0), ik, 2)
        self.assertFalse(check.feasible)
        self.assertIn("관절이", check.reason)

    def test_smooth_path_is_feasible(self):
        check = mf.path_feasible(posx(z=300.0), posx(z=400.0), lambda p, s: ok(), 2)
        self.assertTrue(check.feasible)
        self.assertEqual(check.checked, 3)
        self.assertEqual(check.max_joint_jump_deg, 0.0)

    def test_unknown_ik_does_not_block_the_path(self):
        """ikin이 죽었다고 모든 경로를 막으면 로봇이 통째로 멈춘다."""
        check = mf.path_feasible(posx(z=300.0), posx(z=400.0), lambda p, s: unknown(), 2)
        self.assertTrue(check.feasible)
        self.assertEqual(check.unknown, 3)

    def test_obstacle_blocks_the_path(self):
        obstacle = mf.support_plane_obstacle(floor_z=280.0)
        check = mf.path_feasible(posx(z=300.0), posx(z=260.0), lambda p, s: ok(), 2,
                                 obstacle=obstacle)
        self.assertFalse(check.feasible)
        self.assertIn("지지면", check.reason)


class RoutesTest(unittest.TestCase):
    def test_direct_is_always_first(self):
        options = mf.routes(posx(), posx(x=600.0), transit_z=500.0)
        self.assertEqual(options[0][0], "direct")
        self.assertEqual(options[0][1], [posx(x=600.0)])

    def test_detours_use_the_given_transit_height_only(self):
        """우회 경로가 안전 높이를 낮추지 않는다 — 낮추면 그 높이를 둔 이유가 사라진다."""
        options = mf.routes(posx(z=300.0), posx(x=600.0, z=310.0), transit_z=500.0)
        for name, waypoints in options[1:]:
            heights = [point[2] for point in waypoints[:-1]]
            self.assertTrue(all(abs(h - 500.0) < 1e-6 for h in heights), (name, heights))

    def test_two_detour_orders_are_offered(self):
        names = [name for name, _ in mf.routes(posx(), posx(x=600.0), transit_z=500.0)]
        self.assertEqual(names, ["direct", "rise_rotate_traverse", "rise_traverse_rotate"])


class ObstacleTest(unittest.TestCase):
    def test_unknown_floor_creates_no_check(self):
        """정보가 없으면 막지 않는다 — 무조건 collision 처리하지 않는다."""
        self.assertIsNone(mf.support_plane_obstacle(None))

    def test_unknown_box_creates_no_rim_check(self):
        self.assertIsNone(mf.rim_crossing_obstacle(None, 100.0))
        self.assertIsNone(mf.rim_crossing_obstacle([(0.0, 0.0)], None))

    def test_rim_crossing_is_detected_only_below_the_rim(self):
        corners = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        crosses = mf.rim_crossing_obstacle(corners, rim_z=200.0)
        inside_low, outside_low = posx(50.0, 50.0, 150.0), posx(300.0, 50.0, 150.0)
        inside_high, outside_high = posx(50.0, 50.0, 250.0), posx(300.0, 50.0, 250.0)
        self.assertIn("벽을 가로지른다", crosses(inside_low, outside_low))
        self.assertEqual(crosses(inside_high, outside_high), "",
                         "테두리보다 위로 지나가는 것은 벽과 무관하다")
        self.assertEqual(crosses(inside_low, inside_low), "")

    def test_segment_helper_finds_the_first_violation(self):
        corners = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        crosses = mf.rim_crossing_obstacle(corners, rim_z=200.0)
        samples = [posx(50.0, 50.0, 150.0), posx(300.0, 50.0, 150.0)]
        self.assertIn("벽을 가로지른다", mf.segment_obstacle_hit(samples, crosses))
        self.assertEqual(mf.segment_obstacle_hit(samples, None), "")

    def test_combine_skips_missing_checks(self):
        combined = mf.combine_obstacles(None, mf.support_plane_obstacle(280.0), None)
        self.assertIn("지지면", combined(posx(z=200.0)))
        self.assertEqual(combined(posx(z=300.0)), "")
        self.assertIsNone(mf.combine_obstacles(None, None))


if __name__ == "__main__":
    unittest.main()

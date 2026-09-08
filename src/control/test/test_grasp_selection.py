"""후보 선택 규칙 단위 테스트 — 로봇도 ROS2도 없이 판정 전체를 돌린다.

`grasp_selection`은 IK 질의와 좌표 계산을 콜러블로 받으므로, 여기서 가짜를 넣어
"어떤 후보가 왜 탈락하고 무엇이 뽑히는지"를 그대로 검증할 수 있다. 실물 실행 테스트와
분리해 두는 이유다 — 이 판정들은 로봇을 켜지 않고도 매번 확인돼야 한다.
"""
import math
import pathlib
import sys
import unittest
from types import SimpleNamespace as NS

# 설치본(install/)을 source하지 않아도 돌게 소스 트리를 직접 넣는다 — 이 테스트의 요점은
# "로봇 없이 판정 규칙을 확인한다"이므로 워크스페이스 빌드에 의존하지 않는 편이 낫다.
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))                    # src/control
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "perception_common"))

from control import grasp_selection as gs  # noqa: E402


def pose(x=400.0, y=0.0, z=300.0, quaternion=(0.0, 1.0, 0.0, 0.0)):
    """수직 파지(접근축 = -Z)가 기본. quaternion은 (x, y, z, w)."""
    qx, qy, qz, qw = quaternion
    return NS(position=NS(x=x, y=y, z=z),
              orientation=NS(x=qx, y=qy, z=qz, w=qw))


def candidate(rank=0, score=0.9, width=40.0, x=400.0, y=0.0, z=300.0, object_id="obj_001"):
    return gs.Candidate(candidate_id=f"{object_id}#{rank}", rank=rank, pose=pose(x, y, z),
                        score=score, gripper_width_mm=width, strategy="graspnet_baseline")


def obj(center=(400.0, 0.0, 310.0), height=30.0, depth_ratio=0.9):
    return gs.ObjectContext(object_id="obj_001", center_mm=center,
                            height_mm=height, depth_valid_ratio=depth_ratio)


class FakeIk:
    """ikin/fkin 대역. `fail`에 든 (x,y,z)에 대해 정해진 판정을 돌려준다."""

    def __init__(self, verdicts=None, default_margin=120.0):
        self.verdicts = verdicts or {}
        self.default_margin = default_margin
        self.calls = []

    def __call__(self, posx):
        self.calls.append(list(posx))
        key = tuple(round(v, 3) for v in posx[:3])
        verdict = self.verdicts.get(key)
        if verdict is not None:
            return verdict
        return NS(status=gs.STATUS_VALID, ok=True, known=True,
                  joint_margin_deg=self.default_margin,
                  posj=[0.0, 10.0, 90.0, 0.0, 85.0, 0.0])


def ik_fail():
    return NS(status="unreachable", ok=False, known=True, joint_margin_deg=None, posj=None)


def ik_joint_limit(margin=-12.0):
    return NS(status=gs.IK_JOINT_LIMIT, ok=False, known=True, joint_margin_deg=margin,
              posj=[0.0, 0.0, 162.0, 0.0, 85.0, 0.0])


def ik_unknown():
    return NS(status="unknown", ok=False, known=False, joint_margin_deg=None, posj=None)


def geometry_of(candidate_pose):
    """수직 접근을 가정한 단순 기하 — TCP 오프셋 없이 파지점 그대로.

    실제 pick_server는 dsr_motion.plan_pick_posx를 넘긴다. 여기서 검증하려는 것은
    좌표 변환이 아니라 판정 규칙이므로, 값이 그대로 흐르는 가장 단순한 형태를 쓴다.
    """
    p = candidate_pose.position
    target = [float(p.x), float(p.y), float(p.z), 0.0, 180.0, 0.0]
    approach = [float(p.x), float(p.y), float(p.z) + 80.0, 0.0, 180.0, 0.0]
    return gs.PickGeometry(target_posx=target, approach_posx=approach,
                           approach_axis=[0.0, 0.0, -1.0],
                           pad_reference_mm=[float(p.x), float(p.y), float(p.z)])


def run(candidates, context=None, params=None, ik=None):
    params = params or gs.SelectionParams()
    ik = ik or FakeIk()
    evaluations = gs.evaluate_candidates(candidates, context or obj(), params,
                                         geometry_of, ik)
    return evaluations, gs.select(evaluations)


class TestStatusContract(unittest.TestCase):
    def test_joint_limit_status_matches_dsr_motion(self):
        """grasp_selection이 ROS2를 import하지 않으려고 복제한 문자열이 어긋나지 않았는지.

        어긋나면 관절 한계 탈락이 조용히 ik_failed로 분류된다.
        """
        source = (pathlib.Path(__file__).parents[1] / "control" / "dsr_motion.py"
                  ).read_text(encoding="utf-8")
        self.assertIn(f'IK_JOINT_LIMIT = "{gs.IK_JOINT_LIMIT}"', source)
        self.assertIn(f'IK_OK = "{gs.IK_OK}"', source)

    def test_garbage_joints_from_a_failed_ik_are_not_reported(self):
        """해가 없을 때 ikin이 채워 보내는 관절값으로 여유를 계산하면 안 된다.

        실측(2026-09-07): 팔 밖 목표에 conv_posj=[-7203, -2186, ...]도가 왔고, 그대로
        쓰면 로그에 joint_margin=-14960deg가 찍혀 "관절 한계 문제"로 오해된다.
        """
        garbage = NS(status="unreachable", ok=False, known=True, joint_margin_deg=-14960.0,
                     posj=[-7203.0, -2186.0, -7732.0, -10614.0, 14046.0, -15261.0])
        ik = FakeIk({(400.0, 0.0, 300.0): garbage, (400.0, 0.0, 380.0): garbage})
        evaluations, _ = run([candidate(0)], ik=ik)
        self.assertEqual(evaluations[0].status, gs.STATUS_IK_FAILED)
        self.assertIsNone(evaluations[0].joint_margin_deg)
        self.assertIsNone(evaluations[0].wrist_j5_deg)
        self.assertNotIn("joint_margin=", gs.log_line(evaluations[0]))


class TestCase1AllValid(unittest.TestCase):
    """Case 1 — 후보가 전부 valid면 ranking으로 하나를 고른다."""

    def test_ranking_can_beat_the_top_score(self):
        # #0: 점수는 최고지만 물체 중심에서 25mm 벗어나 있고 개폭이 최대개폭에 가깝다.
        # #1: 점수는 낮지만 중심에 정확히 있고 개폭도 적당하다.
        candidates = [candidate(0, score=0.95, width=105.0, x=425.0),
                      candidate(1, score=0.80, width=40.0, x=400.0)]
        evaluations, selected = run(candidates)
        self.assertEqual([e.status for e in evaluations],
                         [gs.STATUS_VALID, gs.STATUS_SELECTED])
        self.assertEqual(selected.candidate.candidate_id, "obj_001#1")

    def test_top_score_wins_when_everything_else_ties(self):
        """다른 조건이 같으면 GraspNet 점수가 그대로 순위가 된다(회귀 방지)."""
        candidates = [candidate(0, score=0.9), candidate(1, score=0.5)]
        _, selected = run(candidates)
        self.assertEqual(selected.candidate.candidate_id, "obj_001#0")

    def test_tie_falls_back_to_input_rank(self):
        candidates = [candidate(0, score=0.7), candidate(1, score=0.7)]
        _, selected = run(candidates)
        self.assertEqual(selected.candidate.rank, 0)


class TestCase2WidthInvalid(unittest.TestCase):
    """Case 2 — 최고 점수 후보의 개폭을 못 믿으면 다음 후보로 넘어간다."""

    def test_zero_width_top_candidate_is_skipped(self):
        candidates = [candidate(0, score=0.99, width=0.0), candidate(1, score=0.4, width=35.0)]
        evaluations, selected = run(candidates)
        self.assertEqual(evaluations[0].status, gs.STATUS_WIDTH_INVALID)
        self.assertIn("개폭 미상", evaluations[0].rejection_reason)
        self.assertEqual(selected.candidate.candidate_id, "obj_001#1")

    def test_width_below_min_grip_is_rejected(self):
        """닫기 목표가 min_grip_width_mm에서 멈춰 물체에 닿지도 못하는 폭."""
        evaluations, selected = run([candidate(0, width=3.0), candidate(1, width=30.0)])
        self.assertEqual(evaluations[0].status, gs.STATUS_WIDTH_INVALID)
        self.assertEqual(selected.candidate.rank, 1)

    def test_width_above_max_opening_is_rejected(self):
        evaluations, selected = run([candidate(0, width=130.0), candidate(1, width=30.0)])
        self.assertEqual(evaluations[0].status, gs.STATUS_WIDTH_INVALID)
        self.assertEqual(selected.candidate.rank, 1)

    def test_width_invalid_candidate_costs_no_ik_call(self):
        """개폭에서 떨어진 후보는 IK를 묻지 않는다 — ikin 한 번이 약 31ms다."""
        ik = FakeIk()
        run([candidate(0, width=0.0), candidate(1, width=30.0)], ik=ik)
        self.assertEqual(len(ik.calls), 2)   # 살아남은 후보의 접근·파지 2회뿐


class TestCase3IkFailed(unittest.TestCase):
    """Case 3 — IK가 안 풀리는 후보 하나 때문에 pick 전체가 실패하지 않는다."""

    def test_grasp_ik_failure_moves_to_next_candidate(self):
        ik = FakeIk({(400.0, 0.0, 300.0): ik_fail()})
        evaluations, selected = run([candidate(0, x=400.0), candidate(1, x=450.0)], ik=ik)
        self.assertEqual(evaluations[0].status, gs.STATUS_IK_FAILED)
        self.assertIs(evaluations[0].grasp_ik, False)
        self.assertEqual(selected.candidate.rank, 1)

    def test_approach_ik_failure_is_caught_too(self):
        """파지점은 되는데 그 위 접근점이 안 되는 경우 — 실행에서 첫 이동이 실패한다."""
        ik = FakeIk({(400.0, 0.0, 380.0): ik_fail()})
        evaluations, selected = run([candidate(0, x=400.0), candidate(1, x=450.0)], ik=ik)
        self.assertEqual(evaluations[0].status, gs.STATUS_IK_FAILED)
        self.assertIs(evaluations[0].approach_ik, False)
        self.assertEqual(selected.candidate.rank, 1)

    def test_unknown_ik_does_not_reject(self):
        """ikin 무응답은 '불가'가 아니다 — 그렇게 다루면 서비스가 죽었을 때 pick이 전멸한다."""
        ik = FakeIk({(400.0, 0.0, 300.0): ik_unknown(), (400.0, 0.0, 380.0): ik_unknown()})
        evaluations, selected = run([candidate(0)], ik=ik)
        self.assertEqual(evaluations[0].status, gs.STATUS_SELECTED)
        self.assertIsNone(evaluations[0].approach_ik)
        self.assertIsNone(evaluations[0].joint_ok)
        self.assertIsNotNone(selected)


class TestCase4JointLimit(unittest.TestCase):
    """Case 4 — 관절 한계를 넘는 후보는 ik_failed와 구분해 표시하고 건너뛴다."""

    def test_joint_limit_candidate_is_skipped_with_its_own_status(self):
        ik = FakeIk({(400.0, 0.0, 300.0): ik_joint_limit()})
        evaluations, selected = run([candidate(0, x=400.0), candidate(1, x=450.0)], ik=ik)
        self.assertEqual(evaluations[0].status, gs.STATUS_JOINT_LIMIT)
        self.assertIs(evaluations[0].joint_ok, False)
        self.assertEqual(selected.candidate.rank, 1)

    def test_joint_margin_prefers_the_roomier_candidate(self):
        """다른 조건이 같으면 관절 여유가 큰 쪽이 이긴다."""
        tight = NS(status=gs.STATUS_VALID, ok=True, known=True, joint_margin_deg=2.0,
                   posj=[0.0, 0.0, 148.0, 0.0, 85.0, 0.0])
        # #0만 관절 여유가 2도다. 0.5mm 옆으로 옮긴 #1은 중심 근접도에서 아주 조금
        # 손해를 보지만(가중치 0.20 x 0.025), 관절 여유 차이(0.10 x 0.93)가 훨씬 크다.
        ik = FakeIk({(400.0, 0.0, 300.0): tight, (400.0, 0.0, 380.0): tight})
        candidates = [candidate(0, score=0.9, x=400.0), candidate(1, score=0.9, x=400.5)]
        evaluations, selected = run(candidates, ik=ik)
        self.assertAlmostEqual(evaluations[0].joint_margin_deg, 2.0)
        self.assertEqual(selected.candidate.rank, 1)


class TestCase5NoneValid(unittest.TestCase):
    """Case 5 — 전부 탈락하면 선택 없음. 호출부가 로봇을 움직이지 않고 실패로 끝낸다."""

    def test_all_invalid_returns_none(self):
        ik = FakeIk({(400.0, 0.0, 300.0): ik_fail(), (400.0, 0.0, 380.0): ik_fail()})
        evaluations, selected = run([candidate(0, width=0.0), candidate(1)], ik=ik)
        self.assertIsNone(selected)
        self.assertTrue(all(e.rejected_before_motion for e in evaluations))
        tally = gs.counts(evaluations)
        self.assertEqual(tally[gs.STATUS_WIDTH_INVALID], 1)
        self.assertEqual(tally[gs.STATUS_IK_FAILED], 1)
        self.assertEqual(tally[gs.STATUS_VALID], 0)

    def test_empty_candidate_list_selects_nothing(self):
        evaluations, selected = run([])
        self.assertEqual(evaluations, [])
        self.assertIsNone(selected)


class TestMinimumSafety(unittest.TestCase):
    """최소 안전 검사 — 지지면 관통과 설정된 절대 바닥만 본다(collision 검사는 없다)."""

    def test_pad_below_support_is_rejected(self):
        # 지지면 = 310 - 30 = 280mm. 허용 5mm이므로 274mm는 뚫는다.
        evaluations, selected = run([candidate(0, z=274.0), candidate(1, z=300.0)])
        self.assertEqual(evaluations[0].status, gs.STATUS_SAFETY_INVALID)
        self.assertIn("지지면", evaluations[0].rejection_reason)
        self.assertEqual(selected.candidate.rank, 1)

    def test_within_tolerance_is_allowed(self):
        """모델 오차만큼(support_tolerance_mm)은 허용한다 — 아니면 멀쩡한 후보가 죽는다."""
        evaluations, _ = run([candidate(0, z=277.0)])
        self.assertEqual(evaluations[0].status, gs.STATUS_SELECTED)

    def test_unknown_height_skips_the_support_check(self):
        """height_mm이 0(미상)이면 지지면을 모른다 — 모른다고 후보를 죽이지 않는다."""
        evaluations, selected = run([candidate(0, z=100.0)], context=obj(height=0.0))
        self.assertEqual(evaluations[0].status, gs.STATUS_SELECTED)
        self.assertIsNotNone(selected)

    def test_absolute_floor_is_off_by_default(self):
        self.assertIsNone(gs.SelectionParams().min_tcp_z_mm)

    def test_absolute_floor_rejects_when_configured(self):
        params = gs.SelectionParams(min_tcp_z_mm=290.0)
        evaluations, _ = run([candidate(0, z=285.0)], context=obj(height=0.0), params=params)
        self.assertEqual(evaluations[0].status, gs.STATUS_SAFETY_INVALID)
        self.assertIn("작업대 하한", evaluations[0].rejection_reason)


class TestScoreTerms(unittest.TestCase):
    def test_all_terms_are_normalised_to_unit_range(self):
        params = gs.SelectionParams()
        for width in (5.0, 15.0, 40.0, 80.0, 95.0, 110.0):
            self.assertTrue(0.0 <= gs.score_width_fit(width, params) <= 1.0, width)
        for margin in (-5.0, 0.0, 15.0, 30.0, 400.0):
            self.assertTrue(0.0 <= gs.score_joint_margin(margin, params) <= 1.0, margin)

    def test_width_fit_peaks_inside_the_comfortable_band(self):
        params = gs.SelectionParams()
        self.assertEqual(gs.score_width_fit(40.0, params), 1.0)
        self.assertLess(gs.score_width_fit(105.0, params), 1.0)
        self.assertLess(gs.score_width_fit(8.0, params), 1.0)

    def test_height_fit_peaks_at_mid_height(self):
        context = obj(center=(400.0, 0.0, 310.0), height=30.0)   # 지지면 280, 윗면 310
        self.assertAlmostEqual(gs.score_height_fit(candidate(z=295.0), context), 1.0)
        self.assertAlmostEqual(gs.score_height_fit(candidate(z=310.0), context), 0.0)
        self.assertAlmostEqual(gs.score_height_fit(candidate(z=280.0), context), 0.0)

    def test_missing_terms_are_dropped_not_zeroed(self):
        """정보가 없는 항을 0으로 채우면 '모른다'가 '나쁘다'가 된다."""
        terms = {"grasp_score": 1.0, "height_fit": None}
        weights = {"grasp_score": 0.5, "height_fit": 0.5}
        self.assertEqual(gs.combined_score(terms, weights), 1.0)

    def test_center_scale_uses_candidate_widths(self):
        """물체 크기를 모르므로 후보 개폭의 절반을 반폭 추정치로 쓴다."""
        params = gs.SelectionParams()
        self.assertEqual(gs.center_scale_mm([candidate(width=90.0)], params), 45.0)
        # 개폭이 작으면 하한이 쓰인다 — 잡음이 순위를 뒤집지 않게.
        self.assertEqual(gs.center_scale_mm([candidate(width=10.0)], params),
                         params.center_scale_floor_mm)
        self.assertEqual(gs.center_scale_mm([], params), params.center_scale_floor_mm)

    def test_unknown_center_drops_the_proximity_term(self):
        context = gs.ObjectContext(object_id="obj_001", center_mm=None, height_mm=0.0)
        evaluations, _ = run([candidate(0)], context=context)
        self.assertIsNone(evaluations[0].terms["center_proximity"])
        self.assertIsNone(evaluations[0].terms["height_fit"])
        self.assertIsNotNone(evaluations[0].total_score)


class TestEvaluationLimits(unittest.TestCase):
    def test_max_evaluated_caps_the_work(self):
        params = gs.SelectionParams(max_evaluated=3)
        candidates = [candidate(rank=i, score=0.9 - 0.01 * i) for i in range(10)]
        evaluations, selected = run(candidates, params=params)
        self.assertEqual(len(evaluations), 3)
        self.assertIsNotNone(selected)

    def test_log_line_reports_every_check(self):
        evaluations, _ = run([candidate(0)])
        line = gs.log_line(evaluations[0])
        for field in ("candidate=obj_001#0", "width=", "width_valid=true", "approach_ik=true",
                      "grasp_ik=true", "joint_limit=false", "status=selected"):
            self.assertIn(field, line)

    def test_log_line_reports_rejection_reason(self):
        evaluations, _ = run([candidate(0, width=0.0)])
        line = gs.log_line(evaluations[0])
        self.assertIn("status=width_invalid", line)
        self.assertIn("reason=", line)

    def test_wrist_j5_is_recorded_but_not_scored(self):
        """특이점 판정은 미구현이다 — J5는 기록만 하고 점수에 넣지 않는다."""
        evaluations, _ = run([candidate(0)])
        self.assertAlmostEqual(evaluations[0].wrist_j5_deg, 85.0)
        self.assertNotIn("wrist", " ".join(evaluations[0].terms))
        self.assertNotIn("singular", " ".join(evaluations[0].terms))


class TestGeometryIsSharedWithExecution(unittest.TestCase):
    """검사한 좌표와 실행할 좌표가 같아야 한다 — 갈라지면 IK 통과의 근거가 사라진다."""

    def test_selected_carries_the_geometry_used_for_the_ik_check(self):
        ik = FakeIk()
        _, selected = run([candidate(0, x=412.0)], ik=ik)
        self.assertIn(list(selected.geometry.approach_posx), ik.calls)
        self.assertIn(list(selected.geometry.target_posx), ik.calls)
        self.assertAlmostEqual(selected.geometry.target_posx[0], 412.0)


class TestPlanPickPosxMatchesLegacyMath(unittest.TestCase):
    """dsr_motion.plan_pick_posx가 예전 pick_server 계산과 같은 값을 내는지.

    pick_server에서 좌표 계산을 함수로 빼면서 실행 좌표가 달라지면 안 된다 —
    수직 파지에서는 예전 식(base Z에 더하기)과 정확히 같아야 한다.
    """

    def test_vertical_grasp_matches_plain_z_offsets(self):
        from control import dsr_motion

        grasp = pose(400.0, -50.0, 300.0)     # 접근축 -Z인 수직 파지
        target, approach, axis = dsr_motion.plan_pick_posx(
            grasp, [0.0, 0.0, 0.0], approach_height_mm=80.0, depth_extra_mm=10.0)
        self.assertAlmostEqual(axis[2], -1.0, places=6)
        self.assertAlmostEqual(target[0], 400.0, places=6)
        self.assertAlmostEqual(target[1], -50.0, places=6)
        self.assertAlmostEqual(target[2], 290.0, places=6)   # 300 - 10 (하강)
        self.assertAlmostEqual(approach[2], 370.0, places=6)  # 290 + 80 (접근)

    def test_tilted_grasp_offsets_follow_the_approach_axis(self):
        from control import dsr_motion

        # 수직에서 -Y쪽으로 45도 기운 접근축(= X축 둘레 135도 회전). 오프셋이 base Z에만
        # 더해지면 Y 성분이 0으로 나와 이 테스트가 깨진다 — 그게 2026-09-07에 그리퍼가
        # 물체를 옆으로 밀어냈던 버그다.
        half = math.sqrt(0.5)
        angle = math.radians(135.0)
        grasp = pose(400.0, 0.0, 300.0,
                     quaternion=(math.sin(angle / 2), 0.0, 0.0, math.cos(angle / 2)))
        target, approach, axis = dsr_motion.plan_pick_posx(
            grasp, [0.0, 0.0, 0.0], approach_height_mm=100.0, depth_extra_mm=0.0)
        self.assertAlmostEqual(axis[1], -half, places=6)
        self.assertAlmostEqual(axis[2], -half, places=6)
        # 접근점은 접근축 반대 방향으로 100mm — Y와 Z 양쪽으로 움직여야 한다.
        self.assertAlmostEqual(approach[1] - target[1], 100.0 * half, places=4)
        self.assertAlmostEqual(approach[2] - target[2], 100.0 * half, places=4)


if __name__ == "__main__":
    unittest.main()


# --- 물체 밖 자세 차단 (2026-09-08, 2.6차) ------------------------------------
# 주된 판정은 grasp가 포인트클라우드로 한다(cloud_mismatch / no_grip_material).
# control에는 클라우드가 없으므로, 클라우드 없이도 확실한 하나 —
# **파지점이 물체 윗면보다 위** — 만 본다. check_min_safety(지지면=아래쪽)의 반대편이다.

class CheckGraspGeometryTest(unittest.TestCase):

    def setUp(self):
        self.params = gs.SelectionParams()
        # 윗면 z=300, 두께 40 → 물체는 z 260~300 구간이다.
        self.obj = gs.ObjectContext(object_id="obj_001", center_mm=(400.0, 0.0, 300.0),
                                    height_mm=40.0, depth_valid_ratio=0.9)

    def _candidate(self, z):
        return gs.Candidate(candidate_id="obj_001#0", rank=0,
                            pose=NS(position=NS(x=400.0, y=0.0, z=z),
                                    orientation=NS(x=0.0, y=1.0, z=0.0, w=0.0)),
                            score=0.5, gripper_width_mm=40.0)

    def test_inside_the_object_passes(self):
        for z in (262.0, 280.0, 299.0):
            self.assertEqual(gs.check_grasp_geometry(self._candidate(z), self.obj,
                                                     self.params), "")

    def test_top_surface_tolerance_is_allowed(self):
        """손끝 위치가 모델값이라 아래쪽과 같은 크기의 여유를 위쪽에도 준다."""
        edge = 300.0 + self.params.support_tolerance_mm
        self.assertEqual(gs.check_grasp_geometry(self._candidate(edge), self.obj,
                                                 self.params), "")

    def test_above_the_object_is_rejected(self):
        reason = gs.check_grasp_geometry(self._candidate(330.0), self.obj, self.params)
        self.assertNotEqual(reason, "")
        self.assertIn("허공", reason)

    def test_unknown_height_is_not_judged(self):
        """두께를 모르면 판단하지 않는다 — 모른다는 이유로 후보를 죽이지 않는다."""
        obj = gs.ObjectContext(object_id="obj_001", center_mm=(400.0, 0.0, 300.0),
                               height_mm=0.0)
        self.assertEqual(gs.check_grasp_geometry(self._candidate(500.0), obj,
                                                 self.params), "")

    def test_unknown_center_is_not_judged(self):
        obj = gs.ObjectContext(object_id="obj_001", center_mm=None, height_mm=40.0)
        self.assertEqual(gs.check_grasp_geometry(self._candidate(500.0), obj,
                                                 self.params), "")


class GeometryInvalidInPipelineTest(unittest.TestCase):
    """평가 파이프라인에 자연스럽게 연결됐는가 — IK 앞에서 걸러 ikin 왕복을 아낀다."""

    def setUp(self):
        self.obj = gs.ObjectContext(object_id="obj_001", center_mm=(400.0, 0.0, 300.0),
                                    height_mm=40.0, depth_valid_ratio=0.9)
        self.ik_calls = []

    def _evaluate(self, z_values):
        candidates = [
            gs.Candidate(candidate_id=f"obj_001#{i}", rank=i,
                         pose=NS(position=NS(x=400.0, y=0.0, z=z),
                                 orientation=NS(x=0.0, y=1.0, z=0.0, w=0.0)),
                         score=0.9 - 0.1 * i, gripper_width_mm=40.0)
            for i, z in enumerate(z_values)]

        def geometry_of(pose):
            return gs.PickGeometry(
                target_posx=[400.0, 0.0, float(pose.position.z), 0.0, 180.0, 0.0],
                approach_posx=[400.0, 0.0, float(pose.position.z) + 80.0, 0.0, 180.0, 0.0],
                approach_axis=[0.0, 0.0, -1.0],
                pad_reference_mm=[400.0, 0.0, float(pose.position.z)])

        def ik_verdict_of(posx):
            self.ik_calls.append(posx)
            return NS(known=True, ok=True, status="ok", joint_margin_deg=40.0,
                      posj=[0.0, 0.0, 0.0, 0.0, 80.0, 0.0])

        return gs.evaluate_candidates(candidates, self.obj, gs.SelectionParams(),
                                      geometry_of, ik_verdict_of)

    def test_air_candidate_is_marked_and_skipped(self):
        evaluations = self._evaluate([400.0, 280.0])
        self.assertEqual(evaluations[0].status, gs.STATUS_GEOMETRY_INVALID)
        self.assertIsNone(evaluations[0].total_score)
        # 다음 후보는 정상적으로 선택될 수 있어야 한다 (motion 전에 넘어간다)
        self.assertEqual(gs.select(evaluations).candidate.rank, 1)

    def test_ik_is_not_queried_for_geometry_invalid(self):
        """ikin 왕복은 후보당 약 66ms — 못 쓸 후보에 쓰지 않는다."""
        self._evaluate([400.0])
        self.assertEqual(self.ik_calls, [])

    def test_counts_include_the_new_status(self):
        tally = gs.counts(self._evaluate([400.0, 280.0]))
        self.assertEqual(tally[gs.STATUS_GEOMETRY_INVALID], 1)

    def test_rejected_before_motion(self):
        evaluations = self._evaluate([400.0, 280.0])
        self.assertTrue(evaluations[0].rejected_before_motion)

    def test_normal_candidates_are_unaffected(self):
        """Case 5 — 기존 정상 후보는 그대로 통과해야 한다(회귀 없음)."""
        evaluations = self._evaluate([280.0, 275.0, 290.0])
        self.assertTrue(all(e.status in (gs.STATUS_VALID, gs.STATUS_SELECTED)
                            for e in evaluations), [e.status for e in evaluations])

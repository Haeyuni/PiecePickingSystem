"""접근각을 hard filter가 아니라 랭킹으로 다루는 구조 (2026-09-08, 2.5차).

예전에는 grasp 단계에서 접근각으로 후보를 잘라냈다. 그 필터가 막던 것 중 "IK가 안 풀린다"는
이제 control이 직접 확인하므로(접근/파지 IK + 관절 한계), 각도로 미리 자르면 실행 가능한
후보까지 같이 죽는다. 남은 "RG2로 물기 어렵다"는 불가가 아니라 선호도라 여기서 점수로 다룬다.

이 파일이 지키려는 계약:

1. 30도 이하 후보의 평가가 예전과 달라지지 않는다(만점~0.5 구간).
2. 30도를 넘어도 width/IK/joint/safety가 멀쩡하면 **후보로 남아 랭킹 대상이 된다.**
3. GraspNet score가 높아도 접근이 나쁘면 밀리고, score가 조금 낮아도 접근이 자연스러우면
   선택될 수 있다 — 이 역전이 안 일어나면 항을 추가한 의미가 없다.
4. 각도는 탈락한 후보에도 기록된다(hard 상한을 조일지 풀지 판단할 근거).

`test_grasp_selection.py`가 개폭·IK·안전 같은 기존 판정을, 이 파일이 접근 적합도를 본다.
"""
import math
import pathlib
import sys
import unittest
from types import SimpleNamespace as NS

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))                    # src/control

from control import grasp_selection as gs  # noqa: E402

IK_OK = NS(known=True, ok=True, status="ok", joint_margin_deg=40.0,
           posj=[0.0, 0.0, 0.0, 0.0, 80.0, 0.0])
IK_NONE = NS(known=True, ok=False, status="unreachable", joint_margin_deg=None, posj=None)


def axis_at(angle_deg: float):
    """수직 아래에서 `angle_deg`만큼 기운 base 접근축 (xz 평면에서 기울인다)."""
    rad = math.radians(angle_deg)
    return [math.sin(rad), 0.0, -math.cos(rad)]


def candidate(rank=0, score=0.5, width=40.0, cid=None):
    return gs.Candidate(candidate_id=cid or f"obj_001#{rank}", rank=rank,
                        pose=NS(position=NS(x=400.0, y=0.0, z=300.0),
                                orientation=NS(x=0.0, y=1.0, z=0.0, w=0.0)),
                        score=score, gripper_width_mm=width)


def geometry_at(angle_deg, approach_xyz=(400.0, 0.0, 380.0)):
    """그 각도로 접근하는 PickGeometry. 지지면 검사를 통과하도록 손끝은 물체 위에 둔다."""
    return gs.PickGeometry(
        target_posx=[400.0, 0.0, 300.0, 0.0, 180.0, 0.0],
        approach_posx=[*approach_xyz, 0.0, 180.0, 0.0],
        approach_axis=axis_at(angle_deg),
        pad_reference_mm=[400.0, 0.0, 300.0])


def evaluate(angles, scores=None, params=None, obj=None, ik=None, **kw):
    """각도 목록으로 후보를 만들어 평가한다. 반환은 Evaluation 목록."""
    scores = scores or [0.5] * len(angles)
    candidates = [candidate(rank=i, score=scores[i]) for i in range(len(angles))]
    geometries = {id(c.pose): geometry_at(a) for c, a in zip(candidates, angles)}
    # pose 객체가 후보마다 달라야 각도를 구분해 돌려줄 수 있다.
    for c, a in zip(candidates, angles):
        geometries[id(c.pose)] = geometry_at(a)
    return gs.evaluate_candidates(
        candidates,
        obj or gs.ObjectContext(object_id="obj_001", center_mm=(400.0, 0.0, 300.0),
                                height_mm=40.0, depth_valid_ratio=0.9),
        params or gs.SelectionParams(),
        lambda pose: geometries[id(pose)],
        ik or (lambda posx: IK_OK),
        **kw)


class ApproachAngleMathTest(unittest.TestCase):

    def test_straight_down_is_zero(self):
        self.assertAlmostEqual(gs.approach_angle_deg([0.0, 0.0, -1.0]), 0.0, places=6)

    def test_horizontal_is_ninety(self):
        self.assertAlmostEqual(gs.approach_angle_deg([1.0, 0.0, 0.0]), 90.0, places=6)

    def test_from_below_is_beyond_ninety(self):
        """아래에서 위로 찌르는 자세는 180도다 — abs()로 재면 0도로 통과해버린다."""
        self.assertAlmostEqual(gs.approach_angle_deg([0.0, 0.0, 1.0]), 180.0, places=6)

    def test_unknown_axis(self):
        self.assertIsNone(gs.approach_angle_deg(None))
        self.assertIsNone(gs.approach_angle_deg([0.0, 0.0, 0.0]))


class ApproachAngleScoreTest(unittest.TestCase):
    """예전 hard filter의 숫자가 감점 곡선의 꼭짓점으로 남아 있어야 한다."""

    def setUp(self):
        self.params = gs.SelectionParams()

    def test_prefer_band_is_full_marks(self):
        self.assertEqual(gs.score_approach_angle(0.0, self.params), 1.0)
        self.assertEqual(gs.score_approach_angle(15.0, self.params), 1.0)

    def test_old_cutoff_is_exactly_half(self):
        """30도(예전에 잘리던 선)는 0.5 — '탈락'이 '절반짜리'가 된 것이 이번 변경이다."""
        self.assertAlmostEqual(gs.score_approach_angle(30.0, self.params), 0.5, places=6)

    def test_hard_max_is_zero_and_monotonic(self):
        self.assertEqual(gs.score_approach_angle(75.0, self.params), 0.0)
        self.assertEqual(gs.score_approach_angle(120.0, self.params), 0.0)
        values = [gs.score_approach_angle(a, self.params) for a in range(0, 80, 5)]
        self.assertEqual(values, sorted(values, reverse=True), "각도가 커지면 점수는 안 오른다")

    def test_between_soft_and_zero_still_discriminates(self):
        """40도와 60도가 둘 다 0점이면 랭킹이 각도를 구분하지 못한다."""
        self.assertGreater(gs.score_approach_angle(40.0, self.params),
                           gs.score_approach_angle(60.0, self.params))

    def test_unknown_angle(self):
        self.assertIsNone(gs.score_approach_angle(None, self.params))


class CandidateSurvivalTest(unittest.TestCase):

    def test_case1_candidates_within_30_are_valid_as_before(self):
        """Case 1 — 기존 30도 이하 후보는 그대로 정상 평가된다."""
        evaluations = evaluate([5.0, 18.0, 29.0])
        self.assertTrue(all(e.status == gs.STATUS_VALID or e.status == gs.STATUS_SELECTED
                            for e in evaluations))
        self.assertTrue(all(e.total_score is not None for e in evaluations))

    def test_case2_beyond_30_survives_when_feasible(self):
        """Case 2 — 30도 초과라도 width/IK/joint/safety가 멀쩡하면 후보로 남는다.

        예전에는 이 후보들이 grasp 단계에서 사라져 control이 보지도 못했다.
        """
        evaluations = evaluate([38.0, 52.0, 64.0])
        for evaluation in evaluations:
            self.assertEqual(evaluation.status, gs.STATUS_VALID, evaluation.rejection_reason)
            self.assertIsNotNone(evaluation.total_score)
            self.assertIsNotNone(evaluation.legacy_terms["approach_angle"])

    def test_case2_but_they_rank_below_upright_ones(self):
        """살아남되 **우대받지는 않는다** — 수직에 가까운 후보가 위에 있어야 한다."""
        evaluations = evaluate([10.0, 55.0])
        # 접근각은 legacy 랭킹의 항이다(STEP 1에서 enhanced에는 중복해서 넣지 않았다) —
        # 실제 실행이 쓰는 점수가 legacy이므로 여기서도 그 점수로 순위를 본다.
        self.assertGreater(evaluations[0].legacy_score, evaluations[1].legacy_score)

    def test_angle_recorded_even_when_rejected(self):
        """Case 4 근거 — IK로 떨어진 후보의 각도도 남아야 hard 상한을 조정할 수 있다."""
        evaluations = evaluate([47.0], ik=lambda posx: IK_NONE)
        self.assertEqual(evaluations[0].status, gs.STATUS_IK_FAILED)
        self.assertAlmostEqual(evaluations[0].approach_angle_deg, 47.0, places=3)

    def test_ik_still_rejects_regardless_of_angle(self):
        """각도를 풀어준 것이지 IK 검사를 푼 것이 아니다 — 2차 판정은 그대로다."""
        evaluations = evaluate([10.0], ik=lambda posx: IK_NONE)
        self.assertEqual(evaluations[0].status, gs.STATUS_IK_FAILED)
        self.assertIsNone(evaluations[0].total_score)


class ApproachTravelTest(unittest.TestCase):

    def test_travel_and_rotation_recorded(self):
        evaluations = evaluate(
            [10.0], current_posx=[0.0, 0.0, 380.0, 0.0, 180.0, 0.0],
            rotation_diff=lambda a, b: 0.0)
        self.assertAlmostEqual(evaluations[0].approach_travel_mm, 400.0, places=3)
        self.assertAlmostEqual(evaluations[0].approach_rotation_deg, 0.0, places=6)

    def test_term_absent_when_current_pose_unknown(self):
        """현재 자세를 못 읽어도(aux_control 무응답) 나머지 판정은 그대로 돌아야 한다."""
        evaluations = evaluate([10.0])
        self.assertIsNone(evaluations[0].approach_travel_mm)
        self.assertIsNone(evaluations[0].legacy_terms["approach_travel"])
        self.assertIsNotNone(evaluations[0].total_score)

    def test_shorter_travel_scores_higher(self):
        params = gs.SelectionParams()
        near = gs.score_approach_travel(50.0, 10.0, params)
        far = gs.score_approach_travel(390.0, 110.0, params)
        self.assertGreater(near, far)

    def test_unknown_both_is_none(self):
        self.assertIsNone(gs.score_approach_travel(None, None, gs.SelectionParams()))


class RankingReversalTest(unittest.TestCase):
    """Case 4 — GraspNet score만으로 고르던 것을 바꾸는 것이 이 항의 목적이다."""

    def test_better_approach_can_beat_higher_graspnet_score(self):
        # 0번: score 높지만 거의 옆에서 찌른다 / 1번: score 낮지만 수직에 가깝다
        evaluations = evaluate([65.0, 8.0], scores=[0.90, 0.62])
        selected = gs.select(evaluations)[0]
        self.assertEqual(selected.candidate.rank, 1,
                         "접근이 자연스러운 후보가 선택될 수 있어야 한다")

    def test_graspnet_score_still_wins_when_approach_is_comparable(self):
        """각도가 비슷하면 GraspNet 품질이 그대로 순위를 정해야 한다 — 뒤집기만 하면 안 된다."""
        evaluations = evaluate([12.0, 14.0], scores=[0.90, 0.40])
        selected = gs.select(evaluations)[0]
        self.assertEqual(selected.candidate.rank, 0)

    def test_angle_decides_within_the_real_graspnet_score_spread(self):
        """**실물에서 한 물체의 후보 점수는 거의 붙어 있다.**

        2026-09-08 로그: obj_005는 0.372~0.346(폭 0.026), obj_010은 0.613~0.556(폭 0.057).
        이 구간에서는 각도가 순위를 정해야 한다 — 그러라고 넣은 항이다.
        """
        evaluations = evaluate([58.0, 11.0], scores=[0.372, 0.346])
        self.assertEqual(gs.select(evaluations)[0].candidate.rank, 1)

    def test_large_score_gap_still_wins(self):
        """교차점을 못박아 둔다: 각도 만점 차이(0.10) = grasp_score 0.333 차이(0.30 x 0.333).

        그보다 큰 품질 차이는 각도로 못 뒤집는다 — **의도한 동작이다.** GraspNet이 한
        후보만 확실히 좋다고 볼 때까지 각도로 눌러버리면 그건 예전 hard filter와 같아진다.
        기울어진 후보가 계속 실행돼 미끄러지면 skill_params.yaml의 weights.approach_angle을
        올린다(그 값이 이 교차점을 직접 옮긴다).
        """
        evaluations = evaluate([74.0, 5.0], scores=[1.0, 0.55])   # 차이 0.45 > 0.333
        self.assertEqual(gs.select(evaluations)[0].candidate.rank, 0)


class LogLineTest(unittest.TestCase):

    def test_angle_appears_in_log(self):
        """튜닝은 로그를 보고 한다 — 각도가 안 찍히면 조일지 풀지 판단할 수 없다."""
        evaluations = evaluate([33.0], current_posx=[0.0, 0.0, 380.0, 0.0, 180.0, 0.0],
                               rotation_diff=lambda a, b: 12.0)
        line = gs.log_line(evaluations[0])
        self.assertIn("angle=33.0deg", line)
        self.assertIn("travel=400mm", line)
        self.assertIn("rot=12deg", line)


if __name__ == "__main__":
    unittest.main()


# --- 뒤집기 보정과 랭킹의 순서 -------------------------------------------------
# 평행 그리퍼는 닫힘축 부호가 뒤집혀도 물리적으로 같은 파지라, 같은 후보를 172도짜리로도
# 8도짜리로도 표현할 수 있다(dsr_motion.nearest_equivalent_grasp_rotation — 2026-09-07
# 실물에서 172.4도를 헛돌아 접근에만 14초가 걸린 그 문제).
#
# **접근 적합도가 그 보정 전 숫자를 재면 안 된다.** 8도면 되는 후보를 172도짜리로 보고
# 감점하면, 랭킹이 멀쩡한 후보를 표현 방식 때문에 밀어내게 된다. 실행 경로가 쓰는
# geometry를 그대로 재는지 실제 함수로 확인한다(모킹하면 이 성질이 검증되지 않는다).
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "perception_common"))


def _quat_from_matrix(m):
    import numpy as np
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        return ((m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s,
                (m[1, 0] - m[0, 1]) * s, 0.25 / s)
    i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(m[i, i] - m[j, j] - m[k, k] + 1.0) * 2.0
    q = [0.0, 0.0, 0.0]
    q[i], q[j], q[k] = 0.25 * s, (m[j, i] + m[i, j]) / s, (m[k, i] + m[i, k]) / s
    return (*q, (m[k, j] - m[j, k]) / s)


class FlipBeforeRankingTest(unittest.TestCase):
    """랭킹이 재는 회전량은 **뒤집기 보정 후** 값이어야 한다."""

    def setUp(self):
        import numpy as np
        from control import dsr_motion
        from perception_common.geometry import matrix_to_zyz_deg

        self.dsr = dsr_motion
        # 접근축(툴 Z)=base -Z, 닫힘축(툴 X)=base +X 인 파지 자세
        self.R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        flip = np.array(dsr_motion._FLIP_ABOUT_APPROACH)
        rad = math.radians(8.0)
        spin = np.array([[math.cos(rad), -math.sin(rad), 0.0],
                         [math.sin(rad), math.cos(rad), 0.0], [0.0, 0.0, 1.0]])
        # 현재 손목 = 뒤집은 표현에서 8도만 더 돌린 자세 → 원래 표현까지는 172도
        self.current_zyz = matrix_to_zyz_deg(self.R @ flip @ spin)
        qx, qy, qz, qw = _quat_from_matrix(self.R)
        self.pose = NS(position=NS(x=400.0, y=0.0, z=300.0),
                       orientation=NS(x=qx, y=qy, z=qz, w=qw))

    def test_setup_really_is_the_172_vs_8_case(self):
        """전제 확인 — 이게 안 성립하면 아래 테스트가 아무것도 검증하지 않는다."""
        import numpy as np
        flip = np.array(self.dsr._FLIP_ABOUT_APPROACH)

        def sep(a, b):
            trace = float((np.asarray(a).T @ np.asarray(b)).trace())
            return math.degrees(math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0))))

        from perception_common.geometry import posx_to_matrix
        current = posx_to_matrix([0.0, 0.0, 0.0, *self.current_zyz])[:3, :3]
        self.assertAlmostEqual(sep(current, self.R), 172.0, places=3)
        self.assertAlmostEqual(sep(current, self.R @ flip), 8.0, places=3)

    def test_ranking_sees_the_corrected_8_degrees(self):
        target, approach, axis = self.dsr.plan_pick_posx(
            self.pose, [0.0, 0.0, 5.3], 80.0, 0.0, current_zyz_deg=self.current_zyz)
        geometry = gs.PickGeometry(target_posx=target, approach_posx=approach,
                                   approach_axis=axis, pad_reference_mm=target[:3])
        evaluations = gs.evaluate_candidates(
            [candidate()],
            gs.ObjectContext(object_id="obj_001", center_mm=(400.0, 0.0, 300.0),
                             height_mm=40.0, depth_valid_ratio=0.9),
            gs.SelectionParams(),
            lambda pose: geometry,
            lambda posx: IK_OK,
            current_posx=[400.0, 0.0, 380.0, *self.current_zyz],
            rotation_diff=self.dsr.rotation_diff_deg)
        self.assertAlmostEqual(evaluations[0].approach_rotation_deg, 8.0, places=2,
                               msg="172도(보정 전)로 재고 있으면 표현 방식 때문에 감점된다")

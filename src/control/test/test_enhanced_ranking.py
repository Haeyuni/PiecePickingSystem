"""LEGACY vs Enhanced 랭킹, contact_support, robot_comfort (STEP 1, 2026-09-10).

여기서 지키려는 계약:

[A] 기존 release 랭킹은 LEGACY_WEIGHTS로 **그대로** 보존된다 — contact_support 때문에
    다시 튜닝하지 않는다. mode를 주지 않는 호출은 전부 legacy로 고른다.
[J] production runtime은 enhanced 하나만 사용하고 shadow 비교 분기를 남기지 않는다.
[K] 값이 없는 항(contact -1, posj 없음)은 분자·분모 양쪽에서 빠진다 — 정보 없음이
    감점이 되면 안 된다.
[E] contact_support는 soft다. 낮다고 후보를 버리지 않고, 물체 끝을 잡아도 재료만 있으면
    높은 값이 그대로 반영된다.
[I] 첫 후보 IK 실패가 pick 전체를 끝내지 않는다.
"""
import ast
import pathlib
import sys
import unittest
from types import SimpleNamespace as NS

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))                    # src/control
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "perception_common"))

from control import grasp_selection as gs  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[3]


def pose(x=400.0, y=0.0, z=300.0):
    return NS(position=NS(x=x, y=y, z=z), orientation=NS(x=0.0, y=1.0, z=0.0, w=0.0))


def candidate(rank=0, score=0.9, width=40.0, x=400.0, contact=-1.0):
    return gs.Candidate(candidate_id=f"obj_001#{rank}", rank=rank, pose=pose(x),
                        score=score, gripper_width_mm=width,
                        strategy="graspnet_baseline", contact_support_score=contact)


def obj(center=(400.0, 0.0, 310.0), height=30.0, depth_ratio=0.9):
    return gs.ObjectContext(object_id="obj_001", center_mm=center, height_mm=height,
                            depth_valid_ratio=depth_ratio)


def verdict(margin=120.0, posj=(0.0, 10.0, 90.0, 0.0, 85.0, 0.0)):
    return NS(status=gs.IK_OK, ok=True, known=True, joint_margin_deg=margin,
              posj=list(posj) if posj else None)


class FakeIk:
    def __init__(self, verdicts=None):
        self.verdicts = verdicts or {}
        self.calls = []

    def __call__(self, posx):
        self.calls.append(list(posx))
        return self.verdicts.get(tuple(round(v, 3) for v in posx[:3]), verdict())


def ik_fail():
    return NS(status="unreachable", ok=False, known=True, joint_margin_deg=None, posj=None)


def geometry_of(candidate_pose):
    p = candidate_pose.position
    return gs.PickGeometry(target_posx=[float(p.x), float(p.y), float(p.z), 0.0, 180.0, 0.0],
                           approach_posx=[float(p.x), float(p.y), float(p.z) + 80.0,
                                          0.0, 180.0, 0.0],
                           approach_axis=[0.0, 0.0, -1.0],
                           pad_reference_mm=[float(p.x), float(p.y), float(p.z)])


def evaluate(candidates, ik=None, params=None, context=None):
    return gs.evaluate_candidates(candidates, context or obj(),
                                  params or gs.SelectionParams(),
                                  geometry_of, ik or FakeIk())


# --- [A] legacy 보존 ----------------------------------------------------------

class LegacyPreservationTest(unittest.TestCase):
    def test_legacy_weights_match_the_release_ranking(self):
        """기존 release가 쓰던 8항 구성과 값 그대로여야 한다 — 비교 기준이라 흔들리면 안 된다."""
        self.assertEqual(gs.LEGACY_WEIGHTS, {
            "grasp_score": 0.30, "width_fit": 0.15, "center_proximity": 0.15,
            "height_fit": 0.10, "joint_margin": 0.10, "depth_quality": 0.05,
            "approach_angle": 0.10, "approach_travel": 0.05})
        self.assertAlmostEqual(sum(gs.LEGACY_WEIGHTS.values()), 1.0)

    def test_legacy_score_uses_legacy_weights_only(self):
        evaluation = evaluate([candidate(0, contact=1.0)])[0]
        expected = gs.combined_score(evaluation.legacy_terms, gs.LEGACY_WEIGHTS)
        self.assertAlmostEqual(evaluation.legacy_score, expected)
        self.assertNotIn("contact_support", evaluation.legacy_terms)
        self.assertNotIn("robot_comfort", evaluation.legacy_terms)

    def test_default_mode_selects_by_legacy(self):
        """mode를 주지 않으면 legacy — 기존 호출부·테스트가 그대로 이 경로를 탄다."""
        weak_contact = candidate(0, score=0.90, contact=0.05)
        strong_contact = candidate(1, score=0.85, contact=0.95)
        evaluations = evaluate([weak_contact, strong_contact])
        chosen, legacy_best, _ = gs.select(evaluations)
        self.assertIs(chosen, legacy_best)
        self.assertEqual(chosen.candidate.rank, 0)

    def test_contact_support_does_not_change_legacy_score(self):
        with_contact = evaluate([candidate(0, contact=0.95)])[0]
        without_contact = evaluate([candidate(0, contact=-1.0)])[0]
        self.assertAlmostEqual(with_contact.legacy_score, without_contact.legacy_score)


# --- [J] selection_mode -------------------------------------------------------

class SelectionModeTest(unittest.TestCase):
    def _split_case(self):
        # 점수는 #0이 조금 높지만 #1의 손가락 사이 재료 지지가 훨씬 좋다.
        return [candidate(0, score=0.90, contact=0.05), candidate(1, score=0.85, contact=0.95)]

    def test_enhanced_can_prefer_contact_support(self):
        evaluations = evaluate(self._split_case())
        chosen, _, enhanced_best = gs.select(evaluations, mode="enhanced")
        self.assertIs(chosen, enhanced_best)
        self.assertEqual(chosen.candidate.rank, 1)

    def test_shipped_runtime_has_one_enhanced_selector(self):
        config = yaml.safe_load(
            (REPO / "src" / "control" / "config" / "skill_params.yaml").read_text(
                encoding="utf-8"))
        self.assertNotIn("selection_mode", config["grasp_selection"])
        source = (REPO / "src" / "control" / "control" / "pick_server.py").read_text(
            encoding="utf-8")
        self.assertIn('mode="enhanced"', source)
        self.assertNotIn("[RANK_COMPARE]", source)

    def test_enhanced_weights_do_not_duplicate_approach_terms(self):
        weights = gs.SelectionParams().weights
        self.assertNotIn("approach_angle", weights)
        self.assertNotIn("approach_travel", weights)
        self.assertNotIn("top_down_preference", weights)
        self.assertAlmostEqual(sum(weights.values()), 1.0)


# --- [K] None 정규화 ----------------------------------------------------------

class NormalizationTest(unittest.TestCase):
    def test_unknown_contact_is_excluded_not_zeroed(self):
        evaluation = evaluate([candidate(0, contact=-1.0)])[0]
        self.assertIsNone(evaluation.terms["contact_support"])
        # contact를 뺀 나머지 항만으로 정규화된 값과 같아야 한다.
        remaining = {k: v for k, v in evaluation.terms.items() if k != "contact_support"}
        self.assertAlmostEqual(
            evaluation.total_score,
            gs.combined_score(remaining, gs.SelectionParams().weights))

    def test_zero_contact_is_kept_as_a_real_low_score(self):
        """0.0(지지 없음)은 미상이 아니다 — 항에 남아 실제로 감점돼야 한다."""
        zero = evaluate([candidate(0, contact=0.0)])[0]
        unknown = evaluate([candidate(0, contact=-1.0)])[0]
        self.assertEqual(zero.terms["contact_support"], 0.0)
        self.assertLess(zero.total_score, unknown.total_score)

    def test_missing_center_drops_only_that_term(self):
        context = gs.ObjectContext(object_id="obj_001", center_mm=None, height_mm=0.0)
        evaluation = evaluate([candidate(0, contact=0.5)], context=context)[0]
        self.assertIsNone(evaluation.terms["center_proximity"])
        self.assertIsNotNone(evaluation.total_score)


# --- [E] contact_support는 soft ------------------------------------------------

class ContactSupportIsSoftTest(unittest.TestCase):
    def test_low_contact_is_never_a_reject(self):
        evaluation = evaluate([candidate(0, contact=0.0)])[0]
        self.assertEqual(evaluation.status, gs.STATUS_VALID)
        self.assertFalse(evaluation.rejected_before_motion)

    def test_edge_candidate_with_good_contact_can_win_enhanced(self):
        """중심에서 멀어도(center_proximity 손해) 재료 지지가 좋으면 이길 수 있어야 한다."""
        centered_poor_contact = candidate(0, score=0.7, x=400.0, contact=0.05)
        edge_good_contact = candidate(1, score=0.7, x=440.0, contact=0.95)
        evaluations = evaluate([centered_poor_contact, edge_good_contact])
        chosen, _, _ = gs.select(evaluations, mode="enhanced")
        self.assertEqual(chosen.candidate.rank, 1)

    def test_score_function_maps_unknown_to_none(self):
        self.assertIsNone(gs.score_contact_support(candidate(contact=-1.0)))
        self.assertEqual(gs.score_contact_support(candidate(contact=0.0)), 0.0)
        self.assertEqual(gs.score_contact_support(candidate(contact=0.73)), 0.73)


# --- robot_comfort ------------------------------------------------------------

class RobotComfortTest(unittest.TestCase):
    def test_joint_delta_uses_only_the_ik_results_we_already_have(self):
        """추가 서비스 호출을 만들지 않는다 — 인자가 두 IK 판정뿐이다."""
        source = (REPO / "src" / "control" / "control" / "grasp_selection.py").read_text(
            encoding="utf-8")
        func = next(n for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name == "joint_delta_deg")
        self.assertEqual([a.arg for a in func.args.args],
                         ["approach_verdict", "grasp_verdict"])

    def test_large_joint_jump_scores_lower(self):
        smooth = FakeIk()
        jumpy = FakeIk({(400.0, 0.0, 380.0): verdict(posj=(50.0, 10.0, 90.0, 0.0, 85.0, 0.0))})
        smooth_eval = evaluate([candidate(0)], ik=smooth)[0]
        jumpy_eval = evaluate([candidate(0)], ik=jumpy)[0]
        self.assertAlmostEqual(smooth_eval.joint_delta_deg, 0.0)
        self.assertAlmostEqual(jumpy_eval.joint_delta_deg, 50.0)
        self.assertLess(jumpy_eval.terms["robot_comfort"],
                        smooth_eval.terms["robot_comfort"])

    def test_missing_posj_is_unknown_not_zero(self):
        blind = FakeIk({(400.0, 0.0, 300.0): verdict(posj=None),
                        (400.0, 0.0, 380.0): verdict(posj=None)})
        evaluation = evaluate([candidate(0)], ik=blind)[0]
        self.assertIsNone(evaluation.joint_delta_deg)
        self.assertIsNone(evaluation.terms["robot_comfort"])
        self.assertIsNotNone(evaluation.total_score)

    def test_no_hard_reject_from_comfort(self):
        jumpy = FakeIk({(400.0, 0.0, 380.0): verdict(posj=(179.0, 10.0, 90.0, 0.0, 85.0, 0.0))})
        evaluation = evaluate([candidate(0)], ik=jumpy)[0]
        self.assertEqual(evaluation.status, gs.STATUS_VALID)


# --- [I] IK 실패 후 계속 --------------------------------------------------------

class IkChainTest(unittest.TestCase):
    def test_third_candidate_is_reached_after_two_ik_failures(self):
        ik = FakeIk({(400.0, 0.0, 300.0): ik_fail(), (410.0, 0.0, 300.0): ik_fail()})
        candidates = [candidate(0, x=400.0), candidate(1, x=410.0), candidate(2, x=420.0)]
        evaluations = evaluate(candidates, ik=ik)
        chosen, _, _ = gs.select(evaluations, mode="log_only")
        self.assertEqual([e.status for e in evaluations],
                         [gs.STATUS_IK_FAILED, gs.STATUS_IK_FAILED, gs.STATUS_SELECTED])
        self.assertEqual(chosen.candidate.rank, 2)

    def test_no_valid_candidate_returns_none_triple(self):
        ik = FakeIk({(400.0, 0.0, 300.0): ik_fail()})
        evaluations = evaluate([candidate(0)], ik=ik)
        self.assertEqual(gs.select(evaluations, mode="log_only"), (None, None, None))


if __name__ == "__main__":
    unittest.main()

"""검증기 단위 테스트 (stdlib unittest — 런타임 이미지에 테스트 의존성을 추가하지 않는다).

실행: docker compose exec planner python -m unittest discover -s tests -t .

검증기를 통과한 시퀀스는 사람 승인 없이 자동 실행되므로(웹_인터페이스_정의서 1절),
거부되어야 할 입력이 실제로 거부되는지가 이 계층의 유일한 안전장치다. LLM 스모크
테스트(scripts/check_plan_fixtures.py)는 LLM이 그런 시퀀스를 만들어줄 때만 이 경로를
지나므로, 여기서는 검증기를 직접 호출한다.
"""
import unittest

from src.schema import LlmStep
from src.validator import MAX_PAYLOAD_G, Rejected, resolve_profile, validate

BINS = {
    "left_box": {"name_ko": "왼쪽 박스", "pose": {"x": 350, "y": 250, "z": 50}},
    "far_box": {"name_ko": "먼 박스", "pose": {"x": 2000, "y": 0, "z": 50}},
}


def make_object(object_id="obj_001", **overrides):
    obj = {
        "object_id": object_id,
        "class_name": "nail",
        "name_ko": "네일",
        "graspable": True,
        "not_graspable_reason": "",
        "mass_g": 150.0,
        "profile": "normal",
        "needs_confirmation": False,
        "grasp_candidates": [{
            "pose": {"position": {"x": 450.0, "y": 0.0, "z": 80.0},
                     "orientation": {"x": 0.0, "y": 1.0, "z": 0.0, "w": 0.0}},
            "score": 0.9,
            "strategy": "heuristic_pca",
        }],
    }
    obj.update(overrides)
    return obj


def world(*objects):
    return {"objects": list(objects), "needs_reobserve": []}


def pick_place(object_id="obj_001", bin_id="left_box"):
    return [LlmStep(skill="pick", object_id=object_id),
            LlmStep(skill="place_into", object_id=object_id, bin_id=bin_id)]


class TestHappyPath(unittest.TestCase):
    def test_pick_then_place_is_approved(self):
        steps = validate(pick_place(), world(make_object()), BINS)
        self.assertEqual([s.skill for s in steps], ["pick", "place_into"])
        # 파지 자세는 LLM이 아니라 검증기가 후보에서 채운다
        self.assertEqual(steps[0].grasp_pose.position["x"], 450.0)
        self.assertEqual(steps[1].bin_id, "left_box")

    def test_best_scoring_candidate_is_chosen(self):
        obj = make_object()
        obj["grasp_candidates"].append({
            "pose": {"position": {"x": 460.0, "y": 0.0, "z": 80.0},
                     "orientation": {"x": 0.0, "y": 1.0, "z": 0.0, "w": 0.0}},
            "score": 0.99, "strategy": "contact_graspnet",
        })
        steps = validate(pick_place(), world(obj), BINS)
        self.assertEqual(steps[0].grasp_pose.position["x"], 460.0)

    def test_grasp_width_is_carried_into_plan_step(self):
        obj = make_object()
        obj["grasp_candidates"][0]["gripper_width_mm"] = 42.5
        steps = validate(pick_place(), world(obj), BINS)
        self.assertEqual(steps[0].gripper_width_mm, 42.5)

    def test_missing_grasp_width_is_none(self):
        """gripper_width_mm을 못 낸 전략(레거시 후보)이면 None — control이 기본 개폭으로 대체한다."""
        steps = validate(pick_place(), world(make_object()), BINS)
        self.assertIsNone(steps[0].gripper_width_mm)


class TestGrounding(unittest.TestCase):
    def test_unknown_object_id_is_rejected(self):
        with self.assertRaises(Rejected):
            validate(pick_place("obj_999"), world(make_object()), BINS)

    def test_unknown_bin_id_is_rejected(self):
        with self.assertRaises(Rejected):
            validate(pick_place(bin_id="trash_can"), world(make_object()), BINS)


class TestSkillPreconditions(unittest.TestCase):
    def test_place_without_pick_is_rejected(self):
        steps = [LlmStep(skill="place_into", object_id="obj_001", bin_id="left_box")]
        with self.assertRaises(Rejected):
            validate(steps, world(make_object()), BINS)

    def test_two_picks_in_a_row_is_rejected(self):
        """그리퍼는 하나뿐이다 — 놓기 전에 또 집을 수 없다."""
        steps = [LlmStep(skill="pick", object_id="obj_001"),
                 LlmStep(skill="pick", object_id="obj_002")]
        with self.assertRaises(Rejected):
            validate(steps, world(make_object(), make_object("obj_002")), BINS)

    def test_pick_without_place_is_rejected(self):
        with self.assertRaises(Rejected):
            validate([LlmStep(skill="pick", object_id="obj_001")],
                     world(make_object()), BINS)

    def test_not_graspable_object_is_rejected(self):
        obj = make_object(graspable=False, not_graspable_reason="depth_invalid")
        with self.assertRaises(Rejected):
            validate(pick_place(), world(obj), BINS)

    def test_object_without_grasp_candidates_is_rejected(self):
        with self.assertRaises(Rejected):
            validate(pick_place(), world(make_object(grasp_candidates=[])), BINS)


class TestPhysicalLimits(unittest.TestCase):
    def test_overweight_object_is_rejected(self):
        obj = make_object(mass_g=MAX_PAYLOAD_G + 1)
        with self.assertRaises(Rejected) as ctx:
            validate(pick_place(), world(obj), BINS)
        self.assertIn("가반하중", str(ctx.exception))

    def test_grasp_outside_workspace_is_rejected(self):
        obj = make_object()
        obj["grasp_candidates"][0]["pose"]["position"] = {"x": 1200.0, "y": 400.0, "z": 300.0}
        with self.assertRaises(Rejected) as ctx:
            validate(pick_place(), world(obj), BINS)
        self.assertIn("작업반경", str(ctx.exception))

    def test_only_the_unreachable_candidate_is_dropped(self):
        """1순위가 팔 밖이어도 닿는 후보가 있으면 물체를 거부하지 않는다.

        예전에는 점수 최고 후보 하나만 보고 작업반경을 판정해서, 그 하나가 벗어나면
        나머지가 멀쩡해도 물체 전체가 거부됐다.
        """
        obj = make_object()
        obj["grasp_candidates"][0]["score"] = 0.99
        obj["grasp_candidates"][0]["pose"]["position"] = {"x": 1200.0, "y": 400.0, "z": 300.0}
        obj["grasp_candidates"].append({
            "pose": {"position": {"x": 450.0, "y": 0.0, "z": 80.0},
                     "orientation": {"x": 0.0, "y": 1.0, "z": 0.0, "w": 0.0}},
            "score": 0.4, "strategy": "graspnet_baseline", "candidate_id": "obj_001#1",
        })
        steps = validate(pick_place(), world(obj), BINS)
        self.assertEqual(len(steps[0].grasp_candidates), 1)
        self.assertEqual(steps[0].grasp_pose.position["x"], 450.0)

    def test_bin_outside_workspace_is_rejected(self):
        with self.assertRaises(Rejected):
            validate(pick_place(bin_id="far_box"), world(make_object()), BINS)


class TestSafetyGate(unittest.TestCase):
    def test_critical_event_blocks_everything(self):
        events = [{"event_type": "e_stop", "severity": "critical",
                   "description": "비상정지 버튼 눌림"}]
        with self.assertRaises(Rejected) as ctx:
            validate(pick_place(), world(make_object()), BINS, events)
        self.assertIn("안전", str(ctx.exception))

    def test_warning_event_does_not_block(self):
        """사소한 경고로 시스템 전체를 멈추지는 않는다."""
        events = [{"event_type": "limit_violation", "severity": "warning",
                   "description": "속도 제한 근접"}]
        steps = validate(pick_place(), world(make_object()), BINS, events)
        self.assertEqual(len(steps), 2)


class TestProfileForcing(unittest.TestCase):
    def test_unconfirmed_class_is_forced_to_fallback(self):
        """VLM 제안값(normal)을 그대로 쓰면 안 된다 — NFR-03a, FR-05b."""
        obj = make_object(profile="normal", needs_confirmation=True,
                          attr_source="llm_suggested")
        self.assertEqual(resolve_profile(obj), "fragile")
        steps = validate(pick_place(), world(obj), BINS)
        self.assertTrue(all(s.profile == "fragile" for s in steps))

    def test_unknown_profile_value_falls_back(self):
        self.assertEqual(resolve_profile(make_object(profile="turbo")), "fragile")

    def test_confirmed_class_keeps_its_profile(self):
        obj = make_object(profile="deformable", needs_confirmation=False)
        self.assertEqual(resolve_profile(obj), "deformable")


if __name__ == "__main__":
    unittest.main()


class TestCandidateListReachesControl(unittest.TestCase):
    """후보를 하나로 좁히지 않고 전부 넘기는지 — 최종 선택은 control이 한다."""

    def _object_with_two_candidates(self):
        obj = make_object()
        obj["position_base_mm"] = {"x": 450.0, "y": 0.0, "z": 100.0}
        obj["height_mm"] = 40.0
        obj["depth_valid_ratio"] = 0.87
        obj["grasp_candidates"][0].update({"candidate_id": "obj_001#0",
                                           "gripper_width_mm": 30.0,
                                           "grasp_depth_mm": 8.0})
        obj["grasp_candidates"].append({
            "pose": {"position": {"x": 460.0, "y": 0.0, "z": 80.0},
                     "orientation": {"x": 0.0, "y": 1.0, "z": 0.0, "w": 0.0}},
            "score": 0.99, "strategy": "graspnet_baseline",
            "candidate_id": "obj_001#1", "gripper_width_mm": 55.0, "grasp_depth_mm": 12.0,
        })
        return obj

    def test_all_reachable_candidates_are_passed_in_score_order(self):
        steps = validate(pick_place(), world(self._object_with_two_candidates()), BINS)
        candidates = steps[0].grasp_candidates
        self.assertEqual([c.candidate_id for c in candidates], ["obj_001#1", "obj_001#0"])
        self.assertEqual([c.gripper_width_mm for c in candidates], [55.0, 30.0])
        self.assertEqual([c.grasp_depth_mm for c in candidates], [12.0, 8.0])

    def test_grasp_pose_still_holds_the_top_candidate(self):
        """후보 목록을 못 읽는 경로(로그·DB·구 control)를 위해 1순위는 그대로 남는다."""
        steps = validate(pick_place(), world(self._object_with_two_candidates()), BINS)
        self.assertEqual(steps[0].grasp_pose.position["x"], 460.0)
        self.assertEqual(steps[0].gripper_width_mm, 55.0)

    def test_object_context_for_ranking_is_carried(self):
        """control은 /world_state를 안 보므로 물체 중심·높이·depth 신뢰도가 실려 가야 한다."""
        steps = validate(pick_place(), world(self._object_with_two_candidates()), BINS)
        self.assertEqual(steps[0].object_center_mm, {"x": 450.0, "y": 0.0, "z": 100.0})
        self.assertEqual(steps[0].object_height_mm, 40.0)
        self.assertEqual(steps[0].depth_valid_ratio, 0.87)

    def test_place_step_carries_no_candidates(self):
        steps = validate(pick_place(), world(self._object_with_two_candidates()), BINS)
        self.assertEqual(steps[1].grasp_candidates, [])
        self.assertIsNone(steps[1].object_center_mm)

    def test_missing_object_context_stays_none(self):
        """height_mm/중심을 perception이 못 낸 물체 — 0을 실어 보내 '있는 척'하지 않는다."""
        steps = validate(pick_place(), world(make_object()), BINS)
        self.assertIsNone(steps[0].object_center_mm)
        self.assertIsNone(steps[0].object_height_mm)
        self.assertIsNone(steps[0].depth_valid_ratio)

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
        obj["grasp_candidates"][0]["width_mm"] = 42.5
        steps = validate(pick_place(), world(obj), BINS)
        self.assertEqual(steps[0].grasp_width_mm, 42.5)

    def test_missing_grasp_width_is_none(self):
        """width_mm을 못 낸 전략(레거시 후보)이면 None — control이 기본 개폭으로 대체한다."""
        steps = validate(pick_place(), world(make_object()), BINS)
        self.assertIsNone(steps[0].grasp_width_mm)


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

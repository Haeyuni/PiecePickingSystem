"""control이 고른 후보가 스텝에 반영되는지 — place가 놓는 높이가 여기서 나온다.

`_object_bottom_offset_mm`은 직전 pick 스텝의 `grasp_pose`로 "물체가 TCP보다 얼마나
아래로 내려와 있는지"를 계산한다. planner가 채운 1순위 후보가 그대로 남아 있는데
control이 다른 후보를 골랐다면 그 높이가 틀리고, 물체를 바구니 바닥에 찍거나 너무
높은 데서 놓는다(2026-09-06 실물: 바닥에 먼저 닿아 안전모드로 들어갔다).

orchestrator 전체는 FastAPI·DB·planner에 묶여 있어 여기서는 함수 하나만 떼어 검증한다.
"""
import ast
import pathlib
import unittest
from types import SimpleNamespace as NS

SOURCE = pathlib.Path(__file__).parents[1] / "web" / "orchestrator.py"


def _function(name):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec")


def load(logs):
    namespace = {"logger": NS(warning=lambda *a: logs.append(a % () if not a[1:] else a[0] % a[1:]))}
    exec(_function("_record_selected_candidate"), namespace)
    return namespace["_record_selected_candidate"]


def step():
    return {
        "object_id": "obj_001",
        "grasp_pose": {"position": {"x": 400.0, "y": 0.0, "z": 300.0}, "orientation": {}},
        "gripper_width_mm": 30.0,
        "grasp_candidates": [
            {"candidate_id": "obj_001#0", "gripper_width_mm": 30.0,
             "pose": {"position": {"x": 400.0, "y": 0.0, "z": 300.0}, "orientation": {}}},
            {"candidate_id": "obj_001#1", "gripper_width_mm": 55.0,
             "pose": {"position": {"x": 410.0, "y": 5.0, "z": 292.0}, "orientation": {}}},
        ],
    }


class RecordSelectedCandidateTest(unittest.TestCase):
    def test_non_top_candidate_replaces_the_planned_pose(self):
        logs = []
        current = step()
        load(logs)(current, NS(selected_candidate_id="obj_001#1"))
        self.assertEqual(current["grasp_pose"]["position"]["z"], 292.0)
        self.assertEqual(current["gripper_width_mm"], 55.0)
        self.assertEqual(current["selected_candidate_id"], "obj_001#1")
        self.assertEqual(logs, [])

    def test_top_candidate_leaves_the_step_equivalent(self):
        current = step()
        load([])(current, NS(selected_candidate_id="obj_001#0"))
        self.assertEqual(current["grasp_pose"]["position"]["z"], 300.0)
        self.assertEqual(current["gripper_width_mm"], 30.0)

    def test_missing_selection_leaves_the_step_untouched(self):
        """구 control이나 후보 목록 없이 보낸 goal — 1순위를 그대로 둔다."""
        current = step()
        load([])(current, NS(selected_candidate_id=""))
        self.assertEqual(current["grasp_pose"]["position"]["z"], 300.0)
        self.assertNotIn("selected_candidate_id", current)

    def test_unknown_selection_warns_instead_of_guessing(self):
        logs = []
        current = step()
        load(logs)(current, NS(selected_candidate_id="obj_001#9"))
        self.assertEqual(current["grasp_pose"]["position"]["z"], 300.0)
        self.assertEqual(len(logs), 1)


if __name__ == "__main__":
    unittest.main()

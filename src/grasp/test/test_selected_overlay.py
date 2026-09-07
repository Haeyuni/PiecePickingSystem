"""오버레이가 강조하는 후보가 control이 실제로 고른 것인지 — ROS 없이 판정만 떼어 본다.

예전에는 1순위(점수 최고)를 초록으로 칠했다. 이제 실행할 후보는 control이 개폭·IK·
관절·최소안전을 보고 고르므로 1순위가 아닐 수 있고, 화면이 실행과 다른 것을 강조하면
사람이 잘못된 후보를 보고 판단하게 된다.
"""
import ast
import time
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest


@pytest.fixture
def node():
    source = Path(__file__).resolve().parents[2] / "grasp/grasp/node.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GraspNode")
    keep = {"_SELECTED_MAX_AGE_S", "_SELECTED_MATCH_TOL_MM"}
    cls.bases = []
    cls.body = [n for n in cls.body
                if (isinstance(n, ast.FunctionDef) and n.name == "_selected_index")
                or (isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id in keep for t in n.targets))]
    namespace = {"np": np, "time": time}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "<overlay>", "exec"), namespace)
    instance = namespace["GraspNode"]()
    instance._selected_grasp = None
    return instance


def candidate(candidate_id, x=400.0, y=0.0, z=300.0):
    return NS(candidate_id=candidate_id, pose=NS(position=NS(x=x, y=y, z=z)))


def selection(node, candidate_msg, age_s=0.0):
    node._selected_grasp = (time.monotonic() - age_s, candidate_msg)


def test_no_selection_highlights_nothing(node):
    """첫 pick 전에는 고른 것이 없다 — 아무것도 초록으로 칠하지 않는다."""
    assert node._selected_index([candidate("obj_001#0"), candidate("obj_001#1")]) is None


def test_selected_candidate_is_found_by_pose(node):
    """순위는 프레임마다 흔들리므로 id의 순위 부분이 아니라 자세로 맞춘다."""
    candidates = [candidate("obj_001#0", x=400.0), candidate("obj_001#1", x=450.0)]
    selection(node, candidate("obj_001#7", x=450.0))
    assert node._selected_index(candidates) == 1


def test_small_pose_drift_still_matches(node):
    candidates = [candidate("obj_001#0", x=400.0), candidate("obj_001#1", x=450.0)]
    selection(node, candidate("obj_001#1", x=452.0))
    assert node._selected_index(candidates) == 1


def test_far_pose_does_not_match(node):
    """자세가 크게 다르면 다른 후보다 — 엉뚱한 것을 강조하느니 아무것도 안 칠한다."""
    candidates = [candidate("obj_001#0", x=400.0)]
    selection(node, candidate("obj_001#0", x=480.0))
    assert node._selected_index(candidates) is None


def test_other_object_selection_is_ignored(node):
    """다른 물체를 집는 중이면 이 물체의 후보를 강조하면 안 된다."""
    candidates = [candidate("obj_002#0", x=400.0)]
    selection(node, candidate("obj_001#0", x=400.0))
    assert node._selected_index(candidates) is None


def test_stale_selection_expires(node):
    candidates = [candidate("obj_001#0")]
    selection(node, candidate("obj_001#0"), age_s=node._SELECTED_MAX_AGE_S + 1.0)
    assert node._selected_index(candidates) is None


def test_nearest_candidate_wins_when_two_are_close(node):
    candidates = [candidate("obj_001#0", x=400.0), candidate("obj_001#1", x=403.0)]
    selection(node, candidate("obj_001#1", x=403.5))
    assert node._selected_index(candidates) == 1


def test_empty_candidate_list_is_safe(node):
    selection(node, candidate("obj_001#0"))
    assert node._selected_index([]) is None

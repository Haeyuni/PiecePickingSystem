import ast
import dataclasses
import pathlib
from types import SimpleNamespace as NS


SOURCE = pathlib.Path(__file__).parents[1] / "web" / "orchestrator.py"


def _load_snapshot_code():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    wanted = {"PickSnapshot", "_pose_tuple", "_freeze_pick_snapshot"}
    nodes = [node for node in tree.body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted]
    namespace = {"dataclasses": dataclasses}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


def test_snapshot_freezes_actual_selected_pick():
    code = _load_snapshot_code()
    step = {
        "source_observation_id": "obs-17",
        "source_observation_stamp": {"sec": 12, "nanosec": 34},
        "object_id": "obj_002",
        "class_name": "wet_wipes",
        "object_center_mm": {"x": 100, "y": 50, "z": 80},
        "object_height_mm": 20,
        "object_footprint_base_mm": [
            {"x": 90, "y": 40}, {"x": 110, "y": 40},
            {"x": 110, "y": 60}, {"x": 90, "y": 60},
        ],
        "grasp_pose": {
            "position": {"x": 102, "y": 51, "z": 77},
            "orientation": {"x": 0, "y": 1, "z": 0, "w": 0},
        },
    }
    result = NS(selected_candidate_id="obj_002#3",
                executed_tcp_posx=[102, 51, 83, 20, 179, 21])

    snapshot = code["_freeze_pick_snapshot"](step, result)

    assert snapshot.observation_id == "obs-17"
    assert snapshot.selected_candidate_id == "obj_002#3"
    assert snapshot.tcp_to_object_bottom_mm == 23
    try:
        snapshot.object_id = "changed"
        assert False, "frozen snapshot accepted mutation"
    except dataclasses.FrozenInstanceError:
        pass


def test_snapshot_rejects_empty_observation_identity():
    code = _load_snapshot_code()
    step = {
        "source_observation_id": "",
        "source_observation_stamp": {"sec": 12, "nanosec": 34},
        "object_id": "obj_002", "class_name": "wet_wipes",
        "object_center_mm": {"z": 80}, "object_height_mm": 20,
        "object_footprint_base_mm": [], "grasp_pose": {},
    }
    result = NS(selected_candidate_id="obj_002#0",
                executed_tcp_posx=[0, 0, 80, 0, 180, 0])
    assert code["_freeze_pick_snapshot"](step, result) is None

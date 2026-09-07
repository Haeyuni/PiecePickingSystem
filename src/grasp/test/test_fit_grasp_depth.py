"""Pure depth-fit regressions; extract production code without importing ROS."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def node():
    src = Path(__file__).resolve().parents[2]
    tree = ast.parse((src / 'grasp/grasp/node.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GraspNode')
    constants = {'_SUPPORT_CLEARANCE_MM', '_SUPPORT_CLEARANCE_HEIGHT_FRACTION',
                 '_FINGER_REACH_MM'}
    cls.bases = []
    cls.body = [n for n in cls.body
                if (isinstance(n, ast.FunctionDef) and n.name == '_fit_grasp_depth')
                or (isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id in constants for t in n.targets))]
    geometry_tree = ast.parse((src / 'perception_common/perception_common/geometry.py').read_text())
    conversion = next(n for n in geometry_tree.body
                      if isinstance(n, ast.FunctionDef) and n.name == 'quaternion_to_matrix')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[conversion], type_ignores=[]), '<geometry>', 'exec'), namespace)
    namespace['geometry'] = SimpleNamespace(quaternion_to_matrix=namespace['quaternion_to_matrix'])
    exec(compile(ast.Module(body=[cls], type_ignores=[]), '<depth-fit>', 'exec'), namespace)
    instance = namespace['GraspNode']()
    instance.logs = []
    instance.warnings = []
    # list.append also ensures logs cannot silently acquire throttle kwargs.
    instance.get_logger = lambda: SimpleNamespace(info=instance.logs.append,
                                                  warning=instance.warnings.append)
    return instance


def candidate(z, quaternion=(1.0, 0.0, 0.0, 0.0)):
    return {'pose': {'position': {'x': 10.0, 'y': 20.0, 'z': z},
                     'orientation': dict(zip(('x', 'y', 'z', 'w'), quaternion))},
            'score': 0.8, 'width_mm': 30.0, 'strategy': 'test'}


def obj(height):
    return SimpleNamespace(object_id='obj_test', height_mm=height,
                           position_base_mm=SimpleNamespace(z=300.0 + height))


@pytest.mark.parametrize('height,clearance', [(1.0, 0.4), (2.0, 0.8), (5.0, 2.0), (20.0, 2.5)])
@pytest.mark.parametrize('offset', [-0.49, -0.01, 0.0])
def test_floor_without_deadband(node, height, clearance, offset):
    original = candidate(300.0 + clearance + offset)
    snapshot = deepcopy(original)
    result = node._fit_grasp_depth(original, obj(height))
    assert result['pose']['position']['z'] == 300.0 + clearance
    assert result['pose']['orientation'] == original['pose']['orientation']
    assert result['score'] == original['score']
    assert original == snapshot
    assert len(node.logs) == 1
    assert f'unchanged={offset == 0.0}' in node.logs[0]


@pytest.mark.parametrize('z', [299.0, 310.0, 325.0])
def test_support_floor_applies_but_deeper_candidate_is_preserved(node, z):
    result = node._fit_grasp_depth(candidate(z), obj(70.0))
    assert result['pose']['position']['z'] == max(z, 302.5)


def test_tilted_fit_moves_along_own_axis(node):
    angle = np.radians(150.0)
    q = (0.0, float(np.sin(angle / 2)), 0.0, float(np.cos(angle / 2)))
    original = candidate(320.0, q)
    result = node._fit_grasp_depth(original, obj(20.0))
    position = result['pose']['position']
    axis = np.array([0.5, 0.0, -np.sqrt(3) / 2])
    expected = np.array([10.0, 20.0, 320.0]) + ((302.5 - 320.0) / axis[2]) * axis
    assert list(position.values()) == pytest.approx(expected)
    assert result['pose']['orientation'] == original['pose']['orientation']
    assert f'fitted_xyz={tuple(position.values())!r}' in node.logs[0]
    assert f'quaternion_xyzw={q!r}' in node.logs[0]


@pytest.mark.parametrize('z,old,returned,new,height', [
    (300.2, 300.0, 300.2, 302.5, 20.0),
    (302.5, 300.0, 300.0, 302.5, 20.0),
    (310.0, 310.0, 310.0, 310.0, 70.0),
])
def test_full_old_and_new_diagnostics(node, z, old, returned, new, height):
    node._fit_grasp_depth(candidate(z), obj(height))
    log = node.logs[0]
    assert ('OLD zero-clearance: max(min(candidate_z, max(support_z+0, top_z-reach)), '
            f'support_z+0)={old!r}') in log
    assert f'returned_z={returned!r}' in log
    assert 'NEW clearance=min(2.5, 0.4*h)=2.5' in log
    assert (f'max(min(candidate_z, max(support_z+clearance=302.5, '
            f'top_z-reach={300.0 + height - 45.0!r})), support_z+clearance)={new!r}') in log
    assert f'fitted_z={new!r}' in log
    assert f'fitted_xyz={(10.0, 20.0, new)!r}' in log
    assert 'quaternion_xyzw=(1.0, 0.0, 0.0, 0.0)' in log


@pytest.mark.parametrize('height', [0.0, -1.0, float('nan'), float('inf')])
def test_unknown_or_nonfinite_height_is_not_fitted(node, height):
    original = candidate(300.0)
    assert node._fit_grasp_depth(original, obj(height)) is original
    assert len(node.warnings) == 1


@pytest.mark.parametrize('q', [(0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 0.0),
                             (float('nan'), 0.0, 0.0, 1.0),
                             (0.0, np.sqrt(0.5), 0.0, np.sqrt(0.5))])
def test_invalid_or_nondownward_axis_is_not_fitted(node, q):
    original = candidate(300.0, q)
    assert node._fit_grasp_depth(original, obj(20.0)) is original
    assert len(node.warnings) == 1


@pytest.mark.parametrize('field', ['x', 'y', 'z', 'top'])
def test_nonfinite_position_is_not_fitted(node, field):
    original, detected = candidate(300.0), obj(20.0)
    if field == 'top':
        detected.position_base_mm.z = float('inf')
    else:
        original['pose']['position'][field] = float('nan')
    assert node._fit_grasp_depth(original, detected) is original
    assert len(node.warnings) == 1

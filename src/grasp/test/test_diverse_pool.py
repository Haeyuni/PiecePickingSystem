"""Final-pose Diverse-TopK + contact_support (STEP 1, 2026-09-10).

여기서 지키려는 계약:

[F] 다양성은 **_fit_grasp_depth까지 끝난 최종 pose**로 계산한다. 위치·접근축·닫힘축을
    보고, GraspNet 최고점 후보 1개는 pool에 보존한다.
[E] contact_support는 되잡기가 이미 본 정보만 재사용해 0~1로 내고, 미상은 -1이다.
    edge라는 이유로 hard reject하지 않는다.
[C] max_candidates 50 / refine_pool_size 12 / execution_pool_size 10의 의미가 분리돼 있다.
"""
import ast
import pathlib

import numpy as np
import pytest
import yaml

from grasp import diverse_pool
from grasp.strategies import graspnet_baseline as gb

REPO = pathlib.Path(__file__).resolve().parents[3]


def _candidate(x=0.0, y=0.0, z=0.0, approach=(0.0, 0.0, -1.0), closing=(1.0, 0.0, 0.0),
               score=0.9):
    """최종 pose를 직접 지정한 후보 dict (전략이 내는 것과 같은 모양)."""
    approach = np.asarray(approach, dtype=float)
    approach = approach / np.linalg.norm(approach)
    closing = np.asarray(closing, dtype=float)
    closing = closing - approach * float(closing @ approach)
    closing = closing / np.linalg.norm(closing)
    other = np.cross(approach, closing)
    qx, qy, qz, qw = gb._quaternion_from_matrix(
        np.column_stack([closing, other, approach]))
    return {"pose": {"position": {"x": float(x), "y": float(y), "z": float(z)},
                     "orientation": {"x": qx, "y": qy, "z": qz, "w": qw}},
            "score": float(score), "width_mm": 40.0, "strategy": "graspnet_baseline"}


# --- [F] Diverse-TopK ---------------------------------------------------------

def test_axes_round_trip_through_the_quaternion():
    """pose 쿼터니언에서 접근축·닫힘축을 되뽑을 수 있어야 다양성 계산이 성립한다."""
    approach, closing = diverse_pool.axes_from_quaternion(
        _candidate(approach=(0.3, 0.0, -1.0), closing=(0.0, 1.0, 0.0))["pose"]["orientation"])
    expected = np.array([0.3, 0.0, -1.0]) / np.linalg.norm([0.3, 0.0, -1.0])
    assert approach == pytest.approx(expected, abs=1e-6)
    assert closing == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)


def test_highest_score_candidate_is_preserved():
    """seed는 항상 0번(= GraspNet 최고점)이다 — 비교 가능한 fallback을 남긴다."""
    candidates = [_candidate(x=0.0, score=0.9), _candidate(x=1.0, score=0.8),
                  _candidate(x=200.0, score=0.4)]
    chosen, _ = diverse_pool.pool_from_candidates(candidates, pool_size=2)
    assert chosen[0] == 0


def test_similar_candidates_do_not_monopolise_the_pool():
    """같은 자리에 몰린 상위 후보들 대신 실제로 다른 선택지가 들어와야 한다."""
    candidates = [_candidate(x=0.0, score=0.90), _candidate(x=2.0, score=0.89),
                  _candidate(x=3.0, score=0.88), _candidate(x=1.5, score=0.87),
                  _candidate(x=90.0, score=0.50), _candidate(y=90.0, score=0.40)]
    chosen, diagnostics = diverse_pool.pool_from_candidates(candidates, pool_size=3)
    assert chosen[0] == 0
    assert set(chosen) & {4, 5}, "멀리 있는 후보가 최소 하나는 들어와야 한다"
    assert chosen != diagnostics["score_top_ranks"], "score-only Top-K와 같으면 다양성이 없다"
    assert (diagnostics["spread_after"]["position_mm"]
            > diagnostics["spread_before"]["position_mm"])


def test_position_approach_and_closing_all_count():
    """세 축 각각이 다양성 거리에 반영돼야 한다 — 하나라도 빠지면 그 차이를 못 본다."""
    base = _candidate()
    moved = _candidate(x=60.0)
    tilted = _candidate(approach=(0.6, 0.0, -1.0))
    rolled = _candidate(closing=(0.0, 1.0, 0.0))
    for other in (moved, tilted, rolled):
        _, diagnostics = diverse_pool.pool_from_candidates([base, other], pool_size=2)
        assert diagnostics["spread_after"] != {"position_mm": 0.0, "approach_deg": 0.0,
                                               "closing_deg": 0.0}


def test_selection_is_deterministic():
    """같은 장면이면 같은 결과여야 한다 — 무작위 seed를 쓰지 않는다."""
    candidates = [_candidate(x=float(i) * 7.0, score=0.9 - 0.01 * i) for i in range(12)]
    first, _ = diverse_pool.pool_from_candidates(candidates, pool_size=5)
    second, _ = diverse_pool.pool_from_candidates(candidates, pool_size=5)
    assert first == second


def test_pool_is_capped_by_available_candidates():
    chosen, _ = diverse_pool.pool_from_candidates([_candidate()], pool_size=10)
    assert chosen == [0]


def test_diversity_runs_after_depth_fit_in_the_node():
    """**순서 계약**: _fit_grasp_depth → _diverse_pool → _to_msg.

    깊이 맞춤은 접근축을 따라 움직여 기울어진 파지에서는 XY도 바꾼다. 그 전에 다양성을
    고르면 실제 실행될 자세가 아닌 좌표로 고르는 셈이다 — 코드가 그 순서를 유지하는지
    소스에서 직접 확인한다(node.py는 rclpy를 끌어와 import할 수 없다).
    """
    source = (REPO / "src" / "grasp" / "grasp" / "node.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "GraspNode")
    func = next(n for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name == "_candidates_for")
    # ast.walk는 소스 순서를 보장하지 않는다 — 줄 번호로 다시 세운다.
    calls = sorted(((node.lineno, node.func.attr) for node in ast.walk(func)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)))
    order = [name for _, name in calls]
    assert order.index("_fit_grasp_depth") < order.index("_diverse_pool")
    assert order.index("_diverse_pool") < order.index("_to_msg")


# --- [E] contact_support ------------------------------------------------------

def _slab(n=4000, half_x=40.0, half_y=20.0, thickness=10.0, seed=0):
    rng = np.random.default_rng(seed)
    return np.c_[rng.uniform(-half_x, half_x, n), rng.uniform(-half_y, half_y, n),
                 rng.uniform(-thickness, 0.0, n)]


def _vertical_grasp(x=0.0, y=0.0):
    T = np.eye(4)
    T[:3, :3] = np.column_stack([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)])
    T[:3, 3] = (x, y, 0.0)
    return T


def test_contact_support_is_in_unit_range_and_high_on_solid_material():
    _, _, diag = gb._refine_on_cloud(_vertical_grasp(), _slab(), grasp_depth_mm=4.0)
    support = diag["contact_support_score"]
    assert 0.0 <= support <= 1.0
    assert support > 0.5, "재료가 넉넉한 평판 중앙이면 높게 나와야 한다"


def test_edge_grasp_with_material_on_both_fingers_is_not_penalised():
    """물체 끝을 잡아도 **양쪽 손가락에 재료가 있으면** 낮은 값이 아니어야 한다(§6).

    hard reject도 아니다 — 기하 판정(status)은 그대로 ok다.
    """
    points = _slab(half_x=40.0)
    center = gb._refine_on_cloud(_vertical_grasp(x=0.0), points, grasp_depth_mm=4.0)[2]
    edge = gb._refine_on_cloud(_vertical_grasp(x=32.0), points, grasp_depth_mm=4.0)[2]
    assert edge["status"] == gb.GEOMETRY_OK
    assert edge["contact_support_score"] > 0.5
    assert edge["contact_support_score"] == pytest.approx(
        center["contact_support_score"], abs=0.25)


def test_contact_support_reuses_refine_information_only():
    """추가 클라우드 패스를 만들지 않는다 — 이미 계산된 투영/창만 인자로 받는다."""
    source = (REPO / "src" / "grasp" / "grasp" / "strategies" / "graspnet_baseline.py"
              ).read_text(encoding="utf-8")
    func = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == "_contact_support")
    # 인자가 전부 "이미 계산된 것"뿐이다 — 원본 클라우드(points_base)를 받지 않으므로
    # 이 함수 안에서 두 번째 전체 투영을 할 방법이 없다.
    assert [a.arg for a in func.args.args] == [
        "proj_c", "proj_a", "window", "grip_center_c", "min_grip_points"]
    assert "points_base" not in ast.get_source_segment(source, func)


def test_unknown_contact_support_is_minus_one_not_zero():
    """되잡기를 못 한 경우 -1(미상)이다 — 0.0(지지 없음)과 뜻이 다르다."""
    rotation = np.column_stack([(0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)])
    raw = {'score': 0.8, 'width_m': 0.04, 'depth_m': 0.0,
           'rotation_matrix': rotation.tolist(), 'translation_m': [0.0, 0.0, 0.5]}
    picked, _ = gb._select_candidates([raw], np.eye(4), np.eye(4), 45.0, points_base=None)
    assert picked[0]["contact_support_score"] == -1.0


# --- [C] 후보 수 config -------------------------------------------------------

def test_candidate_count_config_is_separated():
    config = yaml.safe_load(
        (REPO / "src" / "grasp" / "config" / "grasp_params.yaml").read_text(encoding="utf-8"))
    graspnet = config["graspnet_baseline"]
    assert graspnet["max_candidates"] == 50
    assert graspnet["refine_pool_size"] == 12
    assert graspnet["execution_pool_size"] == 10
    assert graspnet["refine_pool_size"] >= graspnet["execution_pool_size"]
    assert graspnet["max_candidates"] >= graspnet["refine_pool_size"]

    control_config = REPO / "src" / "control" / "config" / "skill_params.yaml"
    if control_config.is_file():
        # grasp runtime 이미지는 control source를 포함하지 않는다. 저장소 전체가 있는 호스트
        # 통합 테스트에서만 두 패키지 사이의 pool 상한까지 확인한다.
        skill = yaml.safe_load(control_config.read_text(encoding="utf-8"))
        assert skill["grasp_selection"]["max_evaluated"] >= graspnet["execution_pool_size"], (
            "control이 execution pool보다 적게 평가하면 뒤쪽 후보가 조용히 사라진다")


# --- [B] 접근각 정책 (이번 STEP에서 변경 없음) --------------------------------

@pytest.mark.parametrize("tilt_deg,expected", [(44.0, 1), (45.0, 1), (46.0, 0)])
def test_hard_angle_is_45_with_no_widening(tilt_deg, expected):
    """45도 이하 통과, 초과 제외. 예전처럼 좁은 각도부터 넓혀가는 loop는 없다."""
    tilt = np.radians(tilt_deg)
    approach = np.array([np.sin(tilt), 0.0, -np.cos(tilt)])
    helper = np.array([0.0, 1.0, 0.0])
    closing = np.cross(helper, approach)
    closing /= np.linalg.norm(closing)
    other = np.cross(approach, closing)
    raw = {'score': 0.8, 'width_m': 0.04, 'depth_m': 0.0,
           'rotation_matrix': np.column_stack([approach, closing, other]).tolist(),
           'translation_m': [0.0, 0.0, 0.5]}
    picked, diagnostics = gb._select_candidates(
        [raw], np.eye(4), np.eye(4), 45.0, points_base=None)
    assert len(picked) == expected
    assert diagnostics["hard_max_deg"] == 45.0
    assert diagnostics["angle_pass_count"] == expected


def test_config_hard_angle_stays_45():
    config = yaml.safe_load(
        (REPO / "src" / "grasp" / "config" / "grasp_params.yaml").read_text(encoding="utf-8"))
    graspnet = config["graspnet_baseline"]
    assert graspnet["approach_angle_hard_max_deg"] == 45.0
    assert graspnet["approach_angle_prefer_deg"] == 15.0
    assert graspnet["approach_angle_soft_max_deg"] == 30.0


# --- 중복 후보 제거 (2026-09-10) ----------------------------------------------

def test_identical_poses_still_fill_the_pool_but_are_reported():
    """**중복 후보를 버리지 않는다(2026-09-10 사용자 요청).** pool은 요청한 크기대로 채우고,
    실제로 서로 다른 자세가 몇 개인지는 진단(distinct_poses)으로만 남긴다."""
    same = [_candidate(x=0.0, score=0.9 - 0.01 * i) for i in range(8)]
    chosen, diagnostics = diverse_pool.pool_from_candidates(same, pool_size=5)
    assert len(chosen) == 5
    assert chosen[0] == 0
    assert diagnostics["distinct_poses"] == 1, "다양성의 실제 천장은 로그로 드러나야 한다"


def test_distinct_poses_are_all_kept():
    candidates = [_candidate(x=0.0), _candidate(x=60.0), _candidate(x=120.0)]
    chosen, diagnostics = diverse_pool.pool_from_candidates(candidates, pool_size=10)
    assert sorted(chosen) == [0, 1, 2]
    assert diagnostics["distinct_poses"] == 3


def test_distinct_candidates_come_before_duplicates():
    """중복을 버리지는 않되, 서로 다른 후보가 **먼저** 뽑혀야 한다 — 앞쪽 후보가 IK로
    떨어졌을 때 실제로 다른 선택지가 남는 것이 pool을 두는 이유다."""
    candidates = ([_candidate(x=0.0, score=0.9 - 0.01 * i) for i in range(5)]
                  + [_candidate(x=70.0, score=0.5), _candidate(y=70.0, score=0.4)])
    chosen, diagnostics = diverse_pool.pool_from_candidates(candidates, pool_size=10)
    assert chosen[0] == 0, "최고점 seed는 그대로 남는다"
    assert set(chosen[1:3]) == {5, 6}, "서로 다른 자세가 중복보다 먼저 들어와야 한다"
    assert diagnostics["distinct_poses"] == 3

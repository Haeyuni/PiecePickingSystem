import numpy as np
import pytest

from grasp import strategies
from grasp.strategies import graspnet_baseline
from grasp.strategies.exceptions import InferenceBusy


def test_registry_exposes_graspnet_baseline():
    assert strategies.get('graspnet_baseline') is graspnet_baseline.plan


def _raw(rotation, translation_m, *, width_m=0.04, depth_m=0.0, score=0.8):
    return {'score': score, 'width_m': width_m, 'depth_m': depth_m,
            'rotation_matrix': np.asarray(rotation).tolist(),
            'translation_m': list(translation_m)}


def test_camera_meter_output_becomes_base_mm_pose():
    """camera/m 후보 하나가 base/mm TCP 자세로 옮겨지는지 (되잡기 없이, 순수 변환)."""
    T_base_camera_mm = np.eye(4)
    T_base_camera_mm[:3, 3] = (100.0, -50.0, 200.0)
    T_graspnet_tcp_mm = np.eye(4)
    T_graspnet_tcp_mm[:3, 3] = (10.0, 20.0, 30.0)

    # 접근축(회전 1열)이 base -Z를 향해야 기울기 필터를 통과한다.
    rotation = np.column_stack([(0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)])
    picked, _ = graspnet_baseline._select_candidates(
        [_raw(rotation, [0.1, 0.2, 0.3])], T_base_camera_mm, T_graspnet_tcp_mm,
        75.0, points_base=None)

    assert len(picked) == 1
    best = picked[0]
    assert best['strategy'] == 'graspnet_baseline'
    assert best['width_mm'] == pytest.approx(40.0)
    position = best['pose']['position']
    # base 위치 = T_base_camera @ (R @ t_graspnet_tcp + translation)
    expected = T_base_camera_mm[:3, 3] + rotation @ (10.0, 20.0, 30.0) + (100.0, 200.0, 300.0)
    assert (position['x'], position['y'], position['z']) == pytest.approx(tuple(expected))


def test_graspnet_depth_advances_along_the_approach_axis():
    """`depth_m`은 무는 지점까지의 거리다 — 접근축을 따라 전진해야 한다 (버리면 허공)."""
    rotation = np.column_stack([(0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)])
    common = dict(T_base_camera_mm=np.eye(4), T_graspnet_tcp_mm=np.eye(4))
    without, _ = graspnet_baseline._select_candidates(
        [_raw(rotation, [0.0, 0.0, 0.5], depth_m=0.0)],
        common['T_base_camera_mm'], common['T_graspnet_tcp_mm'], 75.0,
        points_base=None)
    with_depth, _ = graspnet_baseline._select_candidates(
        [_raw(rotation, [0.0, 0.0, 0.5], depth_m=0.03)],
        common['T_base_camera_mm'], common['T_graspnet_tcp_mm'], 75.0,
        points_base=None)
    without, with_depth = without[0], with_depth[0]
    # 접근축이 base -Z이므로 30mm 전진은 z가 30mm 낮아지는 것으로 나타난다.
    assert (without['pose']['position']['z']
            - with_depth['pose']['position']['z']) == pytest.approx(30.0)


def test_upright_filter_rejects_candidates_that_stab_from_below():
    """아래에서 위로 찌르는 자세는 abs()로 재면 0도로 통과한다 — 걸러져야 한다."""
    upward = np.column_stack([(0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)])
    picked, diagnostics = graspnet_baseline._select_candidates(
        [_raw(upward, [0.0, 0.0, 0.5])], np.eye(4), np.eye(4), 75.0,
        points_base=None)
    assert picked == []
    assert diagnostics['angles_deg'] == [180.0]


def _plate_cloud(half_x=40.0, half_y=25.0, thickness=8.0, n=6000, seed=0):
    """작업대 위에 놓인 납작한 판. 윗면 z=0, 아래로 thickness 만큼."""
    rng = np.random.default_rng(seed)
    return np.c_[rng.uniform(-half_x, half_x, n),
                 rng.uniform(-half_y, half_y, n),
                 rng.uniform(-thickness, 0.0, n)]


@pytest.mark.parametrize('tilt_deg', [0.0, 15.0, 30.0])
def test_refine_is_not_dragged_sideways_by_a_tilted_approach(tilt_deg):
    """**물체가 가만히 있으면 접근축을 기울여도 파지점은 제자리여야 한다.**

    예전 식은 표면을 클라우드 전체의 접근축 극값으로 잡아, 기울이면 그 극값이 먼 모서리가
    되면서 파지점이 최대 35mm 옆으로 끌려갔다(2026-09-07 실측). GraspNet은 거의 항상
    15~30도 기울어진 자세를 내므로 이 미끄러짐이 매 파지마다 작용했다.
    """
    points = _plate_cloud()
    angle = np.radians(tilt_deg)
    approach = np.array([np.sin(angle), 0.0, -np.cos(angle)])
    closing = np.cross([0.0, 1.0, 0.0], approach)
    closing /= np.linalg.norm(closing)
    other = np.cross(approach, closing)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([closing, other, approach])

    position, _, _ = graspnet_baseline._refine_on_cloud(T, points, grasp_depth_mm=4.0)
    # 판의 중심은 원점이다. 가로로 몇 mm 안쪽이어야 한다.
    assert np.hypot(position[0], position[1]) < 6.0, (
        f"기울기 {tilt_deg}도에서 파지점이 가로로 {np.hypot(position[0], position[1]):.1f}mm 밀렸다")
    # 깊이도 판 안쪽이어야 한다 (윗면 0, 바닥 -8).
    assert -8.0 <= position[2] <= 0.0


def test_refine_width_is_measured_across_the_finger_path():
    """개폭은 손가락이 지나는 창 안에서 재야 한다 — 마스크 전체 범위가 아니라."""
    points = _plate_cloud(half_x=40.0, half_y=25.0)
    T = np.eye(4)                       # closing=+X, other=+Y, approach=+Z
    T[:3, :3] = np.column_stack([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)])
    _, width, _ = graspnet_baseline._refine_on_cloud(T, points, grasp_depth_mm=4.0)
    assert width == pytest.approx(80.0, abs=6.0)


def test_missing_tcp_transform_is_not_replaced_by_pca():
    with pytest.raises(RuntimeError, match='T_graspnet_tcp_mm'):
        graspnet_baseline.plan(
            np.empty((0, 3)),
            {'checkpoint_path': '/does/not/exist/checkpoint.tar'},
            context={'points_cam_mm': np.ones((80, 3)), 'T_base_camera_mm': np.eye(4)},
        )


def test_missing_endpoint_is_not_replaced_by_pca():
    """endpoint가 없으면 조용히 PCA로 떨어지지 않고 **오류를 낸다.**

    이 전략은 상주 추론 서버로만 동작한다(1회용 docker run 경로는 2026-09-07에 제거).
    빈 후보를 돌려주면 planner가 "파지 불가"로 읽어 멀쩡한 물체를 거부하므로,
    설정 누락은 반드시 드러나야 한다.
    """
    with pytest.raises(RuntimeError, match='endpoint'):
        graspnet_baseline.plan(
            np.empty((0, 3)),
            {'T_graspnet_tcp_mm': np.eye(4).tolist()},
            context={'points_cam_mm': np.ones((80, 3)), 'T_base_camera_mm': np.eye(4)},
        )


def test_busy_inference_does_not_become_empty_candidates():
    """이전 추론이 진행 중이면 빈 후보가 아니라 InferenceBusy로 알린다.

    빈 후보는 planner에게 "파지 불가"를 뜻하므로, 단지 바쁜 것과 구분되어야 한다.
    """
    assert graspnet_baseline._INFERENCE_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(InferenceBusy):
            graspnet_baseline.plan(
                np.empty((0, 3)),
                {
                    'endpoint': 'http://localhost:8200',
                    'T_graspnet_tcp_mm': np.eye(4).tolist(),
                },
                context={'points_cam_mm': np.ones((80, 3)), 'T_base_camera_mm': np.eye(4)},
            )
    finally:
        graspnet_baseline._INFERENCE_LOCK.release()


def test_refined_pool_is_kept_in_score_order():
    """**Top-K가 유지돼야 한다.** 예전에는 최고점 하나만 돌려줘서 웹 시각화와 이후의
    최종 선택 알고리즘이 쓸 후보가 남지 않았다(2026-09-07).
    """
    rotation = np.column_stack([(0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)])
    raws = [_raw(rotation, [0.01 * i, 0.0, 0.5], score=0.1 * i) for i in range(1, 8)]

    picked, diagnostics = graspnet_baseline._select_candidates(
        raws, np.eye(4), np.eye(4), 75.0, points_base=None, refine_pool_size=5)

    assert len(picked) == 5, "refine_pool_size 만큼 유지돼야 한다"
    assert diagnostics['raw_count'] == 7
    scores = [c['score'] for c in picked]
    assert scores == sorted(scores, reverse=True), "점수 내림차순이어야 한다"
    assert picked[0]['score'] == pytest.approx(0.7)
    # 1순위에만 좌표체인 진단이 붙는다(로그가 K배로 늘지 않게)
    assert 'debug' in picked[0] and 'debug' not in picked[1]
    # 각 후보가 필요한 필드를 갖는다
    for c in picked:
        assert set(c) >= {'pose', 'width_mm', 'score', 'grasp_depth_mm', 'strategy'}


def test_refined_pool_is_capped_by_available_candidates():
    """후보가 refine_pool_size보다 적으면 있는 만큼만 — 빈 자리를 만들지 않는다."""
    rotation = np.column_stack([(0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)])
    picked, _ = graspnet_baseline._select_candidates(
        [_raw(rotation, [0.0, 0.0, 0.5])], np.eye(4), np.eye(4), 75.0,
        points_base=None, refine_pool_size=10)
    assert len(picked) == 1


# --- 접근각 정책 (2026-09-08, 2.5차) -----------------------------------------
# 예전에는 threshold(15)부터 step(5)씩 넓히다 **처음 통과자가 나오면 멈춰서**, 실제 상한이
# 30도가 아니라 "후보가 하나라도 있는 가장 좁은 5도 구간"이었다. 그 구조를 없애고 hard
# 상한 하나만 남긴 것이 이 정책이다. 아래 테스트가 지키려는 것:
#   - 수직에 가까운 후보 하나가 있어도 기울어진 후보가 같이 살아남는다 (예전엔 죽었다)
#   - 명백히 위험한 각도는 여전히 hard reject
#   - Top-K가 그대로 상한 역할을 한다

def _tilted(angle_deg, score=0.8):
    """수직 아래에서 `angle_deg`만큼 기운 접근축을 가진 raw 후보 (base = camera = 단위행렬)."""
    rad = np.radians(angle_deg)
    approach = np.array([np.sin(rad), 0.0, -np.cos(rad)])
    closing = np.array([0.0, 1.0, 0.0])
    third = np.cross(approach, closing)
    rotation = np.column_stack([approach, closing, third])
    return _raw(rotation, [0.0, 0.0, 0.5], score=score)


def test_upright_candidate_no_longer_starves_the_tilted_ones():
    """**2.5차의 핵심 회귀 방지.** 12도짜리가 하나 있다고 25/40도가 버려지면 안 된다.

    예전 구조에서는 15도 구간에서 멈춰 나머지가 통째로 사라졌다 — 2026-09-08 실물
    로그에서 raw=50인데 filtered=1로 나오던 물체들이 이것이다.
    """
    raws = [_tilted(12.0, score=0.9), _tilted(25.0, score=0.8), _tilted(40.0, score=0.7)]
    picked, diagnostics = graspnet_baseline._select_candidates(
        raws, np.eye(4), np.eye(4), 75.0, points_base=None)
    assert len(picked) == 3, "기울어진 후보가 수직 후보 때문에 죽으면 안 된다"
    assert diagnostics['passed_count'] == 3


def test_extreme_angle_is_hard_rejected():
    """Case 3 — 명백히 위험한 접근각은 grasp 단계에서 자른다."""
    raws = [_tilted(20.0), _tilted(80.0), _tilted(100.0)]
    picked, diagnostics = graspnet_baseline._select_candidates(
        raws, np.eye(4), np.eye(4), 75.0, points_base=None)
    assert diagnostics['passed_count'] == 1
    assert len(picked) == 1
    assert picked[0]['approach_angle_deg'] == pytest.approx(20.0, abs=0.1)


def test_hard_max_is_configurable():
    """상한은 설정값이다 — 실물에서 작업대를 스치면 내릴 수 있어야 한다."""
    raws = [_tilted(50.0)]
    assert graspnet_baseline._select_candidates(
        raws, np.eye(4), np.eye(4), 75.0, points_base=None)[0]
    assert graspnet_baseline._select_candidates(
        raws, np.eye(4), np.eye(4), 45.0, points_base=None)[0] == []


def test_legacy_pass_count_is_reported_for_comparison():
    """정책 변경 효과를 로그로 보려면 '예전 기준이면 몇 개였나'가 함께 있어야 한다."""
    raws = [_tilted(10.0), _tilted(28.0), _tilted(44.0), _tilted(62.0)]
    _, diagnostics = graspnet_baseline._select_candidates(
        raws, np.eye(4), np.eye(4), 75.0, legacy_max_deg=30.0, points_base=None)
    assert diagnostics['legacy_pass_count'] == 2      # 10, 28
    assert diagnostics['passed_count'] == 4
    assert diagnostics['legacy_max_deg'] == 30.0
    assert diagnostics['hard_max_deg'] == 75.0


def test_refine_pool_still_caps_the_widened_policy():
    """Case 5 — 각도를 풀어 후보가 늘어도 되잡기 pool은 refine_pool_size로 묶인다.

    world_state로 나가는 수는 그 뒤 execution_pool_size(다양성 선택)가 다시 묶는다."""
    raws = [_tilted(5.0 + i * 4.0, score=0.9 - i * 0.01) for i in range(15)]
    picked, diagnostics = graspnet_baseline._select_candidates(
        raws, np.eye(4), np.eye(4), 75.0, points_base=None, refine_pool_size=10)
    assert diagnostics['passed_count'] == 15
    assert len(picked) == 10, "refine_pool_size가 상한 역할을 그대로 해야 한다"
    scores = [c['score'] for c in picked]
    assert scores == sorted(scores, reverse=True)


def test_hard_max_default_ignores_the_old_key():
    """예전 키가 남아 있는 설정 파일에서 30도가 그대로 상한이 되면 2.5차가 무효가 된다."""
    assert graspnet_baseline._hard_max_deg({'approach_angle_max_deg': 30.0}) == pytest.approx(45.0)
    assert graspnet_baseline._hard_max_deg({'approach_angle_hard_max_deg': 60.0}) == pytest.approx(60.0)


# --- 되잡기의 역할 경계 (2026-09-08, 2.6차에서 시험하고 되돌린 것) --------------
# 2.6차에서 "GraspNet의 가로 위치를 유지하고 깊이·폭만 보정"으로 바꿔 봤다가 실물에서
# 더 나빠져 되돌렸다(graspnet_baseline._refine_on_cloud docstring 참조). 그때 확인한
# 사실을 테스트로 못박아 둔다 — 같은 시도를 근거 없이 반복하지 않도록.

def _pack_cloud(half_x=95.0, half_y=46.0, thickness=15.0, n=20000, seed=3):
    """물티슈 팩 크기(190 x 92 x 15mm). 윗면 z=0, 중심 (0, 0)."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform([-half_x, -half_y], [half_x, half_y], size=(n, 2))
    z = rng.uniform(-thickness, 0.0, size=(n, 1))
    return np.hstack([xy, z])


def _candidate_frame(x, y, z=40.0, angle_deg=0.0):
    """(x, y, z)에서 내려오는 후보. 닫힘축은 base y(팩의 짧은 축)."""
    rad = np.radians(angle_deg)
    approach = np.array([np.sin(rad), 0.0, -np.cos(rad)])
    closing = np.array([0.0, 1.0, 0.0])
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = closing, np.cross(approach, closing), approach
    T[:3, 3] = [x, y, z]
    return T


def test_refine_recenters_on_the_measured_cloud():
    """**되잡기는 가로 위치를 클라우드 기준으로 다시 잡는다** — 이것이 의도된 동작이다.

    2026-09-08 실물에서 이 재배치를 끄자 파지점이 물체 중심에서 65~88mm 벗어나고
    개폭이 73~141mm로 튀어(4/10이 RG2 110mm 초과로 탈락) 파지가 통째로 나빠졌다.
    GraspNet이 잘린 클라우드를 받는 구성에서는 그 위치 예측을 믿을 수 없다.
    """
    points = _pack_cloud()
    finals = []
    for x in (-70.0, -35.0, 0.0, 35.0, 70.0):
        position, width, diag = graspnet_baseline._refine_on_cloud(
            _candidate_frame(x, 0.0), points, grasp_depth_mm=8.0)
        assert diag['status'] == graspnet_baseline.GEOMETRY_OK
        finals.append(round(float(position[0]), 1))
        assert width == pytest.approx(92.0, abs=6.0), "폭은 실측 클라우드에서 나와야 한다"
    assert max(finals) - min(finals) < 5.0, (
        f"가로 위치는 클라우드 기준으로 모인다(의도된 동작): {finals}")


def test_candidate_lateral_is_reported_as_a_diagnostic():
    """GraspNet 위치가 얼마나 벗어나 있었는지는 **진단으로 남긴다** — 버리는 기준이 아니다."""
    points = _pack_cloud()
    near = graspnet_baseline._refine_on_cloud(
        _candidate_frame(0.0, 0.0), points, grasp_depth_mm=8.0)[2]
    far = graspnet_baseline._refine_on_cloud(
        _candidate_frame(150.0, 0.0), points, grasp_depth_mm=8.0)[2]
    assert near['candidate_lateral_mm'] < 5.0
    assert far['candidate_lateral_mm'] > 100.0
    # 멀어도 버리지 않는다 — 되잡기가 구제하던 후보까지 죽으면 파지가 나빠진다.
    assert far['status'] == graspnet_baseline.GEOMETRY_OK


def test_midair_depth_offset_is_corrected():
    """예전 '허공 8~34mm' 문제는 접근축 보정으로 계속 막힌다."""
    points = _pack_cloud()
    position, _, diag = graspnet_baseline._refine_on_cloud(
        _candidate_frame(0.0, 0.0, z=34.0), points, grasp_depth_mm=8.0)
    assert diag['status'] == graspnet_baseline.GEOMETRY_OK
    assert position[2] < 0.0, "물체 표면 안쪽으로 내려와야 한다"
    assert diag['depth_shift_mm'] > 34.0


def test_window_is_not_widened_to_the_whole_cloud():
    """창에 재료가 없으면 **전체 클라우드로 풀지 않고** invalid로 둔다 (2.6차에서 남긴 것).

    전체로 풀면 물체 반대쪽 끝 점으로 폭을 재서 말도 안 되는 개폭이 나온다.
    """
    # 점 몇 개짜리 성긴 클라우드 — 창을 어떻게 잡아도 최소 점수를 못 채운다.
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                       [200.0, 200.0, -50.0]])
    _, width, diag = graspnet_baseline._refine_on_cloud(
        _candidate_frame(0.0, 0.0), points, grasp_depth_mm=8.0, min_grip_points=20)
    assert diag['status'] == graspnet_baseline.GEOMETRY_NO_MATERIAL
    assert width == 0.0


def test_refined_width_over_gripper_limit_is_rejected():
    """되잡기가 그리퍼 한계를 넘는 개폭으로 다시 재면 그 후보는 버린다.

    raw width(0.04m=40mm)는 추론 서버의 min/max_width_m 필터를 통과하는 값이지만,
    되잡기가 실측 클라우드(_pack_cloud, 개폭 ~92mm)로 다시 재면 max_opening_mm을
    넘을 수 있다 — raw만 걸러서는 못 막는 경우다(2026-09-09, 실물에서 121.2mm짜리가
    raw 필터를 통과한 채 control까지 새 나가 후보 자리만 낭비한 사고 재현).
    """
    points = _pack_cloud()
    T_graspnet_tcp = np.array([[0.0, 0.0, 1.0, 0.0],
                               [1.0, 0.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 1.0]])
    rotation = np.column_stack([(0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)])
    picked, diagnostics = graspnet_baseline._select_candidates(
        [_raw(rotation, [0.0, 0.0, 0.040])], np.eye(4), T_graspnet_tcp, 75.0,
        points_base=points, max_opening_mm=80.0)
    assert picked == []
    assert diagnostics['geometry_rejects'].get('width_invalid') == 1


def test_rejected_candidates_are_backfilled_from_the_next_ones():
    """기하 검사로 빠진 자리는 뒤 후보로 채운다 — Top-K를 둔 이유가 그것이다."""
    points = _pack_cloud()
    T_base_camera = np.eye(4)
    # GraspNet(X=접근) → TCP(Z=접근) 축 맞바꿈. 빼면 되잡기가 접근축을 수평으로 본다.
    T_graspnet_tcp = np.array([[0.0, 0.0, 1.0, 0.0],
                               [1.0, 0.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 1.0]])
    rotation = np.column_stack([(0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)])
    raws = [_raw(rotation, [0.0, 0.0, 0.040], score=0.9 - i * 0.01) for i in range(8)]
    picked, diagnostics = graspnet_baseline._select_candidates(
        raws, T_base_camera, T_graspnet_tcp, 75.0, points_base=points, refine_pool_size=5)
    assert len(picked) == 5
    assert diagnostics['refine_valid'] == 5
    assert diagnostics['examined_count'] >= 5


def test_locally_underestimated_width_is_still_rejected_by_full_extent():
    """손가락 창(국소)이 실제보다 좁게 잰 개폭도 **물체 전체 폭**으로 다시 걸린다.

    2026-09-11 실물(얇은 약상자) — depth가 성기거나 반사되는 얇은 물체는 손가락 창
    (그리퍼 중심선 근처, ±_PAD_HALF_MM 띠) 바로 그 자리의 가장자리 점이 비어 있을 수
    있다. 그러면 창 안 백분위수로 잰 개폭(chosen_width_mm)이 실제보다 좁게 나와
    그리퍼 한계 안으로 들어온 것처럼 보이지만, 물체는 실제로 그보다 넓다 — 실물에서
    손을 못 닫는다. 창을 벗어난 자리(먼 모서리)에는 점이 남아 있어 전체 클라우드
    기준 closing_extent_mm은 이 사실을 놓치지 않는다.
    """
    # 190 x 92mm 판. closing축(X) 중심선 근처(|y|<12, 손가락 창)에서만 x>40 쪽 점을
    # 지워 "창 바로 그 자리의 가장자리가 비어 있다"를 흉내낸다 — 창 밖(|y|>=12)의 먼
    # 모서리는 그대로 남아 있어 전체 폭은 190mm 그대로다.
    points = _pack_cloud()
    # 이 T_graspnet_tcp x rotation 조합에서 closing축은 global X(190mm), other축은
    # global Y(92mm)다 — test_all_geometry_rejected_falls_back_to_pca와 같은 매핑.
    strip = (np.abs(points[:, 1]) < 12.0) & (np.abs(points[:, 0]) > 40.0)
    points = points[~strip]

    T_graspnet_tcp = np.array([[0.0, 0.0, 1.0, 0.0],
                               [1.0, 0.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 1.0]])
    rotation = np.column_stack([(0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0)])
    picked, diagnostics = graspnet_baseline._select_candidates(
        [_raw(rotation, [0.0, 0.0, 0.040])], np.eye(4), T_graspnet_tcp, 75.0,
        points_base=points, max_opening_mm=110.0)

    assert picked == [], (
        "창 안 국소 측정만으로는 통과했겠지만, 물체 전체 폭(190mm)이 그리퍼 한계를 "
        "넘으므로 걸러야 한다")
    assert diagnostics['geometry_rejects'].get('width_invalid') == 1


def test_all_geometry_rejected_falls_back_to_pca(monkeypatch):
    """GraspNet 후보가 전부 기하 검사에서 걸리면(얇고 넓은 물체) heuristic_pca로 떨어진다.

    2026-09-11 실물(약상자) — 긴 축을 가로지르는 후보는 GEOMETRY_WIDTH_INVALID로,
    짧은 축/옆면 후보는 재료 부족(GEOMETRY_NO_MATERIAL)으로 둘 다 걸리면 GraspNet
    쪽에서는 후보가 하나도 안 남는다. 이 안전망이 없으면 no_feasible_grasp로 끝나
    로봇이 아예 움직이지 않는다. **후보가 있을 때는 여전히 쓰지 않는다**
    (test_missing_endpoint_is_not_replaced_by_pca와 대비).
    """
    points = _pack_cloud(half_x=95.0, half_y=46.0, thickness=15.0)
    # closing=+X(폭 190mm, 그리퍼 한계 110mm 초과) 하나만 GraspNet이 낸다 — 기울기는
    # 통과하되(접근 -Z) 되잡기의 GEOMETRY_WIDTH_INVALID로 반드시 떨어지는 자세다.
    rotation = np.column_stack([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)])
    raw = [{'score': 0.8, 'width_m': 0.04, 'depth_m': 0.0,
           'rotation_matrix': rotation.tolist(), 'translation_m': [0.0, 0.0, 0.04]}]

    monkeypatch.setattr(graspnet_baseline, '_infer_via_endpoint', lambda *a, **k: raw)
    candidates = graspnet_baseline.plan(
        points,
        {'endpoint': 'http://localhost:8200',
         'T_graspnet_tcp_mm': np.eye(4).tolist(),
         'max_opening_mm': 110.0, 'min_points': 30},
        context={'points_cam_mm': np.ones((80, 3)), 'T_base_camera_mm': np.eye(4)})

    assert candidates, "안전망이 없으면 no_feasible_grasp로 끝난다"
    assert all(c['strategy'] == 'heuristic_pca' for c in candidates)
    # PCA가 짧은 축(92mm)을 가로지르는 후보를 1순위로 냈어야 한다 — 그리퍼 한계 안이다.
    assert candidates[0]['width_mm'] < 110.0


def test_pca_fallback_never_used_when_graspnet_has_a_valid_candidate():
    """GraspNet 후보가 하나라도 살아 있으면 PCA로 떨어지지 않는다."""
    points = _pack_cloud()
    T_graspnet_tcp = np.array([[0.0, 0.0, 1.0, 0.0],
                               [1.0, 0.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 1.0]])
    rotation = np.column_stack([(0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)])
    picked, _ = graspnet_baseline._select_candidates(
        [_raw(rotation, [0.0, 0.0, 0.040])], np.eye(4), T_graspnet_tcp, 75.0,
        points_base=points, max_opening_mm=100.0)
    assert picked and picked[0]['strategy'] == 'graspnet_baseline'

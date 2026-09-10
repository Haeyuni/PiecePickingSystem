"""Final-pose Diverse-TopK — refined valid pool에서 실행 후보를 고른다 (STEP 1, 2026-09-10).

**왜 필요한가.** GraspNet이 후보를 여러 개 내도 점수 상위끼리는 위치·자세가 몰릴 수 있다.
그대로 점수순 Top-10을 보내면 control이 개폭·IK·관절로 앞 후보를 떨궜을 때 남는 것이
"방금 떨어진 것과 거의 같은 후보"뿐이라, 후보를 여러 개 보내는 의미가 사라진다.

**언제 고르는가가 중요하다.** 반드시 `_fit_grasp_depth`까지 끝난 **최종 pose**로 계산한다
(node._candidates_for). 깊이 맞춤은 접근축을 따라 움직이므로 기울어진 파지에서는 XY와 Z가
같이 바뀐다 — 그 전에 다양성을 재면 실제로 실행될 자세가 아닌 좌표로 고르는 셈이 된다.

**이 모듈은 ROS2도 전략도 import하지 않는다.** 순수 numpy라 로봇·GPU 없이 단위 테스트가
그대로 돈다(grasp/node.py는 rclpy를 끌어온다).
"""
import numpy as np

# 위치(mm)와 각도(도)를 하나의 무차원 거리로 합치기 위한 환산 기준.
# **실물 근거가 아직 없는 초기값이다** — "30mm 떨어진 것"과 "접근축이 30도 다른 것"을
# 같은 크기의 차이로 본다는 뜻이고, 실물에서 후보가 여전히 몰리면 줄이고 물체 범위를
# 벗어나 흩어지면 늘린다. threshold가 아니라 스케일이라 후보를 버리는 데는 쓰이지 않는다.
DEFAULT_POSITION_SCALE_MM = 30.0
DEFAULT_ORIENTATION_SCALE_DEG = 30.0


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)


def _pairwise_angle_deg(axes: np.ndarray) -> np.ndarray:
    """(N,3) 방향벡터들 → (N,N) 각 쌍의 사이각(도)."""
    normed = _unit_rows(np.asarray(axes, dtype=float))
    return np.degrees(np.arccos(np.clip(normed @ normed.T, -1.0, 1.0)))


def distance_matrix(positions_mm, approach_axes, closing_axes,
                    position_scale_mm: float = DEFAULT_POSITION_SCALE_MM,
                    orientation_scale_deg: float = DEFAULT_ORIENTATION_SCALE_DEG
                    ) -> np.ndarray:
    """후보 쌍마다 "파지가 서로 얼마나 다른가"를 하나의 무차원 거리로 합친다.

    보는 것은 셋뿐이다 — 위치, 접근축(어느 방향에서 들어가나), 닫힘축(RG2가 어느 방향으로
    무나). mm와 도는 단위가 달라 그냥 더하면 mm 쪽 숫자가 항상 이기므로 각자 스케일로
    나눈 뒤 유클리드로 합친다.
    """
    positions = np.asarray(positions_mm, dtype=float)
    diff = positions[:, None, :] - positions[None, :, :]
    pos = np.linalg.norm(diff, axis=2) / max(float(position_scale_mm), 1e-6)
    approach = _pairwise_angle_deg(approach_axes) / max(float(orientation_scale_deg), 1e-6)
    closing = _pairwise_angle_deg(closing_axes) / max(float(orientation_scale_deg), 1e-6)
    return np.sqrt(pos ** 2 + approach ** 2 + closing ** 2)


# "사실상 같은 후보"로 보는 거리(무차원, distance_matrix 기준). 0.05는 위치 스케일 30mm
# 기준 1.5mm / 각도 스케일 30도 기준 1.5도에 해당한다.
#
# **이 값으로 후보를 버리지는 않는다(2026-09-10 사용자 요청).** 되잡기가 회전이 같은 후보를
# 같은 최종 pose로 만들기 때문에 pool에 중복이 생기는 것은 사실이지만, 그걸 제거하면
# execution pool이 예고 없이 줄어든다 — 지금은 그 사실을 **진단(distinct_poses)으로만**
# 남기고 pool 크기는 execution_pool_size 그대로 채운다.
DUPLICATE_DISTANCE = 0.05


def select_diverse(distances: np.ndarray, pool_size: int) -> list[int]:
    """deterministic greedy max-min. 입력은 **점수 내림차순**인 후보들의 거리행렬이다.

    첫 seed는 0번 = GraspNet score 최고 후보다. 그 후보는 execution pool에 **보존**된다 —
    다만 이것이 "최고점을 실행하라"는 뜻은 아니다. 최종 선택은 control이 IK·개폭·랭킹으로
    한다(grasp_selection). 여기서는 비교 가능한 기준 후보 하나를 반드시 남기는 것뿐이다.

    그다음부터는 이미 뽑힌 것들과의 **최소거리가 가장 큰** 후보를 차례로 더한다. 동률이면
    `np.argmax`가 먼저 나오는 쪽(=점수가 더 높은 쪽, 입력이 점수순이므로)을 골라 품질이
    자연스러운 tie-break가 된다. 무작위성이 없어 같은 장면이면 같은 결과가 나온다.
    """
    total = int(np.asarray(distances).shape[0])
    k = min(max(1, int(pool_size)), total)
    chosen = [0]
    while len(chosen) < k:
        remaining = [i for i in range(total) if i not in chosen]
        nearest = distances[np.ix_(remaining, chosen)].min(axis=1)
        chosen.append(remaining[int(np.argmax(nearest))])
    return chosen


def spread(positions_mm, approach_axes, closing_axes, indices) -> dict:
    """고른 후보들이 실제로 얼마나 흩어져 있는지 — [POOL_COMPARE] 로그용 진단값.

    거리행렬과 달리 사람이 읽는 단위(mm, 도)를 그대로 쓴다. 최대 쌍거리 = "가장 다른 두
    후보가 얼마나 다른가"라, score-only Top-10과 나란히 찍으면 다양성이 실제로 늘었는지가
    한 줄로 보인다.
    """
    index = list(indices)
    if len(index) < 2:
        return {"position_mm": 0.0, "approach_deg": 0.0, "closing_deg": 0.0}
    positions = np.asarray(positions_mm, dtype=float)[index]
    pos = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    approach = _pairwise_angle_deg(np.asarray(approach_axes, dtype=float)[index])
    closing = _pairwise_angle_deg(np.asarray(closing_axes, dtype=float)[index])
    return {"position_mm": round(float(pos.max()), 1),
            "approach_deg": round(float(approach.max()), 1),
            "closing_deg": round(float(closing.max()), 1)}


def axes_from_quaternion(orientation: dict) -> tuple[np.ndarray, np.ndarray]:
    """후보 pose의 쿼터니언 → (접근축, 닫힘축) base 단위벡터.

    이 프로젝트의 그리퍼 좌표 규약: **X = 닫힘축, Z = 접근축**
    (heuristic_pca._pose_from_closing_axis, graspnet_baseline의 T_graspnet_tcp_mm 축 맞바꿈).
    회전행렬의 0열/2열이 그대로 그 두 축이다.
    """
    x = float(orientation["x"])
    y = float(orientation["y"])
    z = float(orientation["z"])
    w = float(orientation["w"])
    closing = np.array([1.0 - 2.0 * (y * y + z * z),
                        2.0 * (x * y + z * w),
                        2.0 * (x * z - y * w)])
    approach = np.array([2.0 * (x * z + y * w),
                         2.0 * (y * z - x * w),
                         1.0 - 2.0 * (x * x + y * y)])
    return approach, closing


def pool_from_candidates(candidates: list, pool_size: int,
                         position_scale_mm: float = DEFAULT_POSITION_SCALE_MM,
                         orientation_scale_deg: float = DEFAULT_ORIENTATION_SCALE_DEG
                         ) -> tuple[list[int], dict]:
    """전략이 낸 후보 dict 목록(점수 내림차순, **최종 pose**) → (고른 인덱스, 진단).

    진단에는 같은 pool을 score-only Top-K로 잘랐을 때와의 비교가 들어간다 — 추론이나
    되잡기를 다시 돌리지 않고, 이미 계산된 것만 두 가지로 읽는다([POOL_COMPARE]).
    """
    if not candidates:
        return [], {}
    positions = np.array([[c["pose"]["position"]["x"], c["pose"]["position"]["y"],
                           c["pose"]["position"]["z"]] for c in candidates], dtype=float)
    axes = [axes_from_quaternion(c["pose"]["orientation"]) for c in candidates]
    approach_axes = np.array([a for a, _ in axes], dtype=float)
    closing_axes = np.array([c for _, c in axes], dtype=float)

    distances = distance_matrix(positions, approach_axes, closing_axes,
                                position_scale_mm, orientation_scale_deg)
    diverse = select_diverse(distances, pool_size)
    score_only = list(range(min(max(1, int(pool_size)), len(candidates))))
    # 되잡기 뒤 서로 구별되는 자세가 애초에 몇 개인지 — pool 크기를 늘려도 이 수가
    # 천장이다(같은 회전은 같은 최종 pose가 된다). 실물에서 이 값이 작게 나오면
    # refine_pool_size를 키울 게 아니라 GraspNet 입력 쪽을 봐야 한다는 신호다.
    distinct = 0
    seen: list[int] = []
    for i in range(len(candidates)):
        if not seen or distances[np.ix_([i], seen)].min() > DUPLICATE_DISTANCE:
            seen.append(i)
            distinct += 1
    diagnostics = {
        "distinct_poses": distinct,
        "score_top_ranks": score_only,
        "diverse_ranks": diverse,
        "common": sorted(set(score_only) & set(diverse)),
        "replaced": sorted(set(score_only) - set(diverse)),
        "spread_before": spread(positions, approach_axes, closing_axes, score_only),
        "spread_after": spread(positions, approach_axes, closing_axes, diverse),
    }
    return diverse, diagnostics

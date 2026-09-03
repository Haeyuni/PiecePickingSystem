"""파지 전략 레지스트리 (FR-07: 최소 두 가지 전략을 지원해야 한다).

전략은 전부 같은 시그니처를 갖는다:

    plan(points_base: np.ndarray, params: dict) -> list[dict]

`node.py`는 어떤 전략이 도는지 모른 채 이름으로만 고른다. 3단계 비교 실험(BR 4.1절)에서
전략을 바꾸는 것이 노드 수정이 아니라 파라미터 변경이어야 하기 때문이다.
"""
from . import heuristic_pca

# 학습 기반 전략(contact_graspnet, graspnet_baseline)은 가중치와 별도 CUDA 환경이 필요해
# 여기에 아직 등록하지 않는다. 등록되지 않은 이름을 고르면 아래에서 분명한 오류가 난다 —
# 조용히 heuristic으로 대체하면 비교 실험 결과가 통째로 잘못 읽힌다.
_STRATEGIES = {
    heuristic_pca.STRATEGY: heuristic_pca.plan,
}


def available() -> list[str]:
    return sorted(_STRATEGIES)


def get(name: str):
    try:
        return _STRATEGIES[name]
    except KeyError:
        raise ValueError(
            f"등록되지 않은 파지 전략 '{name}'. 사용 가능: {', '.join(available())}"
        ) from None

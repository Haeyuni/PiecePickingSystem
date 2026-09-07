"""파지 전략 레지스트리 (FR-07: 최소 두 가지 전략을 지원해야 한다).

전략은 전부 같은 시그니처를 갖는다:

    plan(points_base: np.ndarray, params: dict, context: dict | None = None) -> list[dict]

`node.py`는 어떤 전략이 도는지 모른 채 이름으로만 고른다. 3단계 비교 실험(BR 4.1절)에서
전략을 바꾸는 것이 노드 수정이 아니라 파라미터 변경이어야 하기 때문이다.
"""
from . import graspnet_baseline, heuristic_pca

# Contact-GraspNet은 가중치·별도 CUDA 환경이 준비되기 전까지 등록하지 않는다. GraspNet은
# Docker adapter가 그 환경을 분리하므로 등록한다. 설정 오류를 PCA로 대체하지 않는다.
_STRATEGIES = {
    heuristic_pca.STRATEGY: heuristic_pca.plan,
    graspnet_baseline.STRATEGY: graspnet_baseline.plan,
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

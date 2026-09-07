"""전략이 오류가 아닌 미완료 상태를 node에 알리는 예외."""


class InferenceBusy(RuntimeError):
    """같은 전략의 이전 추론이 아직 진행 중이다."""

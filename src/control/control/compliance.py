"""토크 기반 접촉감지·파지/배치 성공 판정 (순응제어).

**지금 여기 있는 건 진단용 요약뿐이다 — 성공/실패 판정에는 안 쓴다.** pick의 실제 파지
판정은 이미 RG2 컨트롤러의 'Grip detected' 비트가 맡고 있다(pick_server._grip_detected).
개폭/토크만으로 판정하면 변형체(치약처럼 눌리는 물체)에서 정반대로 틀린다는 게 실물로
확인돼 있어(2026-09-07, pick_server.py 파지 판정 주석 참조), 여기 값으로 그 판정을
대체하지 않는다. 지금 하는 일은 딱 하나 — Result.torque_trace_summary(인터페이스_정의서.md
3.6/4.1절, "접촉~파지 구간 토크 변화 요약값")를 실제 값으로 채우는 것뿐이다.

**검증 범위**: `GetExternalTorque` 서비스의 필드명은 doosan-robot2 GitHub 소스
(dsr_msgs2/srv/GetExternalTorque.srv, jazzy 브랜치, 2026-09-08 확인)로만 확인했다.
이 저장소엔 dsr_msgs2가 마운트로만 존재해(src/control/docker-entrypoint.sh 참조)
실물로는 검증하지 못했다. 서비스가 무응답이면 조용히 건너뛴다 — pick 자체를 막지 않는다.

**하강 중 실제로 멈추는 쪽(place_into 하강)**은 `contact_exceeded`가 맡는다. 이건
`GetExternalTorque`가 아니라 `GetToolForce`를 쓴다 — skill_params.yaml의
`compliance.contact_threshold_n`이 이미 N(뉴턴) 단위로 정의돼 있어(관절
토크 Nm이 아니라 툴에 걸리는 힘) 단위를 맞춰야 하기 때문이다. **기본은 꺼져 있다**
(`compliance.place_descent_enabled: false`, skill_params.yaml) — 임계값도 이 서비스도
실물로 검증하지 못했으니, 켜기 전에 실물에서 낮은 속도로 먼저 확인할 것.
"""
from . import dsr_motion


def torque_norm(external_torque: list[float]) -> float:
    """6축 외부토크 벡터의 크기(L2 노름, Nm). 부호 있는 축별 값보다 "얼마나 걸렸는지"
    하나의 스칼라가 로그·요약에 더 쓸모 있다."""
    return sum(v * v for v in external_torque) ** 0.5


class TorqueTrace:
    """pick 한 사이클 동안 체크포인트별 외부토크 크기를 모은다.

    성공/실패 판정에는 관여하지 않는다(모듈 docstring 참조) — 값을 모아서
    Result.torque_trace_summary로 그대로 내보내는 것이 유일한 역할이다.
    """

    def __init__(self):
        self._checkpoints: list[tuple[str, float]] = []

    def sample(self, client, label: str, timeout_s: float = 0.5) -> float | None:
        """`label` 시점의 외부토크 크기를 읽어 기록한다. 서비스가 무응답이면 None을
        돌려주고 기록하지 않는다 — 호출부는 실패를 무시해도 된다(진단용이라 필수가 아님)."""
        torque = dsr_motion.get_external_torque(client, timeout_s)
        if torque is None:
            return None
        norm = torque_norm(torque)
        self._checkpoints.append((label, norm))
        return norm

    def summary(self) -> list[float]:
        """체크포인트 순서대로 크기값만 뽑는다 — Result.torque_trace_summary에 그대로 싣는다."""
        return [value for _label, value in self._checkpoints]

    def log_line(self) -> str:
        return " ".join(f"{label}={value:.2f}" for label, value in self._checkpoints)


def tool_force_norm(client, ref: int = 0, timeout_s: float = 0.5) -> float | None:
    """`ref` 기준 tool force의 힘 성분(Fx,Fy,Fz) 크기(N). 모멘트(Mx,My,Mz)는 뺀다 —
    place 하강 접촉 판정은 "밀어붙이는 힘"을 보는 것이지 비틀림이 아니다. 무응답이면 None."""
    force = dsr_motion.get_tool_force(client, ref=ref, timeout_s=timeout_s)
    if force is None:
        return None
    return sum(v * v for v in force[:3]) ** 0.5


def contact_exceeded(client, threshold_n: float, ref: int = 0,
                     timeout_s: float = 0.5) -> bool | None:
    """tool force 크기가 `threshold_n`(N)을 넘었는지. 서비스가 무응답이면 None —
    "안 닿았다"가 아니라 "모른다"이므로, 호출부는 None을 접촉 없음으로 취급하면 안 된다
    (내려가는 동안 판단 근거가 없으면 그 스텝은 그냥 계속 내려가는 쪽이 맞다 — 위치제어
    경로가 이미 하던 것과 같아서, 무응답이 새로운 위험을 만들지 않는다)."""
    norm = tool_force_norm(client, ref, timeout_s)
    if norm is None:
        return None
    return norm >= threshold_n

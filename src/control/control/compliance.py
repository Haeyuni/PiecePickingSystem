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

**하강 중 실제로 멈추는 쪽(place_into 하강)**은 `ContactWatch`(그리고 그것이 쓰는
`tool_force_norm`)가 맡는다. 이건 `GetExternalTorque`가 아니라 `GetToolForce`를 쓴다 —
skill_params.yaml의 `compliance.contact_threshold_n`이 이미 N(뉴턴) 단위로 정의돼 있어
(관절 토크 Nm이 아니라 툴에 걸리는 힘) 단위를 맞춰야 하기 때문이다.

**하강은 한 번의 movel로 내려가면서 감시한다**(2026-09-09 구조 변경, 사용자 요청).
예전에는 `place_descent_step_mm`(5mm)으로 쪼개 스텝마다 movel을 새로 보내고 그 사이에만
힘을 봤다 — 멈출 수 있는 지점이 스텝 경계뿐이라 사실상 5mm 단위 위치제어였고, 80mm
하강에 movel이 16번 나갔다. 지금은 목표까지 한 번에 내려가면서 `ContactWatch`가
주기적으로 힘을 보고, 임계값을 넘으면 그 자리에서 movel goal을 취소해 로봇을 세운다
(`dsr_motion.call_action_blocking`의 `stop_when`).

**임계값도 이 서비스도 실물로 충분히 검증되지 않았다** — 임계값이 낮으면 정상 마찰에도
일찍 멈춰 물체가 뜬 채로 놓이고, 높으면 이 기능이 없는 것과 같아진다. 감시가 아예 안
도는 경우(GetToolForce 무응답)와도 구분이 필요해서 `ContactWatch.summary()`를 매 하강마다
로그로 남긴다 — 켜고 쓸 때는 그 줄을 먼저 볼 것.
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


class ContactWatch:
    """하강하는 **동안** 접촉을 감시하는 콜러블. `dsr_motion.move_linear(stop_when=...)`에
    그대로 넘긴다 — movel이 도는 중 주기적으로 불려서 "지금 멈춰야 하나"에 답한다.

    **왜 콜러블인가.** 2026-09-09 이전에는 하강을 `place_descent_step_mm`(5mm)으로 쪼개
    스텝마다 movel을 새로 보내고 그 사이에 힘을 확인했다. 그 구조는 (1) 80mm 하강에
    movel이 16번 나가 사이클이 길고, (2) 스텝 경계에서만 멈출 수 있어 실제로는
    "5mm 단위 위치제어"에 가까웠다. 지금은 목표까지 **한 번의 movel로 내려가면서** 이
    콜러블이 접촉을 보고, 넘으면 그 자리에서 goal을 취소해 로봇을 세운다.

    한 번이라도 임계값을 넘으면 `triggered`가 True로 남는다 — 호출부는 이동이 끝난 뒤
    "접촉으로 멈춘 것"과 "목표까지 다 내려간 것"을 이 값으로 가른다. `call_action_blocking`은
    둘 다 성공으로 돌려주므로(멈춘 것도 의도된 정상 종료다) 그 구분이 여기 말고는 없다.

    **무응답(None)은 '접촉 없음'으로 다룬다** — `contact_exceeded` 주석과 같은 이유로,
    판단 근거가 없을 때 계속 내려가는 것은 순응 하강을 켜기 전(순수 위치제어)과 같은
    동작이라 새로운 위험을 만들지 않는다. 대신 무응답 횟수를 세어 둔다: 서비스가 죽어 있어
    **감시가 사실상 꺼져 있었는데도 조용히 성공으로 끝나는 것**이 이 기능의 가장 나쁜
    실패 모드라, 끝나고 `summary()`를 로그로 남겨 사람이 구분할 수 있게 한다.
    """

    def __init__(self, client, threshold_n: float, ref: int = 0, timeout_s: float = 0.5):
        self._client = client
        self._threshold_n = float(threshold_n)
        self._ref = ref
        self._timeout_s = timeout_s
        self.triggered = False
        self.polls = 0
        self.unknown = 0
        self.peak_n: float | None = None
        self.stop_force_n: float | None = None

    def __call__(self) -> bool:
        self.polls += 1
        norm = tool_force_norm(self._client, self._ref, self._timeout_s)
        if norm is None:
            self.unknown += 1
            return False
        if self.peak_n is None or norm > self.peak_n:
            self.peak_n = norm
        if norm < self._threshold_n:
            return False
        self.triggered = True
        self.stop_force_n = norm
        return True

    def summary(self) -> str:
        """이동이 끝난 뒤 로그에 남길 한 줄. 감시가 실제로 돌았는지가 여기서 드러난다."""
        peak = f"{self.peak_n:.1f}N" if self.peak_n is not None else "미상"
        state = (f"접촉({self.stop_force_n:.1f}N ≥ {self._threshold_n:.1f}N)"
                 if self.triggered else f"접촉 없음(임계값 {self._threshold_n:.1f}N)")
        return (f"{state} | 확인 {self.polls}회 중 무응답 {self.unknown}회 | 최대 {peak}")

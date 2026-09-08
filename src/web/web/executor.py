"""실행 어댑터 경계 (개발계획.md 4.2절 B2).

`web`이 로봇에 도달하는 **유일한 통로**다. mock 구현(`executor_mock.py`)과 ROS2 구현
(`ros_bridge.py`)이 같은 시그니처를 갖고, `MOCK_MODE`로 교체된다. perception/grasp/control이
실물로 붙는 시점에 바꾸는 것은 이 파일이 가리키는 구현체 하나뿐이며, `main.py`·라우터·
프론트엔드는 손대지 않는다.

**`main.py`와 라우터는 rclpy를 import하지 않는다** — ROS2를 아는 코드는 `ros_bridge.py`
하나로 격리한다(시스템명세서 3.1절).
"""
import os
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Protocol

# 실행 단계 콜백. (request_id, phase) — Pick/PlaceInto.action의 Feedback을 그대로 중계한다.
FeedbackCallback = Callable[[str, str], Awaitable[None]]


@dataclass
class SkillGoal:
    """Pick/PlaceInto goal의 공통 형태 (인터페이스_정의서.md 4절)."""

    trace_id: str
    request_id: str
    object_id: str
    profile: str
    source_observation_id: str = ""
    source_observation_stamp: dict | None = None
    class_name: str = ""
    grasp_pose: dict | None = None   # pick — 1순위 후보. 후보 목록이 비었을 때의 대체값이기도 하다
    gripper_width_mm: float | None = None  # pick — 1순위 후보의 GraspCandidate.gripper_width_mm
    # pick — planner가 작업반경으로 거른 후보 전체(점수 내림차순). **실행할 하나는
    # control이 고른다**(개폭·IK·관절·안전 검사가 로봇 쪽에서만 가능하다).
    grasp_candidates: list = field(default_factory=list)
    # pick — control의 후보 랭킹이 쓰는 물체 정보. control은 /world_state를 안 보므로
    # 여기 실어 보내지 않으면 물체 중심·높이·depth 신뢰도를 알 방법이 없다.
    object_center_mm: dict | None = None
    object_height_mm: float | None = None
    depth_valid_ratio: float | None = None
    object_footprint_base_mm: list = field(default_factory=list)
    bin_id: str | None = None        # place_into
    # place_into — 든 물체가 TCP보다 얼마나 아래로 내려와 있는지(mm).
    # orchestrator가 물체 높이와 파지 z로 계산한다. None/0이면 control이 고정 여유만 쓴다.
    object_bottom_offset_mm: float | None = None
    pickup_tcp_posx: list[float] | None = None
    tcp_to_object_bottom_mm: float | None = None
    max_retries: int = 1


@dataclass
class SkillResult:
    success: bool
    failure_reason: str = "none"
    retries_used: int = 0
    cycle_time_ms: float = 0.0
    visual_verification_passed: bool | None = None
    torque_trace: list = field(default_factory=list)
    # 실제로 실행한 파지 후보의 candidate_id (pick만). control이 고른 것 —
    # 후보를 하나도 통과시키지 못했거나 grasp_pose 하나만 보낸 경우는 빈 문자열이다.
    selected_candidate_id: str = ""
    source_observation_id: str = ""
    executed_tcp_posx: list[float] | None = None
    # 사용자가 Stop을 눌러 취소된 결과인지 — orchestrator가 이걸로 "재계획할 실패"와
    # "그만둬야 할 취소"를 구분한다. success=False만 보고는 구분이 안 된다(2026-09-05,
    # Stop 이후 재실행 사고 조사에서 확인 — 취소도 실패와 같은 failure_reason으로
    # 나가서 재계획 로직이 취소를 그냥 실패로 오인해 다시 시도했다).
    cancelled: bool = False


class Executor(Protocol):
    """web ↔ 로봇 경계. 구현체는 executor_mock.MockExecutor / ros_bridge.RosExecutor."""

    async def start(self) -> None: ...
    async def close(self) -> None: ...

    def get_latest_world_state(self) -> dict | None:
        """가장 최근 /world_state. 아직 받은 것이 없으면 None."""

    def robot_state(self) -> dict:
        """최신 RobotState. 최소한 mode를 포함한다 (2.1절 busy 차단 판단에 쓰인다)."""

    def latest_raw_color_jpeg(self) -> bytes | None:
        """리얼센스가 내는 원본 컬러 프레임(JPEG), 오버레이 없음. 기본 "카메라 뷰"가 이걸
        보여준다(2026-09-08) — 명령 사이에도 항상 실시간이라 카메라가 살아있는지 바로
        보인다. 아직 못 받았거나(mock/카메라 미연결) 스트림 자체가 없으면 None."""

    def latest_color_jpeg(self) -> bytes | None:
        """"관측 결과" 뷰의 최근 프레임(JPEG). **원본 RGB가 아니라 grasp의 관측 오버레이다**
        (마스크 윤곽 + 파지 후보 그립 모양) — 화면정의서 2.2.4절 원안(원본만, 오버레이
        없음)은 이후 사용자 요청으로 뒤집혔다가(ros_bridge.py의 `_BridgeNode` 주석 참조,
        D-5), 2026-09-08에 원본 뷰가 기본으로 돌아오면서 이 오버레이는 별도 스트림으로
        분리됐다. 온디맨드 관측이라 명령이 들어와야만 갱신된다(그 전엔 마지막 프레임이
        그대로). 아직 못 받았거나(mock/카메라 미연결) 스트림 자체가 없으면 None."""

    def latest_depth_jpeg(self) -> bytes | None:
        """뎁스 맵을 컬러맵으로 시각화한 최근 프레임(JPEG). 없으면 None."""

    def subscribe_state(self, on_event: Callable[[dict], Awaitable[None]]) -> None:
        """robot_state / safety_event 변화를 web으로 밀어 올리는 콜백을 등록한다."""

    async def observe(self, trace_id: str, mode: str = "full") -> dict | None:
        """온디맨드 관측을 트리거한다 (docs/on-demand-perception.md). perception은 더
        이상 주기 발행하지 않으므로, `/world_state`를 새로 채우려면 이 호출이 있어야 한다.

        mode: "full"(장면 전체를 SAM everything+VLM으로 다시 본다. 명령당 이걸 여러 번
        부르면 API 비용과 지연이 그만큼 쌓인다 — 명령당 1회가 원칙이다) |
        "reprompt"(직전 관측이 남긴 물체들을 재투영 박스로 SAM만 1패스 돌린다. VLM은
        안 부른다. pick/place_into 스텝 사이처럼 자주 불러도 되는 쪽). ROS의
        `Observe.Goal.MODE_*`와 뜻이 같다 — web이 그 enum을 몰라도 되게 문자열로 받는다.

        반환은 `{"success", "failure_reason", "object_count", "cycle_time_ms",
        "cancelled"}` 또는 액션 자체에 접수되지 못했으면 None. **성공해도 `/world_state`가
        그 자리에서 갱신되는 것은 아니다** — grasp가 뒤이어 처리하는 시간이 있다. 호출자는
        이어서 `get_latest_world_state()`의 stamp 변화를 기다려야 한다
        (`orchestrator._wait_for_fresh_observation`이 그 역할이다).
        """

    async def call_pick(self, goal: SkillGoal, on_feedback: FeedbackCallback) -> SkillResult: ...

    async def call_place_into(self, goal: SkillGoal, on_feedback: FeedbackCallback) -> SkillResult: ...

    async def stop(self) -> str | None:
        """진행 중인 goal을 취소한다. 취소된 request_id, 없으면 None."""

    async def home(self) -> None: ...


def is_mock() -> bool:
    return os.environ.get("MOCK_MODE", "1") == "1"


def create_executor() -> Executor:
    """MOCK_MODE에 따라 구현체를 고른다 — 여기가 실물 결합 시 유일한 분기점이다."""
    if is_mock():
        from .executor_mock import MockExecutor

        return MockExecutor()

    # rclpy는 이 시점에만 import된다. mock 개발 중에는 ROS2가 설치되어 있지 않아도 된다.
    from .ros_bridge import RosExecutor

    return RosExecutor()

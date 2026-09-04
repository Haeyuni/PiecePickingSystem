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
    grasp_pose: dict | None = None   # pick
    bin_id: str | None = None        # place_into
    max_retries: int = 1


@dataclass
class SkillResult:
    success: bool
    failure_reason: str = "none"
    retries_used: int = 0
    cycle_time_ms: float = 0.0
    visual_verification_passed: bool | None = None
    torque_trace: list = field(default_factory=list)


class Executor(Protocol):
    """web ↔ 로봇 경계. 구현체는 executor_mock.MockExecutor / ros_bridge.RosExecutor."""

    async def start(self) -> None: ...
    async def close(self) -> None: ...

    def get_latest_world_state(self) -> dict | None:
        """가장 최근 /world_state. 아직 받은 것이 없으면 None."""

    def robot_state(self) -> dict:
        """최신 RobotState. 최소한 mode를 포함한다 (2.1절 busy 차단 판단에 쓰인다)."""

    def latest_color_jpeg(self) -> bytes | None:
        """카메라 원본 스트림의 최근 프레임(JPEG, 화면정의서 2.2.4절 — 마스크 오버레이 없음).
        아직 못 받았거나(mock/카메라 미연결) 스트림 자체가 없으면 None."""

    def latest_depth_jpeg(self) -> bytes | None:
        """뎁스 맵을 컬러맵으로 시각화한 최근 프레임(JPEG). 없으면 None."""

    def subscribe_state(self, on_event: Callable[[dict], Awaitable[None]]) -> None:
        """robot_state / safety_event 변화를 web으로 밀어 올리는 콜백을 등록한다."""

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

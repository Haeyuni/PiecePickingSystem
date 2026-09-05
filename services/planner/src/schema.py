"""planner가 주고받는 데이터 구조 (웹_인터페이스_정의서.md 3절, 인터페이스_정의서.md 3~4절).

**LLM 출력 구조는 여기서 정의한 것만 허용한다.** 자유 텍스트를 파싱하지 않고 JSON Schema로
받는 것이 NFR-03(코드 생성 금지, 사전 정의 스킬 시퀀스로 제한)의 구현이다.

LLM이 정하는 것과 코드가 정하는 것을 나눈다:
- LLM: 어떤 물체를(`object_id`) 어디로(`bin_id`) 어떤 순서로
- 코드: 파지 자세(`grasp_pose`, 후보 중 최고점), 제어 프로파일(`profile`, 속성 DB 기준)

파지 자세와 프로파일은 물리적 안전에 직결되므로 LLM 출력에 맡기지 않는다(NFR-03a).
"""
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0.0"

SkillName = Literal["pick", "place_into"]
Profile = Literal["normal", "fragile", "deformable"]
ValidationStatus = Literal["approved", "rejected"]


# --- LLM 출력 (구조화 출력 스키마) -------------------------------------------

class LlmStep(BaseModel):
    """LLM이 생성하는 스텝 하나. 좌표·힘 같은 물리량은 포함하지 않는다."""

    skill: SkillName
    object_id: str = Field(description="world_state에 실제로 존재하는 object_id만 사용")
    bin_id: str = Field(
        default="",
        description="place_into일 때만 채운다. bins.yaml에 등록된 목적지 중 하나. pick이면 빈 문자열",
    )


class LlmPlan(BaseModel):
    """LLM 응답 전체. 거부도 '자유 텍스트'가 아니라 이 구조 안에서 표현한다."""

    steps: list[LlmStep] = Field(
        description="실행할 스킬 시퀀스. 수행할 수 없는 지시면 빈 배열",
    )
    refusal_reason: str = Field(
        default="",
        description="steps가 비었을 때 그 이유. 수행 가능하면 빈 문자열",
    )


# --- planner 응답 (web 대면) --------------------------------------------------

class Pose(BaseModel):
    """geometry_msgs/Pose와 동일 구조. 위치 단위는 mm (인터페이스_정의서 1.1절)."""

    position: dict[str, float]
    orientation: dict[str, float]


class PlanStep(BaseModel):
    skill: SkillName
    object_id: str
    profile: Profile
    grasp_pose: Pose | None = None   # pick일 때만
    grasp_width_mm: float | None = None  # pick일 때만. 고른 후보의 물체 폭(mm) — 없으면 미상
    bin_id: str | None = None        # place_into일 때만


class PreviousFailure(BaseModel):
    sequence_id: str
    request_id: str
    object_id: str
    failure_reason: str


class PlanRequest(BaseModel):
    schema_version: str = SCHEMA_VERSION
    trace_id: str
    command_text: str
    world_state: dict
    previous_failure: PreviousFailure | None = None
    # 활성 안전 이벤트. FR-12의 게이트 조건이지만 planner는 ROS2를 모르므로 web이 실어 보낸다.
    # 웹_인터페이스_정의서 3절에는 아직 없는 필드 — 8절 미확정 항목으로 등록되어 있다.
    active_safety_events: list[dict] = Field(default_factory=list)


class PlanResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    trace_id: str
    sequence_id: str
    validation_status: ValidationStatus
    validation_reason: str | None = None
    steps: list[PlanStep] = Field(default_factory=list)

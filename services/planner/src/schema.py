"""planner가 주고받는 데이터 구조 (웹_인터페이스_정의서.md 3절, 인터페이스_정의서.md 3~4절).

**LLM 출력 구조는 여기서 정의한 것만 허용한다.** 자유 텍스트를 파싱하지 않고 JSON Schema로
받는 것이 NFR-03(코드 생성 금지, 사전 정의 스킬 시퀀스로 제한)의 구현이다.

LLM이 정하는 것과 코드가 정하는 것을 나눈다:
- LLM: 어떤 물체를(`object_id`) 어디로(`bin_id`) 어떤 순서로
- 코드: 파지 후보 목록(`grasp_candidates`)과 파지력 단계(`grip_level`, 속성 DB 기준).
  **실행할 후보 하나를 최종 선택하는 것은 control이다** — 개폭 유효성·IK·관절 한계는
  로봇에 붙어 있어야 답할 수 있고(ikin 서비스), planner는 ROS2를 모르는 별도 서비스다.
  여기서는 작업반경 안에 있는 후보만 점수 순으로 추려서 넘긴다.

파지 자세와 파지력 단계는 물리적 안전에 직결되므로 LLM 출력에 맡기지 않는다(NFR-03a).
"""
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0.0"

SkillName = Literal["pick", "place_into"]
# 파지력 단계(DetectedObject.grip_level): 1=가장 강하게(40N) ~ 5=가장 약하게(20N).
# LLM은 단계만 고르고 단계→힘 매핑은 control/config/skill_params.yaml이 소유한다.
GripLevel = Literal[1, 2, 3, 4, 5]
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


class GraspCandidateOut(BaseModel):
    """sort_msgs/GraspCandidate 중 control이 후보를 고르는 데 쓰는 필드만."""

    pose: Pose
    score: float = 0.0
    gripper_width_mm: float = 0.0    # 0이면 미상 — control이 그 후보를 탈락시킨다
    candidate_id: str = ""           # "<object_id>#<순위>". 로그·웹·control이 같은 후보를 가리킨다
    # 진단용(control은 쓰지 않는다). 여기서 빠뜨리면 world_state → planner → control
    # 왕복에서 조용히 0이 된다.
    grasp_depth_mm: float = 0.0
    strategy: str = ""


class PlanStep(BaseModel):
    skill: SkillName
    object_id: str
    grip_level: GripLevel
    # 1순위 후보(= grasp_candidates[0]). 후보 목록을 못 읽는 예전 경로와 로그·DB 기록이
    # 그대로 쓴다. **실제로 실행할 후보는 control이 grasp_candidates에서 고른다.**
    grasp_pose: Pose | None = None   # pick일 때만
    gripper_width_mm: float | None = None  # pick일 때만. 1순위 후보의 예측 그리퍼 개폭(mm) — 없으면 미상
    # 작업반경 안에 있는 후보 전체(점수 내림차순). control이 개폭·IK·관절·안전을 보고 고른다.
    grasp_candidates: list[GraspCandidateOut] = Field(default_factory=list)
    # --- control의 후보 랭킹이 쓰는 물체 정보 (DetectedObject에서 그대로 옮긴다) ---
    # control은 /world_state를 구독하지 않으므로 여기 실어 보내지 않으면 볼 방법이 없다.
    object_center_mm: dict[str, float] | None = None   # position_base_mm
    object_height_mm: float | None = None              # height_mm. 미상이면 None
    depth_valid_ratio: float | None = None
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

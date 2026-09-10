"""planner가 주고받는 데이터 구조 (웹_인터페이스_정의서.md 3절, 인터페이스_정의서.md 3~4절).

**LLM 출력 구조는 여기서 정의한 것만 허용한다.** 자유 텍스트를 파싱하지 않고 JSON Schema로
받는 것이 NFR-03(코드 생성 금지, 사전 정의 스킬 시퀀스로 제한)의 구현이다.

LLM이 정하는 것과 코드가 정하는 것을 나눈다:
- LLM: 어떤 물체를(`object_id`) 어디로(`bin_id`) 어떤 순서로
- 코드: 파지 후보 목록(`grasp_candidates`)과 파지력 단계(`grip_level`, 속성 DB 기준).
  **실행할 후보 하나를 최종 선택하는 것은 control이다** — 개폭 유효성·IK·관절 한계는
  로봇에 붙어 있어야 답할 수 있고(ikin 서비스), planner는 ROS2를 모르는 별도 서비스다.
  여기서는 작업반경 안에 있는 후보만 걸러서 **grasp가 만든 순서 그대로** 넘긴다
  (2026-09-10 STEP 1 — 점수로 재정렬하지 않는다).

파지 자세와 파지력 단계는 물리적 안전에 직결되므로 LLM 출력에 맡기지 않는다(NFR-03a).
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0.0"

SkillName = Literal["pick", "place_into"]
# 파지력 단계(DetectedObject.grip_level): 1=가장 강하게(40N) ~ 5=가장 약하게(20N).
# LLM은 단계만 고르고 단계→힘 매핑은 control/config/skill_params.yaml이 소유한다.
GripLevel = Literal[1, 2, 3, 4, 5]
ValidationStatus = Literal["approved", "rejected"]
# 시나리오 도메인. 동일한 명령이라도 도메인마다 VLM 프롬프트가 달라진다.
Domain = Literal["general", "pharmacy", "recycle"]


# --- LLM 출력 (구조화 출력 스키마) -------------------------------------------

class LlmStep(BaseModel):
    """LLM이 생성하는 스텝 하나. 좌표·힘 같은 물리량은 포함하지 않는다."""

    model_config = ConfigDict(extra="forbid")

    skill: SkillName
    object_id: str = Field(description="world_state에 실제로 존재하는 object_id만 사용")
    bin_id: str = Field(
        default="",
        description="place_into일 때만 채운다. bins.yaml에 등록된 목적지 중 하나. pick이면 빈 문자열",
    )


class LlmPlan(BaseModel):
    """LLM 응답 전체. 거부도 '자유 텍스트'가 아니라 이 구조 안에서 표현한다."""

    model_config = ConfigDict(extra="forbid")

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
    # fine-tuning용 원본 클라우드 경로(graspnet_baseline만). control은 안 쓰지만
    # grasp_depth_mm과 같은 이유로 빠뜨리면 안 된다 — orchestrator가 execution_logs에
    # 남기려면 이 왕복(world_state → planner → PlanStep)에서 살아 있어야 한다.
    point_cloud_path: str = ""
    # 손가락 사이 실측 재료 지지도 0~1, **-1이면 미상**(GraspCandidate.msg 주석 참조).
    # control의 enhanced ranking이 쓴다. 기본값을 0.0으로 두면 안 된다 — "미상"이
    # "지지 없음"으로 읽혀 멀쩡한 후보가 매번 감점된다.
    contact_support_score: float = -1.0


class PlanStep(BaseModel):
    skill: SkillName
    object_id: str
    grip_level: GripLevel
    # 1순위 후보(= grasp_candidates[0]). 후보 목록을 못 읽는 예전 경로와 로그·DB 기록이
    # 그대로 쓴다. **실제로 실행할 후보는 control이 grasp_candidates에서 고른다.**
    grasp_pose: Pose | None = None   # pick일 때만
    gripper_width_mm: float | None = None  # pick일 때만. 1순위 후보의 예측 그리퍼 개폭(mm) — 없으면 미상
    # 작업반경 안에 있는 후보 전체. **grasp가 만든 순서 그대로다**(0번=GraspNet 최고점, 그 뒤는
    # 최종 pose Diverse-TopK 순서) — planner는 점수로 재정렬하지 않는다. control이 개폭·IK·
    # 관절·안전·랭킹을 보고 그중 하나를 고른다.
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
    # 시나리오 도메인 (가정/약국/재활용). 일반 명령은 "general". 관측·계획 단계에서
    # 도메인별 VLM 프롬프트를 고르는 데 쓴다.
    domain: Domain = "general"
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

"""LLM 태스크 플래닝 클라이언트. world_state + 자연어 명령 → 스킬 시퀀스 (FR-10).

구조화 출력(JSON Schema)으로 `schema.LlmPlan` 형태만 받는다. 자유 텍스트를 파싱하지
않는 것이 NFR-03(사전 정의된 스킬 시퀀스로 제한)의 구현이다.

프롬프트는 [고정 프리픽스 → 가변 부분] 순서로 만든다. 시스템 지시문과 스킬 설명은
매 요청 동일하므로 프리픽스 캐싱이 걸리고, 매번 바뀌는 월드 상태·명령문은 뒤에 온다.
"""
import logging
import os

from openai import OpenAI

from .schema import LlmPlan

logger = logging.getLogger(__name__)

# 프롬프트를 고칠 때마다 올린다. task_sequences.prompt_version으로 DB에 기록되어
# 평가셋 결과를 프롬프트 버전별로 비교할 수 있게 한다 (NFR-09).
PROMPT_VERSION = "plan-v2"

# 기본값은 이 계정에서 사용 가능한 모델. 변경 시 .env의 OPENAI_MODEL로 덮어쓴다.
DEFAULT_MODEL = "gpt-4o"

SYSTEM_PROMPT = """\
당신은 협동로봇 분류 시스템의 태스크 플래너다. 사용자의 자연어 지시를 로봇이 실행할 \
스킬 시퀀스로 변환한다.

[사용 가능한 스킬 — 이 두 가지 외에는 없다]
- pick: 물체 하나를 집는다. object_id 필요.
- place_into: 들고 있는 물체를 지정한 목적지에 놓는다. object_id와 bin_id 필요.

[규칙]
1. 반드시 아래 "검출된 물체" 목록에 있는 object_id만 사용한다. 목록에 없는 물체를 \
지시받으면 시퀀스를 만들지 말고 steps를 비운 채 refusal_reason에 이유를 적는다.
2. bin_id는 "배치 가능한 목적지" 목록에 있는 것만 사용한다.
3. 하나의 물체는 반드시 pick 다음에 place_into가 이어져야 한다. 여러 물체를 옮길 때는 \
pick→place_into 쌍을 물체 수만큼 반복한다. 한 번에 두 개를 집을 수 없다.
4. 파지 불가로 표시된 물체는 시퀀스에 넣지 않는다. 다만 그 물체 때문에 지시 전체를 \
거부하지는 않는다 — 아래 5번을 따른다.
5. **지시 중 일부만 수행 가능하면, 수행 가능한 부분만 시퀀스로 만든다.** 하나도 수행할 수 \
없을 때만 steps를 비운다. 예: "전부 옮겨줘"인데 3개 중 1개가 파지 불가면, 나머지 2개를 \
옮기는 시퀀스를 만든다.
6. "미확인 신규품목"으로 표시된 물체도 다른 물체와 똑같이 옮길 수 있다. 시스템이 자동으로 \
보수적인 파지 설정을 적용하므로, 확인되지 않았다는 이유로 시퀀스에서 빼지 않는다.
7. 지시가 모호해서 어떤 물체를 뜻하는지 특정할 수 없으면, 추측하지 말고 steps를 비우고 \
refusal_reason에 무엇이 모호한지 적는다.
8. 좌표·파지 자세·파지력은 절대 만들어내지 않는다. 그것은 시스템이 결정한다.
9. refusal_reason은 한국어로, 작업자가 무엇을 고쳐 말해야 할지 알 수 있게 쓴다.
"""


def _client() -> OpenAI:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다 (.env 참조)")
    return OpenAI()


def model_name() -> str:
    return os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)


def build_user_prompt(command_text: str, world_summary: str,
                      previous_failure: dict | None = None) -> str:
    parts = [world_summary, "", f"[지시]\n{command_text}"]
    if previous_failure:
        parts += [
            "",
            "[직전 시도 실패 — 같은 방식을 반복하지 말 것]",
            f"- 대상: {previous_failure.get('object_id')}",
            f"- 실패 사유: {previous_failure.get('failure_reason')}",
            "해당 물체를 건너뛰거나 다른 순서로 재계획한다.",
        ]
    return "\n".join(parts)


def plan(command_text: str, world_summary: str,
         previous_failure: dict | None = None) -> LlmPlan:
    """스킬 시퀀스를 생성한다. 반환값은 스키마가 보장된 LlmPlan."""
    client = _client()
    model = model_name()
    response = client.responses.parse(
        model=model,
        instructions=SYSTEM_PROMPT,          # 고정 프리픽스
        input=build_user_prompt(command_text, world_summary, previous_failure),
        text_format=LlmPlan,
        temperature=0,                        # 같은 입력에 같은 계획 (NFR-09 재현성)
    )
    parsed = response.output_parsed
    logger.info(
        "LLM 계획 생성: model=%s steps=%d refusal=%r",
        model, len(parsed.steps), parsed.refusal_reason,
    )
    return parsed

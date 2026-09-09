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
PROMPT_VERSION = "plan-v7"

# 기본값은 이 계정에서 사용 가능한 모델. 변경 시 .env의 OPENAI_MODEL로 덮어쓴다.
DEFAULT_MODEL = "gpt-4o"

# 도메인별 **계획 해석 규칙** — build_user_prompt가 [장면 맥락]으로 넣는다.
# 같은 문장도 도메인에 따라 목적이 다르다:
# - 약국: 지시가 증상으로 와도 검출된 **약 이름에서 효능을 추론**해 정확한 약을 고른다.
#   먼저 "먹는 약"인지(밴드·연고·안약 등 외용/처치용품 제외)부터 거르고, 그다음 주 용도가
#   지시된 증상과 맞는지로 좁힌다(2026-09-09: medica_band를 두통약으로 잘못 승인한 사고 대응).
# - 재활용: 목적은 **재질별 분리**이므로 이름의 재질과 지시의 목적지를 매칭한다.
# `home`(가정)은 special 규칙이 없어 general과 같으므로 별도 도메인이 아니다.
DOMAIN_CONTEXT_PLAN = {
    "pharmacy": """\
이 작업은 **약국** 시나리오다.
- 지시가 증상·효능으로 와도('머리가 아플 때 먹는 약 줘'), 검출된 **약 이름 자체에서 효능을
  추론**해 그에 맞는 약을 정확히 고른다. e.g. 이부프로펜정/타이레놀정 → 해열진통제(두통),
  비타민C → 영양제.
- **1단계: "먹는 약"인지부터 거른다.** 약국에서 파는 물건이라고 다 "약"이 아니다. 밴드/
  반창고(medica_band 등), 연고·크림(daktarin_cream 등 외용제), 안약(eye_drops), 파스,
  소독약처럼 **바르거나 붙이거나 넣는 제품은 지시가 "먹는 약"을 찾을 때는 후보가 아니다**
  — 지시가 그 제품을 이름으로 콕 집어 부르지 않는 한 1단계에서 통째로 제외한다.
- **2단계: 남은 후보 중에서만 효능으로 고른다.** 이름이 가리키는 **주된 용도**로 판단하고,
  효능이 살짝 겹쳐 보인다고 넓혀 잡지 않는다(예: 종합감기약은 감기가 주 용도이지 두통약이
  아니다 — 지시가 '감기약'이 아니라 '두통약'이면 빼야 한다).
- 예: 검출이 [medica_band(밴드), tylenol(타이레놀정), cold_medicine(종합감기약),
  daktarin_cream(무좀연고), eye_drops(안약)]이고 지시가 '머리가 아플 때 먹는 약 줘'이면
  → 1단계에서 medica_band·daktarin_cream·eye_drops 제외(먹는 게 아님) → 2단계에서
  cold_medicine도 제외(주 용도가 감기이지 두통이 아님) → **tylenol만** 남는다.
- 지시한 효능에 맞는 약이 목록에 없으면, 없는 약을 지어내지 말고 기존 규칙대로 거부한다.
- 여러 약이 같은 효능이면 전부 대상으로 삼되, 파지 불가나 모호한 것은 기존 규칙을 따른다.
- **골랐으면 마지막으로 한 번 더 검산한다**: 최종 대상 하나하나에 대해 "이 이름이 실제로
  ①먹는 약이고 ②지시된 증상/효능과 맞는가"를 스스로 되묻는다. 하나라도 아니다 싶으면
  그 물체를 빼고, 남는 게 없으면 지어내지 말고 거부한다.""",
    "recycle": """\
이 작업은 **재활용(분리수거)** 시나리오다.
- 목적은 **재질별 분리**다. 검출된 물체의 이름에 담긴 재질(플라스틱/캔/유리/종이 등)을 보고
  지시가 말한 목적지(bin)로 보낸다. e.g. pet_plastic_bottle → 왼쪽, aluminum_can → 오른쪽.
- '전부/모두/다'로 오면 재질과 무관하게 재활용 대상 물체를 하나도 빠뜨리지 않는다.
- 이름에서 재질을 특정할 수 없는 물체는 추측하지 않는다 — 기존 규칙 5(가능한 것만)와
  7(모호하면 거부)을 그대로 따른다.""",
}

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
6. "미확인 신규품목"이나 "속성 VLM추정"으로 표시된 물체도 다른 물체와 똑같이 옮길 수 있다. \
파지력과 속도는 시스템이 그 물체의 속성에 맞춰 정하므로, 속성이 확인된 값이 아니라는 이유로 \
시퀀스에서 빼지 않는다.
6-1. **"전부", "모두", "다" 같은 지시를 받으면 조건에 맞는 물체를 하나도 빠뜨리지 않는다.** \
파손위험·투명·미확인·VLM추정처럼 조심스러운 속성이 붙어 있어도 제외 사유가 되지 않는다 — 파지력과 \
속도는 시스템이 속성에 맞게 조절한다. 제외할 수 있는 유일한 경우는 "파지불가" 표시뿐이다. \
시퀀스를 만든 뒤, 대상이 되어야 할 물체가 전부 들어갔는지 목록과 대조해 확인한다.
7. 지시가 모호해서 어떤 물체를 뜻하는지 특정할 수 없으면, 추측하지 말고 steps를 비우고 \
refusal_reason에 무엇이 모호한지 적는다.
7-1. **모호함을 "전부"로 넓혀서 해결하지 않는다.** "그거", "저거", "이것 좀"처럼 지시대명사만 \
있어 어느 물체인지 알 수 없으면 거부한다. 검출된 물체가 하나뿐이라 가리킬 대상이 그것밖에 \
없을 때만 예외다. 6-1은 사용자가 실제로 "전부/모두/다"라고 말했을 때만 적용된다 — 대상이 \
불분명하다고 전부를 옮기면, 사용자가 지시하지 않은 물체까지 옮겨진다.
7-2. **목적지도 같은 기준이다.** "저기", "거기", "적당한 곳"처럼 목적지를 특정할 수 없거나 \
아예 말하지 않았으면, 목적지 목록에서 임의로 고르지 말고 거부한다. 대상이 분명해도 목적지가 \
모호하면 거부다 — 엉뚱한 곳에 놓는 것은 분류 실패이고, 되돌리려면 사람이 다시 꺼내야 한다.
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
                      previous_failure: dict | None = None,
                      domain: str = "general") -> str:
    parts = [world_summary, "", f"[지시]\n{command_text}"]
    if domain and domain != "general":
        context = DOMAIN_CONTEXT_PLAN.get(domain)
        if context:
            parts.insert(1, "[장면 맥락(도메인)]\n" + context)
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
         previous_failure: dict | None = None, domain: str = "general") -> LlmPlan:
    """스킬 시퀀스를 생성한다. 반환값은 스키마가 보장된 LlmPlan."""
    client = _client()
    model = model_name()
    response = client.responses.parse(
        model=model,
        instructions=SYSTEM_PROMPT,          # 고정 프리픽스
        input=build_user_prompt(command_text, world_summary, previous_failure, domain),
        text_format=LlmPlan,
        temperature=0,                        # 같은 입력에 같은 계획 (NFR-09 재현성)
    )
    parsed = response.output_parsed
    logger.info(
        "LLM 계획 생성: model=%s domain=%s steps=%d refusal=%r",
        model, domain, len(parsed.steps), parsed.refusal_reason,
    )
    return parsed

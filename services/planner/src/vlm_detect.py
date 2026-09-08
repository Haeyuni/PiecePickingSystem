"""사진 한 장 → VLM에게 "무엇이 있는가"를 묻는다. YOLO 대신 쓰는 인지 경로.

두 가지 방식이 들어 있다. **쓰는 것은 아래쪽 `label_marks`다.**

- `detect(image)` — VLM에게 물체와 **바운딩박스를 함께** 묻는다. 원래 설계이자
  "사진을 API로 보내면 좌표를 주는가"를 확인하려던 것. gpt-4o에서는 좌표를 못 쓴다
  (docs/vlm_sam_pipeline.md 실측 결과). 더 나은 모델이 붙었을 때 다시 재 볼 수 있게 남겨 둔다.
- `label_marks(image, mark_ids)` — SAM이 먼저 장면을 조각내고 번호를 그려 준 이미지를 받아
  **번호마다 무엇인지와 어떻게 다뤄야 하는지**를 답한다. 좌표는 SAM이, 판단은 VLM이 —
  각자 잘하는 쪽을 맡는다.

어느 쪽이든 3D 좌표와 파지 자세는 만들지 않는다. 마스크와 depth로 코드가 계산한다
(`perception_test_image.py:mask_3d`와 같은 방식).

[속성도 VLM이 판단한다] `label_marks`는 원래 objects.yaml의 등록 클래스 어휘를 프롬프트에
넣고 이름만 받아 왔고, 무게·파손위험·파지 단계는 그 이름으로 objects.yaml에서 조회했다. 지금은
어휘를 주지 않는다 — 사진만 보고 이름과 속성(mass_g/fragile/deformable/transparent)과 파지
단계(grip_level)까지 모델이 정한다. **등록되지 않은 물건도 그 물건에 맞게 다뤄지지만, 파지력을 정하는
값이 모델 출력이 되었다는 뜻이기도 하다**(원래는 NFR-03a가 금지하던 것 — 대신 확신이 없으면
조심스러운 쪽을 고르라고 프롬프트에 박고, `_normalize_marks`가 fragile 조합을 한 번 더
강제한다). 되돌리려면 detector=yolo가 그대로 남아 있다.

[확신이 없으면 web_search로 확인한다] 등록 어휘가 없어졌으니 겉모습만으로 오분류하는 경우가
생긴다(2026-09-08 실측: 빨간 치약 튜브를 "chocolate_bar"로 답했다 — 색과 형태만으로 짐작하고
포장에 적힌 글자는 읽지 않았다). `label_marks`는 `web_search` 도구를 켜 두고, 프롬프트가
겉면 문구를 단서로 실제 검색을 시켜 이름을 확인하게 한다(`SYSTEM_PROMPT_MARKS`의 [확신이
없으면 검색으로 확인한다]). 이미지를 검색하는 것이 아니라 **모델이 사진에서 읽은 글자로
검색어를 만들어** 텍스트 검색을 돌리는 것이다 — OpenAI 웹 검색 도구는 이미지 자체를
질의로 받지 않는다. 물체마다 매번 검색하는 것은 아니다: 모델이 확신이 있으면(정상적인
치약·물티슈처럼 흔한 물건) 검색 없이 바로 답하고, 포장에 글자가 있는데 정확한 제품명을
모를 때만 검색을 쓴다 — 그래서 왕복 시간이 물체 수만큼 늘지는 않는다(검색은 도구 호출이 필요할
때만 같은 요청 안에서 일어난다).

[구조화 출력] `llm_client`와 같은 원칙 — 자유 텍스트를 파싱하지 않고 JSON Schema로만 받는다
(NFR-03).

[지시는 인지에 넣지 않는다] `label_marks`는 사용자의 명령문을 받지 않는다. 이유는 그
함수의 주석에 있다.
"""
import base64
import logging
import mimetypes
import os
import pathlib
from functools import lru_cache
from typing import Literal

import yaml
from openai import OpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# 프롬프트를 고칠 때마다 올린다 (llm_client.PROMPT_VERSION과 같은 이유). 두 경로는 프롬프트가
# 다르므로 버전도 따로 센다.
PROMPT_VERSION = "vlm-detect-v1"        # detect(): VLM에게 박스를 묻는다
MARKS_PROMPT_VERSION = "vlm-marks-v5"   # label_marks(): 도메인(가정/약국/재활용)별 이름 규칙 + 스스로 판단 + 확신 없으면 web_search로 확인 + reasoning

# 미설정 시 OPENAI_MODEL을 따라가고, 그것도 없으면 이 값. 이미지 입력이 되는 모델이어야 한다.
DEFAULT_MODEL = "gpt-4o"

# 시나리오 도메인의 한국어 표기. 프롬프트의 [장면 맥락]에 넣어 VLM이 도메인에 맞게 답하게 한다.
# 도메인별 **이름 규칙** — build_marks_prompt가 [장면 맥락]으로 넣는다 (vlm-detect, label_marks).
# `home`(가정)은 special 규칙이 없어 general과 같으므로 별도 도메인이 아니다 — 가정 버튼도
# general로 온다.
# 같은 물체도 도메인에 따라 이름에 담아야 하는 정보가 다르다:
# - 약국: 어떤 약인지가 중요하므로 **정확한 약품명**(성분/제품명)을 읽어 쓴다.
# - 재활용: 어느 재질로 분리할지가 중요하므로 **재질이 이름에 드러나야** 한다.
# 지시(command_text)는 여기 오지 않는다 — 도메인은 시나리오 컨텍스트일 뿐이다(D-1).
DOMAIN_CONTEXT_MARKS = {
    "pharmacy": """\
이 사진은 **약국** 시나리오다.
- 이름은 **정확한 약품명**이어야 한다. 포장·포일·설명서·병 라벨에 적힌 성분명/제품명을 \
읽어 class_name과 name_ko에 그대로 쓴다 (예: ibuprofen / '이부프로펜정', tylenol / \
'타이레놀정').
- '약', '알약', 'painkiller'처럼 **종류로 얼버무리지 않는다** — 어느 약인지가 분류의 대상이다.
- 포장 글자가 흐려 못 읽으면 `web_search`로 확인한 뒤 확정한다. 그래도 특정할 수 없으면 \
'unknown_medicine'처럼 보수적인 이름 + 낮은 confidence로 답한다.
- 포장 형태(블리스터·병·봉투)나 재질은 이름 대신 속성으로 답한다.""",
    "recycle": """\
이 사진은 **재활용(분리수거)** 시나리오다.
- 이름에 **재질이 반드시 드러나야** 한다. 재질 + 형태 조합으로 지는다 \
(예: pet_plastic_bottle, aluminum_can, glass_bottle, paper_box, hdpe_container, can).
- name_ko도 재질을 담는다 (예: 'PET 페트병', '알루미늄 캔', '유리병', '종이 상자').
- 재질 표시(플라스틱 재질 번호, 캔·병 각인)를 읽고 판단한다. 모르면 `web_search`로 확인한다.
- 재질을 특정할 수 없으면 unknown_material 같은 보수적인 이름 + 낮은 confidence로 답한다 — \
지어내지 않는다.""",
}

OBJECTS_YAML = pathlib.Path(
    os.environ.get("OBJECTS_YAML")
    or pathlib.Path(__file__).resolve().parents[3] / "src" / "perception" / "config" / "objects.yaml"
)


# --- VLM 출력 (구조화 출력 스키마) -------------------------------------------

class VlmObject(BaseModel):
    """검출된 물체 하나. 3D·파지 정보는 없다 — 그건 depth와 grasp의 몫이다."""

    object_id: str = Field(
        description="이 장면 안에서 고유한 식별자. 소문자 슬러그 + 일련번호 (예: toothpaste_1)",
    )
    class_name: str = Field(
        description="아래 [등록된 클래스] 목록에 있으면 그 이름을 그대로. 없으면 새로 지은 "
                    "영문 소문자 스네이크케이스 이름",
    )
    name_ko: str = Field(description="작업자에게 보여줄 한국어 이름")
    is_new_class: bool = Field(description="[등록된 클래스] 목록에 없는 물체면 true")
    box_norm: list[int] = Field(
        description="바운딩박스 [x1, y1, x2, y2]. 이미지 왼쪽 위가 (0,0), 오른쪽 아래가 "
                    "(1000,1000)인 정규화 좌표. x1<x2, y1<y2",
    )
    confidence: float = Field(description="이 판단의 확신도 0.0~1.0")


class VlmScene(BaseModel):
    """한 프레임에 대한 VLM 응답 전체."""

    objects: list[VlmObject] = Field(description="장면에서 집을 수 있는 물체 전부")
    target_object_ids: list[str] = Field(
        description="지시가 가리키는 물체의 object_id. 지시가 없거나 대상이 없으면 빈 배열",
    )
    refusal_reason: str = Field(
        default="",
        description="지시 대상을 장면에서 찾을 수 없거나 지시가 모호할 때 그 이유(한국어). "
                    "대상을 찾았으면 빈 문자열",
    )


# --- 프롬프트 ----------------------------------------------------------------

SYSTEM_PROMPT = """\
당신은 협동로봇 분류 시스템의 시각 인지 모듈이다. 작업대를 위에서 내려다본 사진 한 장을 \
보고, 로봇이 집을 수 있는 물체가 무엇이고 어디에 있는지 답한다.

[해야 할 일]
1. 사진에 있는 **집을 수 있는 물체**를 빠짐없이 찾는다.
2. 각 물체의 바운딩박스를 0~1000 정규화 좌표 [x1, y1, x2, y2]로 준다. 왼쪽 위가 (0,0), \
오른쪽 아래가 (1000,1000)이다. 박스는 물체를 꽉 감싸되 물체 전체를 포함해야 한다 — \
잘리면 뒤 단계(세그멘테이션)에서 물체 일부가 사라진다.
3. 지시문이 함께 오면, 그 지시가 가리키는 물체의 object_id를 target_object_ids에 담는다.

[물체가 아닌 것 — 넣지 않는다]
- 작업대·배경·바구니·상자 자체
- 케이블, 전선, 테이프로 고정된 배선
- 사진에 이미 그려져 있는 검출 결과 오버레이(사각형 테두리, 라벨 글자, 신뢰도 숫자). \
그것은 다른 모델의 출력이지 물체가 아니다. **그 박스를 그대로 베끼지 말고 사진에 실제로 \
찍힌 물체를 직접 보고 판단한다.**

[클래스 이름]
- 아래 [등록된 클래스]에 해당하는 물체면 그 class_name을 **철자 그대로** 쓰고 \
is_new_class=false로 둔다.
- 목록에 없는 물체는 새 이름을 지어 주고(영문 소문자 스네이크케이스) is_new_class=true로 \
둔다. 목록에 없다고 빼지 않는다 — 시스템이 신규품목으로 등록해 보수적으로 다룬다.

[지시 해석 — target_object_ids]
1. 지시는 클래스 이름이 아니라 범주나 생김새로 올 수 있다("화장품", "파란 통", "제일 큰 것"). \
사진과 물체 목록을 근거로 어느 것을 말하는지 고른다.
2. **지시가 가리키는 물체가 사진에 없으면 target_object_ids를 비우고 refusal_reason에 \
무엇이 없는지 적는다.** 비슷한 다른 물체로 바꿔 고르지 않는다 — 사용자가 지시하지 않은 \
물체를 옮기는 것이 못 옮기는 것보다 나쁘다.
3. 지시가 모호해서 어느 물체인지 특정할 수 없으면("그거", "저거") 역시 비우고 이유를 적는다. \
집을 수 있는 물체가 하나뿐이라 가리킬 대상이 그것밖에 없을 때만 예외다.
4. "전부", "모두", "다"라고 하면 조건에 맞는 물체를 하나도 빠뜨리지 않는다.
5. refusal_reason은 작업자가 무엇을 고쳐 말해야 할지 알 수 있게 한국어로 쓴다.

[하지 않는 것]
- 3D 좌표·거리·무게·파지 자세는 절대 만들어내지 않는다. 그것은 depth 센서와 시스템이 정한다.
- 사진에서 안 보이는 물체를 추측해서 넣지 않는다.
"""


@lru_cache(maxsize=1)
def known_classes(path: str | None = None) -> list[tuple[str, str]]:
    """objects.yaml에 등록된 (class_name, name_ko) 목록.

    **쓰는 곳은 `detect()`뿐이다.** 운영 경로인 `label_marks()`는 어휘를 주지 않고 모델이
    직접 이름을 짓는다(모듈 상단 [속성도 VLM이 판단한다]). 어휘를 주면 같은 물체를 매번
    같은 이름으로 부르게 되는 이점이 있어, 좌표까지 묻는 `detect()`에는 남겨 둔다.
    """
    p = pathlib.Path(path) if path else OBJECTS_YAML
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except OSError:
        logger.warning("objects.yaml을 못 읽었다 (%s) — 등록 클래스 없이 진행한다", p)
        return []
    return [
        (name, (spec or {}).get("name_ko") or name)
        for name, spec in (data.get("objects") or {}).items()
    ]


def build_user_prompt(command_text: str | None, image_size: tuple[int, int] | None = None) -> str:
    lines = ["[등록된 클래스]"]
    classes = known_classes()
    lines += [f"- {name}: {ko}" for name, ko in classes] or ["(없음)"]
    if image_size:
        lines += ["", f"[사진 크기] {image_size[0]}x{image_size[1]} px "
                      f"(답은 이 크기와 무관한 0~1000 정규화 좌표로 준다)"]
    lines += ["", "[지시]", command_text or "(지시 없음 — 물체 검출만 하고 "
                                            "target_object_ids는 비워 둔다)"]
    return "\n".join(lines)


# --- 호출 --------------------------------------------------------------------

def _client() -> OpenAI:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다 (.env 참조)")
    return OpenAI()


def model_name() -> str:
    return os.environ.get("VLM_MODEL") or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL


def encode_bytes(data: bytes, mime: str = "image/png") -> str:
    """이미지 바이트를 data URL로. 파일 경로 대신 base64로 보내는 이유는 OpenAI가 우리 파일을
    가져갈 수 있는 URL이 없기 때문이다(로컬 카메라 프레임)."""
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def encode_image(path: pathlib.Path | str) -> str:
    """파일에서 읽어 data URL로."""
    path = pathlib.Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return encode_bytes(path.read_bytes(), mime)


def detect(image: pathlib.Path | str | bytes, command_text: str | None = None,
           image_size: tuple[int, int] | None = None, model: str | None = None,
           detail: str = "high") -> VlmScene:
    """사진(+선택적 지시) → 검출된 물체와 바운딩박스.

    image는 파일 경로 또는 이미 만들어 둔 data URL 문자열/바이트가 아니라 **경로**를 기대한다.
    카메라 프레임을 바로 보내려면 `encode_image` 대신 png로 인코딩해 넘긴다.

    detail="high"는 작은 물체 때문이다 — "low"는 512px 축소본만 보므로 튜브·젤네일처럼
    작은 물체의 박스가 뭉개진다.
    """
    data_url = image if isinstance(image, str) and image.startswith("data:") else encode_image(image)
    client = _client()
    model = model or model_name()
    request = dict(
        model=model,
        instructions=SYSTEM_PROMPT,          # 고정 프리픽스 (프리픽스 캐싱)
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": build_user_prompt(command_text, image_size)},
                {"type": "input_image", "image_url": data_url, "detail": detail},
            ],
        }],
        text_format=VlmScene,
    )
    try:
        response = client.responses.parse(**request, temperature=0)   # 같은 사진에 같은 답
    except Exception as e:
        # gpt-5 계열처럼 temperature를 받지 않는 모델이 있다. 재현성은 포기하고 진행한다.
        if "temperature" not in str(e):
            raise
        logger.warning("%s 모델이 temperature를 거부해 기본값으로 재시도한다", model)
        response = client.responses.parse(**request)

    scene = response.output_parsed
    scene = _normalize(scene)
    logger.info(
        "VLM 검출: model=%s objects=%d targets=%s refusal=%r",
        model, len(scene.objects), scene.target_object_ids, scene.refusal_reason,
    )
    return scene


def _normalize(scene: VlmScene) -> VlmScene:
    """모델 출력의 흔한 흠집을 코드에서 정리한다. 스키마가 못 잡는 것들이다.

    - object_id 중복 (같은 이름을 두 번 붙이는 경우)
    - 박스 좌표 뒤바뀜/범위 초과
    - 존재하지 않는 object_id를 target으로 지목 (그라운딩 위반 — grounding.py와 같은 성격)
    """
    seen: set[str] = set()
    for i, o in enumerate(scene.objects):
        oid = o.object_id or f"{o.class_name or 'object'}_{i + 1}"
        while oid in seen:
            oid = f"{oid}_{i + 1}"
        o.object_id = oid
        seen.add(oid)

        box = list(o.box_norm) + [0] * (4 - len(o.box_norm))
        x1, y1, x2, y2 = (max(0, min(1000, int(v))) for v in box[:4])
        o.box_norm = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    unknown = [t for t in scene.target_object_ids if t not in seen]
    if unknown:
        logger.warning("VLM이 존재하지 않는 object_id를 지목했다: %s — 버린다", unknown)
        scene.target_object_ids = [t for t in scene.target_object_ids if t in seen]
        if not scene.target_object_ids and not scene.refusal_reason:
            scene.refusal_reason = "지시 대상을 사진에서 찾지 못했습니다"
    return scene


# --- 후처리 ------------------------------------------------------------------

def _pixel_box(box_norm: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = box_norm
    px = [round(x1 * width / 1000), round(y1 * height / 1000),
          round(x2 * width / 1000), round(y2 * height / 1000)]
    px[2] = max(px[2], px[0] + 1)      # 폭·높이 0인 박스는 SAM 프롬프트로 쓸 수 없다
    px[3] = max(px[3], px[1] + 1)
    return px


def to_pixels(scene: VlmScene, width: int, height: int) -> list[dict]:
    """정규화 박스를 픽셀 박스로. SAM에 넘길 형태([x1,y1,x2,y2] 픽셀)를 함께 담는다.

    폭·높이가 0인 박스는 SAM 프롬프트로 쓸 수 없으므로 최소 1px을 보장한다.
    """
    out = []
    for o in scene.objects:
        px = _pixel_box(o.box_norm, width, height)
        out.append({
            "object_id": o.object_id,
            "class_name": o.class_name,
            "name_ko": o.name_ko,
            "is_new_class": o.is_new_class,
            "confidence": round(float(o.confidence), 3),
            "box_norm": o.box_norm,
            "box_xyxy": px,
        })
    return out


# =============================================================================
# 마크 라벨링 (Set-of-Mark) — SAM이 먼저, VLM이 나중
# =============================================================================
#
# 위의 detect()는 VLM에게 좌표를 물어본다. **gpt-4o는 그 좌표를 못 맞춘다** —
# 2026-09-07 실측에서 박스가 100px 단위로 뭉개지고 다른 물체 위에 얹혔다
# (docs/vlm_sam_pipeline.md 실측 결과). 좌표를 못 맞추는 모델에게 좌표를 묻는 대신,
# 순서를 뒤집는다:
#
#   SAM everything 모드로 마스크 후보를 전부 뽑는다  ← 좌표는 SAM이 만든다 (정확)
#   → 마스크마다 번호를 그려 넣은 이미지를 VLM에 보낸다
#   → VLM은 "몇 번이 무엇이고 어떻게 다뤄야 하는가"를 답한다  ← 판단은 VLM이 한다
#
# VLM이 잘하는 일(무엇인지 알아보기)과 못하는 일(어디인지 픽셀로 찍기)을 갈라 놓는 것이
# 요점이다. 마스크가 물체 하나를 여러 조각으로 쪼개는 것은 SAM에서 흔하므로,
# `part_of`로 같은 물체를 가리키는 번호를 묶게 한다.

class MarkLabel(BaseModel):
    """번호가 붙은 마스크 하나에 대한 판단. 이름뿐 아니라 **어떻게 다뤄야 하는지**까지 담는다.

    속성(mass_g·fragile·deformable·transparent·grip_level)이 여기 들어온 뒤로 objects.yaml의
    등록 클래스 표를 거치지 않는다 — 모듈 상단 [속성도 VLM이 판단한다] 참조.
    """

    mark_id: int = Field(description="이미지에 그려진 번호")
    is_object: bool = Field(
        description="이 번호가 로봇이 집을 수 있는 물체 하나를 대표하면 true. 작업대·배경·"
                    "바구니·케이블·그림자면 false. 다른 번호가 대표하는 물체의 조각이어도 false",
    )
    part_of: int = Field(
        description="이 번호가 다른 번호가 대표하는 **같은 물체의 조각**이면 그 번호. "
                    "아니면 0. 물체가 아닌 것(배경·케이블·오버레이)은 조각이 아니므로 0",
    )
    class_name: str = Field(
        description="사진을 보고 직접 지은 영문 소문자 스네이크케이스 이름(예: toothpaste). "
                    "브랜드명이 아니라 물건의 종류로 짓는다. is_object=false면 빈 문자열",
    )
    name_ko: str = Field(description="작업자 화면에 띄울 한국어 이름. is_object=false면 빈 문자열")
    mass_g: float = Field(
        description="물체 종류와 사진 속 크기로 추정한 무게(g). 짐작조차 어려우면 0",
    )
    fragile: bool = Field(description="떨어뜨리거나 세게 쥐면 깨지는 물체면 true")
    deformable: bool = Field(description="쥐면 모양이 눌리거나 변하는 물체면 true")
    transparent: bool = Field(description="투명·반투명해서 속이 비쳐 보이면 true")
    grip_level: Literal[1, 2, 3, 4, 5] = Field(
        description="로봇이 이 물체를 쥐는 파지력 단계(1~5). 1=매우 강하게, 5=매우 약하게. "
                    "이 값이 실제 파지력·접근속도가 된다",
    )
    confidence: float = Field(description="이 판단의 확신도 0.0~1.0")
    reasoning: str = Field(
        description="이름·속성·grip_level 판단의 근거를 한 문장으로. 예: '포장 글자로 확인함' "
                    "/ '흔한 물티슈 포장 형태'. is_object=false면 빈 문자열",
    )
    mask_poly: list[list[list[float]]] = Field(
        default_factory=list,
        description="YOLO 학습용 마스크 윤곽선 좌표. 각 요소는 하나의 다각형(polygon), "
                    "다각형은 [[x1,y1],[x2,y2],...] 형태의 정규화 좌표(0~1). "
                    "SAM에서 추출 — 응답에만 사용, VLM 판단과 무관",
    )


def mask_to_polygons(mask: "np.ndarray") -> list[list[list[float]]]:
    """SAM 마스크(bool ndarray) → YOLO 세그멘테이션용 정규화 다각형 좌표.

    각 다각형은 [[x1,y1],[x2,y2],...] 형태로, 좌표는 이미지 크기로 나눠 0~1로 정규화된다.
    음수 좌표(SAM 경계 오류)는 0으로 클리핑한다.
    """
    import cv2
    import numpy as np

    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(
        mask.astype(np.uint8).copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    polys = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        poly = cnt.squeeze(1) if cnt.ndim == 3 else cnt
        poly = poly.astype(float)
        poly[:, 0] = np.clip(poly[:, 0] / w, 0.0, 1.0)
        poly[:, 1] = np.clip(poly[:, 1] / h, 0.0, 1.0)
        polys.append(poly.tolist())
    return polys


class VlmMarkScene(BaseModel):
    """번호별 판단만 담는다. **사용자의 지시는 이 단계에 오지 않는다** — 아래 주석 참조."""

    marks: list[MarkLabel] = Field(description="이미지에 그려진 모든 번호에 대한 판단")


# [grip_level의 단계—힘 매핑은 여기 하드코딩되어 있다] 40/35/30/25/20N은
# perception/config/objects.yaml의 grip_levels 블록과 control/config/skill_params.yaml에 있는
# 값을 프롬프트에 적어 둔 것이다. 모델이 "얼마나 세게 쥐는지"를 모르면 단계를 고를 근거가
# 없어서 넣었는데, 저쪽 값을 고치면 여기도 같이 고쳐야 한다 — 안 고쳐도 오류로는 보이지
# 않고 판단만 틀어진다. **LLM은 단계만 고르고 단계→힘 변환은 하지 않는다.**
SYSTEM_PROMPT_MARKS = """\
당신은 협동로봇 분류 시스템의 시각 인지 모듈이다. 작업대를 위에서 내려다본 사진에 \
세그멘테이션 모델이 뽑은 영역마다 **번호와 윤곽선**이 그려져 있다. 번호마다 그것이 무엇이고 \
로봇이 어떻게 다뤄야 하는지 답한다.

[해야 할 일]
1. 그려진 모든 번호에 대해 판단을 하나씩 낸다. 번호를 빠뜨리지 않는다.
2. 로봇이 집을 수 있는 **물체**면 is_object=true, 아니면 false.
   - false로 두는 것: 작업대·바닥·배경, 바구니나 상자 자체, 케이블·전선, 그림자, \
물체 위의 무늬·라벨·글자만 잘라낸 조각.
   - 사진에 이미 그려져 있는 다른 모델의 검출 오버레이(사각형 테두리, 라벨 글자, 신뢰도 \
숫자)도 물체가 아니다.
3. **한 물체가 여러 조각으로 나뉜 경우**(뚜껑과 몸통이 따로 잡히는 등): 가장 큰 조각만 \
is_object=true로 두고, 나머지 조각은 is_object=false + part_of=대표번호로 둔다. 같은 물체를 \
두 번 세지 않기 위한 규칙이다.
   **part_of는 같은 물체의 조각에만 쓴다.** 2번에서 물체가 아니라고 판단한 것(배경·케이블·\
오버레이·그림자)은 조각이 아니다 — 물체 옆에 붙어 있어도 part_of=0으로 둔다. 여기서 잘못 \
묶으면 배경이 물체의 일부가 되어 로봇이 엉뚱한 곳을 집는다.
4. 좌표는 답하지 않는다 — 이미 세그멘테이션 모델이 정확히 잡아 놓았다. 당신이 할 일은 \
**그것이 무엇이고 어떻게 다뤄야 하는지**를 말하는 것이다.
5. is_object=true인 번호마다 이름(class_name, name_ko)과 물리 속성(mass_g, fragile, \
deformable, transparent), 그리고 파지력 단계(grip_level)를 사진을 보고 직접 판단한다. \
is_object=false면 이름은 빈 문자열, 속성은 전부 false/0, grip_level은 5로 둔다.
6. is_object=true인 번호마다 reasoning에 판단 근거를 한 문장 적는다(예: "포장에 적힌 \
글자로 확인함", "흔한 물티슈 포장 형태와 크기로 판단"). 작업자가 화면에서 왜 이렇게 \
판단했는지 바로 알 수 있어야 한다 — "물체로 보임" 같은 동어반복은 쓰지 않는다.

[이름 — 미리 주어지는 목록은 없다]
당신에게 등록된 클래스 목록을 주지 않는다. 사진에 실제로 보이는 것을 보고 직접 이름을 짓는다.
- class_name은 영문 소문자 스네이크케이스로, **그 물건이 무엇인지**를 담는다 \
(예: toothpaste, wet_wipes, folding_umbrella).
- 브랜드명이나 포장에 적힌 글자를 그대로 옮기지 않는다. 같은 종류의 물건은 다음 사진에서도 \
같은 이름이 나와야 시스템이 같은 물체로 이어 붙인다("치약"은 제품이 달라도 toothpaste다).
- 종류를 특정할 수 없으면 생김새로 짓는다(blue_plastic_bottle). object_1처럼 아무것도 \
알려 주지 않는 이름은 쓰지 않는다.
- name_ko는 작업자가 화면에서 읽을 한국어 이름이다.

[물리 속성 — 사진에 보이는 근거로만 판단한다]
- mass_g: 물건의 종류와 사진 속 크기로 추정한 무게(g). 짐작조차 어려우면 0으로 둔다.
- fragile: 떨어뜨리거나 세게 쥐면 깨지는 것(유리병, 전구, 계란, 얇은 플라스틱 케이스).
- deformable: 쥐면 모양이 변하는 것(치약 튜브, 비닐 팩, 봉제 인형, 종이컵).
- transparent: 투명·반투명해서 속이 비쳐 보이는 것. depth 센서가 표면을 놓치므로 중요하다.
- **겉면 기준으로 본다.** 종이상자에 든 유리병은 로봇이 상자를 쥐므로 fragile이 아니다.

[grip_level — 로봇이 이 물체를 쥐는 파지력 단계]
여기서 고른 값이 **실제 파지력과 접근 속도로 그대로 들어간다.** 1에서 5까지이며, \
**낮을수록 세게** 쥔다. 힘과 속도의 대략적인 느낌은:
- 1 (매우 강하게, 약 40N): 엄청 단단하고 무거워서 세게 물어도 안전한 물체.
- 2 (강하게, 약 35N): 단단한 보통 물체.
- 3 (보통, 약 30N): 무난한 대부분의 물체. 기본값.
- 4 (약하게, 약 25N): 쥐면 쉽게 눌리는 물체(치약 튜브, 비닐 팩, 봉제 인형)나 \
조심해야 하는 물체.
- 5 (매우 약하게, 약 20N): 깨지기 쉬운 것, 또는 **무엇인지 확신이 서지 않는 물체**.
고르는 순서: fragile=true거나 확신이 없으면 5. 아니고 deformable=true면 4. 아니면 3.
**단계만 고르고 힘(N) 값은 직접 말하지 않는다** — 단계→힘 변환은 시스템이 한다. \
**확신이 없으면 한 단계 조심스러운 쪽(숫자를 더 크게)으로 내린다** — 약하게 쥐면 놓치고 \
다시 잡으면 되지만, 세게 쥐면 물체가 부서지고 되돌릴 수 없다.

[확신이 없으면 검색으로 확인한다]
겉모습만으로 종류를 짐작하면 비슷하게 생긴 다른 물건과 혼동한다(예: 치약 튜브를 초콜릿 바로
착각). 그래서:
1. 포장·라벨에 브랜드명, 제품명, 문구가 **읽힌다면** 먼저 그것을 읽는다.
2. 그 글자만으로 정확한 제품(또는 최소한 제품 종류)을 확신할 수 없으면, 읽은 글자를 검색어로
   `web_search` 도구를 **실제로 호출**해 무엇인지 확인한 뒤에 답한다. 짐작으로 채우지 않는다.
3. 글자가 안 보이거나(라벨이 안 보이는 각도, 흐릿함) 흔한 생김새로 충분히 확신되면
   (물티슈 팩, 우산, 봉제 인형처럼) 검색 없이 바로 답한다 — 매번 검색하지 않는다.
4. 검색해도 특정할 수 없으면, 알아낸 범주까지만 이름에 담고(예: 정확한 브랜드 대신
   toothpaste) confidence를 낮춘다. 지어내지 않는다.
**이 확인은 class_name/name_ko뿐 아니라 mass_g·fragile·deformable·transparent·grip_level에도
그대로 적용된다** — 무엇인지 잘못 알면 속성도 따라서 잘못된다.

[하지 않는 것]
- 3D 좌표·거리·파지 자세는 만들어내지 않는다. 그것은 depth 센서와 시스템이 정한다.
- 사진에 보이지 않는 것(내용물, 유통기한, 재질 표기)을 지어내 속성 판단의 근거로 삼지 않는다.

[지시는 오지 않는다]
사용자가 무엇을 옮기라고 했는지는 이 단계에 주어지지 않는다. 그것을 알면 인지가 지시에
끌려간다 — 실제로 "우산을 옮겨라"를 함께 주자 배경 조각을 우산이라고 답한 적이 있다.
당신은 사진에 실제로 있는 것만 답하고, 어느 것을 옮길지는 다음 단계가 이 답을 보고 정한다.
"""


def build_marks_prompt(mark_ids: list[int],
                       image_size: tuple[int, int] | None = None,
                       domain: str = "general") -> str:
    """번호 목록(+사진 크기, 도메인)을 넣는다. **등록 클래스 어휘는 넣지 않는다** — 이름과
    속성을 모델이 스스로 정하는 것이 이 경로의 전제다(모듈 상단 참조).

    도메인은 지시와 다르다. 도메인은 "이 장면이 어느 시나리오(가정/약국/재활용)인가"라는
    컨텍스트일 뿐 사용자가 어떤 물체를 옮기라고 했는지는 담지 않는다 — 그래서 [지시 없는
    인지] 원칙을 깨지 않으면서, 같은 물건이라도 도메인에 맞는 이름·속성으로 답하게 한다.
    """
    lines = [f"[그려진 번호] {', '.join(str(i) for i in mark_ids)} "
             f"(총 {len(mark_ids)}개 — 전부에 대해 답한다)"]
    if image_size:
        lines.append(f"[사진 크기] {image_size[0]}x{image_size[1]} px")
    if domain and domain != "general":
        context = DOMAIN_CONTEXT_MARKS.get(domain)
        if context:
            lines.append("[장면 맥락(도메인)]\n" + context)
    return "\n".join(lines)


def label_marks(image: pathlib.Path | str, mark_ids: list[int],
                image_size: tuple[int, int] | None = None,
                model: str | None = None, detail: str = "high",
                domain: str = "general") -> VlmMarkScene:
    """번호가 그려진 이미지 → 번호별 판단(이름·속성·파지 단계).

    `detect`와 다른 점이 셋 있다. 좌표를 묻지 않고(SAM이 이미 만들었다), 등록 클래스 어휘를
    주지 않으며(모듈 상단 [속성도 VLM이 판단한다]), **사용자의 지시를 넣지 않는다.** 지시를 함께 주면 인지가 지시에 끌려간다 — 2026-09-07 실측에서 "우산
    왼쪽으로"를 같이 주자 배경 조각 하나를 umbrella로 답했고, 무엇을 물어도 target으로 같은
    번호를 돌려줬다. 어느 물체가 지시 대상인지는 이 결과(물체 목록)를 텍스트로 받는
    `llm_client.plan`이 정한다 — 그것이 원래 그 모듈의 일이고(FR-10/FR-11), 검출을 지시와
    분리해 두면 같은 프레임에 다른 지시를 여러 번 물어도 인지 결과가 흔들리지 않는다.

    `domain`은 지시가 아니라 **시나리오 컨텍스트**(가정/약국/재활용)다. build_marks_prompt가
    이걸 [장면 맥락]으로 넣어, 같은 물건이라도 도메인에 맞는 이름으로 답하게 한다 — 지시를
    넣는 것과는 다르다(사용자가 옮기라고 한 대상은 여기서 정하지 않는다).
    """
    data_url = image if isinstance(image, str) and image.startswith("data:") else encode_image(image)
    client = _client()
    model = model or model_name()
    request = dict(
        model=model,
        instructions=SYSTEM_PROMPT_MARKS,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text",
                 "text": build_marks_prompt(mark_ids, image_size, domain)},
                {"type": "input_image", "image_url": data_url, "detail": detail},
            ],
        }],
        # 겉모습만으로 확신할 수 없는 물체는 포장 글자를 검색어로 실제 검색을 시킨다
        # (SYSTEM_PROMPT_MARKS의 [확신이 없으면 검색으로 확인한다]). 이미지 자체를
        # 검색하지는 못한다 — 모델이 사진에서 읽은 텍스트로 질의를 만든다.
        tools=[{"type": "web_search"}],
        text_format=VlmMarkScene,
    )
    try:
        response = client.responses.parse(**request, temperature=0)
    except Exception as e:
        if "temperature" not in str(e):
            raise
        logger.warning("%s 모델이 temperature를 거부해 기본값으로 재시도한다", model)
        response = client.responses.parse(**request)

    scene = response.output_parsed
    searched = sum(1 for item in response.output if item.type == "web_search_call")

    scene = _normalize_marks(scene, mark_ids)
    objects = [m for m in scene.marks if m.is_object]
    logger.info(
        "VLM 마크 라벨링: model=%s marks=%d objects=%d 검색=%d회 (%s)",
        model, len(scene.marks), len(objects), searched,
        ", ".join(f"{m.mark_id}:{m.class_name}/g{m.grip_level}" for m in objects) or "없음",
    )
    return scene


def _normalize_marks(scene: VlmMarkScene, mark_ids: list[int]) -> VlmMarkScene:
    """존재하지 않는 번호를 지목하거나 같은 번호를 두 번 답하는 것을 걸러 낸다.

    `detect`의 `_normalize`와 같은 성격 — 프롬프트로 부탁하는 것과 출력을 실제 목록과
    대조하는 것은 다른 일이다(grounding.py 참조).
    """
    valid = set(mark_ids)
    seen: set[int] = set()
    marks = []
    for m in scene.marks:
        if m.mark_id not in valid or m.mark_id in seen:
            logger.warning("VLM이 없는/중복된 마크 번호를 답했다: %s — 버린다", m.mark_id)
            continue
        seen.add(m.mark_id)
        if m.part_of not in valid or m.part_of == m.mark_id:
            m.part_of = 0
        if m.is_object:
            m.part_of = 0                    # 대표 조각은 누구의 조각도 아니다
            m.mass_g = max(0.0, float(m.mass_g or 0.0))
            # grip_level은 스키마가 1~5로 강제하지만, 속성과 어긋나는 조합은 남는다.
            # 조심스러운 쪽(숫자를 더 큰 쪽)으로만 내린다 — 프롬프트로 부탁한 것을 여기서
            # 한 번 더 지킨다.
            grip = int(m.grip_level)
            if m.fragile:
                m.grip_level = 5             # 깨질 수 있으면 최대한 약하게
            elif m.deformable and grip < 4:
                m.grip_level = 4             # 눌리는 물체는 약하게
        else:
            m.class_name, m.name_ko, m.reasoning = "", "", ""
            m.mass_g = 0.0
            m.fragile = m.deformable = m.transparent = False
            m.grip_level = 5                 # 물체가 아니므로 쓰이지 않지만 값은 보수적으로
        marks.append(m)
    scene.marks = marks

    # 조각이 가리키는 대표가 실제로 물체여야 한다. 배경끼리 묶이거나 조각의 조각이 되는
    # 것을 막는다 — 잘못 묶이면 마스크에 배경이 딸려 들어가 파지점이 엉뚱해진다.
    objects_by_id = {m.mark_id for m in marks if m.is_object}
    for m in marks:
        if m.part_of and m.part_of not in objects_by_id:
            logger.warning("마크 %s가 물체가 아닌 %s의 조각이라고 답했다 — 무시한다",
                           m.mark_id, m.part_of)
            m.part_of = 0

    missing = valid - seen
    if missing:
        logger.warning("VLM이 답하지 않은 마크: %s — 물체가 아닌 것으로 본다", sorted(missing))
    return scene

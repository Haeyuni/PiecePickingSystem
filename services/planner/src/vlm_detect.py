"""사진 한 장 → VLM에게 "무엇이 있는가"를 묻는다. YOLO 대신 쓰는 인지 경로.

`label_marks(image, mark_ids)` — SAM이 먼저 장면을 조각내고 번호를 그려 준 이미지를 받아
**번호마다 무엇인지와 어떻게 다뤄야 하는지**를 답한다. 좌표는 SAM이, 판단은 VLM이 —
각자 잘하는 쪽을 맡는다. 3D 좌표와 파지 자세는 만들지 않는다. 마스크와 depth로 코드가
계산한다(`perception_test_image.py:mask_3d`와 같은 방식).

한때 VLM에게 물체와 **바운딩박스를 함께** 묻는 `detect(image)` 경로도 있었다("사진을
API로 보내면 좌표를 주는가"를 확인하려던 원래 설계) — gpt-4o에서는 좌표를 못 써서
(docs/vlm_sam_pipeline.md 실측 결과) 실제로는 안 쓰였고, 2026-09-09에 코드 전체(SYSTEM_
PROMPT/VlmObject/VlmScene/detect/_normalize/_pixel_box/to_pixels/known_classes/
build_user_prompt)를 지웠다. 되돌리려면 git 이력에서 이 커밋 이전 버전을 보면 된다.

[속성도 VLM이 판단한다] `label_marks`는 원래 objects.yaml의 등록 클래스 어휘를 프롬프트에
넣고 이름만 받아 왔고, 무게·파손위험·파지 단계는 그 이름으로 objects.yaml에서 조회했다. 지금은
어휘를 주지 않는다 — 사진만 보고 이름과 속성(mass_g/fragile/deformable/transparent)과 파지
단계(grip_level)까지 모델이 정한다. **등록되지 않은 물건도 그 물건에 맞게 다뤄지지만, 파지력을 정하는
값이 모델 출력이 되었다는 뜻이기도 하다**(원래는 NFR-03a가 금지하던 것 — 대신 확신이 없으면
조심스러운 쪽을 고르라고 프롬프트에 박고, `_normalize_marks`가 fragile 조합을 한 번 더
강제한다). 되돌리려면 detector=yolo가 그대로 남아 있다.

[web_search는 pharmacy 도메인에서만 켠다] 한때 확신이 없으면 아무 도메인에서나
`web_search`로 포장 글자를 검색해 확인하게 했다(빨간 치약 튜브를 "chocolate_bar"로
오판한 사례 대응). 그러다 2026-09-09 실물에서 응답이 반복해서 `max_output_tokens`(gpt-4o
Responses 상한 16384)를 다 채우고 잘리는 사고가 나서 검색을 아예 껐는데, **그 뒤에도
web_search 없이 똑같이 잘렸다** — 진짜 범인은 검색이 아니라 스키마의 `mask_poly` 필드였다
(MarkLabel 참조, VLM이 안 쓰이는 그 필드에 좌표를 채우려다 토큰을 다 썼다). 그 필드를
없앤 뒤로는 검색을 다시 켜도 안전하다고 보고, **pharmacy 도메인에만** 다시 켠다 — 약
이름을 잘못 읽으면 안전과 직결되므로(엉뚱한 약을 집어 준다) 정확성이 비용보다 중요하고,
general/recycle은 그 정도 정밀도가 필요 없다(DOMAIN_CONTEXT_MARKS 참조). 검색 횟수는
`MARKS_MAX_WEB_SEARCHES`로 여전히 제한한다 — mask_poly를 없앴어도 검색 결과 자체가
출력 토큰을 먹는 것은 그대로이기 때문이다. 검색이 없는 도메인에서 확신이 없으면
`[하지 않는 것]`대로 범주까지만 답하고 confidence를 낮춘다.

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
from typing import Literal

from openai import OpenAI
from openai.lib._parsing._responses import parse_text, type_to_text_format_param
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# 프롬프트를 고칠 때마다 올린다 (llm_client.PROMPT_VERSION과 같은 이유).
MARKS_PROMPT_VERSION = "vlm-marks-v7"   # label_marks(): 도메인별 이름 규칙 + pharmacy만 web_search + reasoning 없음

# 미설정 시 OPENAI_MODEL을 따라가고, 그것도 없으면 이 값. 이미지 입력이 되는 모델이어야 한다.
DEFAULT_MODEL = "gpt-4o"

# label_marks()가 최대로 받을 출력 토큰. gpt-4o Responses API 자체의 상한이 16384라 이
# 이상은 못 올린다. 2026-09-09 실물: 응답이 반복해서 이 상한을 다 채우고 잘렸는데
# (트레이스 tr-e43294e59d56·tr-979c8be0e88b·tr-eda307cfa42b — perception의 observe가
# 그 뒤로 막혀 world_state가 계속 낡은 채로 남았다), 원인은 web_search가 아니라
# MarkLabel의 mask_poly 필드였다(그 필드 참조). 그 필드를 없앤 뒤로는 이 상한에 닿을
# 일이 거의 없지만, 안전판으로 그대로 둔다.
MARKS_MAX_OUTPUT_TOKENS = 16384

# pharmacy 도메인에서 web_search를 허용하는 요청 하나(마크 전부 합쳐)당 총 호출 횟수
# 상한. mask_poly를 없애 토큰 예산에 여유가 생겼어도, 검색 결과(질의문 + 요약)가
# 출력에 그대로 실리는 것은 그대로라 무제한 허용하면 다시 상한을 다 채울 수 있다.
MARKS_MAX_WEB_SEARCHES = 2

# 시나리오 도메인의 한국어 표기. 프롬프트의 [장면 맥락]에 넣어 VLM이 도메인에 맞게 답하게 한다.
# 도메인별 **이름 규칙** — build_marks_prompt가 [장면 맥락]으로 넣는다 (vlm-detect, label_marks).
# `home`(가정)은 special 규칙이 없어 general과 같으므로 별도 도메인이 아니다 — 가정 버튼도
# general로 온다.
# 같은 물체도 도메인에 따라 이름에 담아야 하는 정보가 다르다:
# - 약국: 어떤 약인지가 중요하므로 **정확한 약품명**(성분/제품명)을 읽어 쓴다.
# - 재활용: 어느 재질로 분리할지가 중요하므로 **재질이 이름에 드러나야** 한다(단, 음식물은
#   재질이 아니라 food_waste라는 별도 카테고리).
# 지시(command_text)는 여기 오지 않는다 — 도메인은 시나리오 컨텍스트일 뿐이다(D-1).
DOMAIN_CONTEXT_MARKS = {
    "pharmacy": f"""\
이 사진은 **약국** 시나리오다.
- 이름은 **정확한 약품명**이어야 한다. 포장·포일·설명서·병 라벨에 적힌 성분명/제품명을 \
읽어 class_name과 name_ko에 그대로 쓴다 (예: ibuprofen / '이부프로펜정', tylenol / \
'타이레놀정').
- '약', '알약', 'painkiller'처럼 **종류로 얼버무리지 않는다** — 어느 약인지가 분류의 대상이다.
- **이 도메인에서는 `web_search`가 주어진다** (최대 {MARKS_MAX_WEB_SEARCHES}회, 요청 전체
합산). 포장 글자가 흐려 정확한 약품명을 못 읽으면, 읽히는 부분(성분명 일부·모양·색)을
검색어로 실제로 호출해 확인한 뒤에 답한다 — 약은 잘못 집으면 안전 문제로 이어지므로
짐작으로 채우지 않는다. 그래도 특정할 수 없으면 'unknown_medicine'처럼 보수적인 이름 +
낮은 confidence로 답한다. 여러 개가 애매하면 가장 확신이 없는 것부터 검색한다.
- 포장 형태(블리스터·병·봉투)나 재질은 이름 대신 속성으로 답한다.""",
    "recycle": """\
이 사진은 **재활용(분리수거)** 시나리오다.
- 이름에 **재질이 반드시 드러나야** 한다. 재질 + 형태 조합으로 지는다 \
(예: pet_plastic_bottle, aluminum_can, glass_bottle, paper_box, hdpe_container, can).
- name_ko도 재질을 담는다 (예: 'PET 페트병', '알루미늄 캔', '유리병', '종이 상자').
- **예외: 음식물(음식물쓰레기)은 재질이 아니라 별도 카테고리다.** 먹다 남은 음식·과일 \
껍질·채소 찌꺼기처럼 먹는 것에서 나온 잔재는 재질을 붙이지 말고 class_name='food_waste', \
name_ko='음식물쓰레기'로 통일한다 — 무슨 음식이었는지로 세분화하지 않는다.
- 재질 표시(플라스틱 재질 번호, 캔·병 각인)를 읽고 판단한다. 검색하지 말고(web_search 없음) \
보이는 표시만으로 판단한다.
- 재질을 특정할 수 없으면 unknown_material 같은 보수적인 이름 + 낮은 confidence로 답한다 — \
지어내지 않는다.""",
}


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


# =============================================================================
# 마크 라벨링 (Set-of-Mark) — SAM이 먼저, VLM이 나중
# =============================================================================
#
# VLM에게 좌표를 직접 물어보는 방식은 쓰지 않는다 — **gpt-4o는 그 좌표를 못 맞춘다**
# (2026-09-07 실측: 박스가 100px 단위로 뭉개지고 다른 물체 위에 얹혔다. 이 방식은
# 2026-09-09에 코드에서 걷어냈다 — docs/vlm_sam_pipeline.md 참조). 좌표를 못 맞추는
# 모델에게 좌표를 묻는 대신, 순서를 뒤집는다:
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
    # **여기 reasoning(판단 근거 한 줄) 필드를 넣지 않는다.** 2026-09-09까지는 있었고
    # 화면(ApprovalModal/ObjectList/DatasetPage)에 실제로 표시됐지만, 사용자 요청으로
    # 이 기능 자체를 없앴다 — DetectedObject.msg/dataset_items.reasoning/프론트 표시까지
    # 전부 함께 뗐다(006_drop_reasoning.sql 참조). 다시 붙이려면 이 필드부터 되돌리고
    # 그 하위 배선을 전부 되짚어야 한다.
    # **여기 mask_poly 필드를 넣지 않는다.** 한때 "YOLO 학습용 마스크 윤곽선"이라는 설명으로
    # 있었는데, 이게 이 스키마(text_format=VlmMarkScene)로 구조화 출력을 걸면 **VLM이 그
    # 필드도 직접 채워야 하는 값**이 된다 — 실제 윤곽선은 SAM이 계산해 별도 필드
    # (mask_polys, sam_vlm.py→app.py._marks_to_yolo)로 채워지고 이 필드는 어디서도 읽지
    # 않는데도(2026-09-09 확인: 코드 전체에 mark.mask_poly를 읽는 곳이 없다), 마크마다
    # 자릿수 많은 좌표 배열을 "그럴듯하게" 만들어내려 하면서 output_tokens을 순식간에
    # 다 써버려 응답이 잘렸다(status=incomplete, output_tokens=16384/16384, 2026-09-09
    # 실물 — 4마크 요청에서 재현, web_search를 꺼도 그대로 재현돼 원인이 검색이 아니라
    # 이 필드였음을 확인했다). 정말 필요없는 필드를 스키마에 넣으면 안 쓰는 데이터 때문에
    # 모델이 큰 값을 지어내다 정작 필요한 필드(class_name 등)를 낼 자리가 없어질 수 있다는
    # 교훈 — 서버가 이미 아는 값은 스키마에 넣지 않는다.


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

[검색 없이 사진만으로 판단한다]
겉모습만으로 종류를 짐작하면 비슷하게 생긴 다른 물건과 혼동할 수 있다(예: 치약 튜브를
초콜릿 바로 착각). 그렇더라도 **검색 도구는 쓸 수 없다** — 이 판단은 오직 사진에 보이는
것만으로 한다:
1. 포장·라벨에 브랜드명, 제품명, 문구가 **읽힌다면** 그것을 이름에 반영한다.
2. 읽은 글자나 생김새만으로 정확한 제품을 확신할 수 없으면, 알아낸 범주까지만 이름에 담고
   (예: 정확한 브랜드 대신 toothpaste) confidence를 낮춘다. **짐작을 확신처럼 답하지
   않는다** — 모르면 모른다고 낮은 confidence로 드러내는 편이 틀린 이름을 단정하는 것보다
   낫다.
**이 판단은 class_name/name_ko뿐 아니라 mass_g·fragile·deformable·transparent·grip_level에도
그대로 적용된다** — 무엇인지 잘못 알면 속성도 따라서 잘못된다. 확신이 없는 속성은 항상
조심스러운 쪽(fragile 쪽, grip_level을 크게)으로 둔다.

[하지 않는 것]
- 3D 좌표·거리·파지 자세는 만들어내지 않는다. 그것은 depth 센서와 시스템이 정한다.
- 사진에 보이지 않는 것(내용물, 유통기한, 재질 표기)을 지어내 속성 판단의 근거로 삼지 않는다.
- 검색하지 않는다. 이 판단은 사진에 보이는 것만으로 낸다(위 [검색 없이 사진만으로 판단한다]).

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
        text_format=VlmMarkScene,
        max_output_tokens=MARKS_MAX_OUTPUT_TOKENS,
    )
    # web_search는 pharmacy에만 준다 — SYSTEM_PROMPT_MARKS [web_search는 pharmacy
    # 도메인에서만 켠다] 참조. 다른 도메인은 검색 없이도 충분하고, 검색 결과가 출력
    # 토큰을 먹으므로 꼭 필요한 곳에만 쓴다. max_tool_calls로 이 요청 전체의 호출
    # 총횟수를 잘라 예산을 한 번 더 지킨다(openai 3.8.0 ResponseCreateParams.max_tool_calls).
    if domain == "pharmacy":
        request["tools"] = [{"type": "web_search"}]
        request["max_tool_calls"] = MARKS_MAX_WEB_SEARCHES
    # **`.responses.parse()`가 아니라 `.responses.create()`(원본)를 쓴다.** `.parse()`는
    # 응답을 받자마자 **그 안에서** text_format으로 JSON 파싱까지 해 버린다(openai
    # SDK 3.8.0, lib/_parsing/_responses.py의 parse_response). 그래서 응답이 잘려 JSON이
    # 깨지면 `.parse()` 호출 자체가 pydantic ValidationError를 던지며 죽고, **우리
    # 코드는 status나 usage를 볼 기회조차 없다** — 아래 status/incomplete_details 확인은
    # 그 뒤에 있어도 절대 실행되지 않는 죽은 코드였다(2026-09-08, 잘림이 실제로
    # max_output_tokens 때문인지 확인하려다 발견). create()로 원본 응답을 먼저 받아
    # status·usage(실제 소비 토큰)를 무조건 로그로 남긴 뒤에, 우리가 직접 파싱한다.
    request.pop("text_format", None)
    request["text"] = {"format": type_to_text_format_param(VlmMarkScene)}
    try:
        response = client.responses.create(**request, temperature=0)
    except Exception as e:
        if "temperature" not in str(e):
            raise
        logger.warning("%s 모델이 temperature를 거부해 기본값으로 재시도한다", model)
        response = client.responses.create(**request)

    usage = response.usage
    logger.info(
        "VLM 마크 라벨링 사용량: model=%s status=%s input_tokens=%d output_tokens=%d "
        "(reasoning=%d) total_tokens=%d",
        model, response.status, usage.input_tokens, usage.output_tokens,
        usage.output_tokens_details.reasoning_tokens if usage.output_tokens_details else 0,
        usage.total_tokens,
    )

    # **잘린 응답은 우리가 직접 파싱을 시도하기 전에 여기서 잡는다.** status가
    # incomplete면 스키마가 우연히 맞아떨어져도 마지막 몇 개 마크가 통째로 빠졌을 수
    # 있다 — "번호를 빠뜨리지 않는다"는 프롬프트 전제가 깨진 채로 조용히 넘어가는 것이
    # 파싱 실패보다 나쁘다. reason이 실제로 'max_output_tokens'인지 여기서 확정 확인된다
    # (더 이상 짐작이 아니다 — usage.output_tokens가 요청한 한도에 닿았는지 로그로 보인다).
    if response.status == "incomplete":
        reason = getattr(response.incomplete_details, "reason", "unknown")
        raise RuntimeError(
            f"VLM 응답이 중간에 잘렸다 (status=incomplete, reason={reason}, "
            f"output_tokens={usage.output_tokens}/{MARKS_MAX_OUTPUT_TOKENS}) — "
            f"마크 {len(mark_ids)}개")

    output_text = "".join(
        item.text for out in response.output if out.type == "message"
        for item in out.content if item.type == "output_text")
    scene = parse_text(output_text, text_format=VlmMarkScene)

    scene = _normalize_marks(scene, mark_ids)
    objects = [m for m in scene.marks if m.is_object]
    logger.info(
        "VLM 마크 라벨링: model=%s marks=%d objects=%d (%s)",
        model, len(scene.marks), len(objects),
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
            m.class_name, m.name_ko = "", ""
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

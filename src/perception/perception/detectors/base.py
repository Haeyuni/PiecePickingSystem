"""검출기 경계. `node.py`가 "무엇이 어디에 있는가"를 얻는 유일한 창구다.

**왜 나눴나.** YOLO가 등장하던 곳은 `node.py._build_detections` 한 군데뿐이었고, 그 위(캡처와
입력 가드)와 아래(마스크 → depth → base 3D 변환, 속성 조회, object_id 유지, 발행)는 검출기가
무엇이든 그대로다. 그 한 군데만 갈아끼울 수 있게 하면 SAM+VLM 경로를 붙이면서 나머지를
건드리지 않아도 되고, 같은 프레임에 두 검출기를 번갈아 돌려 비교하는 것도 된다
(`docs/on-demand-perception.md`).

**계약**: `detect()`는 컬러 프레임 하나(+선택적 `trace_id`, `on_phase`)를 받아
`(detections, debug_bgr)`를 돌려준다. `trace_id`는 하위 서비스 로그를 명령과 잇기 위한
것이고, 검출 결과에 영향을 주지 않는다.

`on_phase`는 진행 단계를 알리는 선택적 콜백이다("segmenting", "labeling" 등 검출기가
정하는 토큰 문자열 하나를 받는다). SAM+VLM처럼 10초 안팎 걸리는 검출기가 액션 서버의
피드백(`Observe.action`의 phase)으로 이어 줄 수 있게 하기 위한 것이지 계약의 필수
부분은 아니다 — 빠른 검출기는 호출하지 않아도 된다.

- `detections`: 아래 `detection()`이 만드는 dict의 리스트
- `debug_bgr`: 사람이 눈으로 보라고 그린 오버레이(BGR ndarray) 또는 None.
  `/perception/debug_image`로 나간다.

검출기가 **하지 않는 것**: 3D 좌표, 파지 가능 판정, object_id 부여. 전부 마스크가 나온
뒤의 일이고 검출 방법과 무관하다 — 검출기마다 다시 구현하면 그때부터 갈라진다.

**속성만 예외다.** 원래는 노드가 class_name으로 objects.yaml/DB를 조회하는 것이 전부였는데,
SAM+VLM 경로는 사진을 본 모델이 이름과 함께 속성(무게·파손위험·파지 프로파일)을 답하므로
그 값이 검출기에서 나온다. `detection(..., attrs=...)`으로 실어 보내고, 안 실으면(YOLO)
노드가 예전 조회 경로를 그대로 탄다.
"""
from typing import Callable, Protocol

import numpy as np

OnPhase = Callable[[str], None]


def detection(class_name: str, confidence: float, mask: np.ndarray | None,
              attrs: dict | None = None) -> dict:
    """검출 하나. mask는 **컬러 프레임 해상도**의 bool 배열이어야 한다.

    추론 해상도(YOLO의 640 등)로 돌려주면 마스크와 depth의 픽셀이 어긋나 3D가 조용히
    틀린다 — 되돌리는 것은 검출기의 책임이다(`mask_utils.resize_mask`).
    mask=None은 "이 물체를 봤지만 마스크를 못 냈다"는 뜻이고, 호출자가 파지 불가로 처리한다.

    `attrs`는 검출기가 **속성까지 알아냈을 때만** 채운다(SAM+VLM 경로 — 사진을 본 모델이
    무게·파손위험·파지 프로파일을 함께 답한다). None이면 노드가 예전처럼 class_name으로
    `AttributeSource`(object_attributes → objects.yaml → fallback)를 조회한다. YOLO는 이름만
    알므로 None을 낸다. 키는 `AttributeSource.attributes()`와 같아야 한다.
    """
    return {"class_name": class_name, "confidence": float(confidence), "mask": mask,
            "attrs": attrs}


class Detector(Protocol):
    def detect(self, color_bgr: np.ndarray, trace_id: str = "",
               on_phase: OnPhase | None = None) -> tuple[list[dict], np.ndarray | None]:
        ...

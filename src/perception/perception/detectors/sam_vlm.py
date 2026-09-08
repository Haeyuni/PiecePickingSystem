"""SAM+VLM 검출기: 장면 전체를 조각내고(SAM) 번호에 이름을 붙인다(VLM).

YOLO와 달리 학습한 클래스에 갇히지 않는다 — 처음 보는 물건도 이름이 붙는다.

**objects.yaml을 지나지 않는다.** 이름뿐 아니라 무게·파손위험·변형·투명 여부와 파지
프로파일까지 사진을 본 VLM이 답하고, 그 값이 그대로 `DetectedObject`에 실린다
(`attr_source=llm_suggested`, `needs_confirmation=true`). 등록되지 않은 물건도 그 물건에
맞게 다뤄지지만 파지력을 정하는 값이 모델 출력이 되었다는 뜻이기도 하다 —
`planner/src/vlm_detect.py` 상단의 [속성도 VLM이 판단한다]에 그 맞바꿈을 적어 뒀다.

**VLM은 여기서 부르지 않고 planner에 물어본다.** planner는 ROS를 모르는 서비스이고
OPENAI_API_KEY도 거기에만 있다 — perception이 직접 부르면 ROS 컨테이너에 API 키와 인터넷
의존이 들어간다(docs/on-demand-perception.md 4절). SAM은 로컬 GPU라 여기서 돈다.

**사용자의 지시는 넘기지 않는다.** 지시를 함께 주면 인지가 지시에 끌려간다 — 실측에서
"우산 왼쪽으로"를 같이 주자 배경 조각을 umbrella라고 답했다. 어느 물체를 옮길지는 이
결과를 텍스트로 받는 planner의 계획 단계가 정한다(FR-10/FR-11).
"""
import logging

import numpy as np

from .. import sam_marks
from .base import detection

logger = logging.getLogger(__name__)


class SamVlmDetector:
    def __init__(self, weights: str, planner_url: str, device: str | None = None,
                 points_stride: int = 16, min_area: float = 0.002, max_area: float = 0.35,
                 max_marks: int = 20, timeout_s: float = 120.0, logger_=None):
        self.weights = weights
        self.planner_url = planner_url.rstrip("/")
        self._device = device
        self._points_stride = points_stride
        self._min_area = min_area
        self._max_area = max_area
        self._max_marks = max_marks
        self._timeout_s = timeout_s
        self._log = logger_ or logger
        # 마지막 관측의 번호별 마스크. 재관측(sam_reprompt)이 이어받는다.
        self.last_marks: list[np.ndarray] = []
        self.last_objects: list[dict] = []

    def describe(self) -> str:
        return f"{self.weights} → {self.planner_url}/internal/label-marks"

    def detect(self, color_bgr: np.ndarray, trace_id: str = "",
               on_phase=None) -> tuple[list[dict], np.ndarray | None]:
        if on_phase:
            on_phase("segmenting")
        masks, elapsed = sam_marks.segment_everything(
            color_bgr, self.weights, self._device, self._points_stride,
            self._min_area, self._max_area, self._max_marks)
        self._log.info(f"SAM 전체 분할: 마크 {len(masks)}개 ({elapsed:.1f}s)")
        if not masks:
            # 마크가 하나도 없으면 "아무것도 없다"가 아니라 "못 봤다"이다. 빈 목록을
            # 내보내면 상위가 "물체 없음"으로 판단한다 — 호출자가 구분할 수 있게 둔다.
            self.last_marks, self.last_objects = [], []
            return [], None

        marked = sam_marks.draw_marks(color_bgr, masks)
        if on_phase:
            on_phase("labeling")
        labels = self._label(marked, list(range(1, len(masks) + 1)), trace_id)

        def rejected(piece, parent):
            self._log.warning(f"마크 {piece}는 {parent}의 조각으로 보기 어렵다 — 합치지 않는다")

        objects = sam_marks.merge_marks(labels, masks, on_reject=rejected)
        self.last_marks, self.last_objects = masks, objects
        self._log.info(f"VLM 라벨링: 마크 {len(masks)}개 중 물체 {len(objects)}개")

        # attrs가 함께 나간다 — 이 경로에서는 무게·파손위험·파지 프로파일도 VLM이 답한다
        # (objects.yaml 조회 없음, base.py의 detection() 주석 참조).
        detections = [detection(o["class_name"], o["confidence"], o["mask"], o["attrs"])
                      for o in objects]
        return detections, marked

    def _label(self, marked_bgr: np.ndarray, mark_ids: list[int], trace_id: str) -> list[dict]:
        """번호를 그린 프레임을 planner로 보내 번호별 판단을 받는다."""
        import cv2
        import httpx

        ok, buffer = cv2.imencode(".png", marked_bgr)
        if not ok:
            raise RuntimeError("번호 오버레이를 PNG로 인코딩하지 못했다")

        response = httpx.post(
            f"{self.planner_url}/internal/label-marks",
            files={"image": ("marks.png", buffer.tobytes(), "image/png")},
            data={"mark_ids": ",".join(str(i) for i in mark_ids), "trace_id": trace_id},
            timeout=self._timeout_s,
        )
        response.raise_for_status()
        return response.json()["marks"]

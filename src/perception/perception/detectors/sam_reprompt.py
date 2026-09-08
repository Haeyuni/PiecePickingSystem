"""재관측 검출기: 직전에 알아낸 물체가 지금도 거기 있는지 확인한다 (D-1).

**VLM을 부르지 않는다.** 명령당 VLM 1회 규칙이 여기서 지켜진다 — 스텝 사이마다 다시 물으면
"전부 옮겨" 한 번에 API 왕복이 물체 수만큼 붙고, 인지 지연이 실행 시간을 넘어선다.
직전 관측의 라벨을 그대로 물려받고, SAM에게는 "여기를 잡아라"만 시킨다.

    직전 물체가 차지했던 3D 범위 → (지금 TCP 자세로) 카메라 좌표 → 픽셀 박스
      → 그 박스를 프롬프트로 SAM 1패스
      → 라벨은 직전 것을 상속

**점이 아니라 박스다.** 중심점 하나로는 SAM이 물체 대신 그 위의 무늬를 잡는다 — 실측 IoU가
물티슈 0.07, 섬유탈취제 0.03이었다. 같은 물체에 박스를 주면 0.99 / 0.98이 나온다
(sam_marks.segment_at_boxes 주석). 그래서 직전 관측이 마스크 테두리의 3D 점들
(`mask_utils.mask_extent_3d`)을 남겨 두고, 여기서 그것을 되돌려 박스를 만든다.

eye-in-hand라 카메라가 스텝마다 다른 곳에 있다. 2D 마스크를 그대로 비교하는 대신 base
좌표를 거쳐 되돌리는 이유가 그것이다 — 시점이 바뀌어도 성립한다.

**여기서 판단하지 않는 것**: 되찾은 마스크가 정말 그 물체인지. 프롬프트가 배경을 찍었으면
배경 마스크가 나온다. 그 검사는 3D 좌표가 나온 뒤(node._enrich) 직전 위치와 대조해서 한다
(D-2) — `expected` 목록이 그 대조의 기준이다.
"""
import logging

import numpy as np
from perception_common import geometry

from .. import sam_marks
from .base import detection

logger = logging.getLogger(__name__)


class SamRepromptDetector:
    def __init__(self, weights: str, device: str | None = None, logger_=None):
        self.weights = weights
        self._device = device
        self._log = logger_ or logger
        # prime()이 채운다. detect()가 돌려주는 검출과 순서가 1:1로 맞는다.
        self.expected: list[dict] = []

    def describe(self) -> str:
        return f"{self.weights} (재투영 프롬프트)"

    def prime(self, previous: list[dict], base2gripper, gripper2camera, intrinsics) -> int:
        """직전 관측의 물체들을 지금 프레임의 박스로 되돌린다. 프롬프트로 쓸 개수를 반환한다.

        previous의 각 항목은 `class_name / confidence / attrs / position_base_mm(x,y,z) /
        extent_base_mm([(x,y,z), ...])`를 갖는다.
        **base2gripper는 지금 TCP 자세로 만든 것**이어야 한다 — 그것이 시점 변화를 흡수하는
        지점이다.

        화면 밖이나 카메라 뒤로 간 물체는 프롬프트를 만들 수 없으므로 뺀다. 뺀 것은 이번
        관측에서 사라진 것으로 처리된다(팔이 시야를 가렸거나 이미 옮겨졌다).
        """
        width, height = intrinsics["width"], intrinsics["height"]
        self.expected = []

        for item in previous:
            name = item.get("class_name")
            extent = item.get("extent_base_mm") or []
            if not extent:
                # 범위를 안 남긴 물체는 박스를 만들 수 없다. 중심점으로 대신하지 않는다 —
                # 점 프롬프트는 무늬를 잡는다(모듈 주석). 이번 관측에서 빼는 편이 낫다.
                self._log.warning(f"{name}: 직전 관측에 3D 범위가 없다 — 이번 관측에서 뺀다")
                continue

            pixels = []
            behind = False
            for point_base in extent:
                point_cam = geometry.camera_from_base(point_base, base2gripper, gripper2camera)
                pixel = geometry.project_to_pixel(point_cam, intrinsics)
                if pixel is None:
                    behind = True
                    continue
                pixels.append(pixel)
            if behind and not pixels:
                self._log.warning(f"{name}: 카메라 뒤로 갔다 — 이번 관측에서 뺀다")
                continue

            us = [u for u, _ in pixels]
            vs = [v for _, v in pixels]
            # 일부만 화면 밖으로 나간 물체는 자르지 않고 프레임 안으로 클립해서 쓴다 —
            # 보이는 부분이라도 되찾는 편이 통째로 잃는 것보다 낫다.
            x1, x2 = int(max(0, min(us))), int(min(width - 1, max(us)))
            y1, y2 = int(max(0, min(vs))), int(min(height - 1, max(vs)))
            if x2 <= x1 or y2 <= y1:
                self._log.warning(f"{name}: 화면 밖으로 나갔다 — 이번 관측에서 뺀다")
                continue

            self.expected.append({**item, "box_xyxy": [x1, y1, x2, y2]})

        return len(self.expected)

    def detect(self, color_bgr: np.ndarray, trace_id: str = "",
               on_phase=None) -> tuple[list[dict], np.ndarray | None]:
        if not self.expected:
            return [], None

        if on_phase:
            on_phase("segmenting")
        boxes = [item["box_xyxy"] for item in self.expected]
        masks, elapsed = sam_marks.segment_at_boxes(
            color_bgr, self.weights, boxes, self._device)
        self._log.info(f"SAM 재투영 프롬프트: 박스 {len(boxes)}개 ({elapsed:.1f}s)")

        detections = []
        for item, mask in zip(self.expected, masks):
            # 빈 마스크는 mask=None으로 낸다 — 노드가 REASON_NO_MASK로 처리하고, 좌표를
            # 지어내지 않는다. 라벨과 함께 **속성도** 물려받는다: VLM을 다시 부르지
            # 않으므로 여기서 잃으면 재관측한 물체만 속성 조회 경로로 떨어져 grip_level이
            # 스텝 사이에 달라진다.
            detections.append(detection(item["class_name"], item["confidence"],
                                        mask if mask.any() else None, item.get("attrs")))
        return detections, self._draw(color_bgr, masks)

    def _draw(self, color_bgr, masks):
        """되찾은 마스크와 그것을 부른 박스를 함께 그린다 — 박스가 물체를 안 감쌌다면
        여기서 눈으로 바로 보인다."""
        import cv2

        canvas = color_bgr.copy()
        for index, (item, mask) in enumerate(zip(self.expected, masks)):
            color = sam_marks.COLORS[index % len(sam_marks.COLORS)]
            x1, y1, x2, y2 = item["box_xyxy"]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)
            if mask.any():
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(canvas, contours, -1, color, 2)
            cv2.putText(canvas, item["class_name"], (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        return canvas

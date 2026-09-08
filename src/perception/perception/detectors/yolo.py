"""YOLO11-seg 검출기. 온디맨드 전환 전까지 쓰던 경로를 그대로 옮긴 것이다.

SAM+VLM으로 바꾼 뒤에도 지우지 않는다 — `detector` 파라미터로 돌아올 수 있어야 새 경로가
안 될 때 물러설 곳이 있고, 같은 프레임에서 두 검출기의 마스크 IoU를 재는 것도 이걸로 한다
(`docs/on-demand-perception.md` 10절).
"""
import numpy as np

from .. import mask_utils
from .base import detection


class YoloDetector:
    def __init__(self, model_path: str, attributes, conf: float = 0.25,
                 imgsz: int = 640, device: str | None = None):
        # import 비용이 커서 노드 생성 시점에만 낸다 (모듈 import 시점이 아니라)
        from ultralytics import YOLO

        self._model = YOLO(model_path)
        self._attributes = attributes
        self._conf = conf
        self._imgsz = imgsz
        self._device = device
        self.model_path = model_path

    def describe(self) -> str:
        return f"{self.model_path} (task={self._model.task}, names={self._model.names})"

    def detect(self, color_bgr: np.ndarray, trace_id: str = "",
               on_phase=None) -> tuple[list[dict], np.ndarray | None]:
        if on_phase:
            on_phase("segmenting")
        result = self._model.predict(color_bgr, conf=self._conf, imgsz=self._imgsz,
                                     device=self._device, verbose=False)[0]
        shape_hw = color_bgr.shape[:2]
        count = 0 if result.boxes is None else len(result.boxes)
        detections = []

        for index in range(count):
            # **모델 라벨을 그대로 쓰지 않는다.** objects.yaml의 model_labels 표를 지나야
            # 정식 class_name이 된다 — 안 지나면 네일이 매번 미확인 신규품목으로 떨어지고
            # fragile 프로파일이 강제된다 — 지금은 grip_level 5(가장 약하게)로 강제된다
            model_label = result.names[int(result.boxes.cls[index])]
            class_name = self._attributes.class_name(model_label)
            confidence = float(result.boxes.conf[index])

            if result.masks is None:
                detections.append(detection(class_name, confidence, None))
                continue

            mask = mask_utils.resize_mask(
                result.masks.data[index].cpu().numpy() > 0.5, shape_hw)
            detections.append(detection(class_name, confidence, mask))

        return detections, result.plot()

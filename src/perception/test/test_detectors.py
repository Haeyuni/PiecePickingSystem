"""검출기 경계 계약 (detectors/base.py).

노드가 검출기에게 기대하는 것만 본다 — 모델 성능이 아니라 **모양**이다. 이게 깨지면
`node._enrich`가 조용히 틀린 3D를 낸다(마스크가 추론 해상도로 오면 depth와 픽셀이 어긋난다).
"""
import numpy as np

from perception.detectors import detection
from perception.detectors.base import Detector


class FakeDetector:
    """실제 모델 없이 계약만 지키는 검출기. Protocol 준수 확인용이자 노드 테스트의 스텁."""

    def __init__(self, items):
        self._items = items

    def detect(self, color_bgr, trace_id="", on_phase=None):
        return list(self._items), None


def test_detection_shape():
    mask = np.zeros((4, 6), dtype=bool)
    d = detection("toothpaste", 0.9, mask)

    assert set(d) == {"class_name", "confidence", "mask", "attrs"}
    assert d["class_name"] == "toothpaste"
    assert isinstance(d["confidence"], float)
    assert d["mask"] is mask


def test_detection_accepts_missing_mask():
    """마스크를 못 낸 검출도 버리지 않는다 — 노드가 REASON_NO_MASK로 처리한다."""
    assert detection("tape", 0.5, None)["mask"] is None


def test_detection_coerces_confidence():
    """numpy 스칼라가 그대로 넘어오면 메시지 채울 때 터진다."""
    d = detection("tape", np.float32(0.25), None)

    assert type(d["confidence"]) is float


def test_fake_detector_satisfies_protocol():
    detector: Detector = FakeDetector([detection("tape", 0.5, None)])
    detections, debug = detector.detect(np.zeros((4, 6, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert debug is None


def test_mask_must_match_color_resolution():
    """계약 문서화 — 검출기는 컬러 해상도로 되돌린 마스크를 준다."""
    color = np.zeros((8, 10, 3), dtype=np.uint8)
    detections, _ = FakeDetector(
        [detection("tape", 0.5, np.zeros(color.shape[:2], dtype=bool))]).detect(color)

    assert detections[0]["mask"].shape == color.shape[:2]
    assert detections[0]["mask"].dtype == bool

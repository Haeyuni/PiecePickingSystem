"""음성 인터페이스 (6단계, 2차 범위). faster-whisper(CTranslate2) small/base, CPU INT8 양자화.

브라우저 마이크 입력 → 오디오 blob → 텍스트 변환 → 기존 명령 입력 경로에 그대로 주입 (FR-23).
STT 전용 처리 경로를 별도로 두지 않는다 — 이 모듈은 텍스트로 바꿔서 돌려줄 뿐, 실행하지
않는다(FR-24: 사용자가 확인·수정한 뒤 `/api/commands`를 그대로 호출한다).
"""
import logging
import os
import tempfile

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# 신뢰도가 이 미만이면 low_confidence=true (웹_인터페이스_정의서.md 2.2절, FR-25).
LOW_CONFIDENCE_THRESHOLD = 0.8

_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    """첫 호출에만 모델을 로드한다 — import 시점에 로드하면 모델을 안 쓰는 mock 개발
    환경에서도 매번 다운로드/로드 지연이 생긴다."""
    global _model
    if _model is None:
        size = os.environ.get("STT_MODEL_SIZE", "small")
        logger.info("faster-whisper 모델 로드 중 (size=%s, CPU int8)", size)
        _model = WhisperModel(size, device="cpu", compute_type="int8")
    return _model


def transcribe(audio_bytes: bytes) -> tuple[str, float]:
    """오디오(webm/wav 등) → (인식된 텍스트, 신뢰도 0~1).

    faster-whisper는 파일 경로만 받으므로 임시 파일로 내린다 — ffmpeg(내부적으로
    av 패키지가 씀)가 컨테이너 포맷을 알아서 디코드하므로 webm/wav 어느 쪽이든 된다.
    신뢰도는 세그먼트별 `avg_logprob`(로그 확률)을 `exp`로 0~1 근사치로 바꾼 값의
    평균이다 — Whisper가 confidence를 직접 내지 않아서 나온 근사치다.
    """
    import math

    model = _get_model()
    with tempfile.NamedTemporaryFile(suffix=".audio") as f:
        f.write(audio_bytes)
        f.flush()
        segments, _info = model.transcribe(f.name, language="ko")
        segments = list(segments)

    text = "".join(s.text for s in segments).strip()
    if not segments:
        return "", 0.0
    confidence = sum(math.exp(s.avg_logprob) for s in segments) / len(segments)
    return text, min(1.0, max(0.0, confidence))

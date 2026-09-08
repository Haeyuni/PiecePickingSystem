"""POST /api/stt — 음성 → 텍스트 (웹_인터페이스_정의서.md 2.2절, FR-23~25).

실행하지 않는다. 텍스트로 바꿔서 돌려줄 뿐이다 — 사용자가 확인/수정한 뒤
2.1절의 POST /api/commands를 그대로 호출한다.
"""
import logging

from fastapi import APIRouter, UploadFile
from fastapi.responses import JSONResponse

from .. import stt

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/stt")
async def speech_to_text(audio: UploadFile):
    data = await audio.read()
    if not data:
        return JSONResponse(
            status_code=400,
            content={"schema_version": "1.0.0",
                     "error": {"code": "EMPTY_AUDIO", "message": "오디오 본문이 비어 있습니다"}},
        )

    try:
        text, confidence = stt.transcribe(data)
    except Exception as e:
        logger.exception("STT 변환 실패")
        return JSONResponse(
            status_code=503,
            content={"schema_version": "1.0.0",
                     "error": {"code": "STT_UNAVAILABLE", "message": f"음성 인식에 실패했습니다: {e}"}},
        )

    return {
        "schema_version": "1.0.0",
        "recognized_text": text,
        "confidence": round(confidence, 3),
        # FR-25: 무음/저신뢰는 재입력을 유도한다 — 인식된 텍스트가 있어도 명령으로
        # 자동 전달하지 않는다(호출부가 이 값을 보고 판단).
        "low_confidence": confidence < stt.LOW_CONFIDENCE_THRESHOLD or not text,
    }

"""웨이크워드 감지 브리지 전용 내부 엔드포인트 (tools/voice/wakeword_bridge.py 참조).

브라우저 SpeechRecognition은 "hello rokey"를 신뢰성 있게 못 잡았다(2026-09-09 실측).
로봇 PC 마이크를 openwakeword 커스텀 모델로 직접 듣는 네이티브 프로세스가 감지 시
이 엔드포인트를 부르면, 그 결과를 /ws/live로 그대로 중계해 열려 있는 브라우저 탭에
알린다 — 실제 판정(모델 추론)은 여기서 하지 않는다, 이미 판정된 결과의 통로일 뿐이다.

인증이 없다: 이 프로젝트는 web이 로봇 PC 안에서만 닿는다는 전제로 서 있다(/api/robot/*
와 같은 신뢰 경계 — robot.py 참조). 브리지도 같은 PC에서 돈다.
"""
from fastapi import APIRouter

from ..events import hub

router = APIRouter()


@router.post("/api/internal/wake-detected")
async def wake_detected():
    await hub.broadcast({"type": "wake_word_detected"})
    return {"schema_version": "1.0.0", "broadcast": True}

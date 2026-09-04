"""WebSocket /ws/live (웹_인터페이스_정의서.md 4절).

브라우저는 연결 후 별도 구독 요청 없이 6종 이벤트를 전부 받는다. 연결 직후 현재
로봇 상태를 한 번 보내 화면이 빈 상태로 시작하지 않게 한다.
"""
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..events import hub

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/live")
async def live(websocket: WebSocket):
    await websocket.accept()
    await hub.register(websocket)
    try:
        executor = websocket.app.state.executor
        await websocket.send_json({"type": "robot_state", **executor.robot_state()})
        while True:
            # 브라우저는 보내는 것이 없다. 연결 유지·종료 감지용으로만 받는다.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("WebSocket 오류")
    finally:
        await hub.unregister(websocket)

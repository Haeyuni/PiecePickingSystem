"""WebSocket 이벤트 허브 (/ws/live, 웹_인터페이스_정의서.md 4절).

단일 채널로 5종 이벤트를 전부 내보내고, 브라우저는 `type`으로 구분한다.

**재전송 큐를 두지 않는다.** 연결이 끊겼다 붙으면 브라우저가 `GET /api/traces/{trace_id}`로
현재 스냅샷을 다시 받는 것으로 충분하다 — 놓친 이벤트를 순서대로 재생할 필요가 없다
(1차 범위, 단일 사용자 가정).
"""
import asyncio
import logging

logger = logging.getLogger(__name__)

# 안전 이벤트는 다른 이벤트보다 먼저 나가야 한다 (4절). 전송 실패로 뒤로 밀리지 않도록
# 브로드캐스트 자체를 우선 처리한다.
SAFETY_EVENT = "safety_event"


class EventHub:
    def __init__(self):
        self._clients: set = set()
        self._lock = asyncio.Lock()

    async def register(self, ws) -> None:
        async with self._lock:
            self._clients.add(ws)
        logger.info("WebSocket 연결 (총 %d)", len(self._clients))

    async def unregister(self, ws) -> None:
        async with self._lock:
            self._clients.discard(ws)
        logger.info("WebSocket 해제 (총 %d)", len(self._clients))

    async def broadcast(self, event: dict) -> None:
        async with self._lock:
            clients = list(self._clients)
        if event.get("type") == SAFETY_EVENT:
            logger.warning("안전 이벤트 브로드캐스트: %s", event.get("event_type"))

        dead = []
        for ws in clients:
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.unregister(ws)

    @property
    def client_count(self) -> int:
        return len(self._clients)


hub = EventHub()

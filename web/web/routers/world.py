"""GET /api/world-state — 최신 월드 상태 (웹_인터페이스_정의서.md 2.7절).

화면정의서 2.2.5절의 "탐지 물체 목록"이 최초 렌더 시 채워지려면 필요하다. WebSocket은
변화가 있을 때만 밀어주므로, 새로고침 직후처럼 아무 이벤트도 없는 순간에 화면이 비어 보인다.
`GET /api/traces`가 실행 상태의 폴백인 것과 같은 역할이다.
"""
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/api/world-state")
def get_world_state(request: Request):
    world_state = request.app.state.executor.get_latest_world_state()
    if world_state is None:
        return JSONResponse(
            status_code=503,
            content={"schema_version": "1.0.0",
                     "error": {"code": "NO_WORLD_STATE",
                               "message": "아직 월드 상태를 받지 못했습니다 (perception 미기동)"}},
        )
    return world_state

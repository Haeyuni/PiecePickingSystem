"""GET /api/world-state — 최신 월드 상태 (웹_인터페이스_정의서.md 2.7절).

화면정의서 2.2.5절의 "탐지 물체 목록"이 최초 렌더 시 채워지려면 필요했다 — 그 목록은
온디맨드 전환(docs/on-demand-perception.md D-5)으로 화면에서 빠졌지만, 이 엔드포인트는
그대로 둔다. 프론트가 이제 여기서 쓰는 것은 `stamp`뿐이다 — 관측 패널(CameraViews)의
"n초 전 관측" 표시가 WebSocket 재연결·최초 진입 시에도 바로 채워지려면 필요하다
(`GET /api/traces`가 실행 상태의 폴백인 것과 같은 역할).
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

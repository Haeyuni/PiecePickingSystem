"""POST /api/commands — 명령 입력 (웹_인터페이스_정의서.md 2.1절).

**차단은 여기(백엔드)서 한다.** 프론트엔드의 입력창 비활성화는 UX 보조 수단일 뿐이고,
여러 탭이나 개발자도구로 우회해도 이 검사를 지나갈 수 없다.
"""
import logging
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .. import orchestrator

logger = logging.getLogger(__name__)
router = APIRouter()

# mode별 거부 사유 (2.1절)
BLOCKED_MODES = {
    "busy": ("ROBOT_BUSY", "로봇이 이전 명령을 처리 중입니다"),
    "error": ("ROBOT_ERROR", "로봇이 오류 상태입니다. 상태를 확인하세요"),
    "estopped": ("ROBOT_ESTOPPED", "비상정지 상태입니다. 수동 리셋이 필요합니다"),
}


class CommandRequest(BaseModel):
    schema_version: str = "1.0.0"
    command_text: str


@router.post("/api/commands")
async def create_command(body: CommandRequest, request: Request):
    executor = request.app.state.executor
    mode = executor.robot_state().get("mode", "idle")

    if mode in BLOCKED_MODES:
        code, message = BLOCKED_MODES[mode]
        # planner를 호출하지 않는다 — 실시간 로봇 상태를 아는 쪽은 web이다(NFR-04)
        return JSONResponse(
            status_code=409,
            content={"schema_version": "1.0.0", "error": {"code": code, "message": message}},
        )

    trace_id = f"tr-{uuid.uuid4().hex[:12]}"
    logger.info("명령 접수 trace=%s: %s", trace_id, body.command_text)
    orchestrator.start_command(trace_id, body.command_text, executor)

    # 이후 진행 상황은 폴링이 아니라 WebSocket으로 간다(4절)
    return JSONResponse(
        status_code=202,
        content={"schema_version": "1.0.0", "trace_id": trace_id, "status": "processing"},
    )

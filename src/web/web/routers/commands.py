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
    # 시나리오 도메인 (가정/약국/재활용). 일반 명령은 "general". 이 값이 planner까지
    # 전달되어 VLM 프롬프트가 도메인별로 분기된다.
    domain: str = "general"


@router.post("/api/commands")
async def create_command(body: CommandRequest, request: Request):
    executor = request.app.state.executor
    mode = executor.robot_state().get("mode", "idle")

    # planner/Observe 구간에는 로봇 mode가 아직 idle일 수 있다. mode만 보면 두 명령이
    # 동시에 서로 다른 observation을 만들 수 있으므로 명령 태스크 자체도 확인한다.
    if orchestrator.has_running_command():
        return JSONResponse(
            status_code=409,
            content={"schema_version": "1.0.0", "error": {
                "code": "ROBOT_BUSY", "message": "로봇이 이전 명령을 처리 중입니다"}},
        )

    if mode in BLOCKED_MODES:
        code, message = BLOCKED_MODES[mode]
        # planner를 호출하지 않는다 — 실시간 로봇 상태를 아는 쪽은 web이다(NFR-04)
        return JSONResponse(
            status_code=409,
            content={"schema_version": "1.0.0", "error": {"code": code, "message": message}},
        )

    # 승인 대기는 로봇의 mode가 아니다(orchestrator.has_pending_approval 참조) — 로봇은
    # 가만히 있지만, 이전 명령의 승인이 안 끝났으면 다음 명령을 계획하지 않는다(명령
    # 1건당 승인 1회 원칙이 겹치면 어느 쪽을 승인한 것인지 알 수 없다).
    if orchestrator.has_pending_approval():
        return JSONResponse(
            status_code=409,
            content={"schema_version": "1.0.0",
                     "error": {"code": "AWAITING_APPROVAL",
                               "message": "이전 명령의 승인 대기 중입니다"}},
        )

    trace_id = f"tr-{uuid.uuid4().hex[:12]}"
    logger.info("명령 접수 trace=%s domain=%s: %s", trace_id, body.domain, body.command_text)
    orchestrator.start_command(trace_id, body.command_text, executor, domain=body.domain)

    # 이후 진행 상황은 폴링이 아니라 WebSocket으로 간다(4절)
    return JSONResponse(
        status_code=202,
        content={"schema_version": "1.0.0", "trace_id": trace_id, "status": "processing"},
    )

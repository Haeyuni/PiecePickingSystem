"""GET /api/traces/{trace_id} — 실행 상태 스냅샷 (웹_인터페이스_정의서.md 2.3절).

WebSocket이 기본 경로이고 이 엔드포인트는 **새로고침·재접속 시의 폴백**이다.
그래서 놓친 이벤트를 재생하는 대신 현재 상태를 통째로 돌려준다.
"""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from .. import orchestrator

router = APIRouter()


@router.get("/api/traces/{trace_id}")
def get_trace(trace_id: str):
    trace = orchestrator.snapshot(trace_id)
    if trace is None:
        return JSONResponse(
            status_code=404,
            content={"schema_version": "1.0.0", "trace_id": trace_id,
                     "error": {"code": "TRACE_NOT_FOUND",
                               "message": "해당 trace_id의 실행 이력이 없습니다"}},
        )

    return {
        "schema_version": "1.0.0",
        "trace_id": trace["trace_id"],
        "command_text": trace["command_text"],
        "validation_status": trace["validation_status"],
        "validation_reason": trace["validation_reason"],
        "sequence_id": trace["sequence_id"],
        "steps": [
            {
                "request_id": s["request_id"],
                "skill": s["skill"],
                "object_id": s["object_id"],
                "bin_id": s.get("bin_id"),
                "status": s["status"],
                "phase": s["phase"],
            }
            for s in trace["steps"]
        ],
    }

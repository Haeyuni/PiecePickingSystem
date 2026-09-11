"""POST /api/executions/{trace_id}/approval — 실행 전 승인 (명령 1건당 1회).

`object-confirmations`(FR-05b, 클래스 단위 속성 확정)와는 다른, 병행하는 메커니즘이다.
여기서 승인/거부/라벨수정하는 것은 지금 이 시퀀스 1건이고, orchestrator._await_approval이
그 결과를 기다린다.
"""
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .. import orchestrator

router = APIRouter()


class ApprovalRequest(BaseModel):
    schema_version: str = "1.0.0"
    action: str          # "approve" | "reject" | "correct_label" | "correct_bin"
    object_id: str | None = None
    class_name: str | None = None
    name_ko: str | None = None
    bin_id: str | None = None   # correct_bin일 때만


@router.post("/api/executions/{trace_id}/approval")
def submit_approval(trace_id: str, body: ApprovalRequest):
    if body.action not in ("approve", "reject", "correct_label", "correct_bin"):
        return JSONResponse(
            status_code=400,
            content={"schema_version": "1.0.0",
                     "error": {"code": "BAD_ACTION",
                               "message": f"알 수 없는 action: {body.action}"}},
        )
    if body.action == "correct_label" and not body.object_id:
        return JSONResponse(
            status_code=400,
            content={"schema_version": "1.0.0",
                     "error": {"code": "MISSING_OBJECT_ID",
                               "message": "correct_label은 object_id가 필요합니다"}},
        )
    if body.action == "correct_bin" and not (body.object_id and body.bin_id):
        return JSONResponse(
            status_code=400,
            content={"schema_version": "1.0.0",
                     "error": {"code": "MISSING_BIN_ID",
                               "message": "correct_bin은 object_id와 bin_id가 필요합니다"}},
        )

    ok = orchestrator.resolve_approval(trace_id, body.model_dump())
    if not ok:
        return JSONResponse(
            status_code=409,
            content={"schema_version": "1.0.0",
                     "error": {"code": "NOT_PENDING",
                               "message": f"'{trace_id}'는 승인 대기 상태가 아닙니다"}},
        )
    return {"schema_version": "1.0.0", "trace_id": trace_id, "accepted": True}

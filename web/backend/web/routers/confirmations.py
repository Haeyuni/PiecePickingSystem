"""신규 물체 확인 (웹_인터페이스_정의서.md 2.4절, FR-05b).

이건 실행 승인이 아니라 **속성 확정**이다. 확인 대기 중인 물체가 있어도 다른 명령은
그대로 실행된다 — 다만 그 물체는 확인 전까지 fallback(fragile) 프로파일로만 다뤄진다.
"""
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .. import store

router = APIRouter()


class ConfirmRequest(BaseModel):
    schema_version: str = "1.0.0"
    approved: bool
    corrections: dict | None = None


@router.get("/api/object-confirmations")
def list_confirmations():
    return {"schema_version": "1.0.0", "items": store.pending_confirmations()}


@router.post("/api/object-confirmations/{class_name}")
def confirm(class_name: str, body: ConfirmRequest):
    if not body.approved:
        return JSONResponse(
            status_code=400,
            content={"schema_version": "1.0.0",
                     "error": {"code": "NOT_APPROVED",
                               "message": "approved=false로는 확정할 수 없습니다"}},
        )

    updated = store.confirm_object(class_name, body.corrections)
    if not updated:
        # 없는 클래스와 이미 확인된 클래스를 구분하지 않고 409로 묶는다 — 어느 쪽이든
        # "지금 확정할 수 있는 대상이 아니다"라는 같은 상태 충돌이다(6절).
        return JSONResponse(
            status_code=409,
            content={"schema_version": "1.0.0",
                     "error": {"code": "NOT_PENDING",
                               "message": f"'{class_name}'은 확인 대기 상태가 아닙니다"}},
        )

    return {"schema_version": "1.0.0", "class_name": class_name, "is_confirmed": True}

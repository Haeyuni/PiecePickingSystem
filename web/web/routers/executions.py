"""GET /api/executions — 실행 이력 (웹_인터페이스_정의서.md 2.5절, 화면정의서.md 3절).

web이 DB에 직접 접근한다(planner를 거치지 않는다). 필터 파라미터는 이력 화면의
기간·결과·스킬 필터를 지원하기 위한 확장이다(개발계획.md D-6).
"""
from fastapi import APIRouter, Query

from .. import store

router = APIRouter()


@router.get("/api/executions")
def list_executions(
    trace_id: str | None = None,
    result: str | None = Query(default=None, pattern="^(success|failure)$"),
    skill: str | None = Query(default=None, pattern="^(pick|place_into|stop|home)$"),
    since: str | None = Query(default=None, description="ISO 8601. 이 시각 이후"),
    until: str | None = Query(default=None, description="ISO 8601. 이 시각 이전"),
    limit: int = Query(default=50, ge=1, le=500),
):
    return {
        "schema_version": "1.0.0",
        "items": store.query_executions(
            trace_id=trace_id, result=result, skill=skill,
            since=since, until=until, limit=limit,
        ),
    }

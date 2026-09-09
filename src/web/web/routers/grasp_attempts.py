"""GET /api/grasp-attempts — 파지(grasp) 학습/분석용 데이터 (DatasetPage "그랩 데이터" 섹션).

새 수집 경로가 아니다 — pick 실행마다 execution_logs에 이미 남는 grasp_pose(실행한
자세)+result(성공/실패)를 그대로 노출한다(store.query_grasp_attempts, database/
migrations/001_init.sql). 쓰기는 이 라우터의 일이 아니다 — control이 pick을 실행할 때
이미 기록한다(web/orchestrator.py).
"""
import json

from fastapi import APIRouter, Query
from fastapi.responses import Response

from .. import store

router = APIRouter()

_RESULT_PATTERN = "^(success|failure)$"
_STRATEGY_PATTERN = "^(heuristic_pca|contact_graspnet|graspnet_baseline)$"


@router.get("/api/grasp-attempts")
def list_grasp_attempts(
    result: str | None = Query(default=None, pattern=_RESULT_PATTERN),
    class_name: str | None = None,
    strategy: str | None = Query(default=None, pattern=_STRATEGY_PATTERN),
    since: str | None = Query(default=None, description="ISO 8601. 이 시각 이후"),
    until: str | None = Query(default=None, description="ISO 8601. 이 시각 이전"),
    limit: int = Query(default=100, ge=1, le=500),
):
    return {
        "schema_version": "1.0.0",
        "items": store.query_grasp_attempts(
            result=result, class_name=class_name, strategy=strategy,
            since=since, until=until, limit=limit,
        ),
    }


@router.get("/api/grasp-attempts/export")
def export_grasp_attempts(
    result: str | None = Query(default=None, pattern=_RESULT_PATTERN),
    class_name: str | None = None,
    strategy: str | None = Query(default=None, pattern=_STRATEGY_PATTERN),
    since: str | None = None,
    until: str | None = None,
):
    """지금 가진 최소 신호(실행 자세 + 성공/실패)만 JSON으로 내려준다.

    GraspNet 원 데이터셋(포인트클라우드 + 후보별 품질 점수)에는 못 미친다 — 그건 실행
    하나당 후보 여러 개를 다 채점해야 하는 훨씬 큰 작업이다(tools/training/grasp/
    finetune.py 스텁 참조, "파인튜닝 필요성은 결과를 보고 판단한다"). 지금은 "이 자세로
    집었을 때 실제로 성공했는가"만 있는 가벼운 형태 — 성공/실패 분포부터 보고 다음
    단계(정말 재학습이 필요한지)를 판단하는 데 쓴다.
    """
    items = store.query_grasp_attempts(
        result=result, class_name=class_name, strategy=strategy,
        since=since, until=until, limit=500,
    )
    body = json.dumps(
        {"schema_version": "1.0.0", "count": len(items), "items": items},
        ensure_ascii=False, indent=2,
    ).encode("utf-8")
    return Response(
        content=body, media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="grasp_attempts.json"'},
    )

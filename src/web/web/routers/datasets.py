"""GET /api/datasets — 데이터셋 수집+브라우징+큐레이션 화면 (Roboflow 스타일).

planner의 `/internal/label-marks`가 명령 실행마다 남긴 `dataset_items`를 읽어 이미지
(trace_id) 단위로 묶어 보여준다. 승인/거부·내보내기는 dataset_items를 쓰는 planner로
그대로 넘긴다(web/planner_client.py, web/store.py 상단 주석 참조). 학습 실행 자체
(`tools/training/detector/train.py`)는 이 범위 밖 — 다음 단계다.
"""
from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from .. import planner_client, store
from ..planner_client import PlannerUnavailable

router = APIRouter()


@router.get("/api/datasets")
def list_datasets(
    class_name: str | None = None,
    since: str | None = Query(default=None, description="ISO 8601. 이 시각 이후"),
    until: str | None = Query(default=None, description="ISO 8601. 이 시각 이전"),
    limit: int = Query(default=50, ge=1, le=500),
):
    return {
        "schema_version": "1.0.0",
        "items": store.query_dataset_items(
            class_name=class_name, since=since, until=until, limit=limit,
        ),
    }


@router.get("/api/datasets/{trace_id}/image")
def dataset_image(trace_id: str):
    image_path = store.get_dataset_image_path(trace_id)
    if image_path is None:
        return JSONResponse(
            status_code=404,
            content={"schema_version": "1.0.0",
                     "error": {"code": "NOT_FOUND", "message": f"'{trace_id}'를 찾을 수 없습니다"}},
        )
    return FileResponse(image_path, media_type=("image/png" if image_path.endswith(".png") else "image/jpeg"))


@router.get("/api/datasets/{trace_id}/annotations")
def dataset_annotations(trace_id: str):
    """어노테이션 오버레이용 물체별 폴리곤(정규화 0~1 좌표). store.get_dataset_item_polygons 참조."""
    polygons = store.get_dataset_item_polygons(trace_id)
    if polygons is None:
        return JSONResponse(
            status_code=404,
            content={"schema_version": "1.0.0",
                     "error": {"code": "NOT_FOUND",
                               "message": f"'{trace_id}'의 어노테이션을 찾을 수 없습니다"}},
        )
    return {"schema_version": "1.0.0", "trace_id": trace_id, "polygons": polygons}


@router.post("/api/datasets/{trace_id}/review")
async def review_dataset_item(trace_id: str, request: Request):
    body = await request.json()
    approved = bool(body.get("approved"))
    try:
        result = await planner_client.review_dataset_item(trace_id, approved)
    except PlannerUnavailable as e:
        return JSONResponse(
            status_code=503,
            content={"schema_version": "1.0.0",
                     "error": {"code": "PLANNER_UNAVAILABLE", "message": str(e)}},
        )
    return JSONResponse(status_code=result["status_code"], content=result["body"])


@router.post("/api/datasets/bulk-review")
async def bulk_review_dataset_items(request: Request):
    """DatasetPage의 "전체선택" 승인/거부 — 선택된 trace_id 목록을 한 번에 처리한다."""
    body = await request.json()
    trace_ids = [str(t) for t in (body.get("trace_ids") or [])]
    approved = bool(body.get("approved"))
    try:
        result = await planner_client.bulk_review_dataset_items(trace_ids, approved)
    except PlannerUnavailable as e:
        return JSONResponse(
            status_code=503,
            content={"schema_version": "1.0.0",
                     "error": {"code": "PLANNER_UNAVAILABLE", "message": str(e)}},
        )
    return JSONResponse(status_code=result["status_code"], content=result["body"])


@router.get("/api/datasets/export")
async def export_dataset():
    """승인된 이미지를 YOLO 데이터셋 zip으로 내려준다. `<a href>`로 바로 걸어 브라우저
    네이티브 다운로드로 받는다 — 별도 blob 처리가 필요 없게 GET으로 뒀다."""
    try:
        content, filename = await planner_client.export_dataset()
    except PlannerUnavailable as e:
        return JSONResponse(
            status_code=503,
            content={"schema_version": "1.0.0",
                     "error": {"code": "PLANNER_UNAVAILABLE", "message": str(e)}},
        )
    return Response(
        content=content, media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

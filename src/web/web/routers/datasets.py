"""GET /api/datasets — 데이터셋 수집+브라우징 화면 (Roboflow 스타일).

planner의 `/internal/label-marks`가 명령 실행마다 남긴 `dataset_items`를 읽기만 한다.
학습 트리거는 이 범위 밖이다 — `tools/training/detector/train.py`는 아직 스텁이다.
"""
from fastapi import APIRouter, Query
from fastapi.responses import FileResponse, JSONResponse

from .. import store

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


@router.get("/api/datasets/{item_id}/image")
def dataset_image(item_id: str):
    image_path = store.get_dataset_item_image_path(item_id)
    if image_path is None:
        return JSONResponse(
            status_code=404,
            content={"schema_version": "1.0.0",
                     "error": {"code": "NOT_FOUND", "message": f"'{item_id}'를 찾을 수 없습니다"}},
        )
    return FileResponse(image_path, media_type="image/png")

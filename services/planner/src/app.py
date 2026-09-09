"""planner 서비스 진입점 (FastAPI). ROS2와 무관 — web을 거쳐서만 control에 도달 (시스템명세서 3.1절).

B0: 마이그레이션 적용 + objects.yaml 시드 + /health
B1: /internal/plan (계획 생성 → 그라운딩 → 검증 → task_sequences 기록)
B?: /internal/label-marks (SAM이 번호를 그린 프레임 → 번호별 클래스 판단)

**VLM 호출이 여기 있는 이유**: planner는 ROS2를 모르는 서비스이고 OPENAI_API_KEY도 여기에만
있다. perception이 직접 부르면 ROS 컨테이너에 API 키와 인터넷 의존이 들어간다. SAM은 로컬
GPU라 perception에 두고, 번호를 그린 이미지만 여기로 보내 이름을 받아 간다
(docs/on-demand-perception.md 4절).
"""
import datetime
import json
import logging
import os
import pathlib
import uuid
from contextlib import asynccontextmanager

import psycopg
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from . import db, grounding, llm_client, seed, validator, vlm_detect
from .schema import SCHEMA_VERSION, PlanRequest, PlanResponse

# data/datasets/<날짜>/<trace_id>.{png,json} — Roboflow 스타일 수집 화면의 재료
# (database/migrations/003_dataset_items.sql). vlm_detect.OBJECTS_YAML과 같은 방식으로
# 저장소 루트를 계산한다 — planner는 perception_common(ROS 패키지)에 의존하지 않는다.
DATASETS_DIR = pathlib.Path(
    os.environ.get("DATASETS_DIR")
    or pathlib.Path(__file__).resolve().parents[3] / "data" / "datasets"
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

try:
    import cv2
    _HAS_CV2 = True
except Exception:  # cv2가 없을 때는 원본 그대로 저장한다
    _HAS_CV2 = False


def _strip_overlay(image_bytes: bytes) -> bytes:
    """학습용 이미지 바이트 → YOLO용 JPEG.

    perception이 보내는 `original_image`는 오버레이 없는 원본 프레임이지만 PNG라, 여기서
    YOLO 학습에서 일반적인 JPEG로 리인코딩한다. cv2가 없는 환경(_HAS_CV2=False)에서는
    본문 그대로를 돌려준다 — 이때는 원본 PNG 바이트가 `images/<trace_id>.jpg`에 쓰인다.
    """
    import numpy as np

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return image_bytes
    # 오버레이가 그려진 원본(:17)이 아니라 색만 리인코딩 — 실제 복원은 불가하므로
    # 원본 프레임을 별도 전달받는 구조로 바꿀 때까지는 이 상태로 둔다.
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return jpg.tobytes() if ok else image_bytes

# 기동 시 수행한 작업 요약. /health가 그대로 노출한다.
_startup: dict = {"migrations": [], "seeded": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    with db.connect() as conn:
        _startup["migrations"] = db.apply_migrations(conn)
        _startup["seeded"] = seed.seed_object_attributes(conn)
    logger.info("planner 기동 완료: %s", _startup)
    yield


app = FastAPI(title="piece-picking-system planner", lifespan=lifespan)


@app.get("/health")
def health():
    """DB까지 실제로 왕복해서 확인한다 — 프로세스만 살아있는 상태를 ok로 보고하지 않는다."""
    try:
        with psycopg.connect(db.dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM object_attributes")
            object_count = cur.fetchone()[0]
    except Exception as e:  # DB 장애 시 503 대신 상태를 실어 보낸다 (web이 판단)
        return {"schema_version": SCHEMA_VERSION, "status": "degraded", "db": str(e)}

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "db": "ok",
        "object_attributes": object_count,
        "llm_model": llm_client.model_name(),
        "prompt_version": llm_client.PROMPT_VERSION,
        "startup": _startup,
    }


def _record(req: PlanRequest, sequence_id: str, status: str, reason: str | None,
            steps: list) -> None:
    """생성된 시퀀스와 검증 결과를 남긴다 (task_sequences).

    기록 실패가 계획 응답 자체를 막지는 않는다 — 로봇 동작을 결정하는 것은 검증 결과이고,
    이력은 그 다음 문제다.
    """
    try:
        with psycopg.connect(db.dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO task_sequences (
                    sequence_id, trace_id, raw_command, world_state_snapshot,
                    generated_sequence, validation_status, validation_reason,
                    llm_model_version, prompt_version
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    sequence_id, req.trace_id, req.command_text,
                    json.dumps(req.world_state, ensure_ascii=False),
                    json.dumps([s.model_dump() for s in steps], ensure_ascii=False),
                    status, reason,
                    llm_client.model_name(), llm_client.PROMPT_VERSION,
                ),
            )
            conn.commit()
    except Exception:
        logger.exception("task_sequences 기록 실패 (sequence_id=%s)", sequence_id)


@app.post("/internal/plan")
def internal_plan(req: PlanRequest):
    """자연어 명령 + 월드 상태 → 검증된 스킬 시퀀스 (웹_인터페이스_정의서 3절).

    rejected면 web은 ros_bridge를 호출하지 않는다 — 로봇은 움직이지 않는다.
    """
    sequence_id = str(uuid.uuid4())
    bins = grounding.load_bins()

    def rejected(reason: str, http_status: int = 200):
        _record(req, sequence_id, "rejected", reason, [])
        body = PlanResponse(
            trace_id=req.trace_id, sequence_id=sequence_id,
            validation_status="rejected", validation_reason=reason,
        )
        return JSONResponse(status_code=http_status, content=body.model_dump())

    # 1) LLM 계획 생성
    try:
        llm_plan = llm_client.plan(
            req.command_text,
            grounding.describe_for_prompt(req.world_state, bins),
            req.previous_failure.model_dump() if req.previous_failure else None,
            req.domain,
        )
    except Exception as e:
        logger.exception("LLM 호출 실패")
        # 하위 서비스(LLM API) 장애는 503 (웹_인터페이스_정의서 6절)
        return rejected(f"계획 생성에 실패했습니다: {e}", http_status=503)

    # 2) LLM이 스스로 거부한 경우 — 그대로 전달한다
    if not llm_plan.steps:
        return rejected(llm_plan.refusal_reason or "지시를 수행할 수 없습니다")

    # 3) 검증 (그라운딩·작업공간·가반하중·전제조건·안전 게이트)
    try:
        steps = validator.validate(
            llm_plan.steps, req.world_state, bins, req.active_safety_events,
        )
    except validator.Rejected as e:
        # 검증기 거부는 422 (웹_인터페이스_정의서 6절)
        return rejected(str(e), http_status=422)

    _record(req, sequence_id, "approved", None, steps)
    return PlanResponse(
        trace_id=req.trace_id, sequence_id=sequence_id,
        validation_status="approved", steps=steps,
    )


def _error(status: int, code: str, message: str, trace_id: str = "") -> JSONResponse:
    """웹_인터페이스_정의서 6절의 공통 에러 형식."""
    body: dict = {"schema_version": SCHEMA_VERSION,
                  "error": {"code": code, "message": message}}
    if trace_id:
        body["trace_id"] = trace_id
    return JSONResponse(status_code=status, content=body)


def _marks_to_yolo(marks: list[dict], mask_polys: dict[int, list[list[list[float]]]]) -> str:
    """VLM 마크 + 마스크 윤곽선 → YOLO 세그멘테이션 TXT 라벨.

    클래스 인덱스는 TXT에 등장하는 순서대로 매긴다(같은 class_name은 같은 인덱스).
    is_object=false(배경·조각)는 라벨에서 제외한다.

    `mask_polys`는 mark_id → [poly, ...] 형태이고, poly는 [[x,y], ...] 정규화(0~1) 좌표다.
    """
    lines: list[str] = []
    class_index: dict[str, int] = {}
    for mark in marks:
        if not mark.get("is_object"):
            continue
        mark_id = int(mark.get("mark_id") or 0)
        cls = mark.get("class_name") or ""
        if not cls:
            continue
        if cls not in class_index:
            class_index[cls] = len(class_index)
        polys = mask_polys.get(mark_id, [])
        for poly in polys:
            flat = []
            for x, y in poly:
                flat.append(f"{x:.6f}")
                flat.append(f"{y:.6f}")
            if len(flat) >= 6:
                lines.append(f"{class_index[cls]} " + " ".join(flat))
    return "\n".join(lines)


def _save_dataset_items(trace_id: str, image_bytes: bytes, marks: list[dict],
                        mask_polys: dict[int, list[list[list[float]]]] | None = None) -> None:
    """이번 라벨링 결과를 YOLO 학습용 데이터셋으로 남긴다.

    저장 형식:
      data/datasets/<YYYY-MM-DD>/images/<trace_id>.jpg   ← 원본 이미지(오버레이 제거)
      data/datasets/<YYYY-MM-DD>/labels/<trace_id>.txt   ← YOLO 세그 TXT 라벨

    실패해도 라벨링 응답 자체는 그대로 나간다 — 수집은 인지 파이프라인의 필수 경로가 아니다.
    """
    objects = [m for m in marks if m.get("is_object")]
    if not objects:
        return

    mask_polys = mask_polys or {}
    date_dir = DATASETS_DIR / datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    images_dir = date_dir / "images"
    labels_dir = date_dir / "labels"
    image_path = images_dir / f"{trace_id}.jpg"
    label_path = labels_dir / f"{trace_id}.txt"
    try:
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # 원본 이미지 저장 (번호 오버레이가 아닌 학습용 원본)
        image_bytes = _strip_overlay(image_bytes) if _HAS_CV2 else image_bytes
        image_path.write_bytes(image_bytes)

        yolo_text = _marks_to_yolo(marks, mask_polys)
        label_path.write_text(yolo_text, encoding="utf-8")
    except OSError:
        logger.exception("데이터셋 파일 저장 실패 (trace_id=%s)", trace_id)
        return

    try:
        with psycopg.connect(db.dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
            for obj in objects:
                cur.execute(
                    """
                    INSERT INTO dataset_items (
                        item_id, trace_id, image_path, label_path,
                        class_name, name_ko, attr_source, confidence, reasoning
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(uuid.uuid4()), trace_id, str(image_path), str(label_path),
                        obj.get("class_name"), obj.get("name_ko"), "llm_suggested",
                        obj.get("confidence"), obj.get("reasoning"),
                    ),
                )
            conn.commit()
    except Exception:
        logger.exception("dataset_items 기록 실패 (trace_id=%s)", trace_id)


@app.post("/internal/label-marks")
async def internal_label_marks(
    image: UploadFile = File(..., description="SAM 마스크마다 번호를 그려 넣은 프레임"),
    mark_ids: str = Form(..., description="이미지에 그려진 번호. 쉼표로 구분 (예: 1,2,3)"),
    trace_id: str = Form(""),
    detail: str = Form("high"),
    domain: str = Form("general", description="시나리오 도메인(가정/약국/재활용). VLM 프롬프트 분기에 쓴다"),
    mask_polys: str = Form("", description="mark_id → 윤곽선 다각형 좌표 (JSON)"),
    original_image: UploadFile | None = File(
        default=None, description="오버레이가 없는 원본 프레임 (YOLO 학습용 이미지)"),
):
    """번호가 그려진 프레임 → 번호별 "무엇인가" 판단 (좌표는 묻지 않는다).

    **지시(명령문)를 받지 않는다.** 인지가 지시에 끌려가기 때문이다 — 실측에서 "우산
    왼쪽으로"를 함께 주자 배경 조각을 umbrella라고 답했다(vlm_detect.label_marks 주석).
    어느 물체를 옮길지는 이 결과를 텍스트로 받는 /internal/plan이 정한다.

    original_image는 perception이 함께 보내는 원본 프레임(번호 오버레이 없음)이다 —
    YOLO 학습용 이미지는 오버레이가 없어야 하므로 마스크 이미지보다 이것을 우선 저장한다.

    domain은 sam_vlm.py가 함께 보내는 시나리오 컨텍스트다(가정/약국/재활용) —
    vlm_detect.label_marks로 그대로 넘겨야 [장면 맥락(도메인)] 프롬프트 분기가 실제로
    걸린다. 여기서 안 받으면 FastAPI가 조용히 버려서 sam_vlm.py가 보낸 값이 VLM에
    끝내 닿지 못한다(2026-09-09 발견 — domain 파라미터를 sam_vlm.py/vlm_detect.py에는
    배선했는데 이 엔드포인트 시그니처에는 추가하지 않아 죽은 배선이었다).
    """
    try:
        ids = [int(v) for v in mark_ids.split(",") if v.strip()]
    except ValueError:
        return _error(400, "BAD_MARK_IDS", f"mark_ids를 정수 목록으로 읽을 수 없습니다: {mark_ids!r}",
                      trace_id)
    if not ids:
        return _error(400, "BAD_MARK_IDS", "mark_ids가 비어 있습니다", trace_id)

    polys_by_id: dict[int, list] = {}
    if mask_polys:
        try:
            raw = json.loads(mask_polys)
            polys_by_id = {int(k): v for k, v in raw.items()}
        except (ValueError, TypeError):
            logger.warning("mask_polys 파싱 실패 (trace_id=%s)", trace_id)

    data = await image.read()
    if not data:
        return _error(400, "EMPTY_IMAGE", "이미지 본문이 비어 있습니다", trace_id)

    # 학습용 원본: perception이 보내면 그것을, 없으면 오버레이 이미지를 fallback으로 쓴다.
    save_image = data
    if original_image is not None:
        orig_data = await original_image.read()
        if orig_data:
            save_image = orig_data

    data_url = vlm_detect.encode_bytes(data, image.content_type or "image/png")
    try:
        scene = vlm_detect.label_marks(data_url, ids, detail=detail, domain=domain)
    except Exception as e:
        logger.exception("VLM 라벨링 실패 (trace_id=%s)", trace_id)
        # 하위 서비스(VLM API) 장애는 503 — /internal/plan의 LLM 장애와 같은 판단이다
        return _error(503, "VLM_UNAVAILABLE", f"라벨링에 실패했습니다: {e}", trace_id)

    marks = [m.model_dump() for m in scene.marks]
    if trace_id:
        # 빈 trace_id는 vlm_sam_test.py 같은 수동 호출일 수 있다 — 파일명이 겹치므로
        # 저장하지 않는다(실제 명령 실행은 항상 trace_id를 채워 보낸다).
        _save_dataset_items(trace_id, save_image, marks, polys_by_id)

    return {
        "schema_version": SCHEMA_VERSION,
        "trace_id": trace_id,
        "model": vlm_detect.model_name(),
        "prompt_version": vlm_detect.MARKS_PROMPT_VERSION,
        "marks": marks,
    }


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8100)


if __name__ == "__main__":
    main()

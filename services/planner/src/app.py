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
import random
import shutil
import uuid
import zipfile
from contextlib import asynccontextmanager

import psycopg
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from . import db, grounding, llm_client, validator, vlm_detect
from .schema import SCHEMA_VERSION, PlanRequest, PlanResponse

# data/datasets/<날짜>/<trace_id>.{png,json} — Roboflow 스타일 수집 화면의 재료
# (database/migrations/003_dataset_items.sql). __file__ 기준으로 저장소 루트를 계산한다
# — planner는 perception_common(ROS 패키지)에 의존하지 않는다.
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
_startup: dict = {"migrations": []}


@asynccontextmanager
async def lifespan(app: FastAPI):
    with db.connect() as conn:
        _startup["migrations"] = db.apply_migrations(conn)
    logger.info("planner 기동 완료: %s", _startup)
    yield


app = FastAPI(title="piece-picking-system planner", lifespan=lifespan)


@app.get("/health")
def health():
    """DB까지 실제로 왕복해서 확인한다 — 프로세스만 살아있는 상태를 ok로 보고하지 않는다."""
    try:
        with psycopg.connect(db.dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
    except Exception as e:  # DB 장애 시 503 대신 상태를 실어 보낸다 (web이 판단)
        return {"schema_version": SCHEMA_VERSION, "status": "degraded", "db": str(e)}

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "db": "ok",
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


def _assign_class_indices(marks: list[dict]) -> dict[str, int]:
    """이미지 한 장(marks) 안에서 클래스명 → 로컬 인덱스. 등장 순서대로 매긴다.

    **이 인덱스는 이미지마다 독립적이다** — 이미지 A의 0번과 이미지 B의 0번이 다른
    클래스일 수 있다. `_marks_to_yolo`(라벨 TXT 생성)와 `_save_dataset_items`(DB에
    local_class_index 저장, database/migrations/007 참조) 둘 다 이 함수 **하나**를 써야
    한다 — 각자 따로 계산하면 다시 어긋난다. 여러 이미지를 하나의 학습셋으로 합칠 때
    이 로컬 인덱스를 전역 인덱스로 치환하는 쪽은 export 쪽(/internal/datasets/export)의 일이다.
    """
    class_index: dict[str, int] = {}
    for mark in marks:
        if not mark.get("is_object"):
            continue
        cls = mark.get("class_name") or ""
        if cls and cls not in class_index:
            class_index[cls] = len(class_index)
    return class_index


def _marks_to_yolo(marks: list[dict], mask_polys: dict[int, list[list[list[float]]]],
                   class_index: dict[str, int]) -> str:
    """VLM 마크 + 마스크 윤곽선 → YOLO 세그멘테이션 TXT 라벨.

    is_object=false(배경·조각)는 라벨에서 제외한다. `class_index`는 `_assign_class_indices`가
    이 marks에 대해 만든 것과 같은 것이어야 한다(호출부가 하나만 계산해 공유한다).

    `mask_polys`는 mark_id → [poly, ...] 형태이고, poly는 [[x,y], ...] 정규화(0~1) 좌표다.
    """
    lines: list[str] = []
    for mark in marks:
        if not mark.get("is_object"):
            continue
        mark_id = int(mark.get("mark_id") or 0)
        cls = mark.get("class_name") or ""
        if not cls or cls not in class_index:
            continue
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
    # _marks_to_yolo(라벨 TXT)와 아래 INSERT(local_class_index)가 같은 매핑을 보게
    # 여기서 한 번만 계산한다 — _assign_class_indices 주석 참조.
    class_index = _assign_class_indices(marks)
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

        yolo_text = _marks_to_yolo(marks, mask_polys, class_index)
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
                        class_name, name_ko, attr_source, confidence, local_class_index
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(uuid.uuid4()), trace_id, str(image_path), str(label_path),
                        obj.get("class_name"), obj.get("name_ko"), "llm_suggested",
                        obj.get("confidence"), class_index.get(obj.get("class_name") or ""),
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


class ReviewRequest(BaseModel):
    approved: bool


@app.post("/internal/datasets/{trace_id}/review")
def review_dataset_item(trace_id: str, req: ReviewRequest):
    """이미지(trace_id) 하나 단위로 데이터셋 큐레이션 승인/거부 (web DatasetPage).

    dataset_items는 물체 하나당 한 행이라 이미지 하나에 여러 행이 붙는다(같은
    image_path/label_path). 물체 단위로 승인/거부하려면 "이 물체의 폴리곤 줄만 라벨
    파일에서 빼기"가 필요한데 그건 사실상 라벨 편집이라 범위 밖이다 — 그래서 이미지
    (trace_id) 전체를 단위로 다룬다. 승인=reviewed를 true로, 거부=행을 아예 지운다
    (이미지/라벨 파일 자체는 안 지운다 — 디스크 정리는 범위 밖).
    """
    with psycopg.connect(db.dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
        if req.approved:
            cur.execute("UPDATE dataset_items SET reviewed = true WHERE trace_id = %s",
                       (trace_id,))
        else:
            cur.execute("DELETE FROM dataset_items WHERE trace_id = %s", (trace_id,))
        affected = cur.rowcount
        conn.commit()
    if affected == 0:
        return _error(404, "NOT_FOUND", f"trace_id={trace_id}인 데이터셋 항목이 없습니다")
    return {"schema_version": SCHEMA_VERSION, "trace_id": trace_id,
            "approved": req.approved, "rows_affected": affected}


class BulkReviewRequest(BaseModel):
    trace_ids: list[str]
    approved: bool


@app.post("/internal/datasets/bulk-review")
def bulk_review_dataset_items(req: BulkReviewRequest):
    """DatasetPage의 "전체선택" 승인/거부 — review_dataset_item과 같은 규칙(이미지 단위)을
    trace_id 목록 전체에 한 번의 쿼리로 적용한다. 프론트가 선택한 개수만큼 요청을
    왕복하지 않게 하려는 것뿐, 판정 로직 자체는 단건과 동일하다."""
    if not req.trace_ids:
        return _error(400, "EMPTY_TRACE_IDS", "trace_ids가 비어 있습니다")
    with psycopg.connect(db.dsn(), connect_timeout=5) as conn, conn.cursor() as cur:
        if req.approved:
            cur.execute("UPDATE dataset_items SET reviewed = true WHERE trace_id = ANY(%s)",
                       (req.trace_ids,))
        else:
            cur.execute("DELETE FROM dataset_items WHERE trace_id = ANY(%s)", (req.trace_ids,))
        affected = cur.rowcount
        conn.commit()
    return {"schema_version": SCHEMA_VERSION, "trace_ids": req.trace_ids,
            "approved": req.approved, "rows_affected": affected}


@app.get("/internal/datasets/export")
def export_dataset():
    """승인된(reviewed=true) 이미지를 YOLO 세그멘테이션 데이터셋(zip)으로 묶어 돌려준다.

    각 이미지의 라벨 TXT는 그 이미지 **안에서만** 유효한 로컬 클래스 인덱스를 쓴다
    (_assign_class_indices 참조) — 그래서 그대로 합치면 클래스가 뒤섞인다. 여기서
    전체 대상의 class_name을 모아 정렬한 **전역** 인덱스를 만들고, 이미지별로 원본
    라벨의 로컬 인덱스를 전역 인덱스로 치환해서 내보낸다. local_class_index가 없는
    행(마이그레이션 007 이전에 쌓인 것)은 치환할 방법이 없어 제외한다.
    """
    with psycopg.connect(db.dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT trace_id, image_path, label_path, class_name, local_class_index
            FROM dataset_items
            WHERE reviewed = true AND local_class_index IS NOT NULL
            """
        )
        rows = cur.fetchall()

    if not rows:
        return _error(404, "EMPTY_EXPORT",
                     "승인된 항목이 없습니다(또는 전부 local_class_index가 없는 옛 항목입니다)")

    # trace_id(이미지)별로 묶는다 — 한 이미지 안의 로컬 인덱스 → class_name 매핑.
    images: dict[str, dict] = {}
    for trace_id, image_path, label_path, class_name, local_idx in rows:
        entry = images.setdefault(
            trace_id, {"image_path": image_path, "label_path": label_path, "local_to_class": {}})
        entry["local_to_class"][local_idx] = class_name

    # 전역 클래스 목록 — 정렬해서 재현 가능한 순서로 만든다(내보낼 때마다 같은 입력이면
    # 같은 data.yaml이 나와야 diff로 비교할 수 있다).
    all_classes = sorted({cls for e in images.values() for cls in e["local_to_class"].values()})
    global_index = {cls: i for i, cls in enumerate(all_classes)}

    # 85/15 train/val, 고정 시드 — 같은 승인 목록이면 같은 분할이 나온다.
    trace_ids = sorted(images.keys())
    shuffled = trace_ids[:]
    random.Random(0).shuffle(shuffled)
    split_at = max(1, int(len(shuffled) * 0.85)) if len(shuffled) > 1 else len(shuffled)
    train_ids = set(shuffled[:split_at])

    export_id = f"export-{datetime.datetime.now(datetime.timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
    export_root = DATASETS_DIR / "exports" / export_id
    for split in ("train", "val"):
        (export_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (export_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    exported = 0
    for trace_id, entry in images.items():
        split = "train" if trace_id in train_ids else "val"
        src_image = pathlib.Path(entry["image_path"])
        src_label = pathlib.Path(entry["label_path"])
        if not src_image.exists() or not src_label.exists():
            logger.warning("내보내기: 원본 파일 없음 (trace_id=%s) — 건너뜀", trace_id)
            continue
        remapped_lines = []
        for line in src_label.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            local_idx_str, rest = line.split(" ", 1)
            cls = entry["local_to_class"].get(int(local_idx_str))
            if cls is None:
                # 이 줄의 로컬 인덱스가 이번 승인 대상 행에 없다 — 거부된 물체이거나
                # DB와 파일이 어긋난 것. 조용히 빼는 게 지어내는 것보다 안전하다.
                continue
            remapped_lines.append(f"{global_index[cls]} {rest}")
        dest_image = export_root / "images" / split / f"{trace_id}.jpg"
        dest_label = export_root / "labels" / split / f"{trace_id}.txt"
        shutil.copyfile(src_image, dest_image)
        dest_label.write_text("\n".join(remapped_lines), encoding="utf-8")
        exported += 1

    data_yaml = (
        f"path: {export_root}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(all_classes)}\n"
        f"names: {json.dumps(all_classes, ensure_ascii=False)}\n"
    )
    (export_root / "data.yaml").write_text(data_yaml, encoding="utf-8")

    zip_path = export_root.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in export_root.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(export_root.parent))

    logger.info("데이터셋 내보내기 완료: export_id=%s 이미지=%d/%d 클래스=%d",
               export_id, exported, len(images), len(all_classes))
    return FileResponse(zip_path, media_type="application/zip", filename=f"{export_id}.zip")


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8100)


if __name__ == "__main__":
    main()

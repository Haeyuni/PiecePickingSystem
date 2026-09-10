"""DB 접근 계층. web은 execution_logs에 직접 접근한다
(웹_인터페이스_정의서.md 2.5절 — 이력 조회는 planner를 거치지 않는다).

trace 진행 상태는 DB가 아니라 메모리에 둔다. 실행 중 스냅샷은 수명이 짧고 단일 사용자
가정이라(8절), WebSocket 재연결 시 되돌려줄 용도로만 필요하다. 영속 기록은 execution_logs다.
"""
import json
import logging
import os
import pathlib
import uuid

import psycopg

logger = logging.getLogger(__name__)


def dsn() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL이 설정되지 않았습니다 (.env 참조)")
    return url


def connect():
    return psycopg.connect(dsn(), connect_timeout=5)


# --- 실행 로그 --------------------------------------------------------------

def insert_execution_log(**kw) -> None:
    """스킬 실행 1건 기록. stop/home은 trace_id·sequence_id 없이 들어온다(2.6절)."""
    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO execution_logs (
                    log_id, sequence_id, trace_id, request_id, object_id, class_name,
                    skill_name, grip_level_used, bin_id, grasp_pose, torque_trace,
                    result, failure_reason,
                    retry_count, cycle_time_ms
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(uuid.uuid4()),
                    kw.get("sequence_id"), kw.get("trace_id"), kw.get("request_id"),
                    kw.get("object_id"), kw.get("class_name"),
                    kw["skill_name"], kw.get("grip_level_used"), kw.get("bin_id"),
                    json.dumps(kw["grasp_pose"]) if kw.get("grasp_pose") else None,
                    json.dumps(kw["torque_trace"]) if kw.get("torque_trace") else None,
                    kw["result"], kw.get("failure_reason", "none"),
                    kw.get("retry_count", 0), kw.get("cycle_time_ms"),
                ),
            )
            conn.commit()
    except Exception:
        # 기록 실패가 로봇 동작을 막지는 않는다 — 다만 조용히 넘어가지도 않는다.
        logger.exception("execution_logs 기록 실패 (%s)", kw.get("skill_name"))


def query_executions(trace_id: str | None = None, result: str | None = None,
                     skill: str | None = None, since: str | None = None,
                     until: str | None = None, limit: int = 50) -> list[dict]:
    """이력 화면용 조회 (화면정의서.md 3절 필터, 개발계획 D-6)."""
    where, params = [], []
    if trace_id:
        where.append("trace_id = %s")
        params.append(trace_id)
    if result:
        where.append("result = %s")
        params.append(result)
    if skill:
        where.append("skill_name = %s")
        params.append(skill)
    if since:
        where.append("executed_at >= %s")
        params.append(since)
    if until:
        where.append("executed_at <= %s")
        params.append(until)

    sql = """
        SELECT log_id, trace_id, sequence_id, request_id, object_id, class_name,
               skill_name, grip_level_used, bin_id, grasp_strategy,
               result, failure_reason, retry_count,
               cycle_time_ms, executed_at
        FROM execution_logs
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY executed_at DESC LIMIT %s"
    params.append(min(limit, 500))

    with connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        columns = [c.name for c in cur.description]
        rows = [dict(zip(columns, row)) for row in cur.fetchall()]

    for row in rows:
        row["log_id"] = str(row["log_id"])
        row["sequence_id"] = str(row["sequence_id"]) if row["sequence_id"] else None
        row["executed_at"] = row["executed_at"].isoformat()
    return rows


def query_grasp_attempts(result: str | None = None, class_name: str | None = None,
                         strategy: str | None = None, since: str | None = None,
                         until: str | None = None, limit: int = 100) -> list[dict]:
    """파지(grasp) 학습/분석용 조회 — DatasetPage의 "그랩 데이터" 섹션.

    **새 수집 경로가 아니다.** pick 실행마다 execution_logs에 이미 grasp_pose(실제로
    실행한 파지 자세)와 result(success/failure)가 남고 있었다(database/migrations/
    001_init.sql) — GraspNet류 재학습에 필요한 최소 신호(자세 + 성공 여부)가 이미
    쌓이는 중이었다는 뜻이라, 여기서는 pick만 걸러 그대로 노출한다.

    query_executions와 다른 함수로 둔 이유: 이력 화면(HistoryPage)은 grasp_pose 같은
    무거운 jsonb를 안 보여줘도 되고, 여기는 그게 핵심이라 SELECT 컬럼이 다르다.
    """
    where = ["skill_name = 'pick'"]
    params: list = []
    if result:
        where.append("result = %s")
        params.append(result)
    if class_name:
        where.append("class_name = %s")
        params.append(class_name)
    if strategy:
        where.append("grasp_strategy = %s")
        params.append(strategy)
    if since:
        where.append("executed_at >= %s")
        params.append(since)
    if until:
        where.append("executed_at <= %s")
        params.append(until)

    sql = (
        "SELECT log_id, trace_id, object_id, class_name, grip_level_used, "
        "grasp_strategy, grasp_pose, result, "
        "failure_reason, executed_at "
        "FROM execution_logs WHERE " + " AND ".join(where) +
        " ORDER BY executed_at DESC LIMIT %s"
    )
    params.append(min(limit, 500))

    with connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        columns = [c.name for c in cur.description]
        rows = [dict(zip(columns, row)) for row in cur.fetchall()]

    for row in rows:
        row["log_id"] = str(row["log_id"])
        row["executed_at"] = row["executed_at"].isoformat()
    return rows


def object_class_map(world_state: dict | None) -> dict[str, str]:
    """object_id → class_name. execution_logs에 속성 조인용 class_name을 남기기 위해 쓴다."""
    if not world_state:
        return {}
    return {o["object_id"]: o.get("class_name") for o in world_state.get("objects", [])}


# --- 데이터셋 화면 (Roboflow 스타일 수집+브라우징, database/migrations/003/007) -----------
#
# 승인/거부·내보내기 같은 **쓰기**는 planner가 한다(dataset_items를 원래 쓰던 서비스이고,
# data/datasets에 대한 rw 마운트도 거기 있다 — routers/datasets.py의 planner_client 호출
# 참조). 여기 store.py는 web 화면이 읽는 것만 맡는다.

def query_dataset_items(class_name: str | None = None, since: str | None = None,
                        until: str | None = None, limit: int = 50) -> list[dict]:
    """이미지(trace_id) 단위로 묶어 반환한다.

    dataset_items는 물체 하나당 한 행이라, 이미지 하나에 물체가 여럿이면 행도 여럿이다
    (같은 image_path/label_path) — 화면 카드 하나 = 이미지 하나로 보여주려면 여기서 묶어야
    한다. class_name 필터는 "그 조건에 맞는 물체가 하나라도 있는 이미지"를 고르는 데만
    쓰고, 실제로 담아 돌려주는 objects는 그 이미지의 물체 전부다 — 필터에 안 걸린 다른
    물체를 화면에서 숨기면 안 된다(그 물체도 같은 사진에 있다는 사실 자체가 정보다).

    지금 규모(초기 수집 단계)에서는 후보 행을 전부 읽어와 파이썬에서 묶은 뒤 이미지
    개수로 자른다 — 데이터셋이 커지면 SQL 쪽에서 trace_id를 먼저 LIMIT하는 방식으로
    바꿀 것.
    """
    where, params = [], []
    if class_name:
        where.append("class_name = %s")
        params.append(class_name)
    if since:
        where.append("captured_at >= %s")
        params.append(since)
    if until:
        where.append("captured_at <= %s")
        params.append(until)

    match_sql = "SELECT DISTINCT trace_id FROM dataset_items"
    if where:
        match_sql += " WHERE " + " AND ".join(where)

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT trace_id, captured_at, class_name, name_ko, reviewed
            FROM dataset_items
            WHERE trace_id IN ({match_sql})
            ORDER BY captured_at DESC
            """,
            params,
        )
        columns = [c.name for c in cur.description]
        rows = [dict(zip(columns, row)) for row in cur.fetchall()]

    images: dict[str, dict] = {}
    order: list[str] = []
    for row in rows:
        trace_id = row["trace_id"]
        if trace_id not in images:
            images[trace_id] = {
                "trace_id": trace_id,
                "captured_at": row["captured_at"].isoformat(),
                "reviewed": True,
                "objects": [],
            }
            order.append(trace_id)
        images[trace_id]["objects"].append(
            {"class_name": row["class_name"], "name_ko": row["name_ko"]})
        # 한 이미지의 행 중 하나라도 reviewed=false면 그 이미지는 아직 대기 중이다 —
        # 승인은 트레이스 전체를 한 번에 true로 돌리므로(planner review 엔드포인트)
        # 정상 상태에서는 다 같은 값이지만, 방어적으로 "전부 true여야 true"로 둔다.
        if not row["reviewed"]:
            images[trace_id]["reviewed"] = False

    return [images[t] for t in order[:min(limit, 500)]]


def get_dataset_image_path(trace_id: str) -> str | None:
    """이미지 서빙(`GET /api/datasets/{trace_id}/image`)용. 없으면 None."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT image_path FROM dataset_items WHERE trace_id = %s LIMIT 1",
                    (trace_id,))
        row = cur.fetchone()
    return row[0] if row else None


def get_dataset_item_polygons(trace_id: str) -> list[dict] | None:
    """어노테이션 오버레이용 — 그 이미지의 물체별 폴리곤(정규화 0~1 좌표, DatasetPage).

    YOLO 라벨 TXT는 그 이미지 **안에서만** 유효한 로컬 클래스 인덱스를 쓴다(planner
    app.py._assign_class_indices와 같은 이유) — local_class_index 컬럼(마이그레이션 007)
    으로 로컬 인덱스 → class_name 매핑을 만든 뒤 라벨 파일을 직접 읽어 해석한다.
    web_ros는 이미 `/data`를 읽기 전용으로 마운트하고 있어(docker-compose.yml) 새 마운트가
    필요 없다. 매핑을 만들 수 없으면(전부 옛 데이터라 local_class_index가 없으면) None.
    """
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT label_path, class_name, name_ko, local_class_index "
            "FROM dataset_items WHERE trace_id = %s",
            (trace_id,),
        )
        rows = cur.fetchall()
    if not rows:
        return None

    label_path = rows[0][0]
    local_to_class: dict[int, tuple[str | None, str | None]] = {}
    for _, class_name, name_ko, local_idx in rows:
        if local_idx is not None:
            local_to_class[local_idx] = (class_name, name_ko)
    if not local_to_class:
        return None

    try:
        text = pathlib.Path(label_path).read_text(encoding="utf-8")
    except OSError:
        logger.warning("데이터셋 라벨 파일을 읽지 못했습니다: %s", label_path)
        return None

    polygons = []
    for line in text.splitlines():
        if not line.strip():
            continue
        local_idx_str, rest = line.split(" ", 1)
        pair = local_to_class.get(int(local_idx_str))
        if pair is None:
            continue
        class_name, name_ko = pair
        values = [float(v) for v in rest.split()]
        points = [[values[i], values[i + 1]] for i in range(0, len(values) - 1, 2)]
        polygons.append({"class_name": class_name, "name_ko": name_ko, "points": points})
    return polygons

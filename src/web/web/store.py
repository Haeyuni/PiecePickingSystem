"""DB 접근 계층. web은 execution_logs에 직접 접근한다
(웹_인터페이스_정의서.md 2.5절 — 이력 조회는 planner를 거치지 않는다).

trace 진행 상태는 DB가 아니라 메모리에 둔다. 실행 중 스냅샷은 수명이 짧고 단일 사용자
가정이라(8절), WebSocket 재연결 시 되돌려줄 용도로만 필요하다. 영속 기록은 execution_logs다.
"""
import json
import logging
import os
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
                    visual_verification_passed, result, failure_reason,
                    retry_count, cycle_time_ms
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(uuid.uuid4()),
                    kw.get("sequence_id"), kw.get("trace_id"), kw.get("request_id"),
                    kw.get("object_id"), kw.get("class_name"),
                    kw["skill_name"], kw.get("grip_level_used"), kw.get("bin_id"),
                    json.dumps(kw["grasp_pose"]) if kw.get("grasp_pose") else None,
                    json.dumps(kw["torque_trace"]) if kw.get("torque_trace") else None,
                    kw.get("visual_verification_passed"),
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
               visual_verification_passed, result, failure_reason, retry_count,
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


def object_class_map(world_state: dict | None) -> dict[str, str]:
    """object_id → class_name. execution_logs에 속성 조인용 class_name을 남기기 위해 쓴다."""
    if not world_state:
        return {}
    return {o["object_id"]: o.get("class_name") for o in world_state.get("objects", [])}


# --- 데이터셋 화면 (Roboflow 스타일 수집+브라우징, database/migrations/003) ---------------

def query_dataset_items(class_name: str | None = None, since: str | None = None,
                        until: str | None = None, limit: int = 50) -> list[dict]:
    """`query_executions`와 같은 필터 패턴. planner의 /internal/label-marks가 쓴 행을
    web이 읽기만 한다 — 쓰기는 이 서비스의 일이 아니다."""
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

    sql = """
        SELECT item_id, trace_id, captured_at, image_path, label_path,
               class_name, name_ko, attr_source, confidence, reviewed
        FROM dataset_items
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY captured_at DESC LIMIT %s"
    params.append(min(limit, 500))

    with connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        columns = [c.name for c in cur.description]
        rows = [dict(zip(columns, row)) for row in cur.fetchall()]

    for row in rows:
        row["item_id"] = str(row["item_id"])
        row["captured_at"] = row["captured_at"].isoformat()
    return rows


def get_dataset_item_image_path(item_id: str) -> str | None:
    """이미지 서빙(`GET /api/datasets/{item_id}/image`)용. 없으면 None."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT image_path FROM dataset_items WHERE item_id = %s", (item_id,))
        row = cur.fetchone()
    return row[0] if row else None

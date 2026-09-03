"""PostgreSQL 클라이언트. object_attributes / task_sequences / execution_logs 등 접근 (시스템명세서 2절).

마이그레이션 적용도 여기서 담당한다 — 별도 마이그레이션 도구를 두기에는
파일이 적고, planner가 DB를 쓰는 유일한 서비스 중 하나라 기동 시점에 맞추는 편이 단순하다.
"""
import logging
import os
import pathlib
import time

import psycopg

logger = logging.getLogger(__name__)

# db/migrations/*.sql — 저장소 루트 기준. 컨테이너에서는 /migrations로 마운트된다.
MIGRATIONS_DIR = pathlib.Path(
    os.environ.get(
        "MIGRATIONS_DIR",
        pathlib.Path(__file__).resolve().parents[2] / "db" / "migrations",
    )
)


def dsn() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL이 설정되지 않았습니다 (.env 참조)")
    return url


def connect(retries: int = 30, delay_s: float = 1.0) -> psycopg.Connection:
    """DB 연결. compose에서 db보다 planner가 먼저 뜰 수 있어 재시도한다."""
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return psycopg.connect(dsn())
        except psycopg.OperationalError as e:
            last_error = e
            logger.info("DB 연결 대기 중 (%d/%d)", attempt, retries)
            time.sleep(delay_s)
    raise RuntimeError(f"DB에 연결하지 못했습니다: {last_error}")


def apply_migrations(conn: psycopg.Connection) -> list[str]:
    """미적용 마이그레이션을 파일명 순으로 적용하고, 적용한 버전 목록을 반환한다.

    각 SQL은 자체적으로 재실행 가능(IF NOT EXISTS)하지만, 이미 적용된 것은
    schema_migrations를 보고 건너뛴다.
    """
    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not files:
        raise RuntimeError(f"마이그레이션 파일이 없습니다: {MIGRATIONS_DIR}")

    applied: set[str] = set()
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('schema_migrations')")
        if cur.fetchone()[0] is not None:
            cur.execute("SELECT version FROM schema_migrations")
            applied = {row[0] for row in cur.fetchall()}

    newly_applied = []
    for path in files:
        version = path.stem
        if version in applied:
            continue
        logger.info("마이그레이션 적용: %s", version)
        with conn.cursor() as cur:
            cur.execute(path.read_text(encoding="utf-8"))
        conn.commit()
        newly_applied.append(version)
    return newly_applied

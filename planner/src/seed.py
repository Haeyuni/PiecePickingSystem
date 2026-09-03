"""perception/config/objects.yaml의 초기값을 object_attributes 테이블로 적재 (시스템명세서 2.1절).

멱등하게 동작해야 한다 — 재기동할 때마다 돌지만, **이미 있는 행은 건드리지 않는다**.
사람이 확인한 값(source='user_confirmed')이나 VLM 제안값을 yaml 기본값으로
되돌려버리면 FR-05b가 무너지기 때문이다.
"""
import logging
import os
import pathlib

import psycopg
import yaml

logger = logging.getLogger(__name__)

OBJECTS_YAML = pathlib.Path(
    os.environ.get(
        "OBJECTS_YAML",
        pathlib.Path(__file__).resolve().parents[2] / "perception" / "config" / "objects.yaml",
    )
)


def load_yaml(path: pathlib.Path | None = None) -> dict:
    path = path or OBJECTS_YAML
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_object_attributes(conn: psycopg.Connection, config: dict | None = None) -> int:
    """objects.yaml의 objects 항목을 적재하고, 새로 넣은 행 수를 반환한다."""
    config = config if config is not None else load_yaml()
    objects = config.get("objects") or {}

    inserted = 0
    with conn.cursor() as cur:
        for class_name, attrs in objects.items():
            cur.execute(
                """
                INSERT INTO object_attributes (
                    class_name, name_ko, mass_g, fragile, deformable, transparent,
                    profile, is_confirmed, source
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, true, 'yaml_seed')
                ON CONFLICT (class_name) DO NOTHING
                """,
                (
                    class_name,
                    attrs.get("name_ko"),
                    attrs.get("mass_g"),
                    attrs.get("fragile", True),
                    attrs.get("deformable", False),
                    attrs.get("transparent", False),
                    attrs.get("profile", "fragile"),
                ),
            )
            inserted += cur.rowcount
    conn.commit()
    logger.info("object_attributes 시드: %d개 신규 적재 (yaml %d개)", inserted, len(objects))
    return inserted


def fallback_profile(config: dict | None = None) -> dict:
    """DB에 없는 신규 클래스에 강제 적용할 보수적 기본값 (FR-05, NFR-03a).

    yaml의 fallback 블록을 그대로 돌려준다 — 값을 코드에 중복 정의하지 않는다.
    """
    config = config if config is not None else load_yaml()
    return config.get("fallback") or {}

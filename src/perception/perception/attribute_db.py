"""물체 속성 조회: 모델 라벨 → class_name → 속성 (FR-05, FR-05b).

두 곳에서 값을 찾는다.

1. **DB `object_attributes`** — 사람이 확인한 값(`user_confirmed`)과 VLM 제안값이 들어 있는
   런타임 마스터다(시스템명세서 2.2절). 여기 있는 값이 최신이다.
2. **`objects.yaml`** — 초기값(seed). DB에 닿지 못할 때의 대비책이자 `model_labels`의 출처.

DB에도 yaml에도 없는 클래스는 **신규 클래스**로 보고 `fallback`(fragile) 프로파일을 붙이고
`needs_confirmation=true`로 표시한다. 사진에서 추정한 값을 그대로 파지력에 반영하지 않기
위한 것이다(NFR-03a) — 확인 전에는 무조건 조심스럽게 다룬다.
"""
import logging
import os
import pathlib

import yaml

logger = logging.getLogger(__name__)

# 신규 클래스에 강제하는 프로파일. yaml의 fallback 블록이 우선이고, 이 값은 최후의 보루다.
HARD_FALLBACK = {
    "name_ko": "",
    "mass_g": 0.0,
    "fragile": True,
    "deformable": False,
    "transparent": False,
    "profile": "fragile",
}


def objects_yaml_path() -> pathlib.Path:
    """objects.yaml 위치. 설치본과 소스 양쪽에서 찾는다.

    `place_server.bins_yaml_path()`와 같은 순서다 — `__file__` 상대경로만 쓰면 colcon
    설치본에서 조용히 빈 목록이 된다.
    """
    override = os.environ.get("OBJECTS_YAML")
    if override:
        return pathlib.Path(override)
    try:
        from ament_index_python.packages import get_package_share_directory

        installed = pathlib.Path(get_package_share_directory("perception")) / "config" / "objects.yaml"
        if installed.is_file():
            return installed
    except Exception:
        pass
    return pathlib.Path(__file__).resolve().parents[1] / "config" / "objects.yaml"


class AttributeSource:
    """objects.yaml + object_attributes 테이블을 합쳐 속성을 돌려준다.

    DB는 **없어도 동작한다**. 카메라와 로봇만 있으면 검출은 되어야 하고, DB 장애로 인식
    전체가 멈추면 원인을 찾기도 어렵다. DB에 못 닿으면 yaml seed로 답하고 그 사실을 남긴다.
    """

    def __init__(self, yaml_path: pathlib.Path | None = None, database_url: str | None = None):
        self._yaml_path = yaml_path or objects_yaml_path()
        with self._yaml_path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        self._objects: dict = config.get("objects") or {}
        self._fallback: dict = {**HARD_FALLBACK, **(config.get("fallback") or {})}
        self._model_labels: dict = config.get("model_labels") or {}

        self._database_url = database_url if database_url is not None else os.environ.get("DATABASE_URL")
        self._db_warned = False

    # --- 모델 라벨 → class_name -------------------------------------------
    def class_name(self, model_label: str) -> str:
        """검출 모델이 낸 라벨을 시스템의 `class_name`으로 옮긴다 (개발계획.md D-7).

        표에 없는 라벨은 **그대로 통과시킨다.** 재학습으로 클래스가 늘었는데 표를 갱신하지
        않은 경우인데, 여기서 버리면 새 물체가 통째로 보이지 않게 된다. 통과시키면 아래
        속성 조회에서 신규 클래스로 잡혀 확인 대기로 올라간다 — 사람이 알아챌 수 있는 쪽이다.
        """
        mapped = self._model_labels.get(model_label)
        if mapped is None and self._model_labels:
            logger.warning("model_labels에 없는 라벨 '%s' — 신규 클래스로 처리한다", model_label)
        return mapped or model_label

    # --- 속성 조회 ---------------------------------------------------------
    def attributes(self, class_name: str) -> dict:
        """`DetectedObject`의 속성 필드를 채울 dict.

        반환 키: name_ko, mass_g, fragile, deformable, transparent, profile,
                attr_source, needs_confirmation
        """
        row = self._from_db(class_name)
        if row is not None:
            return row

        seed = self._objects.get(class_name)
        if seed is not None:
            return {
                "name_ko": seed.get("name_ko") or "",
                "mass_g": float(seed.get("mass_g") or 0.0),
                "fragile": bool(seed.get("fragile", True)),
                "deformable": bool(seed.get("deformable", False)),
                "transparent": bool(seed.get("transparent", False)),
                "profile": seed.get("profile") or "fragile",
                "attr_source": "yaml_seed",
                "needs_confirmation": False,
            }

        # 신규 클래스 — 확인 전까지 보수적으로 (FR-05b, NFR-03a)
        return {
            "name_ko": self._fallback.get("name_ko") or "",
            "mass_g": float(self._fallback.get("mass_g") or 0.0),
            "fragile": bool(self._fallback.get("fragile", True)),
            "deformable": bool(self._fallback.get("deformable", False)),
            "transparent": bool(self._fallback.get("transparent", False)),
            "profile": self._fallback.get("profile") or "fragile",
            "attr_source": "yaml_seed",
            "needs_confirmation": True,
        }

    def _from_db(self, class_name: str) -> dict | None:
        if not self._database_url:
            return None
        try:
            import psycopg
        except ImportError:
            self._warn_db_once("psycopg가 설치되어 있지 않다")
            return None
        try:
            with psycopg.connect(self._database_url, connect_timeout=3) as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT name_ko, mass_g, fragile, deformable, transparent, profile,"
                    " source, is_confirmed FROM object_attributes WHERE class_name = %s",
                    (class_name,),
                )
                row = cur.fetchone()
        except Exception as e:
            self._warn_db_once(str(e))
            return None
        if row is None:
            return None
        name_ko, mass_g, fragile, deformable, transparent, profile, source, is_confirmed = row
        return {
            "name_ko": name_ko or "",
            "mass_g": float(mass_g or 0.0),
            "fragile": bool(fragile),
            "deformable": bool(deformable),
            "transparent": bool(transparent),
            "profile": profile or "fragile",
            "attr_source": source or "yaml_seed",
            "needs_confirmation": not bool(is_confirmed),
        }

    def _warn_db_once(self, detail: str) -> None:
        """DB 장애 경고는 한 번만 낸다 — 프레임마다 찍으면 로그가 그것만 남는다."""
        if not self._db_warned:
            logger.warning("object_attributes 조회 실패, objects.yaml seed로 대신한다: %s", detail)
            self._db_warned = True

"""물체 속성 조회: 모델 라벨 → class_name → 속성 (FR-05, FR-05b).

값의 출처는 하나, **`objects.yaml`**뿐이다(초기값이자 유일한 소스, `model_labels`의
출처이기도 하다). 예전에는 DB `object_attributes` 테이블을 먼저 보고 없으면 yaml로
내려오는 2단 구조였지만, 그 테이블에 실제로 값을 쓰는 코드가 없었다 — 웹 확인 UI는
있었어도(`ConfirmModal.tsx`) 그 대상을 채워 넣는 VLM 제안 기록(`vlm_client.py`)이 TODO
스텁으로 끝까지 구현되지 않아 확인 대기 목록이 항상 비어 있었다 — 그래서 테이블째
걷어냈다(`database/migrations/006_drop_object_attributes.sql`).

yaml에 없는 클래스는 **신규 클래스**로 보고 `fallback`(grip_level 5=가장 약하게)을
붙이고 `needs_confirmation=true`로 표시한다. 사진에서 추정한 값을 그대로 파지력에 반영하지
않기 위한 것이다(NFR-03a) — 확인 전에는 무조건 조심스럽게 다룬다.
"""
import logging
import os
import pathlib

import yaml

logger = logging.getLogger(__name__)

# 신규 클래스에 강제하는 파지력 단계. yaml의 fallback 블록이 우선이고, 이 값은 최후의 보루다.
# 5 = 가장 약하게(20N) — 미확인 물체는 조심스럽게 다루는 것이 FR-05b의 기본이다.
HARD_FALLBACK = {
    "name_ko": "",
    "mass_g": 0.0,
    "fragile": True,
    "deformable": False,
    "transparent": False,
    "grip_level": 5,
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
    """objects.yaml에서 속성을 돌려준다. 등록 안 된 클래스는 fallback으로 조심스럽게."""

    def __init__(self, yaml_path: pathlib.Path | None = None):
        self._yaml_path = yaml_path or objects_yaml_path()
        with self._yaml_path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        self._objects: dict = config.get("objects") or {}
        self._fallback: dict = {**HARD_FALLBACK, **(config.get("fallback") or {})}
        self._model_labels: dict = config.get("model_labels") or {}

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

        반환 키: name_ko, mass_g, fragile, deformable, transparent, grip_level,
                attr_source, needs_confirmation
        """
        seed = self._objects.get(class_name)
        if seed is not None:
            return {
                "name_ko": seed.get("name_ko") or "",
                "mass_g": float(seed.get("mass_g") or 0.0),
                "fragile": bool(seed.get("fragile", True)),
                "deformable": bool(seed.get("deformable", False)),
                "transparent": bool(seed.get("transparent", False)),
                "grip_level": int(seed.get("grip_level") or 5),
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
            "grip_level": int(self._fallback.get("grip_level") or 5),
            "attr_source": "yaml_seed",
            "needs_confirmation": True,
        }

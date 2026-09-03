"""실제 검출된 object_id / 등록된 bin_id로만 그라운딩. 미검출 물체 지시는 거부 응답 반환 (FR-11).

LLM 프롬프트에 "검출된 것만 쓰라"고 적는 것만으로는 보장되지 않는다. 출력이 나온 뒤
여기서 실제 목록과 대조하는 것이 그라운딩의 실체다.
"""
import os
import pathlib

import yaml

BINS_YAML = pathlib.Path(
    os.environ.get(
        "BINS_YAML",
        pathlib.Path(__file__).resolve().parents[2] / "control" / "config" / "bins.yaml",
    )
)


def load_bins(path: pathlib.Path | None = None) -> dict:
    """bins.yaml의 목적지 정의를 읽는다. LLM이 고를 수 있는 bin_id의 전체 집합."""
    path = path or BINS_YAML
    with path.open(encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("bins") or {}


def world_objects(world_state: dict) -> dict[str, dict]:
    """object_id → DetectedObject 매핑."""
    return {o["object_id"]: o for o in world_state.get("objects", [])}


def describe_for_prompt(world_state: dict, bins: dict) -> str:
    """LLM 프롬프트에 넣을 월드 상태 요약.

    좌표·파지 후보는 넣지 않는다 — LLM이 정할 것이 아니고, 넣으면 토큰만 늘고
    LLM이 좌표를 지어낼 여지가 생긴다.
    """
    lines = ["[검출된 물체]"]
    objects = world_state.get("objects", [])
    if not objects:
        lines.append("(없음)")
    for o in objects:
        attrs = []
        if o.get("fragile"):
            attrs.append("파손위험")
        if o.get("deformable"):
            attrs.append("변형가능")
        if o.get("transparent"):
            attrs.append("투명")
        if o.get("needs_confirmation"):
            attrs.append("미확인 신규품목")
        if not o.get("graspable", True):
            attrs.append(f"파지불가({o.get('not_graspable_reason') or '사유없음'})")
        mass = o.get("mass_g") or 0
        lines.append(
            f"- {o['object_id']}: {o.get('name_ko') or o.get('class_name')}"
            f" (class={o.get('class_name')}, 무게={mass:g}g"
            + (f", {', '.join(attrs)}" if attrs else "")
            + ")"
        )

    lines.append("")
    lines.append("[배치 가능한 목적지]")
    if not bins:
        lines.append("(없음)")
    for bin_id, spec in bins.items():
        lines.append(f"- {bin_id}: {spec.get('name_ko') or bin_id}")
    return "\n".join(lines)


def check_grounding(steps: list, world_state: dict, bins: dict) -> str | None:
    """그라운딩 위반 사유를 반환한다. 문제가 없으면 None.

    steps는 LlmStep 리스트.
    """
    objects = world_objects(world_state)
    for i, step in enumerate(steps):
        where = f"{i + 1}번째 스텝({step.skill})"
        if step.object_id not in objects:
            return f"{where}: '{step.object_id}'는 현재 인지된 물체 목록에 없습니다"
        if step.skill == "place_into":
            if not step.bin_id:
                return f"{where}: place_into에 bin_id가 없습니다"
            if step.bin_id not in bins:
                return f"{where}: '{step.bin_id}'는 등록된 목적지가 아닙니다"
    return None

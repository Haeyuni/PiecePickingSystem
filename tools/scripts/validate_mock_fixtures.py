#!/usr/bin/env python3
"""mock 픽스처가 sort_msgs 스키마와 어긋나지 않는지 검사한다.

개발계획.md 7절 리스크 "mock과 실제 world_state의 괴리"에 대한 방어선이다.
`.msg` 파일의 필드 목록을 파싱해 `data/mock/*.json`의 키와 대조한다 — 필드를
추가·개명하고 픽스처를 안 고치면 여기서 걸린다.

사용법: python3 scripts/validate_mock_fixtures.py
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MSG_DIR = ROOT / "sort_msgs" / "msg"
MOCK_DIR = ROOT / "data" / "mock"

# ROS2 msg의 상수 정의(`string FOO=bar`)와 필드 정의(`string foo`)를 구분한다.
FIELD_RE = re.compile(r"^\s*([A-Za-z_][\w/\[\]]*)\s+([a-z_][a-z0-9_]*)\s*(?:#.*)?$")


def msg_fields(name: str) -> set[str]:
    fields = set()
    for line in (MSG_DIR / f"{name}.msg").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0]
        if not line.strip() or "=" in line:
            continue
        m = FIELD_RE.match(line)
        if m:
            fields.add(m.group(2))
    return fields


def check(obj: dict, expected: set[str], where: str, errors: list[str]) -> None:
    actual = set(obj)
    for missing in sorted(expected - actual):
        errors.append(f"{where}: 필드 누락 '{missing}'")
    for extra in sorted(actual - expected):
        errors.append(f"{where}: 스키마에 없는 필드 '{extra}'")


def main() -> int:
    world_fields = msg_fields("WorldState")
    object_fields = msg_fields("DetectedObject")
    candidate_fields = msg_fields("GraspCandidate")

    # ROS2의 builtin_interfaces/Time은 JSON에서 stamp 객체로 직렬화된다
    files = sorted(MOCK_DIR.glob("world_state_*.json"))
    if not files:
        print(f"픽스처가 없습니다: {MOCK_DIR}", file=sys.stderr)
        return 1

    errors: list[str] = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        check(data, world_fields, path.name, errors)

        for obj in data.get("objects", []):
            where = f"{path.name}:{obj.get('object_id', '?')}"
            check(obj, object_fields, where, errors)
            for i, cand in enumerate(obj.get("grasp_candidates", [])):
                check(cand, candidate_fields, f"{where}.grasp_candidates[{i}]", errors)

    for e in errors:
        print(f"NG {e}")
    print(f"\n픽스처 {len(files)}개 검사, 불일치 {len(errors)}건")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

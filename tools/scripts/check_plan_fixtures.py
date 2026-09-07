#!/usr/bin/env python3
"""mock 픽스처 5종을 /internal/plan에 넣고 기대 동작을 확인한다 (개발계획.md B1 완료 기준).

LLM을 실제로 호출하므로 API 비용이 든다(요청 5건). 평가셋 전체 실행(B5)이 아니라
계층이 붙었는지 확인하는 스모크 테스트다.

사용법: python3 scripts/check_plan_fixtures.py [--url http://localhost:8100]
"""
import argparse
import json
import pathlib
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
MOCK_DIR = ROOT / "data" / "mock"

# (픽스처, 명령, 기대 상태, 추가 검사)
CASES = [
    ("world_state_normal", "투명한 것만 왼쪽 박스로 옮겨줘", "approved", "transparent_only"),
    ("world_state_normal", "전부 왼쪽 박스로 옮겨줘", "approved", "all_three"),
    ("world_state_unknown_class", "전부 오른쪽 박스에 넣어줘", "approved", "fallback_forced"),
    ("world_state_missing_target", "선크림을 왼쪽 박스에 넣어줘", "rejected", None),
    ("world_state_not_graspable", "전부 왼쪽 박스로 옮겨줘", "approved", "excludes_obj_005"),
    ("world_state_empty", "네일을 왼쪽 박스로 옮겨줘", "rejected", None),
]


def post(url: str, payload: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def extra_check(name: str, world: dict, steps: list) -> str | None:
    """시나리오별 추가 검사. 문제가 있으면 사유를 반환한다."""
    if name == "transparent_only":
        # 테이프(transparent)만 옮겨야 한다. 현재 클래스 3종 중 투명한 것은 하나뿐이다.
        picked = {s["object_id"] for s in steps if s["skill"] == "pick"}
        if picked != {"obj_003"}:
            return f"투명한 물체만 집어야 하는데 {sorted(picked)}를 집었다"
    elif name == "fallback_forced":
        # 미확인 신규 클래스도 "전부" 지시에 포함되어야 한다 (FR-05b: 확인 전이라고 빼는 게 아니라,
        # 보수적 프로파일로 다루는 것이다). 빠지면 아래 프로파일 검사가 헛통과한다.
        touched = {s["object_id"] for s in steps}
        if "obj_004" not in touched:
            return "미확인 클래스 obj_004가 시퀀스에서 통째로 빠졌다 (제외가 아니라 fallback 처리 대상)"
        for s in steps:
            if s["object_id"] == "obj_004" and s["profile"] != "fragile":
                return f"미확인 클래스 obj_004의 프로파일이 {s['profile']} (fragile로 강제되어야 함)"
    elif name == "all_three":
        # "전부"는 파지 가능한 물체를 하나도 빠뜨리지 않아야 한다. 조심스러운 속성
        # (파손위험·투명)은 제외 사유가 아니다 — 프로파일로 다뤄질 뿐이다.
        picked = {s["object_id"] for s in steps if s["skill"] == "pick"}
        expected = {o["object_id"] for o in world["objects"] if o.get("graspable", True)}
        if picked != expected:
            return f"'전부' 지시인데 {sorted(expected - picked)}가 누락됐다 (집은 것: {sorted(picked)})"
    elif name == "excludes_obj_005":
        if any(s["object_id"] == "obj_005" for s in steps):
            return "파지 불가 물체 obj_005가 시퀀스에 포함됐다"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8100")
    args = ap.parse_args()

    failures = []
    for fixture, command, expected, check_name in CASES:
        world = json.loads((MOCK_DIR / f"{fixture}.json").read_text(encoding="utf-8"))
        status, body = post(f"{args.url}/internal/plan", {
            "schema_version": "1.0.0",
            "trace_id": f"tr-check-{fixture}",
            "command_text": command,
            "world_state": world,
        })
        actual = body.get("validation_status")
        steps = body.get("steps") or []
        ok = actual == expected
        detail = extra_check(check_name, world, steps) if ok and check_name else None
        if detail:
            ok = False

        print(f"{'OK ' if ok else 'NG '} {fixture:28s} → {actual} (HTTP {status})")
        if actual == "rejected":
            print(f"      사유: {body.get('validation_reason')}")
        for s in steps:
            extra = s.get("bin_id") or ("grasp_pose" if s.get("grasp_pose") else "")
            print(f"      - {s['skill']:11s} {s['object_id']} profile={s['profile']} {extra}")
        if not ok:
            failures.append(f"{fixture}: {detail or f'기대 {expected}, 실제 {actual}'}")

    print()
    for f in failures:
        print(f"실패 — {f}")
    print(f"{len(CASES) - len(failures)}/{len(CASES)} 통과")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

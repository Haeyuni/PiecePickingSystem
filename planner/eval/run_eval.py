#!/usr/bin/env python3
"""평가셋 채점 (개발계획.md 4.2절 B5, BR 4.1절 4단계 산출물).

`test_set.jsonl`의 각 행을 `/internal/plan`에 넣고 기대 결과와 대조해 **명령 정확도**와
**거부 정확도**(BR 3.1절)를 산출한다. 결과는 `eval_runs`/`eval_results`에 `prompt_version`
별로 남는다 — 같은 조건을 다시 돌리지 않기 위한 기록이기도 하다(7절 LLM 비용 리스크).

**LLM을 케이스마다 한 번씩 호출한다.** 프롬프트 버전이 바뀌지 않았다면 다시 돌릴 이유가
없다. `--dry-run`으로 호출 없이 평가셋만 검사할 수 있다.

**planner 컨테이너 안에서 돌린다.** psycopg와 DATABASE_URL이 거기 있고, 호스트에 파이썬
의존성을 새로 깔지 않아도 된다. `tests/`를 이미지에 굽는 것과 같은 이유다.

사용법:
    docker compose exec planner python eval/run_eval.py            # 전체 실행 + DB 기록
    docker compose exec planner python eval/run_eval.py --dry-run  # LLM 호출 없이 평가셋 검사
    docker compose exec planner python eval/run_eval.py --category "모호한 지시"
    docker compose exec planner python eval/run_eval.py --no-db    # 기록 없이 점수만

    python3 planner/eval/run_eval.py --dry-run   # 호스트에서도 검사만은 의존성 없이 된다
"""
import argparse
import json
import os
import pathlib
import sys
import urllib.error
import urllib.request
import uuid

ROOT = pathlib.Path(__file__).resolve().parents[2]
TEST_SET = pathlib.Path(__file__).resolve().parent / "test_set.jsonl"
# 컨테이너에서는 ./data가 /data로 마운트된다 (docker-compose.yml). 호스트에서 직접 돌릴 때는
# 저장소 구조를 그대로 쓴다.
MOCK_DIR = pathlib.Path(os.environ.get("MOCK_DIR", ROOT / "data" / "mock"))

# 평가셋 자체의 버전. 케이스를 고치면 올린다 — 점수는 평가셋이 같을 때만 비교 가능하다.
DATASET_VERSION = "eval-v2"

# 기대 결과에서 쓸 수 있는 키. 오타가 조용히 "검사 안 함"이 되지 않도록 화이트리스트로 막는다.
EXPECT_KEYS = {"status", "pick_objects", "pick_objects_exclude", "pick_objects_include",
               "bins", "profiles"}


def load_cases(path: pathlib.Path) -> list[dict]:
    cases = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path}:{i} JSON 파싱 실패 — {e}")
    return cases


def validate_cases(cases: list[dict]) -> list[str]:
    """평가셋 자체의 결함을 잡는다. 채점 전에 돌린다 — 틀린 평가셋으로 낸 점수는 무의미하다."""
    problems = []
    seen = set()
    for c in cases:
        cid = c.get("case_id", "(case_id 없음)")
        if cid in seen:
            problems.append(f"{cid}: case_id 중복")
        seen.add(cid)
        for field in ("category", "fixture", "command", "expect", "why"):
            if not c.get(field):
                problems.append(f"{cid}: '{field}' 누락")
        fixture = MOCK_DIR / f"{c.get('fixture')}.json"
        if not fixture.is_file():
            problems.append(f"{cid}: 픽스처 없음 — {fixture}")
        expect = c.get("expect") or {}
        for key in set(expect) - EXPECT_KEYS:
            problems.append(f"{cid}: 알 수 없는 기대 키 '{key}'")
        if expect.get("status") not in ("approved", "rejected", "any"):
            problems.append(f"{cid}: status는 approved/rejected/any 중 하나여야 한다")
        # 승인 케이스가 아무 것도 검사하지 않으면 통과가 의미 없다
        if expect.get("status") == "approved" and not (
                set(expect) & {"pick_objects", "pick_objects_include",
                               "pick_objects_exclude", "bins", "profiles"}):
            problems.append(f"{cid}: approved 케이스인데 시퀀스 내용을 하나도 검사하지 않는다")
        if not fixture.is_file():
            continue
        world = json.loads(fixture.read_text(encoding="utf-8"))
        known = {o["object_id"] for o in world.get("objects", [])}
        for key in ("pick_objects", "pick_objects_include", "pick_objects_exclude"):
            for oid in expect.get(key) or []:
                if oid not in known:
                    problems.append(f"{cid}: {key}의 '{oid}'가 픽스처에 없다")
        for oid in list(expect.get("bins") or {}) + list(expect.get("profiles") or {}):
            if oid not in known:
                problems.append(f"{cid}: '{oid}'가 픽스처에 없다")
    return problems


def post(url: str, payload: dict, timeout: float = 180.0) -> tuple[int, dict]:
    req = urllib.request.Request(
        url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def grade(expect: dict, body: dict) -> list[str]:
    """기대와 실제의 차이를 사람이 읽을 수 있는 문장으로 반환한다. 빈 리스트면 통과."""
    fails = []
    status = body.get("validation_status")
    steps = body.get("steps") or []
    picked = [s["object_id"] for s in steps if s["skill"] == "pick"]
    picked_set = set(picked)

    wanted_status = expect.get("status")
    if wanted_status != "any" and status != wanted_status:
        fails.append(f"상태가 {status} (기대 {wanted_status})")
        # 거부된 응답에는 검사할 시퀀스가 없다
        if status == "rejected":
            return fails

    if status == "rejected":
        return fails

    if "pick_objects" in expect:
        wanted = set(expect["pick_objects"])
        if picked_set != wanted:
            missing = sorted(wanted - picked_set)
            extra = sorted(picked_set - wanted)
            detail = []
            if missing:
                detail.append(f"누락 {missing}")
            if extra:
                detail.append(f"불필요 {extra}")
            fails.append("집은 물체 불일치 — " + ", ".join(detail))
    for oid in expect.get("pick_objects_include") or []:
        if oid not in picked_set:
            fails.append(f"'{oid}'가 시퀀스에 없다")
    for oid in expect.get("pick_objects_exclude") or []:
        if oid in picked_set:
            fails.append(f"'{oid}'가 시퀀스에 들어갔다 (들어가면 안 되는 물체)")

    # 같은 물체를 두 번 집으면 두 번째 place_into는 갈 곳이 없다
    if len(picked) != len(picked_set):
        fails.append(f"같은 물체를 여러 번 집는다 — {picked}")

    placed = {s["object_id"]: s.get("bin_id") for s in steps if s["skill"] == "place_into"}
    for oid, want_bin in (expect.get("bins") or {}).items():
        if placed.get(oid) != want_bin:
            fails.append(f"'{oid}'의 목적지가 {placed.get(oid)} (기대 {want_bin})")

    profiles = {s["object_id"]: s.get("profile") for s in steps}
    for oid, want_profile in (expect.get("profiles") or {}).items():
        if profiles.get(oid) != want_profile:
            fails.append(f"'{oid}'의 프로파일이 {profiles.get(oid)} (기대 {want_profile})")

    # 집었으면 놓아야 한다. 검증기가 막지만, 막혔는지 여기서도 본다.
    for oid in picked_set:
        if oid not in placed:
            fails.append(f"'{oid}'를 집고 놓지 않는다")
    return fails


def record(rows: list[dict], prompt_version: str, model: str) -> str | None:
    """eval_runs/eval_results에 남긴다. 기록 실패가 점수 산출을 막지는 않는다."""
    try:
        import psycopg
    except ImportError:
        print("psycopg가 없어 DB 기록을 건너뛴다 (--no-db와 동일)", file=sys.stderr)
        return None
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("DATABASE_URL이 없어 DB 기록을 건너뛴다", file=sys.stderr)
        return None

    run_id = str(uuid.uuid4())
    try:
        with psycopg.connect(url, connect_timeout=5) as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO eval_runs (run_id, prompt_version, llm_model_version, dataset_version)"
                " VALUES (%s, %s, %s, %s)",
                (run_id, prompt_version, model, DATASET_VERSION),
            )
            for r in rows:
                cur.execute(
                    "INSERT INTO eval_results (result_id, run_id, test_case_id, category,"
                    " expected_pass, actual_pass, model_output) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (str(uuid.uuid4()), run_id, r["case_id"], r["category"],
                     r["expected_status"] != "rejected", r["passed"],
                     json.dumps(r["response"], ensure_ascii=False)),
                )
            conn.commit()
        return run_id
    except Exception as e:
        print(f"eval 기록 실패: {e}", file=sys.stderr)
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("PLANNER_URL", "http://localhost:8100"))
    ap.add_argument("--test-set", type=pathlib.Path, default=TEST_SET)
    ap.add_argument("--category", action="append", help="이 카테고리만 실행 (여러 번 지정 가능)")
    ap.add_argument("--case", action="append", help="이 case_id만 실행")
    ap.add_argument("--no-db", action="store_true", help="eval_runs/eval_results에 기록하지 않는다")
    ap.add_argument("--dry-run", action="store_true", help="LLM 호출 없이 평가셋 유효성만 검사")
    ap.add_argument("--json-out", type=pathlib.Path, help="케이스별 결과를 JSON으로 저장")
    args = ap.parse_args()

    cases = load_cases(args.test_set)
    problems = validate_cases(cases)
    if problems:
        print(f"평가셋 결함 {len(problems)}건:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"평가셋 검사 통과: {len(cases)}케이스 ({DATASET_VERSION})")
    if args.dry_run:
        return 0

    if args.category:
        cases = [c for c in cases if c["category"] in args.category]
    if args.case:
        cases = [c for c in cases if c["case_id"] in args.case]
    if not cases:
        print("실행할 케이스가 없다")
        return 1

    try:
        with urllib.request.urlopen(f"{args.url}/health", timeout=10) as r:
            health = json.loads(r.read())
    except Exception as e:
        print(f"planner에 연결하지 못했다 ({args.url}): {e}")
        return 1
    prompt_version = health.get("prompt_version", "unknown")
    model = health.get("llm_model", "unknown")
    print(f"planner: prompt_version={prompt_version} model={model}\n")

    rows = []
    for c in cases:
        world = json.loads((MOCK_DIR / f"{c['fixture']}.json").read_text(encoding="utf-8"))
        http_status, body = post(f"{args.url}/internal/plan", {
            "schema_version": "1.0.0",
            "trace_id": f"tr-eval-{c['case_id']}",
            "command_text": c["command"],
            "world_state": world,
        })
        fails = grade(c["expect"], body)
        passed = not fails
        rows.append({
            "case_id": c["case_id"], "category": c["category"],
            "phrasing": c.get("phrasing"), "command": c["command"],
            "fixture": c["fixture"], "expected_status": c["expect"].get("status"),
            "actual_status": body.get("validation_status"),
            "http_status": http_status, "passed": passed, "fails": fails,
            "response": body,
        })

        mark = "OK " if passed else "NG "
        print(f"{mark} [{c['category']}] {c['case_id']}")
        print(f"      명령: {c['command']}")
        print(f"      결과: {body.get('validation_status')} (HTTP {http_status})")
        if body.get("validation_reason"):
            print(f"      사유: {body['validation_reason']}")
        for s in body.get("steps") or []:
            print(f"      - {s['skill']:11s} {s['object_id']} profile={s['profile']}"
                  f" {s.get('bin_id') or ''}")
        for f in fails:
            print(f"      ✗ {f}")
        print()

    # --- 지표 (BR 3.1절) ---------------------------------------------------
    # 명령 정확도: 수행되어야 하는 지시를 올바른 시퀀스로 옮긴 비율
    # 거부 정확도: 불가능·모호한 지시를 실제로 거부한 비율
    command_cases = [r for r in rows if r["expected_status"] != "rejected"]
    reject_cases = [r for r in rows if r["expected_status"] == "rejected"]
    def rate(items):
        return (sum(1 for i in items if i["passed"]) / len(items) * 100) if items else float("nan")

    print("=" * 68)
    print(f"명령 정확도  {rate(command_cases):5.1f}%  "
          f"({sum(1 for r in command_cases if r['passed'])}/{len(command_cases)})")
    print(f"거부 정확도  {rate(reject_cases):5.1f}%  "
          f"({sum(1 for r in reject_cases if r['passed'])}/{len(reject_cases)})")
    print(f"전체        {rate(rows):5.1f}%  "
          f"({sum(1 for r in rows if r['passed'])}/{len(rows)})")
    print()
    print("카테고리별")
    for cat in dict.fromkeys(r["category"] for r in rows):
        sub = [r for r in rows if r["category"] == cat]
        print(f"  {cat:18s} {rate(sub):5.1f}%  ({sum(1 for r in sub if r['passed'])}/{len(sub)})")

    failed = [r for r in rows if not r["passed"]]
    if failed:
        print(f"\n실패 {len(failed)}건")
        for r in failed:
            print(f"  - {r['case_id']}: {'; '.join(r['fails'])}")

    run_id = None
    if not args.no_db:
        run_id = record(rows, prompt_version, model)
        if run_id:
            print(f"\neval_runs 기록: run_id={run_id} dataset={DATASET_VERSION}"
                  f" prompt={prompt_version}")

    if args.json_out:
        args.json_out.write_text(json.dumps({
            "dataset_version": DATASET_VERSION, "prompt_version": prompt_version,
            "llm_model": model, "run_id": run_id, "results": rows,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"결과 저장: {args.json_out}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""web 백엔드 차단 규칙 확인 (웹_인터페이스_정의서.md 2.1·2.6절).

LLM을 호출하지 않는다 — `/api/robot/home`으로 로봇을 busy로 만들고 그 사이의 동작을 본다.
비용 없이 매번 돌릴 수 있어야 하는 검사다.

확인 항목:
- busy일 때 새 명령은 백엔드가 409로 막는다 (UI 비활성화는 보조 수단일 뿐)
- busy일 때 홈 이동도 막힌다
- **정지만은 busy에서도 항상 허용된다** — busy일 때 가장 필요한 동작이므로

사용법: python3 scripts/check_web_gating.py [--url http://localhost:8000]
"""
import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request


def http(method: str, url: str, payload: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    args = ap.parse_args()
    base = args.url

    failures = []

    def expect(label: str, actual, wanted):
        ok = actual == wanted
        print(f"{'OK ' if ok else 'NG '} {label}: {actual} (기대 {wanted})")
        if not ok:
            failures.append(label)

    expect("idle에서 홈 이동 허용", http("POST", f"{base}/api/robot/home")[0], 202)

    # 홈 이동을 백그라운드로 걸어 busy 상태를 만든다
    threading.Thread(target=lambda: http("POST", f"{base}/api/robot/home"), daemon=True).start()
    time.sleep(0.3)

    mode = http("GET", f"{base}/health")[1]["robot_state"]["mode"]
    expect("홈 이동 중 mode", mode, "busy")

    status, body = http("POST", f"{base}/api/commands",
                        {"schema_version": "1.0.0", "command_text": "캔을 왼쪽 박스로"})
    expect("busy에서 명령 거부", status, 409)
    expect("거부 코드", body.get("error", {}).get("code"), "ROBOT_BUSY")

    expect("busy에서 홈 이동 거부", http("POST", f"{base}/api/robot/home")[0], 409)

    # 정지는 2.1절 차단 규칙의 유일한 예외 — busy에서도 통과해야 한다
    status, body = http("POST", f"{base}/api/robot/stop")
    expect("busy에서도 정지 허용", status, 200)
    expect("정지 응답", body.get("stopped"), True)

    time.sleep(1.2)
    expect("복귀 후 mode", http("GET", f"{base}/health")[1]["robot_state"]["mode"], "idle")
    expect("idle 복귀 후 홈 이동 허용", http("POST", f"{base}/api/robot/home")[0], 202)

    # stop/home도 이력에 남아야 한다 (2.6절 — 분류 작업과 시간순으로 같이 보이도록)
    time.sleep(1.2)
    items = http("GET", f"{base}/api/executions?skill=stop&limit=5")[1].get("items", [])
    expect("stop 이력 기록", len(items) > 0, True)
    items = http("GET", f"{base}/api/executions?skill=home&limit=5")[1].get("items", [])
    expect("home 이력 기록", len(items) > 0, True)

    print()
    for f in failures:
        print(f"실패 — {f}")
    print("차단 규칙 통과" if not failures else f"{len(failures)}건 실패")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

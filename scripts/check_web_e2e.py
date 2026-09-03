#!/usr/bin/env python3
"""web 백엔드 E2E 확인 (개발계획.md B2 완료 기준).

MOCK_MODE=1에서 로봇·GPU 없이: 명령 전송 → WebSocket으로 진행 상황 수신 → 실행 결과 →
이력 조회까지 한 바퀴를 돈다. planner를 실제로 호출하므로 LLM 비용이 든다(명령 1건).

사용법: python3 scripts/check_web_e2e.py [--url http://localhost:8000]
"""
import argparse
import asyncio
import json
import sys
import urllib.request

import websockets


def http(method: str, url: str, payload: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    args = ap.parse_args()
    ws_url = args.url.replace("http://", "ws://") + "/ws/live"

    failures = []
    events = []

    async with websockets.connect(ws_url) as ws:
        # 1) 명령 전송
        status, body = http("POST", f"{args.url}/api/commands",
                            {"schema_version": "1.0.0",
                             "command_text": "깨지기 쉬운 것만 왼쪽 박스로 옮겨줘"})
        print(f"POST /api/commands → HTTP {status} {body}")
        if status != 202:
            return 1
        trace_id = body["trace_id"]

        # 2) 실행이 끝날 때까지 이벤트 수신
        try:
            while True:
                event = json.loads(await asyncio.wait_for(ws.recv(), timeout=180))
                events.append(event)
                if event["type"] == "execution_progress":
                    print(f"  진행: {event['skill']:11s} {event['phase']}")
                elif event["type"] == "execution_result":
                    print(f"  결과: request={event['request_id']} success={event['success']}")
                elif event["type"] == "robot_state":
                    print(f"  상태: mode={event['mode']} skill={event['current_skill']}")
                # 마지막 place_into 결과 후 idle로 돌아오면 종료
                if (event["type"] == "robot_state" and event["mode"] == "idle"
                        and any(e["type"] == "execution_result" for e in events)):
                    snap = http("GET", f"{args.url}/api/traces/{trace_id}")[1]
                    if all(s["status"] in ("success", "failure") for s in snap["steps"]):
                        break
        except asyncio.TimeoutError:
            failures.append("WebSocket 이벤트 타임아웃")

    # 3) 검증
    kinds = [e["type"] for e in events]
    if "execution_progress" not in kinds:
        failures.append("execution_progress 이벤트가 오지 않았다")
    if "execution_result" not in kinds:
        failures.append("execution_result 이벤트가 오지 않았다")
    if "robot_state" not in kinds:
        failures.append("robot_state 이벤트가 오지 않았다")

    # 진행 phase가 액션 정의 순서대로 왔는가
    pick_phases = [e["phase"] for e in events
                   if e["type"] == "execution_progress" and e["skill"] == "pick"]
    if pick_phases != ["approaching", "contact_detected", "lifting", "verifying"]:
        failures.append(f"pick phase 순서가 액션 정의와 다르다: {pick_phases}")

    # 4) 스냅샷 폴백 (2.3절)
    status, snap = http("GET", f"{args.url}/api/traces/{trace_id}")
    print(f"\nGET /api/traces/{trace_id} → HTTP {status} "
          f"validation={snap.get('validation_status')} steps={len(snap.get('steps', []))}")
    if snap.get("validation_status") != "approved":
        failures.append(f"trace 스냅샷의 검증 상태가 approved가 아니다: {snap.get('validation_status')}")
    if not all(s["status"] == "success" for s in snap.get("steps", [])):
        failures.append("모든 스텝이 성공으로 끝나지 않았다")

    # 5) 이력 기록 (2.5절)
    status, hist = http("GET", f"{args.url}/api/executions?trace_id={trace_id}")
    print(f"GET /api/executions → {len(hist.get('items', []))}건 기록")
    for item in hist.get("items", []):
        print(f"  {item['skill_name']:11s} {item['object_id']} "
              f"profile={item['profile_used']} result={item['result']} "
              f"visual={item['visual_verification_passed']}")
    if len(hist.get("items", [])) != len(snap.get("steps", [])):
        failures.append("execution_logs 건수가 시퀀스 스텝 수와 다르다")

    print()
    for f in failures:
        print(f"실패 — {f}")
    print("E2E 통과" if not failures else f"{len(failures)}건 실패")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

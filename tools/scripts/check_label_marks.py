#!/usr/bin/env python3
"""planner의 /internal/label-marks를 실제 VLM으로 한 번 돌려 본다 (스모크 체크).

단위 테스트(services/planner/tests/test_label_marks.py)는 VLM을 부르지 않는다 — 요청 파싱과
에러 형식만 본다. 여기서는 진짜 모델이 번호가 그려진 프레임을 읽고 등록된 클래스 어휘로
답하는지를 본다. **명령 1건만큼 과금된다.**

  docker compose up -d db planner
  python3 tools/scripts/check_label_marks.py                     # 기본 프레임 사용
  python3 tools/scripts/check_label_marks.py --image 다른_before.png --marks 1,2,3

기본 프레임은 vlm_sam_test.py가 남긴 *_before.png(LLM에 보내는 번호 오버레이)다.
없으면 먼저 그것을 돌린다:
  .venv/bin/python tools/scripts/vlm_sam_test.py test_image/scene1.png --device cpu
"""
import argparse
import json
import pathlib
import sys
import urllib.error
import urllib.request
import uuid

REPO = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_IMAGE = REPO / "test_result" / "scene1_before.png"


def multipart(fields: dict, image_name: str, image_bytes: bytes) -> tuple[bytes, str]:
    """requests 없이 multipart 본문을 만든다 — 저장소의 다른 check_* 스크립트처럼
    stdlib만 쓴다(런타임 이미지에 없는 의존성을 검증 스크립트가 요구하지 않게)."""
    boundary = f"----check{uuid.uuid4().hex}"
    parts = []
    for key, value in fields.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n"
                     f"{value}\r\n".encode())
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
        f"filename=\"{image_name}\"\r\nContent-Type: image/png\r\n\r\n".encode())
    parts.append(image_bytes + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8100")
    ap.add_argument("--image", default=str(DEFAULT_IMAGE), help="번호가 그려진 프레임")
    ap.add_argument("--marks", default="1,2,3,4,5,6,7,8", help="쉼표로 구분한 번호")
    args = ap.parse_args()

    image = pathlib.Path(args.image)
    if not image.exists():
        print(f"이미지가 없다: {image}", file=sys.stderr)
        return 1

    marks = args.marks
    trace_id = f"tr-check-{uuid.uuid4().hex[:8]}"
    body, content_type = multipart(
        {"mark_ids": marks, "trace_id": trace_id}, image.name, image.read_bytes())

    print(f"POST {args.url}/internal/label-marks  image={image.name} marks={marks}")
    request = urllib.request.Request(f"{args.url}/internal/label-marks", data=body,
                                     headers={"Content-Type": content_type}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            status, payload = response.status, json.loads(response.read())
    except urllib.error.HTTPError as e:
        status, payload = e.code, json.loads(e.read() or b"{}")
    except urllib.error.URLError as e:
        print(f"planner에 연결하지 못했다 ({e}) — docker compose up -d db planner", file=sys.stderr)
        return 1

    print(f"HTTP {status}")
    if status != 200:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1

    failures = []
    if payload.get("trace_id") != trace_id:
        failures.append(f"trace_id가 돌아오지 않았다: {payload.get('trace_id')!r}")

    objects = [m for m in payload["marks"] if m["is_object"]]
    print(f"model={payload['model']} prompt={payload['prompt_version']} "
          f"마크 {len(payload['marks'])}개 중 물체 {len(objects)}개")
    for m in payload["marks"]:
        note = (f"{m['class_name']} ({m['name_ko']})" if m["is_object"]
                else (f"→ {m['part_of']}번의 조각" if m["part_of"] else "물체 아님"))
        # 속성도 VLM이 판단하므로 여기서 눈으로 확인할 수 있어야 한다 — 파지력을 정하는
        # 값이라, 이름만 맞고 grip_level이 엉뚱하면 로봇이 물체를 부순다.
        extra = ""
        if m["is_object"]:
            flags = [k for k in ("fragile", "deformable", "transparent") if m.get(k)]
            extra = (f"  g{m['grip_level']} {m['mass_g']:g}g"
                     + (f" [{','.join(flags)}]" if flags else ""))
        print(f"  [{m['mark_id']}] {note}  conf={m['confidence']:.2f}{extra}")

    if not objects:
        failures.append("물체로 판단된 마크가 하나도 없다 — 프레임이나 프롬프트를 확인한다")
    for m in objects:
        if not m["class_name"]:
            failures.append(f"마크 {m['mark_id']}: is_object인데 class_name이 비었다")
        if int(m.get("grip_level") or 0) not in (1, 2, 3, 4, 5):
            failures.append(f"마크 {m['mark_id']}: 모르는 grip_level {m.get('grip_level')!r} — "
                            "이 값이 그대로 파지력이 된다")
        if m["fragile"] and int(m.get("grip_level") or 0) != 5:
            failures.append(f"마크 {m['mark_id']}: fragile인데 grip_level={m.get('grip_level')} — "
                            "planner의 _normalize_marks가 5로 걸렀어야 한다")

    for f in failures:
        print(f"  실패: {f}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

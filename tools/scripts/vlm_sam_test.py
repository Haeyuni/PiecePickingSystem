#!/usr/bin/env python3
"""VLM 검출 → SAM 세그멘테이션 파이프라인을 정지 이미지 한 장으로 확인한다 (ROS 불필요).

확인하려는 것은 두 가지다.

  (1) 사진을 API로 보냈을 때 무엇이 있는지 알아보고 바운딩박스 좌표를 주는가
  (2) 그 좌표를 프롬프트로 SAM이 마스크를 뽑는가

두 가지 순서로 돌려 볼 수 있다. 기본값 --mode som은 SAM이 먼저 장면을 조각내고 VLM은 번호에
이름만 붙인다. --mode box는 반대로 VLM에게 박스를 물어 SAM 프롬프트로 쓴다 — 원래 설계이고
**gpt-4o에서는 som이 비교가 안 되게 정확해서** 기본값을 som으로 둔다
(이유와 실측은 docs/vlm_sam_pipeline.md 참조).

  # 기본 경로: SAM 먼저, VLM이 이름 붙이고, planner가 지시를 해석한다
  .venv/bin/python tools/scripts/vlm_sam_test.py services/planner/image.png \
      --command "치약 왼쪽으로"

  # SAM 전체 분할(CPU 26초)만 재사용하고 지시만 바꿔 가며
  .venv/bin/python tools/scripts/vlm_sam_test.py services/planner/image.png \
      --reuse-marks --command "우산 왼쪽으로"

  # 원래 물어본 경로: VLM 박스 → SAM 프롬프트
  .venv/bin/python tools/scripts/vlm_sam_test.py services/planner/image.png \
      --mode box --command "치약 왼쪽으로" --plan

  # VLM 응답까지 통째로 재사용 — API 비용 없이 뒤 단계만 반복
  .venv/bin/python tools/scripts/vlm_sam_test.py services/planner/image.png \
      --from-json data/samples/vlm_sam/image_scene.json

결과는 --out 디렉터리(기본 data/samples/vlm_sam)에 저장된다:
  *_marks.png   SAM 번호 오버레이(som)   *_marks.npz   번호별 마스크(som)
  *_boxes.png   물체 박스 오버레이        *_masks.png   물체 마스크 오버레이
  *_scene.json  VLM 원본 응답            *_result.json 마스크 통계까지 포함한 전체 결과

**여기서 나오는 마스크는 2D까지다.** 3D 좌표는 같은 프레임의 aligned depth가 있어야 하고,
같은 stem의 *_depth.npy / *_info.json이 있으면 마스크 기준 카메라 좌표(mm)까지 뽑는다
(perception_test_image.py:mask_3d와 같은 계산).
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "data" / "samples" / "vlm_sam"
# MobileSAM 가중치(~40MB). models/에 있으면 그것을 쓰고, 없으면 ultralytics가 이름을 보고
# 자동으로 내려받는다(첫 실행 때 한 번). *.pt는 커밋하지 않는다(.gitignore).
DEFAULT_SAM = str(REPO / "models" / "mobile_sam.pt") \
    if (REPO / "models" / "mobile_sam.pt").exists() else "mobile_sam.pt"

# 오버레이 색 (BGR). 물체 순서대로 돌려 쓴다.
COLORS = [(0, 200, 255), (0, 255, 120), (255, 120, 0), (255, 0, 200),
          (60, 60, 255), (255, 255, 0), (180, 0, 255), (0, 140, 255)]


def load_dotenv(path: pathlib.Path) -> None:
    """.env를 os.environ에 얹는다. 이미 셸에 있는 값은 덮지 않는다.

    python-dotenv를 쓰지 않는 이유는 이 스크립트 하나 때문에 planner/perception의
    requirements에 없는 의존성을 늘리지 않기 위해서다.
    """
    import os
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        # 값 뒤 인라인 주석 제거 (POSTGRES_DB=x   # 설명)
        if "  #" in value:
            value = value.split("  #")[0].strip()
        os.environ.setdefault(key, value)


def import_perception_module(name: str):
    """src/perception의 모듈을 불러온다. ROS를 import하지 않는 모듈만 여기로 온다
    (sam_marks는 numpy+cv2+ultralytics만 쓴다)."""
    package = REPO / "src" / "perception"
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
    return __import__(f"perception.{name}", fromlist=[name])


def import_planner_module(name: str):
    """services/planner/src의 모듈을 패키지 컨텍스트째로 불러온다.

    llm_client가 `from .schema import ...`를 쓰므로 파일 하나만 떼어 로드할 수 없다.
    """
    planner = REPO / "services" / "planner"
    if str(planner) not in sys.path:
        sys.path.insert(0, str(planner))
    return __import__(f"src.{name}", fromlist=[name])


# --- (1) VLM ------------------------------------------------------------------

def run_vlm(image_path: pathlib.Path, command: str | None, size: tuple[int, int],
            model: str | None, detail: str) -> tuple[dict, float]:
    vlm_detect = import_planner_module("vlm_detect")
    t0 = time.perf_counter()
    scene = vlm_detect.detect(image_path, command_text=command, image_size=size,
                              model=model, detail=detail)
    elapsed = time.perf_counter() - t0
    return {
        "mode": "box",
        "model": model or vlm_detect.model_name(),
        "prompt_version": vlm_detect.PROMPT_VERSION,
        "command_text": command,
        "image": str(image_path),
        "image_size": {"width": size[0], "height": size[1]},
        "raw": scene.model_dump(),
        "objects": vlm_detect.to_pixels(scene, *size),
        "target_object_ids": scene.target_object_ids,
        "refusal_reason": scene.refusal_reason,
        "elapsed_s": round(elapsed, 2),
    }, elapsed


def print_objects(scene: dict) -> None:
    objects = scene["objects"]
    if not objects:
        print("  (검출된 물체 없음)")
    for o in objects:
        x1, y1, x2, y2 = o["box_xyxy"]
        target = " ←지시대상" if o["object_id"] in scene["target_object_ids"] else ""
        new = " [신규클래스]" if o["is_new_class"] else ""
        print(f"  {o['object_id']:<18} {o['class_name']:<14} {o['name_ko']:<8} "
              f"conf={o['confidence']:.2f} box=({x1},{y1})-({x2},{y2}) "
              f"{x2 - x1}x{y2 - y1}px{new}{target}")
    # som 모드는 이 단계에서 지시를 보지 않으므로 대상/거부가 비어 있다 — 그건 (3)에서 나온다.
    if scene["refusal_reason"]:
        print(f"  거부: {scene['refusal_reason']}")
    elif scene["target_object_ids"]:
        print(f"  지시대상: {scene['target_object_ids']}")


# --- (2) SAM ------------------------------------------------------------------

def run_sam(image_bgr, boxes: list[list[int]], weights: str, device: str | None):
    """박스 프롬프트로 마스크를 뽑는다. 반환은 (H,W) bool 마스크 리스트."""
    from ultralytics import SAM

    model = SAM(weights)
    t0 = time.perf_counter()
    # 박스를 한 번에 넘긴다 — 이미지 임베딩을 한 번만 계산하므로 물체마다 부르는 것보다 빠르다.
    result = model.predict(image_bgr, bboxes=boxes, device=device, verbose=False)[0]
    elapsed = time.perf_counter() - t0
    if result.masks is None:
        return [], elapsed
    masks = result.masks.data.cpu().numpy() > 0.5
    h, w = image_bgr.shape[:2]
    if masks.shape[1:] != (h, w):        # 추론 해상도로 나오면 원본으로 되돌린다
        import cv2
        masks = np.stack([
            cv2.resize(m.astype(np.uint8), (w, h),
                       interpolation=cv2.INTER_NEAREST).astype(bool) for m in masks])
    return list(masks), elapsed


def sam_everything(image_bgr, weights: str, device: str | None, points_stride: int,
                   min_area: float, max_area: float, max_marks: int):
    """SAM everything 모드. 구현은 perception.sam_marks에 있다 — 노드와 이 스크립트가 같은
    코드를 써야 여기서 검증한 것이 노드에서도 성립한다."""
    return import_perception_module("sam_marks").segment_everything(
        image_bgr, weights, device, points_stride, min_area, max_area, max_marks)


def draw_marks(image_bgr, masks: list[np.ndarray]):
    return import_perception_module("sam_marks").draw_marks(image_bgr, masks)


def marks_to_objects(labels, masks: list[np.ndarray]) -> tuple[list[dict], list[np.ndarray]]:
    """VLM 응답(pydantic) → 물체 목록 + 물체별 마스크.

    합치기 규칙은 perception.sam_marks.merge_marks가 갖고 있다. 여기서는 object_id와
    박스만 덧붙인다 — 노드에서는 그 둘을 ObjectTracker와 _enrich가 만든다.
    """
    sam_marks = import_perception_module("sam_marks")
    merged = sam_marks.merge_marks(
        [m.model_dump() for m in labels.marks], masks,
        on_reject=lambda piece, parent:
            print(f"  마크 {piece}는 {parent}의 조각으로 보기 어렵다 — 합치지 않는다"))

    objects, out_masks = [], []
    counter: dict[str, int] = {}
    for item in merged:
        mask = item["mask"]
        ys, xs = np.nonzero(mask)
        class_name = item["class_name"] or "object"
        counter[class_name] = counter.get(class_name, 0) + 1
        objects.append({
            "object_id": f"{class_name}_{counter[class_name]}",
            "class_name": item["class_name"],
            "name_ko": item["name_ko"],
            "is_new_class": item["is_new_class"],
            "confidence": round(item["confidence"], 3),
            "mark_id": item["mark_id"],
            "merged_marks": item["merged_marks"],
            # 박스는 VLM이 아니라 마스크에서 뽑는다 — 이 경로의 핵심이다.
            "box_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        })
        out_masks.append(mask)
    return objects, out_masks


def rebuild_masks(objects: list[dict], mark_masks: list[np.ndarray]) -> list[np.ndarray]:
    """저장된 scene.json + marks.npz로 물체별 마스크를 되살린다 (--from-json 용)."""
    out = []
    for o in objects:
        mask = mark_masks[o["mark_id"] - 1].copy()
        for other in o.get("merged_marks", []):
            mask |= mark_masks[other - 1]
        out.append(mask)
    return out


def mask_stats(mask: np.ndarray, box: list[int]) -> dict:
    """마스크가 박스 프롬프트에 대해 말이 되는지 판단할 수 있는 값들.

    box_fill이 지나치게 낮으면 SAM이 물체가 아니라 무늬 일부를 잡은 것이고, outside_ratio가
    높으면 박스 밖으로 새어 나간 것(대개 배경/그림자까지 삼킨 경우)이다.
    """
    x1, y1, x2, y2 = box
    total = int(mask.sum())
    inside = int(mask[y1:y2, x1:x2].sum())
    box_area = max(1, (x2 - x1) * (y2 - y1))
    ys, xs = np.nonzero(mask)
    return {
        "mask_px": total,
        "box_area_px": box_area,
        "box_fill": round(inside / box_area, 3),            # 박스 안을 얼마나 채웠나
        "outside_ratio": round((total - inside) / max(1, total), 3),  # 박스 밖으로 샌 비율
        "centroid_xy": [int(xs.mean()), int(ys.mean())] if total else None,
        "bbox_of_mask": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if total else None,
    }


def mask_3d(mask: np.ndarray, depth_mm: np.ndarray, info: dict):
    """마스크 안쪽 유효 depth의 median으로 카메라 좌표 3D 점(mm).
    perception_test_image.py와 같은 계산 — 여기서도 같은 값이 나와야 파이프라인이 바뀐 게 없다."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None, 0.0
    z = depth_mm[ys, xs].astype(np.float32)
    valid = z > 0
    ratio = float(valid.mean())
    if not valid.any():
        return None, ratio
    zc = float(np.median(z[valid]))
    uc, vc = float(xs[valid].mean()), float(ys[valid].mean())
    return ((uc - info["cx"]) * zc / info["fx"],
            (vc - info["cy"]) * zc / info["fy"], zc), ratio


# --- 오버레이 -----------------------------------------------------------------

def draw_boxes(image_bgr, objects: list[dict], targets: list[str]):
    import cv2
    canvas = image_bgr.copy()
    for i, o in enumerate(objects):
        x1, y1, x2, y2 = o["box_xyxy"]
        color = COLORS[i % len(COLORS)]
        thickness = 4 if o["object_id"] in targets else 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        label = f"{o['object_id']} {o['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(canvas, label, (x1 + 2, max(th, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def draw_masks(image_bgr, objects: list[dict], masks: list[np.ndarray]):
    import cv2
    canvas = image_bgr.copy()
    for i, (o, mask) in enumerate(zip(objects, masks)):
        color = np.array(COLORS[i % len(COLORS)], dtype=np.float32)
        canvas[mask] = (canvas[mask] * 0.45 + color * 0.55).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, COLORS[i % len(COLORS)], 2)
        cv2.putText(canvas, o["object_id"], tuple(o["box_xyxy"][:2]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


# --- (3) 계획 (선택) ----------------------------------------------------------

def build_world_state(objects: list[dict]) -> dict:
    """VLM 검출 → planner가 먹는 world_state 모양.

    3D 좌표·파지 후보는 없다(depth와 grasp의 몫). planner의 계획 생성은 object_id와 속성만
    보므로 이 상태로도 시퀀스를 만들 수 있지만, **검증기(validator)는 좌표를 보므로 여기서
    만든 world_state로는 통과하지 못한다** — 이 옵션이 확인하는 것은 LLM이 지시를 스킬
    시퀀스로 옮기는 부분까지다.
    """
    import yaml
    attrs = (yaml.safe_load((REPO / "src" / "perception" / "config" / "objects.yaml")
                            .read_text(encoding="utf-8")) or {})
    table, fallback = attrs.get("objects") or {}, attrs.get("fallback") or {}
    out = []
    for o in objects:
        spec = table.get(o["class_name"])
        known = spec is not None
        spec = spec or fallback
        out.append({
            "object_id": o["object_id"],
            "class_name": o["class_name"],
            "name_ko": o["name_ko"] or (spec.get("name_ko") if known else o["class_name"]),
            "confidence": o["confidence"],
            "graspable": True,
            "not_graspable_reason": "",
            "mass_g": spec.get("mass_g"),
            "fragile": bool(spec.get("fragile")),
            "deformable": bool(spec.get("deformable")),
            "transparent": bool(spec.get("transparent")),
            "profile": spec.get("profile", "fragile"),
            "attr_source": "yaml_seed" if known else "vlm_new_class",
            "needs_confirmation": not known,
            "grasp_candidates": [],
        })
    return {"schema_version": "1.0.0", "frame_id": "base", "objects": out,
            "needs_reobserve": []}


def run_plan(command: str, world_state: dict) -> None:
    # grounding.BINS_YAML의 기본값은 컨테이너 레이아웃(services/control/config)을 가리켜
    # 호스트에서는 없는 경로다. 컨테이너는 Dockerfile이 BINS_YAML을 넣어 주므로 문제가
    # 드러나지 않는다 — 여기서는 저장소 실제 경로를 넣어 준다.
    import os
    os.environ.setdefault("BINS_YAML", str(REPO / "src" / "control" / "config" / "bins.yaml"))
    grounding = import_planner_module("grounding")
    llm_client = import_planner_module("llm_client")
    bins = grounding.load_bins()
    summary = grounding.describe_for_prompt(world_state, bins)
    print(f"\n[3] planner 계획 생성 — model={llm_client.model_name()} "
          f"prompt={llm_client.PROMPT_VERSION}")
    plan = llm_client.plan(command, summary)
    if not plan.steps:
        print(f"  거부: {plan.refusal_reason}")
        return
    for i, step in enumerate(plan.steps, 1):
        print(f"  {i}. {step.skill} object_id={step.object_id}"
              + (f" bin_id={step.bin_id}" if step.bin_id else ""))


# --- main ---------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", help="테스트할 이미지 파일")
    ap.add_argument("--command", default=None, help='자연어 지시 (예: "화장품 왼쪽으로")')
    ap.add_argument("--mode", default="som", choices=["som", "box"],
                    help="som(기본): SAM 전체 분할 → VLM이 번호에 이름만 붙임 / "
                         "box: VLM에게 박스를 물어 SAM 프롬프트로 씀. gpt-4o에서는 som이 "
                         "훨씬 정확해서 기본값이다 (docs/vlm_sam_pipeline.md 실측 결과)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--model", default=None, help="VLM 모델 (기본: .env의 VLM_MODEL/OPENAI_MODEL)")
    ap.add_argument("--detail", default="high", choices=["high", "low", "auto"],
                    help="이미지 입력 해상도. low는 싸지만 작은 물체 박스가 뭉개진다")
    ap.add_argument("--sam-model", default=DEFAULT_SAM, help="SAM 가중치 (기본 MobileSAM)")
    ap.add_argument("--device", default=None, help="0 이면 GPU, cpu 면 CPU")
    ap.add_argument("--points-stride", type=int, default=16,
                    help="som 모드의 SAM 샘플 격자. 촘촘할수록 느리고 조각이 늘어난다")
    ap.add_argument("--min-area", type=float, default=0.002,
                    help="som 모드에서 마크로 남길 최소 면적 (이미지 대비 비율)")
    ap.add_argument("--max-area", type=float, default=0.35,
                    help="som 모드에서 마크로 남길 최대 면적 — 작업대·배경을 여기서 뺀다")
    ap.add_argument("--max-marks", type=int, default=20,
                    help="som 모드에서 VLM에 보낼 최대 마크 수")
    ap.add_argument("--reuse-marks", action="store_true",
                    help="som 모드에서 SAM 전체 분할을 건너뛰고 앞선 실행의 *_marks.npz를 "
                         "그대로 쓴다. VLM 쪽만 바꿔 가며 볼 때 (SAM 전체 분할은 CPU에서 25초)")
    ap.add_argument("--from-json", default=None,
                    help="VLM을 부르지 않고 이전 *_scene.json을 재사용한다 "
                         "(som 모드는 같은 실행이 남긴 *_marks.npz도 함께 읽는다)")
    ap.add_argument("--no-sam", action="store_true", help="(1) VLM 검출까지만 (box 모드 전용)")
    ap.add_argument("--plan", action="store_true",
                    help="검출 결과로 planner 스킬 시퀀스까지 생성 (LLM 1회 추가 호출)")
    args = ap.parse_args()

    import cv2

    load_dotenv(REPO / ".env")
    image_path = pathlib.Path(args.image)
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"이미지를 읽을 수 없다: {image_path}", file=sys.stderr)
        return 1
    h, w = image.shape[:2]
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    print(f"이미지 {image_path} ({w}x{h})")

    depth_p, info_p = (image_path.with_name(f"{stem}_depth.npy"),
                       image_path.with_name(f"{stem}_info.json"))
    depth = np.load(depth_p) if depth_p.exists() else None
    info = json.loads(info_p.read_text()) if info_p.exists() else None
    cached = json.loads(pathlib.Path(args.from_json).read_text(encoding="utf-8")) \
        if args.from_json else None
    # 저장된 응답을 되쓸 때는 그때의 모드를 따라간다 — 기본값(som)이 붙은 채로 box 모드
    # 결과를 읽으면 있지도 않은 *_marks.npz를 찾는다.
    mode = (cached or {}).get("mode", args.mode)
    if cached and mode != args.mode:
        print(f"--from-json이 {mode} 모드 결과다 — --mode {args.mode} 대신 그쪽을 따른다")
    masks: list[np.ndarray] = []
    sam_meta = None

    if mode == "som":
        # --- (1) SAM 전체 분할 → 번호 붙이기 -------------------------------
        marks_npz = out_dir / f"{stem}_marks.npz"
        if cached or args.reuse_marks:
            store = np.load(marks_npz)
            mark_masks = [store[f"mark_{i}"] for i in range(1, len(store.files) + 1)]
            sam_elapsed = ((cached or {}).get("sam_everything") or {}).get("elapsed_s", 0.0)
            print(f"마크 재사용: {marks_npz}")
        else:
            mark_masks, sam_elapsed = sam_everything(
                image, args.sam_model, args.device, args.points_stride,
                args.min_area, args.max_area, args.max_marks)
            np.savez_compressed(marks_npz,
                                **{f"mark_{i}": m for i, m in enumerate(mark_masks, 1)})
        print(f"\n[1] SAM 전체 분할 — {args.sam_model} device={args.device or 'auto'} "
              f"stride={args.points_stride} → 마크 {len(mark_masks)}개 ({sam_elapsed:.1f}s)")
        if not mark_masks:
            print("  마크가 하나도 없다 — --min-area/--max-area를 확인한다", file=sys.stderr)
            return 1
        marks_png = out_dir / f"{stem}_marks.png"
        cv2.imwrite(str(marks_png), draw_marks(image, mark_masks))
        print(f"  -> {marks_png}")
        sam_meta = {"weights": args.sam_model, "device": args.device or "auto",
                    "mode": "everything", "points_stride": args.points_stride,
                    "marks": len(mark_masks), "elapsed_s": round(sam_elapsed, 2)}

        # --- (2) VLM은 번호에 이름만 붙인다 --------------------------------
        if cached:
            scene = cached
            objects = scene["objects"]
            masks = rebuild_masks(objects, mark_masks)
            print(f"VLM 응답 재사용: {args.from_json}")
        else:
            vlm_detect = import_planner_module("vlm_detect")
            t0 = time.perf_counter()
            # 지시(args.command)는 일부러 넘기지 않는다 — 인지가 지시에 끌려간다
            # (vlm_detect.label_marks 주석). 지시는 아래 (3) planner 단계로만 간다.
            labels = vlm_detect.label_marks(
                marks_png, list(range(1, len(mark_masks) + 1)),
                image_size=(w, h), model=args.model, detail=args.detail)
            elapsed = time.perf_counter() - t0
            objects, masks = marks_to_objects(labels, mark_masks)
            scene = {
                "mode": "som",
                "model": args.model or vlm_detect.model_name(),
                "prompt_version": vlm_detect.MARKS_PROMPT_VERSION,
                "command_text": args.command,
                "image": str(image_path),
                "image_size": {"width": w, "height": h},
                "raw": labels.model_dump(),
                "objects": objects,
                # 지시 대상 선정은 이 단계의 일이 아니다 — (3) planner가 정한다.
                "target_object_ids": [],
                "refusal_reason": "",
                "elapsed_s": round(elapsed, 2),
                "sam_everything": sam_meta,
            }
            (out_dir / f"{stem}_scene.json").write_text(
                json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[2] VLM 라벨링 — model={scene['model']} "
              f"prompt={scene['prompt_version']} ({scene['elapsed_s']:.1f}s)")
        print_objects(scene)

    else:
        # --- (1) VLM이 박스를 준다 -----------------------------------------
        if cached:
            scene = cached
            print(f"VLM 응답 재사용: {args.from_json}")
        else:
            scene, _ = run_vlm(image_path, args.command, (w, h), args.model, args.detail)
            (out_dir / f"{stem}_scene.json").write_text(
                json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[1] VLM 검출 — model={scene['model']} "
              f"prompt={scene['prompt_version']} ({scene['elapsed_s']:.1f}s)")
        print_objects(scene)
        objects = scene["objects"]

        # --- (2) 그 박스를 프롬프트로 SAM ------------------------------------
        if not args.no_sam and objects:
            boxes = [o["box_xyxy"] for o in objects]
            masks, elapsed = run_sam(image, boxes, args.sam_model, args.device)
            print(f"\n[2] SAM 세그멘테이션 — {args.sam_model} "
                  f"device={args.device or 'auto'} ({elapsed:.1f}s, 박스 {len(boxes)}개 일괄)")
            if len(masks) != len(boxes):
                print(f"  경고: 박스 {len(boxes)}개에 마스크 {len(masks)}개 — 짝이 맞지 않는다")
            sam_meta = {"weights": args.sam_model, "device": args.device or "auto",
                        "mode": "box_prompt", "elapsed_s": round(elapsed, 2)}

    boxes_png = out_dir / f"{stem}_boxes.png"
    cv2.imwrite(str(boxes_png), draw_boxes(image, objects, scene["target_object_ids"]))
    print(f"  -> {boxes_png}")

    # --- 마스크 품질 + (있으면) depth 3D ------------------------------------
    stats = []
    for o, mask in zip(objects, masks):
        s = mask_stats(mask, o["box_xyxy"])
        line = (f"  {o['object_id']:<18} mask_px={s['mask_px']:<7} "
                f"box_fill={s['box_fill']:.2f} 박스밖={s['outside_ratio']:.2f} "
                f"중심={s['centroid_xy']}")
        if depth is not None and info is not None:
            p, ratio = mask_3d(mask, depth, info)
            s["depth_valid_ratio"] = round(ratio, 3)
            s["camera_mm"] = [round(v, 1) for v in p] if p else None
            line += (f" depth_valid={ratio:.2f}" +
                     (f" cam_mm=({p[0]:.0f},{p[1]:.0f},{p[2]:.0f})" if p else " cam_mm=None"))
        print(line)
        stats.append({"object_id": o["object_id"], **s})

    if masks:
        masks_png = out_dir / f"{stem}_masks.png"
        cv2.imwrite(str(masks_png), draw_masks(image, objects, masks))
        np.savez_compressed(out_dir / f"{stem}_masks.npz",
                            **{o["object_id"]: m for o, m in zip(objects, masks)})
        print(f"  -> {masks_png}")
        print(f"  -> {out_dir / f'{stem}_masks.npz'}")

    # --- (3) 계획 ------------------------------------------------------------
    # som 모드에서는 지시 해석이 여기서만 일어난다(VLM 단계는 지시를 안 본다). 그래서
    # --command가 있으면 --plan 없이도 돈다 — 안 그러면 지시가 어디에도 쓰이지 않는다.
    if args.plan and not args.command:
        print("\n[3] --plan에는 --command가 필요하다", file=sys.stderr)
    elif args.command and (args.plan or mode == "som"):
        run_plan(args.command, build_world_state(objects))

    result = {**{k: v for k, v in scene.items() if k != "raw"},
              "sam": {**(sam_meta or {}), "masks": stats} if sam_meta else None}
    (out_dir / f"{stem}_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과: {out_dir / f'{stem}_result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

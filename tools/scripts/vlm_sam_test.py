#!/usr/bin/env python3
"""VLM 검출 → SAM 세그멘테이션 파이프라인을 정지 이미지 한 장으로 확인한다 (ROS 불필요).

확인하려는 것은 두 가지다.

  (1) 사진을 API로 보냈을 때 무엇이 있는지 알아보고 바운딩박스 좌표를 주는가
  (2) 그 좌표를 프롬프트로 SAM이 마스크를 뽑는가

두 가지 순서로 돌려 볼 수 있다. 기본값 --mode som은 SAM이 먼저 장면을 조각내고 VLM은 번호에
이름만 붙인다. --mode box는 반대로 VLM에게 박스를 물어 SAM 프롬프트로 쓴다 — 원래 설계이고
**gpt-4o에서는 som이 비교가 안 되게 정확해서** 기본값을 som으로 둔다
(이유와 실측은 docs/vlm_sam_pipeline.md 참조).

  # SAM 마스킹만 볼 때 (openai 불필요, OPENAI_API_KEY 없어도 된다)
  .venv/bin/python tools/scripts/vlm_sam_test.py --no-vlm

  # test_image/에 넣은 파일 그대로: 인자 없이 돌리면 test_image/ 안의 이미지를 전부 돈다
  .venv/bin/python tools/scripts/vlm_sam_test.py --command "치약 왼쪽으로"

  # 특정 파일 하나만 (test_image/ 밖도 가능 — 경로 아무거나)
  .venv/bin/python tools/scripts/vlm_sam_test.py test_image/scene1.png \
      --command "치약 왼쪽으로"

  # SAM 전체 분할(CPU 26초)만 재사용하고 지시만 바꿔 가며
  .venv/bin/python tools/scripts/vlm_sam_test.py test_image/scene1.png \
      --reuse-marks --command "우산 왼쪽으로"

  # 원래 물어본 경로: VLM 박스 → SAM 프롬프트
  .venv/bin/python tools/scripts/vlm_sam_test.py test_image/scene1.png \
      --mode box --command "치약 왼쪽으로" --plan

SAM 파라미터(--points-stride/--min-area/--max-area/--max-marks/--sam-model/--device)는
perception 노드가 쓰는 것과 같은 이름·같은 기본값이다(src/perception/perception/node.py의
sam_points_stride 등 declare_parameter). 여기서 바꿔 보고 마음에 드는 값을 그쪽 ROS
파라미터에 반영하면 된다.

결과는 --out 디렉터리(기본 test_result/, 입력은 기본 test_image/ 전체)에 저장된다 — json은
안 만들고, "LLM에 보내기 전"과 "VLM이 걸러낸 후"를 나란히 볼 수 있게 둘 다 남긴다
(--no-vlm이면 VLM을 안 부르니 *_marks.png는 안 만든다):
  *_marks.png               SAM 원본 조각에 번호만 붙인 것 — LLM에 실제로 보낸 그림 (som)
  *_masks.png               전체 이미지에 물체별 윤곽선+라벨을 그린 최종 결과(웹 화면과 같은 스타일)
  *_objects/<object_id>.png 물체별로 배경을 지우고 마스크만 잘라낸 컷아웃(투명 PNG)

**여기서 나오는 마스크는 2D까지다.** 3D 좌표는 같은 프레임의 aligned depth가 있어야 하고,
같은 stem의 *_depth.npy / *_info.json이 있으면 마스크 기준 카메라 좌표(mm)까지 뽑는다
(perception_test_image.py:mask_3d와 같은 계산).
"""
import argparse
import json
import pathlib
import shutil
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_DIR = REPO / "test_image"
DEFAULT_OUT = REPO / "test_result"
# SAM2 base 가중치(~310MB). models/에 있으면 그것을 쓰고, 없으면 ultralytics가 이름을 보고
# 자동으로 내려받는다(첫 실행 때 한 번). *.pt는 커밋하지 않는다(.gitignore).
# 2026-09-08: mobile_sam → SAM2 base로 기본값 교체 (docker-compose.yml SAM_MODEL과 맞춤 —
# 실측 근거는 docs/vlm_sam_pipeline.md). mobile_sam으로 되돌리려면 --sam-model mobile_sam.pt.
DEFAULT_SAM = str(REPO / "models" / "sam2_b.pt") \
    if (REPO / "models" / "sam2_b.pt").exists() else "sam2_b.pt"

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


def collect_images(arg: str | None) -> list[pathlib.Path]:
    """이미지 인자를 파일 목록으로 푼다.

    인자가 없으면 test_image/ 전체(넣기만 하면 도는 환경), 디렉터리면 그 안의 이미지 전부,
    파일이면 그 하나만 — perception_test_image.py의 collect()와 같은 규칙이다.
    """
    exts = {".png", ".jpg", ".jpeg"}
    path = pathlib.Path(arg) if arg else DEFAULT_IMAGE_DIR
    if path.is_dir():
        return sorted(q for q in path.iterdir() if q.suffix.lower() in exts)
    if path.is_file():
        return [path]
    return []


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
        # detect 경로는 등록 어휘 대비 신규 여부를, som 경로는 VLM이 판단한 속성을 낸다.
        attrs = o.get("attrs")
        if attrs:
            flags = [k for k in ("fragile", "deformable", "transparent") if attrs[k]]
            note = (f" {attrs['profile']} {attrs['mass_g']:g}g"
                    + (f" [{','.join(flags)}]" if flags else ""))
        else:
            note = " [신규클래스]" if o.get("is_new_class") else ""
        print(f"  {o['object_id']:<18} {o['class_name']:<14} {o['name_ko']:<8} "
              f"conf={o['confidence']:.2f} box=({x1},{y1})-({x2},{y2}) "
              f"{x2 - x1}x{y2 - y1}px{note}{target}")
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
            # 무게·파손위험·파지 프로파일도 VLM이 답한다 — som 경로는 objects.yaml을
            # 지나지 않는다(planner/src/vlm_detect.py 상단 [속성도 VLM이 판단한다]).
            "attrs": item["attrs"],
            "confidence": round(item["confidence"], 3),
            "mark_id": item["mark_id"],
            "merged_marks": item["merged_marks"],
            # 박스는 VLM이 아니라 마스크에서 뽑는다 — 이 경로의 핵심이다.
            "box_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        })
        out_masks.append(mask)
    return objects, out_masks


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

def crop_object_rgba(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """마스크의 외접 사각형만큼 잘라내고, 마스크 바깥은 알파 0으로 지운 BGRA 이미지.

    "배경 없이 객체만" 보고 싶을 때 쓴다 — png 알파 채널을 지원하는 뷰어라면 배경이
    실제로 비어 보인다.
    """
    import cv2

    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    b, g, r = cv2.split(image_bgr[y1:y2, x1:x2])
    alpha = (mask[y1:y2, x1:x2].astype(np.uint8) * 255)
    return cv2.merge([b, g, r, alpha])


# 2026-09-08: 0.35/두께1이던 예전 값은 1280x720 사진에서 실제로 읽기 힘들었다(사람이
# 직접 확인). production(src/grasp/grasp/node.py:_draw_label)은 원래도 0.5를 쓰고 있었으니
# 그쪽과도 어긋나 있었다 — 여기서는 그보다 더 키운다. 이 스크립트는 사람이 결과를 눈으로
# 검수하는 용도라 production 실시간 스트림보다 커도 된다.
_LABEL_FONT_SCALE = 0.8
_LABEL_THICKNESS = 2
_LABEL_PAD = 6


def _draw_label(canvas, origin: tuple[int, int], text: str, color: tuple[int, int, int]) -> None:
    """색 배경 박스 위에 흰 글씨를 올려 마스크 윤곽선과 겹쳐도 읽히게 한다."""
    import cv2

    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, _LABEL_FONT_SCALE, _LABEL_THICKNESS)
    x, y = origin
    y = max(y, th + baseline + _LABEL_PAD)
    cv2.rectangle(canvas, (x - _LABEL_PAD, y - th - baseline - _LABEL_PAD),
                  (x + tw + _LABEL_PAD, y + _LABEL_PAD), color, -1)
    # 밝은 색(노랑 등) 배경에서 흰 글씨가 묻히므로 배경 밝기에 따라 글자색을 바꾼다.
    luma = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]  # color는 BGR
    text_color = (0, 0, 0) if luma > 150 else (255, 255, 255)
    cv2.putText(canvas, text, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX,
               _LABEL_FONT_SCALE, text_color, _LABEL_THICKNESS, cv2.LINE_AA)


def draw_masks(image_bgr, objects: list[dict], masks: list[np.ndarray]):
    """웹 화면에서 실제로 보이는 /grasp/debug_image와 같은 스타일 —
    색을 채우지 않고 윤곽선만 그린다. 마스크를 반투명으로 칠하면 조각이 거칠거나
    겹칠 때 지저분해 보인다(vlm_sam_test.py 예전 버전이 그랬다) — 웹 쪽은 애초에
    채우기 없이 윤곽선 + 라벨 박스만 쓴다(src/grasp/grasp/node.py:_publish_debug_image)."""
    import cv2
    canvas = image_bgr.copy()
    for i, (o, mask) in enumerate(zip(objects, masks)):
        color = COLORS[i % len(COLORS)]
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, 1)
        ys, xs = np.nonzero(mask)
        if xs.size:
            _draw_label(canvas, (int(xs.min()), int(ys.min())), o["object_id"], color)
    return canvas


# --- (3) 계획 (선택) ----------------------------------------------------------

def build_world_state(objects: list[dict]) -> dict:
    """VLM 검출 → planner가 먹는 world_state 모양.

    3D 좌표·파지 후보는 없다(depth와 grasp의 몫). planner의 계획 생성은 object_id와 속성만
    보므로 이 상태로도 시퀀스를 만들 수 있지만, **검증기(validator)는 좌표를 보므로 여기서
    만든 world_state로는 통과하지 못한다** — 이 옵션이 확인하는 것은 LLM이 지시를 스킬
    시퀀스로 옮기는 부분까지다.

    속성의 출처는 경로마다 다르다. som 경로(운영 경로)는 VLM이 낸 `attrs`를 그대로 쓰고,
    박스를 묻는 detect 경로는 예전처럼 objects.yaml에서 class_name으로 찾는다 — 그쪽은
    아직 등록 어휘를 프롬프트에 넣기 때문이다.
    """
    out = []
    for o in objects:
        attrs = o.get("attrs") or _yaml_attributes(o["class_name"])
        out.append({
            "object_id": o["object_id"],
            "class_name": o["class_name"],
            "name_ko": o["name_ko"] or attrs["name_ko"] or o["class_name"],
            "confidence": o["confidence"],
            "graspable": True,
            "not_graspable_reason": "",
            "mass_g": attrs["mass_g"],
            "fragile": attrs["fragile"],
            "deformable": attrs["deformable"],
            "transparent": attrs["transparent"],
            "profile": attrs["profile"],
            "attr_source": attrs["attr_source"],
            "needs_confirmation": attrs["needs_confirmation"],
            "grasp_candidates": [],
        })
    return {"schema_version": "1.0.0", "frame_id": "base", "objects": out,
            "needs_reobserve": []}


def _yaml_attributes(class_name: str) -> dict:
    """objects.yaml에서 속성을 찾는다 (detect 경로 전용 — som 경로는 VLM이 낸 값을 쓴다).

    노드에서 이 일을 하는 것은 `perception.attribute_db.AttributeSource`지만 그쪽은 DB까지
    보므로, 스크립트에서는 yaml만 읽는다.
    """
    import yaml

    config = (yaml.safe_load((REPO / "src" / "perception" / "config" / "objects.yaml")
                             .read_text(encoding="utf-8")) or {})
    spec = (config.get("objects") or {}).get(class_name)
    known = spec is not None
    spec = spec or (config.get("fallback") or {})
    return {
        "name_ko": (spec.get("name_ko") or "") if known else "",
        "mass_g": spec.get("mass_g"),
        "fragile": bool(spec.get("fragile")),
        "deformable": bool(spec.get("deformable")),
        "transparent": bool(spec.get("transparent")),
        "profile": spec.get("profile", "fragile"),
        "attr_source": "yaml_seed" if known else "vlm_new_class",
        "needs_confirmation": not known,
    }


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
    ap.add_argument("image", nargs="?", default=None,
                    help="테스트할 이미지 파일 또는 디렉터리 (생략하면 test_image/ 전체)")
    ap.add_argument("--command", default=None, help='자연어 지시 (예: "화장품 왼쪽으로")')
    ap.add_argument("--mode", default="som", choices=["som", "box"],
                    help="som(기본): SAM 전체 분할 → VLM이 번호에 이름만 붙임 / "
                         "box: VLM에게 박스를 물어 SAM 프롬프트로 씀. gpt-4o에서는 som이 "
                         "훨씬 정확해서 기본값이다 (docs/vlm_sam_pipeline.md 실측 결과)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--model", default=None, help="VLM 모델 (기본: .env의 VLM_MODEL/OPENAI_MODEL)")
    ap.add_argument("--detail", default="high", choices=["high", "low", "auto"],
                    help="이미지 입력 해상도. low는 싸지만 작은 물체 박스가 뭉개진다")
    ap.add_argument("--sam-model", default=DEFAULT_SAM, help="SAM 가중치 (기본 SAM2 base)")
    ap.add_argument("--device", default=None, help="0 이면 GPU, cpu 면 CPU")
    ap.add_argument("--points-stride", type=int, default=10,
                    help="som 모드의 SAM 샘플 격자. 촘촘할수록 느리고 조각이 늘어난다")
    ap.add_argument("--min-area", type=float, default=0.008,
                    help="som 모드에서 마크로 남길 최소 면적 (이미지 대비 비율)")
    ap.add_argument("--max-area", type=float, default=0.35,
                    help="som 모드에서 마크로 남길 최대 면적 — 작업대·배경을 여기서 뺀다")
    ap.add_argument("--max-marks", type=int, default=20,
                    help="som 모드에서 VLM에 보낼 최대 마크 수")
    ap.add_argument("--reuse-marks", action="store_true",
                    help="som 모드에서 SAM 전체 분할을 건너뛰고 앞선 실행의 *_marks.npz를 "
                         "그대로 쓴다. VLM 쪽만 바꿔 가며 볼 때 (SAM 전체 분할은 CPU에서 25초)")
    ap.add_argument("--no-sam", action="store_true", help="(1) VLM 검출까지만 (box 모드 전용)")
    ap.add_argument("--no-vlm", action="store_true",
                    help="som 모드에서 VLM 라벨링을 건너뛰고 SAM 조각을 그대로 물체로 쓴다 "
                         "(OPENAI_API_KEY 없이 SAM 마스킹만 볼 때. box 모드는 VLM이 있어야 "
                         "애초에 박스가 나오므로 지원 안 함)")
    ap.add_argument("--plan", action="store_true",
                    help="검출 결과로 planner 스킬 시퀀스까지 생성 (LLM 1회 추가 호출)")
    args = ap.parse_args()
    if args.no_vlm and args.mode != "som":
        print("--no-vlm은 --mode som에서만 된다 (box 모드는 VLM 박스로 시작한다)",
              file=sys.stderr)
        return 1

    import cv2

    load_dotenv(REPO / ".env")

    images = collect_images(args.image)
    if not images:
        target = args.image or f"{DEFAULT_IMAGE_DIR} (비어 있음)"
        print(f"입력 이미지가 없다: {target}", file=sys.stderr)
        return 1
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    exit_code = 0
    for image_path in images:
        exit_code = process_image(image_path, args, out_dir) or exit_code
    return exit_code


def process_image(image_path: pathlib.Path, args, out_dir: pathlib.Path) -> int:
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"이미지를 읽을 수 없다: {image_path}", file=sys.stderr)
        return 1
    h, w = image.shape[:2]
    stem = image_path.stem
    print(f"\n== 이미지 {image_path} ({w}x{h}) ==")

    depth_p, info_p = (image_path.with_name(f"{stem}_depth.npy"),
                       image_path.with_name(f"{stem}_info.json"))
    depth = np.load(depth_p) if depth_p.exists() else None
    info = json.loads(info_p.read_text()) if info_p.exists() else None
    mode = args.mode
    masks: list[np.ndarray] = []

    if mode == "som":
        # --- (1) SAM 전체 분할 → 번호 붙이기 -------------------------------
        before_npz = out_dir / f"{stem}_before.npz"
        if args.reuse_marks:
            store = np.load(before_npz)
            mark_masks = [store[f"mark_{i}"] for i in range(1, len(store.files) + 1)]
            sam_elapsed = 0.0
            print(f"마크 재사용: {before_npz}")
        else:
            mark_masks, sam_elapsed = sam_everything(
                image, args.sam_model, args.device, args.points_stride,
                args.min_area, args.max_area, args.max_marks)
            np.savez_compressed(before_npz,
                                **{f"mark_{i}": m for i, m in enumerate(mark_masks, 1)})
        print(f"\n[1] SAM 전체 분할 — {args.sam_model} device={args.device or 'auto'} "
              f"stride={args.points_stride} → 마크 {len(mark_masks)}개 ({sam_elapsed:.1f}s)")
        if not mark_masks:
            print("  마크가 하나도 없다 — --min-area/--max-area를 확인한다", file=sys.stderr)
            return 1

        # --- (2) VLM은 번호에 이름만 붙인다 (--no-vlm이면 건너뛴다) --------
        if args.no_vlm:
            objects, masks = [], list(mark_masks)
            for i, mask in enumerate(mark_masks, 1):
                ys, xs = np.nonzero(mask)
                objects.append({
                    "object_id": f"mark_{i}",
                    "class_name": "", "name_ko": "", "attrs": None,
                    "confidence": 0.0,
                    "box_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                })
            print(f"\n[2] VLM 라벨링 생략(--no-vlm) — SAM 조각 {len(objects)}개를 "
                  "그대로 물체로 쓴다 (openai 불필요)")
        else:
            # 번호 오버레이 — LLM에 실제로 보내는 그림이라 결과 폴더에 남겨서
            # "보내기 전"(_before)과 "VLM이 걸러낸 후"(_after)를 나란히 비교할 수 있게 한다.
            before_png = out_dir / f"{stem}_before.png"
            cv2.imwrite(str(before_png), draw_marks(image, mark_masks))
            print(f"  -> {before_png}  (LLM에 보내기 전 — SAM 원본 조각)")

            vlm_detect = import_planner_module("vlm_detect")
            t0 = time.perf_counter()
            # 지시(args.command)는 일부러 넘기지 않는다 — 인지가 지시에 끌려간다
            # (vlm_detect.label_marks 주석). 지시는 아래 (3) planner 단계로만 간다.
            labels = vlm_detect.label_marks(
                before_png, list(range(1, len(mark_masks) + 1)),
                image_size=(w, h), model=args.model, detail=args.detail)
            elapsed = time.perf_counter() - t0
            objects, masks = marks_to_objects(labels, mark_masks)
            scene = {
                "model": args.model or vlm_detect.model_name(),
                "prompt_version": vlm_detect.MARKS_PROMPT_VERSION,
                "objects": objects,
                "target_object_ids": [],
                "refusal_reason": "",
                "elapsed_s": round(elapsed, 2),
            }
            print(f"\n[2] VLM 라벨링 — model={scene['model']} "
                  f"prompt={scene['prompt_version']} ({scene['elapsed_s']:.1f}s)")
            print_objects(scene)

    else:
        # --- (1) VLM이 박스를 준다 -----------------------------------------
        scene, _ = run_vlm(image_path, args.command, (w, h), args.model, args.detail)
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

    # --- 마스크 품질 + (있으면) depth 3D (콘솔 확인용, 파일로는 안 남긴다) --
    for o, mask in zip(objects, masks):
        s = mask_stats(mask, o["box_xyxy"])
        line = (f"  {o['object_id']:<18} mask_px={s['mask_px']:<7} "
                f"box_fill={s['box_fill']:.2f} 박스밖={s['outside_ratio']:.2f} "
                f"중심={s['centroid_xy']}")
        if depth is not None and info is not None:
            _, ratio = mask_3d(mask, depth, info)
            line += f" depth_valid={ratio:.2f}"
        print(line)

    # --- 결과 이미지: 전체 마스킹 이미지 + 물체별 배경 없는 컷아웃 ----------
    if masks:
        masks_png = out_dir / f"{stem}_masks.png"
        cv2.imwrite(str(masks_png), draw_masks(image, objects, masks))
        print(f"  -> {masks_png}")

        objects_dir = out_dir / f"{stem}_objects"
        # 이전 실행의 컷아웃이 남아 섞이지 않게 매번 새로 비운다(예: 지난번 --no-vlm의
        # mark_1..15가 이번 물체 이름과 뒤섞여 보이는 문제).
        if objects_dir.exists():
            shutil.rmtree(objects_dir)
        objects_dir.mkdir(parents=True)
        for o, mask in zip(objects, masks):
            rgba = crop_object_rgba(image, mask)
            if rgba is None:
                continue
            obj_png = objects_dir / f"{o['object_id']}.png"
            cv2.imwrite(str(obj_png), rgba)
        print(f"  -> {objects_dir}/ ({len(masks)}개)")

    # --- (3) 계획 ------------------------------------------------------------
    # som 모드에서는 지시 해석이 여기서만 일어난다(VLM 단계는 지시를 안 본다). 그래서
    # --command가 있으면 --plan 없이도 돈다 — 안 그러면 지시가 어디에도 쓰이지 않는다.
    if args.no_vlm:
        if args.plan or args.command:
            print("\n[3] --no-vlm은 물체 이름이 없어 계획을 만들지 않는다", file=sys.stderr)
    elif args.plan and not args.command:
        print("\n[3] --plan에는 --command가 필요하다", file=sys.stderr)
    elif args.command and (args.plan or mode == "som"):
        run_plan(args.command, build_world_state(objects))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

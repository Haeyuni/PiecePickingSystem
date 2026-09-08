"""SAM으로 장면을 조각내고 번호를 그린다 (Set-of-Mark). VLM에게 좌표를 묻지 않기 위한 장치다.

**왜 이 순서인가.** VLM에게 바운딩박스를 물어보면 gpt-4o는 좌표를 못 맞춘다 — 2026-09-07
실측에서 박스가 100px 단위로 뭉개지고 다른 물체 위에 얹혔다(docs/vlm_sam_pipeline.md).
그래서 좌표는 SAM이 만들고(정확), 이름은 VLM이 붙인다(정확). 각자 잘하는 쪽을 맡는다.

    segment_everything()  프롬프트 없이 장면 전체를 조각냄 → 마스크 후보
      → draw_marks()      조각마다 번호를 그린 이미지 → VLM에 보낸다
      → (VLM 응답)         번호별 is_object / class_name / part_of
      → merge_marks()     조각을 물체로 합침

ROS와 무관하다(numpy + cv2 + ultralytics만 쓴다). perception 노드와
tools/scripts/vlm_sam_test.py가 **같은 코드를 쓰도록** 여기 둔다 — 갈라지면 스크립트로
검증한 것이 노드에서 성립하지 않는다.
"""
import time

import numpy as np

# 오버레이 색 (BGR). 번호 순서대로 돌려 쓴다.
COLORS = [(0, 200, 255), (0, 255, 120), (255, 120, 0), (255, 0, 200),
          (60, 60, 255), (255, 255, 0), (180, 0, 255), (0, 140, 255)]


def _release_cuda_cache() -> None:
    """추론에 쓴 GPU 캐시를 드라이버에 돌려준다.

    아래 두 함수는 SAM 모델을 매 호출 새로 만든다(오래 들고 있지 않음 — 관측 사이에는
    아무것도 GPU에 안 남기려는 의도). 그런데 함수가 끝나도 PyTorch의 caching allocator는
    한 번 받은 GPU 메모리를 드라이버에 곧바로 돌려주지 않고 재사용을 위해 쥐고 있다.
    이 프로세스만 쓰면 문제가 안 되지만, graspnet_baseline이 **같은 8GB GPU**를 나눠
    쓰고 있어(compose가 GPU를 분리하지 않는다) 관측을 반복할수록(온디맨드라 명령마다
    최소 1회) 이 프로세스가 캐시로 쥔 양이 계속 늘어난다. 2026-09-08 실측: 재시작 전
    7.6GB, 재시작 후 1.4GB — 그 차이가 전부 이 캐시였고, 그동안 graspnet_baseline은
    256MiB를 못 늘려 CUDA OOM으로 매번 추론에 실패했다(파지 후보 0개).
    """
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def segment_everything(image_bgr, weights: str, device: str | None = None,
                       points_stride: int = 16, min_area: float = 0.002,
                       max_area: float = 0.35, max_marks: int = 20):
    """프롬프트 없이 장면 전체를 조각낸다 → (마스크 목록, 걸린 시간).

    everything 모드는 물체가 아닌 것(작업대·그림자·라벨 조각)까지 전부 뱉으므로 여기서 한 번
    거른다. 무엇이 물체인지 최종 판단은 VLM이 하지만, 마크가 100개면 이미지가 번호로 덮여
    VLM이 아무것도 못 본다.
    """
    from ultralytics.models.sam import Predictor as SAMPredictor

    predictor = SAMPredictor(overrides=dict(
        conf=0.25, task="segment", mode="predict", imgsz=1024,
        model=weights, device=device, verbose=False, save=False))
    result = None
    try:
        predictor.set_image(image_bgr)
        started = time.perf_counter()
        result = predictor(points_stride=points_stride, crop_n_layers=0)[0]
        elapsed = time.perf_counter() - started
        predictor.reset_image()
        if result.masks is None:
            return [], elapsed

        masks = result.masks.data.cpu().numpy() > 0.5
        height, width = image_bgr.shape[:2]
        total_px = height * width
        areas = masks.reshape(len(masks), -1).sum(1)
        order = np.argsort(-areas)                  # 큰 것부터 — 겹칠 때 큰 쪽을 남긴다

        kept: list[np.ndarray] = []
        for index in order:
            area = int(areas[index])
            if not (min_area * total_px <= area <= max_area * total_px):
                continue
            mask = masks[index]
            # 이미 남긴 것과 사실상 같은 마스크는 버린다. 조금 겹치는 것(뚜껑/몸통)은 남겨
            # 두고 VLM의 part_of로 묶는다.
            if any((mask & k).sum() / max(1, min(area, int(k.sum()))) > 0.9 for k in kept):
                continue
            kept.append(mask)
            if len(kept) >= max_marks:
                break
        return kept, elapsed
    finally:
        # predictor/result가 들고 있는 GPU 텐서 참조를 끊어야 아래 empty_cache()가
        # 실제로 회수할 게 생긴다 — 참조가 남아 있으면 caching allocator가 계속 쥔다.
        del predictor
        del result
        _release_cuda_cache()


def mark_anchor(mask: np.ndarray) -> tuple[int, int]:
    """번호를 찍을 지점. 무게중심은 초승달 모양 마스크에서 바깥으로 나가므로,
    마스크 안쪽에서 경계로부터 가장 먼 점을 쓴다."""
    import cv2

    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(distance.argmax()), distance.shape)
    return int(x), int(y)


def draw_marks(image_bgr, masks: list[np.ndarray]):
    """VLM에 보낼 번호 오버레이. 번호와 윤곽선만 그리고 마스크는 옅게만 칠한다 —
    진하게 칠하면 물체가 안 보여서 VLM이 무엇인지 판단할 수 없다."""
    import cv2

    canvas = image_bgr.copy()
    # 번호 크기를 이미지 해상도에 맞춘다 — 원래 고정 픽셀값(font_scale=1.0, 여백=12px)은
    # 1280x720 기준이라, 그보다 작은 사진(예: 402x228)에서는 번호가 물체를 통째로
    # 가려서 VLM이 마스크를 못 알아본다.
    reference_dim = 720
    scale = min(canvas.shape[0], canvas.shape[1]) / reference_dim
    scale = max(0.35, min(1.0, scale))
    font_scale = 1.0 * scale
    text_thickness = max(1, round(3 * scale))
    contour_thickness = max(1, round(3 * scale))
    pad = max(4, round(12 * scale))
    for number, mask in enumerate(masks, 1):
        color = COLORS[(number - 1) % len(COLORS)]
        canvas[mask] = (canvas[mask] * 0.82 + np.array(color, np.float32) * 0.18).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, contour_thickness)
        x, y = mark_anchor(mask)
        label = str(number)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
        cv2.circle(canvas, (x, y), max(tw, th) // 2 + pad, (255, 255, 255), -1)
        cv2.circle(canvas, (x, y), max(tw, th) // 2 + pad, color, contour_thickness)
        cv2.putText(canvas, label, (x - tw // 2, y + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), text_thickness, cv2.LINE_AA)
    return canvas


def bbox_area(mask: np.ndarray) -> int:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return 0
    return (int(xs.max() - xs.min()) + 1) * (int(ys.max() - ys.min()) + 1)


def merge_marks(marks: list[dict], mark_masks: list[np.ndarray],
                max_growth: float = 1.6, on_reject=None) -> list[dict]:
    """VLM의 번호별 판단 → 물체 목록. 쪼개진 조각(part_of)은 합친다.

    조각 합치기는 VLM의 판단이므로 기하로 한 번 더 거른다. 실제로 2026-09-07 테스트에서
    gpt-4o가 사진에 그려져 있던 YOLO 라벨 배너("...toothpaste 0.98")를 치약의 조각이라고
    답했다 — 글자를 읽고 같은 물체라고 본 것이다. 조각 하나가 물체의 외접 사각형을
    max_growth배 넘게 키우거나 대표 조각보다 크면, 조각이 아니라 다른 영역으로 보고 버린다.

    marks는 `{mark_id, is_object, part_of, class_name, name_ko, is_new_class, confidence}`
    dict 리스트다(planner /internal/label-marks 응답 그대로). 반환값에는 합쳐진 `mask`가
    붙지만 **object_id는 붙이지 않는다** — 그건 ObjectTracker의 일이다.
    """
    count = len(mark_masks)
    pieces_of: dict[int, list[int]] = {}
    for mark in marks:
        parent = int(mark.get("part_of") or 0)
        if parent and not mark.get("is_object"):
            pieces_of.setdefault(parent, []).append(int(mark["mark_id"]))

    objects = []
    for mark in marks:
        if not mark.get("is_object"):
            continue
        mark_id = int(mark["mark_id"])
        if not 1 <= mark_id <= count:
            continue
        main = mark_masks[mark_id - 1]
        mask, merged = main.copy(), []
        for other in pieces_of.get(mark_id, []):
            if not 1 <= other <= count:
                continue
            piece = mark_masks[other - 1]
            grown = mask | piece
            if piece.sum() > main.sum() or bbox_area(grown) > bbox_area(mask) * max_growth:
                if on_reject:
                    on_reject(other, mark_id)
                continue
            mask = grown
            merged.append(other)
        if not mask.any():
            continue
        objects.append({
            "mark_id": mark_id,
            "class_name": mark.get("class_name") or "",
            "name_ko": mark.get("name_ko") or "",
            "is_new_class": bool(mark.get("is_new_class")),
            "confidence": float(mark.get("confidence") or 0.0),
            "merged_marks": merged,
            "mask": mask,
        })
    return objects


def segment_at_boxes(image_bgr, weights: str, boxes: list[list[int]],
                     device: str | None = None):
    """박스 프롬프트로 마스크를 뽑는다 → (마스크 목록, 걸린 시간). boxes와 1:1 대응한다.

    재관측(D-1)이 쓰는 경로다. everything 모드와 달리 **무엇을 찾는지 이미 알고** 있으므로
    장면 전체를 훑을 이유가 없다 — 박스를 한 번에 넘겨 이미지 임베딩을 한 번만 만든다.
    실측(CPU): everything 26.6초 vs 박스 3개 0.2초.

    **점이 아니라 박스인 이유.** 2026-09-07 실측에서 마스크 중심을 점 프롬프트로 주자 SAM이
    물체가 아니라 그 위의 무늬 조각을 잡았다 — 직전 마스크와의 IoU가 물티슈 0.07,
    섬유탈취제 0.03이었다(치약만 0.89). 점 하나는 "이 부분"인지 "이 물체"인지 모호하다.
    같은 물체에 박스를 주자 0.99 / 0.98 / 0.89가 나왔다. 다점(9점)도 물티슈에서는 실패했다.

    프롬프트가 배경을 감쌌으면 그 자리의 배경 마스크가 나온다. 그것이 물체인지는 여기서
    판단하지 않는다 — 호출자가 3D 위치를 직전 관측과 대조해 거른다(D-2).
    """
    from ultralytics import SAM

    if not boxes:
        return [], 0.0

    model = SAM(weights)
    result = None
    try:
        started = time.perf_counter()
        result = model.predict(image_bgr, bboxes=boxes, device=device, verbose=False)[0]
        elapsed = time.perf_counter() - started
        height, width = image_bgr.shape[:2]
        if result.masks is None:
            return [np.zeros((height, width), dtype=bool) for _ in boxes], elapsed

        masks = result.masks.data.cpu().numpy() > 0.5
        if masks.shape[1:] != (height, width):     # 추론 해상도로 나오면 원본으로 되돌린다
            import cv2

            masks = np.stack([
                cv2.resize(m.astype(np.uint8), (width, height),
                           interpolation=cv2.INTER_NEAREST).astype(bool) for m in masks])
        out = list(masks)
        # 박스 수와 마스크 수가 어긋나면 짝이 밀려 엉뚱한 라벨이 붙는다 — 빈 마스크로 채운다.
        while len(out) < len(boxes):
            out.append(np.zeros((height, width), dtype=bool))
        return out[:len(boxes)], elapsed
    finally:
        # segment_everything과 같은 이유(모듈 상단 _release_cuda_cache 참조) —
        # model/result의 GPU 텐서 참조를 끊어야 empty_cache()가 실제로 회수한다.
        del model
        del result
        _release_cuda_cache()

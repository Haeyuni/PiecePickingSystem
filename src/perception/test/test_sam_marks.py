"""조각 합치기 규칙 (sam_marks.merge_marks).

VLM이 "이 번호는 저 물체의 조각이다"라고 답한 것을 그대로 믿지 않는다는 것이 요점이다.
2026-09-07 실측에서 gpt-4o가 사진에 그려져 있던 YOLO 라벨 배너를 치약의 조각이라고 답했다 —
글자를 읽고 같은 물체로 본 것이다. 잘못 합치면 배경이 물체의 일부가 되어 마스크 중심이
밀리고 로봇이 엉뚱한 곳을 집는다.
"""
import numpy as np

from perception import sam_marks


def box_mask(x1, y1, x2, y2, shape=(100, 200)):
    mask = np.zeros(shape, dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def mark(mark_id, *, is_object=False, part_of=0, class_name="", name_ko="",
         mass_g=0.0, fragile=False, deformable=False, transparent=False,
         profile="normal", confidence=0.9):
    return {"mark_id": mark_id, "is_object": is_object, "part_of": part_of,
            "class_name": class_name, "name_ko": name_ko, "mass_g": mass_g,
            "fragile": fragile, "deformable": deformable, "transparent": transparent,
            "profile": profile, "confidence": confidence}


def test_object_without_pieces_passes_through():
    masks = [box_mask(10, 10, 40, 40)]
    objects = sam_marks.merge_marks([mark(1, is_object=True, class_name="tape")], masks)

    assert len(objects) == 1
    assert objects[0]["class_name"] == "tape"
    assert objects[0]["merged_marks"] == []
    assert objects[0]["mask"].sum() == masks[0].sum()


def test_adjacent_piece_is_merged():
    """뚜껑/몸통처럼 붙어 있는 작은 조각은 합쳐야 한다 — 안 합치면 물체가 잘린다."""
    masks = [box_mask(10, 10, 50, 50), box_mask(50, 10, 58, 50)]
    objects = sam_marks.merge_marks(
        [mark(1, is_object=True, class_name="spray"), mark(2, part_of=1)], masks)

    assert objects[0]["merged_marks"] == [2]
    assert objects[0]["mask"].sum() == masks[0].sum() + masks[1].sum()


def test_piece_that_blows_up_the_bbox_is_rejected():
    """배너 사례. 붙어 있어도 외접 사각형을 1.6배 넘게 키우면 조각이 아니다."""
    masks = [box_mask(10, 10, 50, 50), box_mask(50, 10, 190, 16)]
    rejected = []
    objects = sam_marks.merge_marks(
        [mark(1, is_object=True, class_name="toothpaste"), mark(2, part_of=1)], masks,
        on_reject=lambda piece, parent: rejected.append((piece, parent)))

    assert objects[0]["merged_marks"] == []
    assert rejected == [(2, 1)]
    assert objects[0]["mask"].sum() == masks[0].sum()


def test_piece_bigger_than_the_main_region_is_rejected():
    """조각이 대표 조각보다 크면 그것은 조각이 아니라 다른 영역이다."""
    masks = [box_mask(10, 10, 20, 20), box_mask(20, 10, 60, 60)]
    objects = sam_marks.merge_marks(
        [mark(1, is_object=True, class_name="gel_nail"), mark(2, part_of=1)], masks)

    assert objects[0]["merged_marks"] == []


def test_non_objects_are_dropped():
    masks = [box_mask(10, 10, 40, 40), box_mask(60, 10, 90, 40)]
    objects = sam_marks.merge_marks(
        [mark(1, is_object=True, class_name="tape"), mark(2)], masks)

    assert [o["mark_id"] for o in objects] == [1]


def test_part_of_pointing_at_an_object_mark_is_ignored():
    """물체로 판단된 번호를 조각으로 지목하면 무시한다 — 두 물체가 하나로 합쳐지면
    한쪽이 통째로 사라진다."""
    masks = [box_mask(10, 10, 40, 40), box_mask(41, 10, 70, 40)]
    objects = sam_marks.merge_marks(
        [mark(1, is_object=True, class_name="tape"),
         mark(2, is_object=True, part_of=1, class_name="nail")], masks)

    assert [o["mark_id"] for o in objects] == [1, 2]
    assert objects[0]["merged_marks"] == []


def test_out_of_range_mark_ids_are_ignored():
    """VLM이 없는 번호를 답해도 IndexError로 관측 전체가 죽지 않아야 한다."""
    masks = [box_mask(10, 10, 40, 40)]
    objects = sam_marks.merge_marks(
        [mark(1, is_object=True, class_name="tape"), mark(9, part_of=1),
         mark(7, is_object=True, class_name="ghost")], masks)

    assert [o["mark_id"] for o in objects] == [1]


def test_bbox_area_of_empty_mask():
    assert sam_marks.bbox_area(np.zeros((10, 10), dtype=bool)) == 0


def test_vlm_attributes_ride_along_as_attrs():
    """이 경로에는 objects.yaml 조회가 없다 — VLM이 낸 값이 그대로 DetectedObject가 된다."""
    masks = [box_mask(10, 10, 40, 40)]
    objects = sam_marks.merge_marks(
        [mark(1, is_object=True, class_name="toothpaste", name_ko="치약",
              mass_g=150.0, deformable=True, profile="deformable")], masks)

    attrs = objects[0]["attrs"]
    assert attrs["mass_g"] == 150.0
    assert attrs["deformable"] is True
    assert attrs["profile"] == "deformable"
    # 사람이 확인한 값이 아니라는 표시가 함께 간다 (FR-05b의 확인 대기 목록)
    assert attrs["attr_source"] == "llm_suggested"
    assert attrs["needs_confirmation"] is True


def test_unknown_profile_value_falls_back_to_fragile():
    """스크립트 경로(tools/scripts)는 planner의 _normalize_marks를 안 지난다."""
    masks = [box_mask(10, 10, 40, 40)]
    objects = sam_marks.merge_marks(
        [mark(1, is_object=True, class_name="mystery", profile="turbo")], masks)

    assert objects[0]["attrs"]["profile"] == "fragile"

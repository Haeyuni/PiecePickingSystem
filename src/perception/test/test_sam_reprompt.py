"""재투영 프롬프트 만들기 (sam_reprompt.prime).

SAM은 부르지 않는다 — 여기서 보는 것은 "직전 물체를 지금 프레임의 어느 픽셀로 되돌렸나"와
"되돌릴 수 없는 것을 조용히 통과시키지 않나"다. 화면 밖 좌표를 프롬프트로 주면 SAM이 엉뚱한
곳의 마스크를 돌려주고, 그것이 직전 물체의 것으로 라벨링되어 로봇이 잘못된 곳을 집는다.
"""
import numpy as np
from perception_common import geometry

from perception.detectors.sam_reprompt import SamRepromptDetector

INTRINSICS = {"fx": 600.0, "fy": 600.0, "cx": 640.0, "cy": 360.0,
              "width": 1280, "height": 720}
# 카메라가 base 원점에서 +z를 보고 있는 가장 단순한 배치
IDENTITY = np.eye(4)


def previous(class_name="tape", position=(0.0, 0.0, 500.0), extent=None, **extra):
    """직전 관측 한 건. extent는 마스크 테두리를 역투영한 base 점들(mask_extent_3d)."""
    if extent is None and position is not None:
        x, y, z = position
        extent = [(x - 20, y - 15, z), (x + 20, y - 15, z),
                  (x - 20, y + 15, z), (x + 20, y + 15, z)]
    return {"class_name": class_name, "confidence": 0.9,
            "attrs": {"name_ko": "테이프", "mass_g": 80.0, "fragile": False,
                      "deformable": False, "transparent": True, "grip_level": 3,
                      "attr_source": "llm_suggested", "needs_confirmation": True},
            "position_base_mm": position, "extent_base_mm": extent or [], **extra}


def make():
    return SamRepromptDetector("unused.pt")


def test_box_surrounds_the_object_centre():
    detector = make()

    assert detector.prime([previous()], IDENTITY, IDENTITY, INTRINSICS) == 1
    x1, y1, x2, y2 = detector.expected[0]["box_xyxy"]
    assert x1 < 640 < x2 and y1 < 360 < y2


def test_box_corners_match_the_forward_projection():
    """prime이 쓰는 변환이 geometry의 정방향과 같은지 — 다르면 조용히 어긋난다."""
    detector = make()
    item = previous(position=(40.0, -25.0, 480.0))
    detector.prime([item], IDENTITY, IDENTITY, INTRINSICS)

    pixels = [geometry.project_to_pixel(
        geometry.camera_from_base(p, IDENTITY, IDENTITY), INTRINSICS)
        for p in item["extent_base_mm"]]
    expected = [int(min(u for u, _ in pixels)), int(min(v for _, v in pixels)),
                int(max(u for u, _ in pixels)), int(max(v for _, v in pixels))]

    assert detector.expected[0]["box_xyxy"] == expected


def test_object_without_extent_is_dropped():
    """중심점으로 대신하지 않는다 — 점 프롬프트는 물체 대신 무늬를 잡는다(실측 IoU 0.03)."""
    detector = make()

    assert detector.prime([previous(extent=[])], IDENTITY, IDENTITY, INTRINSICS) == 0


def test_partially_offscreen_object_is_clipped_not_dropped():
    """보이는 부분이라도 되찾는 편이 통째로 잃는 것보다 낫다."""
    detector = make()
    # fx=600, z=600이라 u = cx + x. 중심을 u=0에 두면 테두리 절반이 화면 밖으로 나간다.
    item = previous(position=(-640.0, 0.0, 600.0))

    assert detector.prime([item], IDENTITY, IDENTITY, INTRINSICS) == 1
    assert detector.expected[0]["box_xyxy"][0] == 0


def test_labels_are_inherited_not_recomputed():
    """VLM을 다시 부르지 않는다는 것이 코드로 지켜지는지 (명령당 1회 규칙).

    이름뿐 아니라 **속성도** 물려받아야 한다 — 재관측 경로는 VLM도 objects.yaml도 안 부르므로
    여기서 잃으면 그 물체만 grip_level이 달라진다.
    """
    detector = make()
    detector.prime([previous(class_name="toothpaste")], IDENTITY, IDENTITY, INTRINSICS)

    assert detector.expected[0]["class_name"] == "toothpaste"
    assert detector.expected[0]["attrs"]["grip_level"] == 3
    assert detector.expected[0]["attrs"]["attr_source"] == "llm_suggested"


def test_object_behind_the_camera_is_dropped():
    detector = make()
    item = previous(position=(0.0, 0.0, -300.0),
                    extent=[(-20, -15, -300), (20, 15, -300)])

    assert detector.prime([item], IDENTITY, IDENTITY, INTRINSICS) == 0
    assert detector.expected == []


def test_object_outside_the_frame_is_dropped():
    """팔이 물체를 시야 밖으로 밀어냈을 때. 클램프해서 가장자리 픽셀을 찍으면 안 된다."""
    detector = make()

    assert detector.prime([previous(position=(5000.0, 0.0, 500.0))],
                          IDENTITY, IDENTITY, INTRINSICS) == 0


def test_object_without_position_is_dropped():
    """좌표를 못 냈던 물체(depth 무효)는 되돌릴 기준이 없다."""
    detector = make()

    assert detector.prime([previous(position=None, extent=[])],
                          IDENTITY, IDENTITY, INTRINSICS) == 0


def test_detect_without_prime_returns_nothing():
    detections, debug = make().detect(np.zeros((720, 1280, 3), dtype=np.uint8))

    assert detections == []
    assert debug is None

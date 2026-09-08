"""/internal/label-marks 단위 테스트 (stdlib unittest — test_validator.py와 같은 이유).

**VLM은 부르지 않는다.** 여기서 보는 것은 요청 파싱과 에러 형식이다. 실제 모델이 사진을
제대로 읽는지는 tools/scripts/vlm_sam_test.py와 check_label_marks.py가 본다 — 단위 테스트가
매번 과금되면 아무도 안 돌린다.

실행: docker compose exec planner python -m unittest discover -s tests -t .
"""
import io
import unittest

from fastapi.testclient import TestClient

from src import app as app_module
from src import vlm_detect

# TestClient를 with 없이 쓰면 lifespan(마이그레이션·시드)이 돌지 않는다 — 이 엔드포인트는
# DB를 쓰지 않으므로 그대로 둔다.
client = TestClient(app_module.app)

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32


def post(image: bytes = PNG, **form):
    data = {"mark_ids": "1,2", **form}
    return client.post("/internal/label-marks",
                       files={"image": ("marks.png", io.BytesIO(image), "image/png")},
                       data=data)


class LabelMarksTest(unittest.TestCase):
    def setUp(self):
        self._real = vlm_detect.label_marks
        self.calls = []

        def fake(image, mark_ids, **kwargs):
            self.calls.append((image, mark_ids, kwargs))
            return vlm_detect.VlmMarkScene(marks=[
                vlm_detect.MarkLabel(mark_id=1, is_object=True, part_of=0,
                                     class_name="toothpaste", name_ko="치약",
                                     mass_g=150.0, fragile=False, deformable=True,
                                     transparent=False, profile="deformable",
                                     confidence=0.9, reasoning="흔한 치약 튜브 형태"),
                vlm_detect.MarkLabel(mark_id=2, is_object=False, part_of=0, class_name="",
                                     name_ko="", mass_g=0.0, fragile=False,
                                     deformable=False, transparent=False,
                                     profile="fragile", confidence=0.9, reasoning=""),
            ])

        vlm_detect.label_marks = fake

    def tearDown(self):
        vlm_detect.label_marks = self._real

    def test_returns_marks_and_versions(self):
        response = post(trace_id="tr-1")
        self.assertEqual(response.status_code, 200)

        body = response.json()
        self.assertEqual(body["trace_id"], "tr-1")
        self.assertEqual(body["prompt_version"], vlm_detect.MARKS_PROMPT_VERSION)
        self.assertEqual([m["mark_id"] for m in body["marks"]], [1, 2])
        self.assertEqual(body["marks"][0]["class_name"], "toothpaste")

    def test_attributes_reach_perception(self):
        """무게·파지 프로파일까지 응답에 실려야 한다 — perception은 objects.yaml을 안 본다."""
        mark = post().json()["marks"][0]

        self.assertEqual(mark["mass_g"], 150.0)
        self.assertTrue(mark["deformable"])
        self.assertEqual(mark["profile"], "deformable")

    def test_reasoning_reaches_the_response(self):
        """판단 근거가 화면에 뜨려면 이 응답에 실려 나가야 한다 (물체 판단 근거 표시)."""
        marks = post().json()["marks"]

        self.assertEqual(marks[0]["reasoning"], "흔한 치약 튜브 형태")
        self.assertEqual(marks[1]["reasoning"], "")   # is_object=false는 빈 문자열

    def test_registered_classes_are_not_sent_to_the_vlm(self):
        """등록 클래스 어휘를 주지 않는 것이 이 경로의 전제다 (vlm_detect 상단 주석).

        어휘를 넣으면 모델이 그 목록 안에서만 답해 처음 보는 물건이 목록의 이름으로
        불리고, 그 물체에 맞지 않는 속성이 붙는다.
        """
        prompt = vlm_detect.build_marks_prompt([1, 2])

        self.assertNotIn("등록된 클래스", prompt)
        self.assertIn("[그려진 번호] 1, 2", prompt)

    def test_image_reaches_vlm_as_data_url(self):
        """파일 경로가 아니라 본문을 그대로 넘겨야 한다 — planner에는 그 파일이 없다."""
        post()

        image, mark_ids, _ = self.calls[0]
        self.assertTrue(image.startswith("data:image/png;base64,"))
        self.assertEqual(mark_ids, [1, 2])

    def test_command_is_never_forwarded(self):
        """지시를 인지 단계에 넣지 않는다는 결정이 코드로 지켜지는지 (D-1 근거)."""
        post()

        _, _, kwargs = self.calls[0]
        self.assertNotIn("command_text", kwargs)

    def test_rejects_unparsable_mark_ids(self):
        response = post(mark_ids="1,둘,3")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "BAD_MARK_IDS")

    def test_rejects_mark_ids_without_numbers(self):
        """구분자만 있는 경우. 필드 자체가 빠지면 FastAPI가 먼저 422로 막는다 —
        /internal/plan의 스키마 오류와 같은 처리라 그대로 둔다."""
        response = post(mark_ids=" , ")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "BAD_MARK_IDS")

    def test_rejects_empty_image(self):
        response = post(image=b"")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "EMPTY_IMAGE")

    def test_vlm_failure_is_503(self):
        """하위 서비스 장애는 503 (웹_인터페이스_정의서 6절). 400으로 내면 web이
        '요청이 잘못됐다'고 판단해 재시도하지 않는다."""
        def boom(*args, **kwargs):
            raise RuntimeError("openai timeout")

        vlm_detect.label_marks = boom
        response = post(trace_id="tr-2")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "VLM_UNAVAILABLE")
        self.assertEqual(response.json()["trace_id"], "tr-2")


if __name__ == "__main__":
    unittest.main()

"""label_marks()의 잘린 응답(response.status == "incomplete") 처리 테스트.

**VLM은 부르지 않는다** (test_label_marks.py와 같은 이유) — `vlm_detect._client()`가
돌려주는 객체를 가짜로 바꿔서, 실제 API가 토큰 부족으로 응답을 자르면 나오는 모양
(status="incomplete")을 흉내 낸다.

**`.responses.create()`를 모킹한다 — `.parse()`가 아니다.** 최초 구현은 `.parse()`를
썼는데, 그건 응답을 받자마자 SDK 내부에서 곧바로 JSON 파싱까지 해 버린다(openai
3.8.0, lib/_parsing/_responses.py). 그래서 응답이 잘려 있으면 `.parse()` 호출 자체가
pydantic ValidationError로 죽어서, 그 뒤에 있던 `response.status == "incomplete"` 확인은
**한 번도 실행되지 못하는 죽은 코드**였다(2026-09-08 실물에서 발견 — 잘림이 실제로
max_output_tokens 때문인지 확인하려고 원본 응답을 들여다보다가 알았다). 지금은
`.create()`(원본, 자동 파싱 없음)로 status/usage를 먼저 보고, 파싱은 우리가 한다.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src import vlm_detect


def _message_output(text: str):
    return SimpleNamespace(
        type="message",
        content=[SimpleNamespace(type="output_text", text=text)],
    )


def _usage(output_tokens: int = 100):
    return SimpleNamespace(
        input_tokens=500, output_tokens=output_tokens,
        output_tokens_details=SimpleNamespace(reasoning_tokens=0),
        total_tokens=500 + output_tokens,
    )


def _fake_client(status: str, reason: str | None = None, output_text: str = "{}"):
    response = SimpleNamespace(
        status=status,
        incomplete_details=SimpleNamespace(reason=reason) if reason else None,
        output=[_message_output(output_text)] if status == "completed" else [],
        usage=_usage(),
    )
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **_: response))
    return client


class LabelMarksTruncationTest(unittest.TestCase):
    def test_incomplete_response_raises_clear_error(self):
        with patch.object(vlm_detect, "_client",
                          return_value=_fake_client("incomplete", "max_output_tokens")):
            with self.assertRaises(RuntimeError) as ctx:
                vlm_detect.label_marks("data:image/png;base64,AA==", [1, 2])
        self.assertIn("incomplete", str(ctx.exception))
        self.assertIn("max_output_tokens", str(ctx.exception))

    def test_completed_response_passes_through(self):
        with patch.object(
                vlm_detect, "_client",
                return_value=_fake_client("completed", output_text='{"marks": []}')):
            scene = vlm_detect.label_marks("data:image/png;base64,AA==", [1, 2])
        self.assertEqual(scene.marks, [])

    def test_request_asks_for_max_output_tokens_and_structured_text_format(self):
        captured = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                status="completed", incomplete_details=None,
                output=[_message_output('{"marks": []}')], usage=_usage())

        client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))
        with patch.object(vlm_detect, "_client", return_value=client):
            vlm_detect.label_marks("data:image/png;base64,AA==", [1, 2])
        self.assertEqual(captured.get("max_output_tokens"),
                         vlm_detect.MARKS_MAX_OUTPUT_TOKENS)
        self.assertNotIn("text_format", captured)
        self.assertIn("format", captured.get("text", {}))

    def test_pharmacy_domain_gets_web_search_with_call_cap(self):
        """2026-09-09 — pharmacy는 약 이름을 잘못 읽으면 안전 문제라 web_search를 다시
        켠다(mask_poly를 없애 truncation 원인을 없앴으니 안전). 검색 결과가 여전히
        출력 토큰을 먹으므로 max_tool_calls로 횟수는 계속 제한한다."""
        captured = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                status="completed", incomplete_details=None,
                output=[_message_output('{"marks": []}')], usage=_usage())

        client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))
        with patch.object(vlm_detect, "_client", return_value=client):
            vlm_detect.label_marks("data:image/png;base64,AA==", [1, 2], domain="pharmacy")
        self.assertEqual(captured.get("tools"), [{"type": "web_search"}])
        self.assertEqual(captured.get("max_tool_calls"), vlm_detect.MARKS_MAX_WEB_SEARCHES)

    def test_other_domains_get_no_web_search(self):
        """general/recycle은 그 정도 정밀도가 필요 없다 — 검색 없이 출력 토큰을 아낀다."""
        for domain in ("general", "recycle"):
            captured = {}

            def fake_create(**kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    status="completed", incomplete_details=None,
                    output=[_message_output('{"marks": []}')], usage=_usage())

            client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))
            with patch.object(vlm_detect, "_client", return_value=client):
                vlm_detect.label_marks("data:image/png;base64,AA==", [1, 2], domain=domain)
            self.assertNotIn("tools", captured, f"domain={domain}")
            self.assertNotIn("max_tool_calls", captured, f"domain={domain}")

    def test_usage_is_logged_so_actual_token_spend_is_visible(self):
        """2026-09-08 사용자 질문 — "토큰이 부족한지, 실제 얼마나 썼는지 어디서 보나."
        추측이 아니라 매 호출마다 실제 usage를 로그로 남긴다."""
        with self.assertLogs(vlm_detect.logger, level="INFO") as ctx:
            with patch.object(
                    vlm_detect, "_client",
                    return_value=_fake_client("completed", output_text='{"marks": []}')):
                vlm_detect.label_marks("data:image/png;base64,AA==", [1, 2])
        self.assertTrue(any("input_tokens=500" in line and "output_tokens=100" in line
                            for line in ctx.output))


if __name__ == "__main__":
    unittest.main()

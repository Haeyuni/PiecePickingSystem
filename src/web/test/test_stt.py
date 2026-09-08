"""POST /api/stt — 실행하지 않고 텍스트만 돌려주는지, 저신뢰도 플래그가 맞는지(FR-24/25).

전체 앱(lifespan이 executor를 띄운다)을 부르지 않고 stt.router만 올린다 — 여기서 볼 것은
요청/응답 형태와 신뢰도 임계값 로직이지 로봇 상태가 아니다. 모델 로드는 무겁고 네트워크가
필요할 수 있어 `stt.transcribe`를 가짜로 바꾼다.
"""
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from web import stt
from web.routers import stt as stt_router

app = FastAPI()
app.include_router(stt_router.router)
client = TestClient(app)


class SttTest(unittest.TestCase):
    def setUp(self):
        self._real = stt.transcribe

    def tearDown(self):
        stt.transcribe = self._real

    def post(self, audio: bytes = b"fake-audio-bytes"):
        return client.post("/api/stt", files={"audio": ("cmd.webm", audio, "audio/webm")})

    def test_high_confidence_is_not_flagged(self):
        stt.transcribe = lambda data: ("치약 왼쪽으로", 0.93)

        response = self.post()

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["recognized_text"], "치약 왼쪽으로")
        self.assertEqual(body["confidence"], 0.93)
        self.assertFalse(body["low_confidence"])

    def test_low_confidence_is_flagged_but_still_returns_text(self):
        """FR-25: 낮은 신뢰도는 재입력을 유도하되, 텍스트 자체는 감추지 않는다
        (사용자가 보고 직접 고칠 수 있어야 한다)."""
        stt.transcribe = lambda data: ("음... 저거", 0.4)

        body = self.post().json()

        self.assertEqual(body["recognized_text"], "음... 저거")
        self.assertTrue(body["low_confidence"])

    def test_silence_is_flagged_even_with_no_confidence_signal(self):
        stt.transcribe = lambda data: ("", 0.0)

        body = self.post().json()

        self.assertEqual(body["recognized_text"], "")
        self.assertTrue(body["low_confidence"])

    def test_empty_audio_is_rejected_before_transcribing(self):
        calls = []
        stt.transcribe = lambda data: calls.append(data) or ("", 0.0)

        response = self.post(audio=b"")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "EMPTY_AUDIO")
        self.assertEqual(calls, [])

    def test_transcribe_failure_is_503(self):
        def boom(data):
            raise RuntimeError("model not loaded")

        stt.transcribe = boom
        response = self.post()

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "STT_UNAVAILABLE")

    def test_never_calls_the_command_pipeline(self):
        """STT는 실행하지 않는다 — orchestrator를 아예 참조하지 않는다(FR-23/24)."""
        import web.routers.stt as module

        self.assertNotIn("orchestrator", dir(module))


if __name__ == "__main__":
    unittest.main()

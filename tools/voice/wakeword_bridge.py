"""'hello rokey' 웨이크워드 감지 → web 백엔드로 중계.

브라우저 SpeechRecognition(Web Speech API)은 사전에 없는 단어("rokey")를 신뢰성 있게
못 잡는다 — 실측(2026-09-09)으로 확인된 문제다(web/frontend/src/hooks/useWakeWord.ts
이력 참조, 그 접근은 이 브리지로 대체됐다). corecode/VoiceProcessing의 openwakeword
커스텀 모델(hello_rokey_8332_32.tflite)로 로봇 PC에 붙은 마이크를 직접 들으면 훨씬
안정적으로 잡힌다(`python3 corecode/VoiceProcessing/wakeup_word.py`로 실측 확인).

감지되면 web의 /api/internal/wake-detected를 호출한다 — web이 그걸 그대로
/ws/live로 중계해 열려 있는 모든 브라우저 탭에 알린다(routers/voice.py).

**드라이버처럼 계속 떠 있어야 하는 프로세스이고, 사용자가 직접 띄우고 끈다**
(bringup과 같은 원칙 — tools/scripts/run_bringup.sh 참조). corecode는 이 저장소가
관리하지 않는 로봇 PC 로컬 자산(학습된 모델 파일 포함)이라 자동 기동에 넣지 않았고,
그 코드를 복사하지 않고 경로로만 참조한다 — 모델이 갱신되면 따로 손댈 곳이 없다.

실행:
    python3 tools/voice/wakeword_bridge.py

환경변수:
    VOICE_PROCESSING_DIR   corecode/VoiceProcessing 경로 (기본 ~/corecode/VoiceProcessing)
    WEB_BASE_URL            web 백엔드 주소 (기본 http://localhost:8000 — web_ros가
                            network_mode: host라 다른 서비스와 같은 이유로 이렇게 부른다)
    VOICE_MIC_DEVICE        WakeupWord가 그대로 쓴다(audio_device.py 참조)
"""
import os
import sys
import time
import urllib.error
import urllib.request

VOICE_PROCESSING_DIR = os.environ.get(
    "VOICE_PROCESSING_DIR", os.path.expanduser("~/corecode/VoiceProcessing"))
WEB_BASE_URL = os.environ.get("WEB_BASE_URL", "http://localhost:8000")
WAKE_ENDPOINT = f"{WEB_BASE_URL}/api/internal/wake-detected"

# 같은 발화가 연속 프레임에서 다시 감지돼 중복 트리거되지 않게 두는 최소 간격.
# CommandInput.tsx의 WAKE_RECORD_MS(4000ms) 녹음 창보다 넉넉히 잡는다.
RETRIGGER_COOLDOWN_S = 5.0

sys.path.insert(0, VOICE_PROCESSING_DIR)
from wakeup_word import WakeupWord  # noqa: E402  corecode 쪽 코드라 경로를 먼저 더해야 한다

# wakeup_word.py의 MODEL_NAME은 상대 경로("hello_rokey_8332_32.tflite")라 openwakeword가
# 그걸 cwd 기준으로 찾는다 — `cd corecode/VoiceProcessing && python3 wakeup_word.py`로 돌릴
# 때는 저절로 맞았지만, 여기서는 이 프로세스가 어디서 실행되든 맞게 우리가 cwd를 옮겨야 한다.
os.chdir(VOICE_PROCESSING_DIR)


def notify_web() -> bool:
    request = urllib.request.Request(WAKE_ENDPOINT, method="POST", data=b"")
    try:
        with urllib.request.urlopen(request, timeout=2.0):
            return True
    except urllib.error.URLError as exc:
        print(f"[wakeword_bridge] web 알림 실패 (web이 안 떠 있나?): {exc}")
        return False


def main() -> None:
    wakeup = WakeupWord()
    wakeup.open()
    print(f"[wakeword_bridge] 대기 중 (모델={VOICE_PROCESSING_DIR}, web={WEB_BASE_URL}) "
          "— Ctrl+C로 종료")
    last_trigger = 0.0
    try:
        while True:
            if wakeup.is_wakeup():
                now = time.monotonic()
                if now - last_trigger < RETRIGGER_COOLDOWN_S:
                    continue
                last_trigger = now
                if notify_web():
                    print("[wakeword_bridge] 감지 → web에 알림")
    except KeyboardInterrupt:
        pass
    finally:
        wakeup.close()


if __name__ == "__main__":
    main()

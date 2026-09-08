/** "Hello Rokey" 웨이크워드 — 상시 대기 후 감지되면 콜백을 부른다.
 *
 * 브라우저 내장 Web Speech API(`SpeechRecognition`, continuous 모드)로 구현한다.
 * 별도 라이브러리·API 키 없이 가장 빠르게 붙는 방법이지만 트레이드오프가 있다:
 * Chrome/Edge 전용이고, 음성이 브라우저 벤더 서버로 전송된다(완전 오프라인이 아니다).
 * 오프라인이 필요해지면 Picovoice Porcupine Web SDK로 "Hello Rokey" 커스텀 키워드를
 * 학습시켜 교체하는 걸 다음 단계로 남겨 둔다.
 *
 * 상시 마이크 대기는 프라이버시에 민감하므로 기본값은 꺼짐이고, 사용자가 명시적으로
 * 켜야 한다(옵트인) — `enabled`/`setEnabled`로 그 상태를 그대로 노출한다.
 */
import { useEffect, useRef, useState } from 'react'

// 표준 DOM 타입에 아직 없다(lib.dom.d.ts 미포함) — 여기서 쓰는 최소한만 선언한다.
interface SpeechRecognitionResultLike {
  [index: number]: { transcript: string }
}
interface SpeechRecognitionEventLike {
  resultIndex: number
  results: { [index: number]: SpeechRecognitionResultLike; length: number }
}
interface SpeechRecognitionLike {
  continuous: boolean
  interimResults: boolean
  lang: string
  onresult: ((event: SpeechRecognitionEventLike) => void) | null
  onend: (() => void) | null
  onerror: (() => void) | null
  start(): void
  stop(): void
}

declare global {
  interface Window {
    SpeechRecognition?: new () => SpeechRecognitionLike
    webkitSpeechRecognition?: new () => SpeechRecognitionLike
  }
}

const WAKE_LANG = 'en-US'   // "hello rokey"는 영문 문구라 인식 언어도 영문으로 맞춘다

export function useWakeWord(phrase: string, onDetected: () => void) {
  const [enabled, setEnabled] = useState(false)
  const supported = typeof window !== 'undefined'
    && !!(window.SpeechRecognition || window.webkitSpeechRecognition)
  const onDetectedRef = useRef(onDetected)
  onDetectedRef.current = onDetected

  useEffect(() => {
    if (!enabled || !supported) return

    const Ctor = window.SpeechRecognition ?? window.webkitSpeechRecognition!
    const recognition = new Ctor()
    recognition.continuous = true
    recognition.interimResults = true
    recognition.lang = WAKE_LANG

    let stopped = false
    recognition.onresult = (event) => {
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript.toLowerCase()
        if (transcript.includes(phrase.toLowerCase())) {
          onDetectedRef.current()
        }
      }
    }
    // continuous 모드도 브라우저가 일정 시간 뒤 스스로 끊을 수 있다 — 계속 켜 두려면
    // 다시 시작해야 한다(사용자가 끈 경우는 stopped로 구분해 재시작하지 않는다).
    recognition.onend = () => {
      if (!stopped) recognition.start()
    }
    recognition.onerror = () => { /* 권한 거부 등 — onend가 뒤이어 오므로 여기선 무시 */ }

    recognition.start()
    return () => {
      stopped = true
      recognition.stop()
    }
  }, [enabled, supported, phrase])

  return { enabled, setEnabled, supported }
}

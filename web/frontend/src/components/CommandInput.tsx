/** 명령 입력창 (화면정의서.md 2.2.9)
 *
 * mode !== 'idle'이면 비활성화하지만, 이건 **UX 보조 수단**이다. 실제 차단은 항상
 * web 백엔드가 하며(웹_인터페이스_정의서 2.1절), 여기서 막지 못하고 넘어가도 409로 돌아온다.
 * 그 409 사유를 그대로 보여주는 것이 이 컴포넌트의 두 번째 역할이다.
 */
import { useRef, useState } from 'react'
import { sendCommand, transcribeAudio } from '../api'
import type { ApiError } from '../api'
import { useWakeWord } from '../hooks/useWakeWord'
import type { Domain, RobotMode } from '../types'

const WAKE_PHRASE = 'hello rokey'
// 웨이크워드 감지 후 명령을 녹음할 고정 길이. 무음 감지 대신 단순 타이머로 자른다 —
// push-to-talk(수동 토글)과 달리 언제 끝났는지 판단할 신호가 없다.
const WAKE_RECORD_MS = 4000

const DISABLED_REASON: Record<string, string> = {
  busy: '로봇이 동작 중입니다 — 완료 후 다시 시도하세요',
  error: '로봇이 오류 상태입니다 — 상태를 확인하세요',
  estopped: '비상정지 상태입니다 — 수동 리셋이 필요합니다',
}

// 시나리오별 계획 능력을 보여주는 테스트용 원클릭 명령. 단순히 문구를 그대로 보내는 것이
// 아니라 `domain`과 함께 보내, 백엔드가 어떤 시나리오인지 알고 **도메인에 맞는 다른
// VLM 프롬프트**로 분기 처리하게 한다 (vlm_detect.py의 도메인 프롬프트 참조).
const SCENARIOS: { label: string; domain: Domain; text: string }[] = [
  { label: '가정', domain: 'general', text: '생활용품 왼쪽으로' },
  { label: '약국', domain: 'pharmacy', text: '머리가 아플 때 먹는 약 줘' },
  { label: '재활용', domain: 'recycle', text: '플라스틱 왼쪽, 캔 오른쪽' },
]

export default function CommandInput({
  mode, onAccepted,
}: {
  mode: RobotMode
  onAccepted: (traceId: string, commandText: string) => void
}) {
  const [text, setText] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [sending, setSending] = useState(false)
  const [recording, setRecording] = useState(false)
  const [transcribing, setTranscribing] = useState(false)
  const [sttWarning, setSttWarning] = useState<string | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const blocked = mode !== 'idle'

  const submit = async (
    commandText: string = text.trim(),
    domain: Domain = 'general',
  ) => {
    if (!commandText) return
    setSending(true)
    setError(null)
    try {
      const { trace_id } = await sendCommand(commandText, domain)
      onAccepted(trace_id, commandText)
      setText('')
    } catch (e) {
      setError((e as ApiError).message ?? '명령을 보내지 못했습니다')
    } finally {
      setSending(false)
    }
  }

  const runScenario = (scenarioText: string, domain: Domain) => {
    setText(scenarioText)
    submit(scenarioText, domain)
  }

  // 마이크 버튼(2.2.9절) — 클릭으로 녹음 시작/종료를 토글한다(누르고 있는 방식이 아니다).
  // 종료 시 /api/stt로 변환한다. FR-24에 따라 인식 결과는 입력창에 채우기만 하고
  // 자동 전송하지 않는다 — 사용자가 확인/수정 후 직접 전송 버튼을 눌러야 한다.
  const toggleRecording = () => {
    if (recording) stopRecording()
    else void startRecording()
  }

  const startRecording = async () => {
    setSttWarning(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)
      chunksRef.current = []
      recorder.ondataavailable = (e) => chunksRef.current.push(e.data)
      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop())
        void handleRecorded(new Blob(chunksRef.current, { type: recorder.mimeType }))
      }
      recorder.start()
      recorderRef.current = recorder
      setRecording(true)
    } catch {
      setSttWarning('마이크를 사용할 수 없습니다 — 권한을 확인하세요')
    }
  }

  const stopRecording = () => {
    recorderRef.current?.stop()
    setRecording(false)
  }

  const handleRecorded = async (audio: Blob) => {
    setTranscribing(true)
    try {
      const result = await transcribeAudio(audio)
      setText(result.recognized_text)
      if (result.low_confidence) {
        setSttWarning('인식 결과가 정확하지 않을 수 있습니다 — 확인 후 전송하세요')
      }
    } catch (e) {
      setSttWarning((e as ApiError).message ?? '음성 인식에 실패했습니다')
    } finally {
      setTranscribing(false)
    }
  }

  // "Hello Rokey" 감지 시 자동으로 녹음을 시작하고 고정 시간 뒤 스스로 끊는다 — 이미
  // 녹음/변환/전송 중이거나 명령을 받을 수 없는 상태면 무시한다(중복 트리거 방지).
  const handleWakeDetected = () => {
    if (recording || transcribing || blocked || sending) return
    void startRecording()
    window.setTimeout(stopRecording, WAKE_RECORD_MS)
  }

  const { enabled: wakeEnabled, setEnabled: setWakeEnabled, supported: wakeSupported } =
    useWakeWord(WAKE_PHRASE, handleWakeDetected)

  return (
    <div>
      <div className="scenarios">
        {SCENARIOS.map((s) => (
          <button
            key={s.label}
            type="button"
            disabled={blocked || sending}
            title={s.text}
            onClick={() => runScenario(s.text, s.domain)}
          >
            {s.label}
          </button>
        ))}
        <button
          type="button"
          className={wakeEnabled ? 'wake-armed' : undefined}
          disabled={!wakeSupported}
          title={wakeSupported
            ? `켜면 "${WAKE_PHRASE}"라고 말했을 때 자동으로 녹음을 시작합니다 — 상시 마이크 대기`
            : '이 브라우저는 지원하지 않습니다 (Chrome/Edge 권장)'}
          onClick={() => setWakeEnabled(!wakeEnabled)}
        >
          {wakeEnabled ? `🔴 "${WAKE_PHRASE}" 대기 중` : `🎙️ "${WAKE_PHRASE}" 켜기`}
        </button>
      </div>
      <div className="command">
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && !blocked && submit()}
          placeholder="예: 깨지기 쉬운 것만 왼쪽 박스로 옮겨줘"
          disabled={blocked || sending}
        />
        <button
          type="button"
          className={recording ? 'mic-recording' : undefined}
          disabled={blocked || sending || transcribing}
          onClick={toggleRecording}
          title="클릭하여 녹음 시작/종료"
        >
          {recording ? '● 녹음 중 (클릭하여 종료)' : transcribing ? '변환 중…' : '🎤'}
        </button>
        <button disabled={blocked || sending || !text.trim()} onClick={() => submit()}>
          {sending ? '전송 중…' : '전송'}
        </button>
      </div>
      {blocked && <div className="command-hint muted">{DISABLED_REASON[mode]}</div>}
      {sttWarning && <div className="command-hint hint-error">{sttWarning}</div>}
      {error && <div className="command-hint hint-error">{error}</div>}
    </div>
  )
}

/** 명령 입력창 (화면정의서.md 2.2.9)
 *
 * mode !== 'idle'이면 비활성화하지만, 이건 **UX 보조 수단**이다. 실제 차단은 항상
 * web 백엔드가 하며(웹_인터페이스_정의서 2.1절), 여기서 막지 못하고 넘어가도 409로 돌아온다.
 * 그 409 사유를 그대로 보여주는 것이 이 컴포넌트의 두 번째 역할이다.
 */
import { useState } from 'react'
import { sendCommand } from '../api'
import type { ApiError } from '../api'
import type { RobotMode } from '../types'

const DISABLED_REASON: Record<string, string> = {
  busy: '로봇이 동작 중입니다 — 완료 후 다시 시도하세요',
  error: '로봇이 오류 상태입니다 — 상태를 확인하세요',
  estopped: '비상정지 상태입니다 — 수동 리셋이 필요합니다',
}

// 시나리오별 계획 능력을 보여주는 테스트용 원클릭 명령. 인지 로직은 바꾸지 않고
// 기존 /api/commands 파이프라인에 미리 정해 둔 문구를 그대로 보낸다.
const SCENARIOS: { label: string; text: string }[] = [
  { label: '가정', text: '생활용품 왼쪽으로' },
  { label: '약국', text: '머리가 아플 때 먹는 약 줘' },
  { label: '재활용', text: '플라스틱 왼쪽, 캔 오른쪽' },
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
  const blocked = mode !== 'idle'

  const submit = async (commandText: string = text.trim()) => {
    if (!commandText) return
    setSending(true)
    setError(null)
    try {
      const { trace_id } = await sendCommand(commandText)
      onAccepted(trace_id, commandText)
      setText('')
    } catch (e) {
      setError((e as ApiError).message ?? '명령을 보내지 못했습니다')
    } finally {
      setSending(false)
    }
  }

  const runScenario = (scenarioText: string) => {
    setText(scenarioText)
    submit(scenarioText)
  }

  return (
    <div>
      <div className="scenarios">
        {SCENARIOS.map((s) => (
          <button
            key={s.label}
            type="button"
            disabled={blocked || sending}
            title={s.text}
            onClick={() => runScenario(s.text)}
          >
            {s.label}
          </button>
        ))}
      </div>
      <div className="command">
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && !blocked && submit()}
          placeholder="예: 깨지기 쉬운 것만 왼쪽 박스로 옮겨줘"
          disabled={blocked || sending}
        />
        {/* 마이크 버튼은 6단계(STT)에서 붙인다 — 2.2.9절 */}
        <button disabled={blocked || sending || !text.trim()} onClick={() => submit()}>
          {sending ? '전송 중…' : '전송'}
        </button>
      </div>
      {blocked && <div className="command-hint muted">{DISABLED_REASON[mode]}</div>}
      {error && <div className="command-hint hint-error">{error}</div>}
    </div>
  )
}

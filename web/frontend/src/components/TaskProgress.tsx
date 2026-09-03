/** 작업 진행도 (화면정의서.md 2.2.8)
 *
 * phase → 진행률 매핑은 프론트가 갖는 테이블이다(2.2.8절). 액션 정의의 Feedback 상수와
 * 같은 문자열을 키로 쓰므로, 스킬에 phase가 추가되면 여기도 함께 고쳐야 한다.
 */
import type { Trace, TraceStep } from '../types'

const PHASE_PERCENT: Record<string, number> = {
  // Pick.action
  approaching: 25, contact_detected: 50, lifting: 75,
  // PlaceInto.action
  moving: 25, inserting: 50, releasing: 75,
  // 두 액션 공통
  verifying: 90,
}

const PHASE_LABEL: Record<string, string> = {
  approaching: '접근 중', contact_detected: '접촉 감지', lifting: '들어올리는 중',
  moving: '이동 중', inserting: '넣는 중', releasing: '놓는 중', verifying: '검증 중',
}

const SKILL_LABEL: Record<string, string> = { pick: 'Pick', place_into: 'Place into' }

function percentOf(step: TraceStep): number {
  if (step.status === 'success') return 100
  if (step.status === 'failure') return 100
  if (step.status === 'pending') return 0
  return PHASE_PERCENT[step.phase ?? ''] ?? 10
}

function statusText(step: TraceStep): string {
  if (step.status === 'success') return '완료'
  if (step.status === 'failure') return '실패'
  if (step.status === 'pending') return '대기 중'
  return PHASE_LABEL[step.phase ?? ''] ?? (step.phase ?? '진행 중')
}

export default function TaskProgress({ trace }: { trace: Trace | null }) {
  if (!trace) {
    return <div className="empty">대기 중인 작업이 없습니다</div>
  }

  return (
    <div>
      <div style={{ marginBottom: 6 }}>
        현재 작업 — <strong>&ldquo;{trace.command_text}&rdquo;</strong>
      </div>

      {trace.validation_status === 'rejected' && (
        <div className="command-hint hint-error">
          검증 거부: {trace.validation_reason ?? '사유 없음'}
        </div>
      )}

      {trace.steps.length === 0 && trace.validation_status !== 'rejected' && (
        <div className="muted">계획 생성 중…</div>
      )}

      {trace.steps.map((step) => (
        <div className="progress-step" key={step.request_id}>
          <span className="label">
            {SKILL_LABEL[step.skill] ?? step.skill} — {step.bin_id ?? step.object_id}
          </span>
          <div
            className={`bar${step.status === 'success' ? ' done' : ''}${
              step.status === 'failure' ? ' failed' : ''
            }`}
          >
            <div style={{ width: `${percentOf(step)}%` }} />
          </div>
          <span className="phase">{statusText(step)}</span>
        </div>
      ))}
    </div>
  )
}

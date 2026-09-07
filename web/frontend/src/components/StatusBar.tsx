/** 상태바 (화면정의서.md 2.2.2) */
import type { RobotState, SafetyEvent } from '../types'

const MODE_LABEL: Record<string, string> = {
  idle: '대기', busy: '동작 중', error: '오류', estopped: '비상정지',
}

export default function StatusBar({
  robot, connected, warning,
}: {
  robot: RobotState
  connected: boolean
  warning: SafetyEvent | null
}) {
  return (
    <div className="statusbar">
      <span className={`mode mode-${robot.mode}`}>
        <span className="dot" />
        {MODE_LABEL[robot.mode] ?? robot.mode}
      </span>
      <span className="muted">
        {robot.current_skill && robot.current_skill !== 'none'
          ? `현재 스킬: ${robot.current_skill}`
          : '현재 스킬 없음'}
      </span>
      {/* warning 등급은 배너로 키우지 않고 작게 상시 노출한다 (2.2.3) */}
      {warning && <span className="warn-chip">⚠ {warning.description}</span>}
      <span className="spacer" />
      <span className={connected ? 'muted' : 'hint-error'}>
        {connected ? '연결됨' : '연결 끊김 — 재연결 중'}
      </span>
    </div>
  )
}

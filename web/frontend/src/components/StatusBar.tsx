/** 상태바 (화면정의서.md 2.2.2) */
import type { RobotState, SafetyEvent } from '../types'

const MODE_LABEL: Record<string, string> = {
  idle: '대기', busy: '동작 중', error: '오류', estopped: '비상정지',
}

export default function StatusBar({
  robot, connected, warning, awaitingApproval,
}: {
  robot: RobotState
  connected: boolean
  warning: SafetyEvent | null
  awaitingApproval?: boolean
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
      {/* 로봇 mode와는 별개다 — 승인 대기는 로봇이 아니라 사람을 기다리는 상태다 */}
      {awaitingApproval && <span className="warn-chip">⏸ 승인 대기 중</span>}
      {/* warning 등급은 배너로 키우지 않고 작게 상시 노출한다 (2.2.3) */}
      {warning && <span className="warn-chip">⚠ {warning.description}</span>}
      {/* bringup 직후 TCP 오프셋이 풀렸는지 눈으로 바로 확인하기 위한 것 — get_current_tcp를
          CLI로 매번 찍어보던 걸 화면으로 옮겼다. 좌표가 아니라 **설정 자체**만 본다 —
          tcp_configured가 기대값(GripperDA_v1)과 일치하는지의 결과다(types.ts 참조). */}
      <span className={robot.tcp_configured ? 'muted' : 'hint-error'}
            title={robot.tcp_name ? `TCP: ${robot.tcp_name}` : 'get_current_tcp 응답 없음'}>
        {robot.tcp_configured ? 'TCP 설정됨' : 'TCP 설정 안 됨'}
      </span>
      <span className="spacer" />
      <span className={connected ? 'muted' : 'hint-error'}>
        {connected ? '연결됨' : '연결 끊김 — 재연결 중'}
      </span>
    </div>
  )
}

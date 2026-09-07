/** 안전 배너 (화면정의서.md 2.2.3) — severity=critical일 때만 노출 */
import type { SafetyEvent } from '../types'

export default function SafetyBanner({ event }: { event: SafetyEvent }) {
  return (
    <div className="safety-banner" role="alert">
      <span>⚠</span>
      <span>{event.description}</span>
      {event.requires_manual_reset && <span className="muted">수동 리셋 필요</span>}
    </div>
  )
}

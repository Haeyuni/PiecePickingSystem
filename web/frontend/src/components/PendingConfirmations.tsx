/** 신규 물체 확인 대기 목록.
 *
 * 온디맨드 전환 전에는 이 진입점이 "탐지 물체 목록"의 "확인 필요" 배지였다
 * (ObjectList.tsx, 화면정의서.md 2.2.5). 그 목록을 걷어내면서(2.2.4 — 화면에는 이제
 * 캡처 사진·마스크·파지 후보만 남긴다) 확인 흐름 자체가 막히면 안 되므로, 이미 별도로
 * 폴링하던 `GET /api/object-confirmations` 결과(`pending`)를 바로 여기서 보여준다 —
 * 물체 목록과 무관하게 항상 동작한다.
 */
import type { ObjectConfirmation } from '../types'

export default function PendingConfirmations({
  items, onClick,
}: {
  items: ObjectConfirmation[]
  onClick: (className: string) => void
}) {
  if (items.length === 0) {
    return <div className="empty">확인 대기 중인 신규 물체 없음</div>
  }

  return (
    <div className="objects">
      {items.map((item) => (
        <div className="object-row" key={item.class_name}>
          <span className="name">{item.suggested_name_ko || item.class_name}</span>
          <span
            className="badge badge-confirm"
            onClick={() => onClick(item.class_name)}
            role="button"
          >
            확인 필요
          </span>
        </div>
      ))}
    </div>
  )
}

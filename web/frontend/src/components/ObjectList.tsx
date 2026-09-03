/** 탐지 물체 목록 (화면정의서.md 2.2.5) */
import type { DetectedObject } from '../types'

const PROFILE_LABEL: Record<string, string> = {
  normal: '일반', fragile: '파손위험', deformable: '변형가능',
}

export default function ObjectList({
  objects, onConfirmClick,
}: {
  objects: DetectedObject[]
  onConfirmClick: (className: string) => void
}) {
  if (objects.length === 0) {
    return <div className="empty">탐지된 물체 없음</div>
  }

  return (
    <div className="objects">
      {objects.map((o) => (
        <div className="object-row" key={o.object_id}>
          <span className="name">{o.name_ko || o.class_name}</span>
          {!o.graspable && (
            <span className="badge badge-nograsp" title={o.not_graspable_reason}>파지불가</span>
          )}
          {o.needs_confirmation && (
            <span
              className="badge badge-confirm"
              onClick={() => onConfirmClick(o.class_name)}
              role="button"
            >
              확인 필요
            </span>
          )}
          <span className={`badge badge-${o.profile}`}>{PROFILE_LABEL[o.profile] ?? o.profile}</span>
        </div>
      ))}
    </div>
  )
}

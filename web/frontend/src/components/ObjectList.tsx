/** 인식된 물체 목록 (화면정의서.md 2.2.5, 2026-09-08 재도입).
 *
 * 온디맨드 전환(D-5)으로 한 번 걷어냈던 목록이다 — perception이 명령 사이엔 조용하니
 * 계속 들고 있을 이유가 없다는 게 그때 이유였는데, 사용자가 "지금 뭘 인식했는지 바로
 * 보고 싶다"고 다시 요청해 되살렸다. 최신 world_state의 스냅샷일 뿐이라 다음 관측 전까지는
 * 그대로 멈춰 있다(CameraViews의 "n초 전 관측"과 같은 신선도를 공유한다).
 *
 * "확인 필요" 배지는 PendingConfirmations와 같은 모달을 연다 — 물체 하나가 두 목록
 * 모두에 걸릴 수 있어(여기서도, 확인 대기 목록에서도) 어느 쪽을 눌러도 같은 동작이어야
 * 한다.
 */
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
    return <div className="empty">인식된 물체 없음</div>
  }

  return (
    <div className="objects">
      {objects.map((o) => (
        <div className="object-row" key={o.object_id}>
          <span className="name" title={o.reasoning || undefined}>{o.name_ko || o.class_name}</span>
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
          {o.reasoning && <div className="reasoning muted">{o.reasoning}</div>}
        </div>
      ))}
    </div>
  )
}

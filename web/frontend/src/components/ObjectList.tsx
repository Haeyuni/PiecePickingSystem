/** 인식된 물체 목록 (화면정의서.md 2.2.5, 2026-09-08 재도입).
 *
 * 온디맨드 전환(D-5)으로 한 번 걷어냈던 목록이다 — perception이 명령 사이엔 조용하니
 * 계속 들고 있을 이유가 없다는 게 그때 이유였는데, 사용자가 "지금 뭘 인식했는지 바로
 * 보고 싶다"고 다시 요청해 되살렸다. 최신 world_state의 스냅샷일 뿐이라 다음 관측 전까지는
 * 그대로 멈춰 있다(CameraViews의 "n초 전 관측"과 같은 신선도를 공유한다).
 *
 * `needs_confirmation`은 여전히 신규(미등록) 클래스를 표시하지만(fallback 강제,
 * NFR-03a), 클릭해서 확정하는 절차(FR-05b, `ConfirmModal.tsx`)는 2026-09-09
 * 제거되었다 — 확인 UI 자체는 있었지만, 그 대상을 채워 넣을 저장소(DB
 * `object_attributes`)에 값을 쓰는 코드(planner `vlm_client.py`)가 TODO 스텁으로
 * 끝까지 구현되지 않아 목록이 항상 비어 있었다. 그래서 여기서는 안내용 배지만
 * 보여주고 클릭 동작은 없다. (실행 승인 모달의 라벨 수정(`correct_label`)은 이것과
 * 무관한 별개 기능이라 그대로 남아 있다 — `ApprovalModal.tsx`.)
 */
import type { DetectedObject, GripLevel } from '../types'

const GRIP_LEVEL_LABEL: Record<GripLevel, string> = {
  1: '매우 강하게', 2: '강하게', 3: '보통', 4: '약하게', 5: '매우 약하게',
}

export default function ObjectList({ objects }: { objects: DetectedObject[] }) {
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
            <span className="badge badge-confirm" title="미등록 클래스 — fallback으로 조심스럽게 다룸">
              신규
            </span>
          )}
          <span className={`badge badge-g${o.grip_level}`}>
            {GRIP_LEVEL_LABEL[o.grip_level] ?? `g${o.grip_level}`}
          </span>
          {o.reasoning && <div className="reasoning muted">{o.reasoning}</div>}
        </div>
      ))}
    </div>
  )
}

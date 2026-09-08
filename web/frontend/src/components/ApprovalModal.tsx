/** 실행 전 승인 모달 (명령 1건당 1회).
 *
 * 검증기를 통과한 시퀀스도 여기서 사람이 승인해야 실행된다(orchestrator.py 모듈
 * docstring). 라벨이 틀렸으면 고쳐서 재계획을 요청할 수 있다 — 그러면 이 컴포넌트는
 * 닫히지 않고 다음 execution_approval_needed 이벤트로 내용만 갱신된다(ControlPage가
 * 이벤트를 그대로 다시 넘겨준다).
 */
import { useState } from 'react'
import { submitApproval } from '../api'
import type { ApiError } from '../api'
import type { ApprovalNeededEvent } from '../types'

const SKILL_LABEL: Record<string, string> = { pick: 'Pick', place_into: 'Place into' }
const PROFILE_LABEL: Record<string, string> = {
  normal: '일반', fragile: '파손위험', deformable: '변형가능',
}

export default function ApprovalModal({
  event, onResolved,
}: {
  event: ApprovalNeededEvent
  onResolved: () => void
}) {
  const [editingId, setEditingId] = useState<string | null>(null)
  const [draftClassName, setDraftClassName] = useState('')
  const [draftNameKo, setDraftNameKo] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const objectsById = new Map(event.objects.map((o) => [o.object_id, o]))

  const startEdit = (objectId: string) => {
    const obj = objectsById.get(objectId)
    setEditingId(objectId)
    setDraftClassName(obj?.class_name ?? '')
    setDraftNameKo(obj?.name_ko ?? '')
  }

  const approve = () => act({ action: 'approve' }, onResolved)
  const reject = () => act({ action: 'reject' }, onResolved)
  const correctLabel = () =>
    act(
      { action: 'correct_label', object_id: editingId!, class_name: draftClassName, name_ko: draftNameKo },
      () => setEditingId(null),
    )

  const act = async (body: Parameters<typeof submitApproval>[1], after: () => void) => {
    setBusy(true)
    setError(null)
    try {
      await submitApproval(event.trace_id, body)
      after()
    } catch (e) {
      setError((e as ApiError).message ?? '처리하지 못했습니다')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="modal-backdrop">
      <div className="modal">
        <h3>실행 승인</h3>

        {event.validation_status === 'rejected' ? (
          <div className="command-hint hint-error">
            검증 거부: {event.validation_reason ?? '사유 없음'}
          </div>
        ) : (
          <div className="muted" style={{ fontSize: 12 }}>
            아래 시퀀스를 실행하기 전에 확인하세요. 라벨이 틀렸으면 고친 뒤 재계획할 수
            있습니다.
          </div>
        )}

        <div className="approval-steps">
          {event.steps.map((step, i) => {
            const obj = objectsById.get(step.object_id)
            return (
              <div className="approval-step" key={i}>
                <div>
                  <strong>{SKILL_LABEL[step.skill] ?? step.skill}</strong>{' '}
                  {obj?.name_ko || step.object_id}
                  {step.bin_id && ` → ${step.bin_id}`}
                  {obj && (
                    <span className={`badge badge-${obj.profile}`} style={{ marginLeft: 6 }}>
                      {PROFILE_LABEL[obj.profile] ?? obj.profile}
                    </span>
                  )}
                </div>
                {obj?.reasoning && <div className="reasoning muted">{obj.reasoning}</div>}

                {editingId === step.object_id ? (
                  <div className="field">
                    <input type="text" value={draftClassName}
                           onChange={(e) => setDraftClassName(e.target.value)}
                           placeholder="class_name" disabled={busy} />
                    <input type="text" value={draftNameKo}
                           onChange={(e) => setDraftNameKo(e.target.value)}
                           placeholder="한국어 이름" disabled={busy} />
                    <button disabled={busy} onClick={correctLabel}>수정 후 재계획</button>
                    <button disabled={busy} onClick={() => setEditingId(null)}>취소</button>
                  </div>
                ) : (
                  <button disabled={busy} onClick={() => startEdit(step.object_id)}>
                    라벨 수정
                  </button>
                )}
              </div>
            )
          })}
        </div>

        {error && <div className="command-hint hint-error">{error}</div>}

        <div className="modal-actions">
          <button disabled={busy} onClick={reject}>거부</button>
          <button disabled={busy} onClick={approve}>{busy ? '처리 중…' : '승인'}</button>
        </div>
      </div>
    </div>
  )
}

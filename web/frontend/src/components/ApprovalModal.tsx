/** 실행 전 승인 모달 (명령 1건당 1회).
 *
 * 검증기를 통과한 시퀀스도 여기서 사람이 승인해야 실행된다(orchestrator.py 모듈
 * docstring). 라벨이 틀렸으면 고쳐서 재계획을 요청할 수 있다 — 그러면 이 컴포넌트는
 * 닫히지 않고 다음 execution_approval_needed 이벤트로 내용만 갱신된다(ControlPage가
 * 이벤트를 그대로 다시 넘겨준다).
 */
import { useEffect, useState } from 'react'
import { getBins, submitApproval } from '../api'
import type { ApiError } from '../api'
import type { ApprovalNeededEvent, BinOption, GripLevel } from '../types'

const SKILL_LABEL: Record<string, string> = { pick: 'Pick', place_into: 'Place into' }
const GRIP_LEVEL_LABEL: Record<GripLevel, string> = {
  1: '매우 강하게', 2: '강하게', 3: '보통', 4: '약하게', 5: '매우 약하게',
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
  // 목적지(bin) 수정 — 라벨 수정과 별개 편집 상태다. 둘을 동시에 열 이유가 없어
  // editingBinId만 따로 둔다(라벨 편집은 재계획을 부르고, 이건 그 자리에서 바로 바뀐다).
  const [editingBinId, setEditingBinId] = useState<string | null>(null)
  const [draftBinId, setDraftBinId] = useState('')
  const [binOptions, setBinOptions] = useState<BinOption[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const objectsById = new Map(event.objects.map((o) => [o.object_id, o]))

  useEffect(() => {
    // 목적지 목록은 명령마다 바뀌지 않으므로(bins.yaml, 캘리브레이션 때만 바뀐다) 모달이
    // 뜰 때 한 번만 받는다. 못 받아도 조용히 빈 목록으로 둔다 — "라벨 수정"은 그대로 동작
    // 해야 하고, 목적지 수정 버튼만 못 쓰게 하면 된다.
    getBins().then((res) => setBinOptions(res.bins)).catch(() => setBinOptions([]))
  }, [])

  const startEdit = (objectId: string) => {
    const obj = objectsById.get(objectId)
    setEditingId(objectId)
    setDraftClassName(obj?.class_name ?? '')
    setDraftNameKo(obj?.name_ko ?? '')
  }

  const startEditBin = (objectId: string, currentBinId: string) => {
    setEditingBinId(objectId)
    setDraftBinId(currentBinId)
  }

  const approve = () => act({ action: 'approve' }, onResolved)
  const reject = () => act({ action: 'reject' }, onResolved)
  const correctLabel = () =>
    act(
      { action: 'correct_label', object_id: editingId!, class_name: draftClassName, name_ko: draftNameKo },
      () => setEditingId(null),
    )
  const correctBin = () =>
    act(
      { action: 'correct_bin', object_id: editingBinId!, bin_id: draftBinId },
      () => setEditingBinId(null),
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
                  {step.bin_id && ` → ${step.bin_name_ko ?? step.bin_id}`}
                  {obj && (
                    <span className={`badge badge-g${obj.grip_level}`} style={{ marginLeft: 6 }}>
                      {GRIP_LEVEL_LABEL[obj.grip_level] ?? `g${obj.grip_level}`}
                    </span>
                  )}
                </div>

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

                {/* 목적지 수정은 place_into 스텝에만 있다 — pick 스텝은 bin_id가 없다. */}
                {step.skill === 'place_into' && (
                  editingBinId === step.object_id ? (
                    <div className="field">
                      <select value={draftBinId} disabled={busy}
                              onChange={(e) => setDraftBinId(e.target.value)}>
                        {binOptions.length === 0 && step.bin_id && (
                          <option value={step.bin_id}>{step.bin_id}</option>
                        )}
                        {binOptions.map((b) => (
                          <option key={b.bin_id} value={b.bin_id}>
                            {b.name_ko || b.bin_id}
                          </option>
                        ))}
                      </select>
                      <button disabled={busy || !draftBinId} onClick={correctBin}>
                        목적지 변경
                      </button>
                      <button disabled={busy} onClick={() => setEditingBinId(null)}>취소</button>
                    </div>
                  ) : (
                    <button disabled={busy || binOptions.length === 0}
                            onClick={() => startEditBin(step.object_id, step.bin_id ?? '')}>
                      목적지 수정
                    </button>
                  )
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

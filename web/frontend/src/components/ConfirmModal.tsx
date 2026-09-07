/** 신규 물체 확인 모달 (화면정의서.md 2.2.7, FR-05b)
 *
 * 별도 화면이 아니라 제어 화면 위 모달이다 — 맥락 전환 없이 바로 처리하기 위해서다.
 * 이건 실행 승인이 아니라 **속성 확정**이므로, 떠 있는 동안에도 명령 입력은 그대로 동작한다.
 */
import { useState } from 'react'
import { confirmObject } from '../api'
import type { ApiError } from '../api'
import type { ObjectConfirmation, Profile } from '../types'

export default function ConfirmModal({
  item, onClose, onConfirmed,
}: {
  item: ObjectConfirmation
  onClose: () => void
  onConfirmed: () => void
}) {
  const [nameKo, setNameKo] = useState(item.suggested_name_ko ?? '')
  const [massG, setMassG] = useState(item.suggested_mass_g ?? 0)
  const [fragile, setFragile] = useState(item.suggested_fragile ?? true)
  const [deformable, setDeformable] = useState(item.suggested_deformable ?? false)
  const [transparent, setTransparent] = useState(item.suggested_transparent ?? false)
  const [profile, setProfile] = useState<Profile>(item.suggested_profile ?? 'fragile')
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)

  const submit = async () => {
    setSaving(true)
    setError(null)
    try {
      await confirmObject(item.class_name, {
        name_ko: nameKo, mass_g: massG, fragile, deformable, transparent, profile,
      })
      onConfirmed()
    } catch (e) {
      setError((e as ApiError).message ?? '확정에 실패했습니다')
      setSaving(false)
    }
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h3>신규 물체 확인 — {item.class_name}</h3>
        <div className="muted" style={{ fontSize: 12 }}>
          {item.suggested_by_model
            ? `${item.suggested_by_model}의 제안입니다. 확인 전까지는 보수적 설정으로만 다뤄집니다.`
            : '확인 전까지는 보수적 설정으로만 다뤄집니다.'}
        </div>

        {/* 제안의 근거가 된 크롭 이미지. 정적 서빙 경로는 D-5와 함께 정한다 */}
        <div className="crop">
          {item.image_ref ? `크롭 이미지: ${item.image_ref}` : '이미지 없음'}
        </div>

        <div className="field">
          <label htmlFor="name-ko">이름</label>
          <input id="name-ko" type="text" value={nameKo} onChange={(e) => setNameKo(e.target.value)} />
        </div>
        <div className="field">
          <label htmlFor="mass">무게(g)</label>
          <input id="mass" type="number" value={massG}
                 onChange={(e) => setMassG(Number(e.target.value))} />
        </div>
        <div className="field">
          <label htmlFor="profile">프로파일</label>
          <select id="profile" value={profile}
                  onChange={(e) => setProfile(e.target.value as Profile)}>
            <option value="normal">일반</option>
            <option value="fragile">파손위험</option>
            <option value="deformable">변형가능</option>
          </select>
        </div>
        <div className="field">
          <label>속성</label>
          <label className="muted">
            <input type="checkbox" checked={fragile}
                   onChange={(e) => setFragile(e.target.checked)} /> 파손위험
          </label>
          <label className="muted">
            <input type="checkbox" checked={deformable}
                   onChange={(e) => setDeformable(e.target.checked)} /> 변형가능
          </label>
          <label className="muted">
            <input type="checkbox" checked={transparent}
                   onChange={(e) => setTransparent(e.target.checked)} /> 투명
          </label>
        </div>

        {error && <div className="command-hint hint-error">{error}</div>}

        <div className="modal-actions">
          <button onClick={onClose} disabled={saving}>취소</button>
          <button onClick={submit} disabled={saving}>
            {saving ? '저장 중…' : '확정'}
          </button>
        </div>
      </div>
    </div>
  )
}

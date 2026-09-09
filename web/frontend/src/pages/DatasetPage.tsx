/** 데이터셋 화면 (Roboflow 스타일 수집+브라우징, 화면정의서.md 확장 — 학습 트리거는 범위 밖)
 *
 * 이력 화면과 같은 원칙 — 정적 조회, WebSocket 없음. planner의 /internal/label-marks가
 * 명령 실행마다 남긴 dataset_items를 읽기만 한다. 실제 재학습 실행(train.py)은 아직
 * 스텁이라 이 화면에는 "학습 시작" 버튼이 없다.
 */
import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getDatasetItems } from '../api'
import type { DatasetItem } from '../types'

const SOURCE_LABEL: Record<string, string> = {
  yaml_seed: '등록 클래스', llm_suggested: 'VLM 추정', user_confirmed: '사용자 확인',
}

function formatTime(iso: string): string {
  const d = new Date(iso)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
}

export default function DatasetPage() {
  const [items, setItems] = useState<DatasetItem[]>([])
  const [loading, setLoading] = useState(true)
  const [className, setClassName] = useState('')
  const [since, setSince] = useState('')
  const [selected, setSelected] = useState<DatasetItem | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    const params: Record<string, string> = { limit: '200' }
    if (className) params.class_name = className
    if (since) params.since = new Date(since).toISOString()
    try {
      setItems((await getDatasetItems(params)).items)
    } catch {
      setItems([])
    } finally {
      setLoading(false)
    }
  }, [className, since])

  useEffect(() => { void load() }, [load])

  return (
    <div className="app">
      <div className="nav">
        <h1>데이터셋 ({items.length})</h1>
        <Link to="/">← 제어 화면</Link>
      </div>

      <div className="filters">
        <label className="muted" htmlFor="ds-class">클래스</label>
        <input
          id="ds-class" type="text" placeholder="예: toothpaste"
          value={className} onChange={(e) => setClassName(e.target.value)}
        />

        <label className="muted" htmlFor="ds-since">기간</label>
        <input id="ds-since" type="date" value={since} onChange={(e) => setSince(e.target.value)} />

        <button onClick={() => void load()}>새로고침</button>
      </div>

      <div className="dataset-grid">
        {items.map((item) => (
          <div className="dataset-card" key={item.item_id} onClick={() => setSelected(item)}>
            <img className="dataset-thumb" src={`/api/datasets/${item.item_id}/image`}
                 alt={item.name_ko ?? item.class_name ?? ''} loading="lazy" />
            <div className="dataset-card-body">
              <div>{item.name_ko || item.class_name || '(이름 없음)'}</div>
              <div className="muted" style={{ fontSize: 12 }}>{formatTime(item.captured_at)}</div>
              {item.attr_source && (
                <span className="badge badge-confirm">{SOURCE_LABEL[item.attr_source] ?? item.attr_source}</span>
              )}
            </div>
          </div>
        ))}
      </div>

      {!loading && items.length === 0 && <div className="empty">수집된 데이터가 없습니다</div>}
      {loading && <div className="empty">불러오는 중…</div>}

      {selected && (
        <div className="modal-backdrop" onClick={() => setSelected(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3>{selected.name_ko || selected.class_name}</h3>
            <img className="dataset-detail-image"
                 src={`/api/datasets/${selected.item_id}/image`} alt="" />
            <dl>
              <dt>class_name</dt><dd>{selected.class_name ?? '—'}</dd>
              <dt>trace_id</dt><dd>{selected.trace_id}</dd>
              <dt>출처</dt><dd>{SOURCE_LABEL[selected.attr_source ?? ''] ?? selected.attr_source ?? '—'}</dd>
              <dt>확신도</dt><dd>{selected.confidence?.toFixed(2) ?? '—'}</dd>
              <dt>수집 시각</dt><dd>{formatTime(selected.captured_at)}</dd>
            </dl>
            <div className="modal-actions">
              <button onClick={() => setSelected(null)}>닫기</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

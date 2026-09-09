/** 그랩(파지) 학습 데이터 섹션 — DatasetPage의 두 번째 탭.
 *
 * **새 수집 경로가 아니다.** pick을 실행할 때마다 execution_logs에 이미 쌓이고 있던
 * grasp_pose(실행한 자세)+result(성공/실패)를 그대로 보여준다(api.ts getGraspAttempts,
 * web/store.py query_grasp_attempts 참조) — GraspNet류 재학습에 필요한 최소 신호가
 * 이미 있었다는 뜻이라, 여기서는 노출·필터·내보내기만 한다.
 */
import { useCallback, useEffect, useState } from 'react'
import { getGraspAttempts, GRASP_ATTEMPTS_EXPORT_URL } from '../api'
import type { GraspAttempt } from '../types'

const STRATEGY_LABEL: Record<string, string> = {
  heuristic_pca: 'PCA(휴리스틱)', contact_graspnet: 'Contact-GraspNet',
  graspnet_baseline: 'GraspNet',
}

function formatTime(iso: string): string {
  const d = new Date(iso)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
}

export default function GraspAttemptsSection() {
  const [items, setItems] = useState<GraspAttempt[]>([])
  const [loading, setLoading] = useState(true)
  const [result, setResult] = useState<'' | 'success' | 'failure'>('')
  const [className, setClassName] = useState('')

  const load = useCallback(async () => {
    setLoading(true)
    const params: Record<string, string> = { limit: '200' }
    if (result) params.result = result
    if (className) params.class_name = className
    try {
      setItems((await getGraspAttempts(params)).items)
    } catch {
      setItems([])
    } finally {
      setLoading(false)
    }
  }, [result, className])

  useEffect(() => { void load() }, [load])

  const successCount = items.filter((i) => i.result === 'success').length

  const exportUrl = (() => {
    const params = new URLSearchParams()
    if (result) params.set('result', result)
    if (className) params.set('class_name', className)
    const qs = params.toString()
    return `${GRASP_ATTEMPTS_EXPORT_URL}${qs ? `?${qs}` : ''}`
  })()

  return (
    <div>
      <div className="filters">
        <label className="muted" htmlFor="ga-class">클래스</label>
        <input
          id="ga-class" type="text" placeholder="예: toothpaste"
          value={className} onChange={(e) => setClassName(e.target.value)}
        />

        <label className="muted" htmlFor="ga-result">결과</label>
        <select id="ga-result" value={result}
                onChange={(e) => setResult(e.target.value as '' | 'success' | 'failure')}>
          <option value="">전체</option>
          <option value="success">성공</option>
          <option value="failure">실패</option>
        </select>

        <button onClick={() => void load()}>새로고침</button>
        <a href={exportUrl}>JSON으로 내보내기 ↓</a>
      </div>

      {!loading && items.length > 0 && (
        <div className="muted" style={{ marginTop: 8, fontSize: 13 }}>
          {items.length}건 중 {successCount}건 성공
          ({((successCount / items.length) * 100).toFixed(0)}%)
        </div>
      )}

      <div className="dataset-grid">
        {items.map((item) => (
          <div className="dataset-card" key={item.log_id} style={{ cursor: 'default' }}>
            {item.trace_id && (
              <img
                className="dataset-thumb" src={`/api/datasets/${item.trace_id}/image`}
                alt={item.class_name ?? ''} loading="lazy"
                // 이 pick과 같은 관측(trace_id)의 이미지가 dataset_items에 이미 없을 수
                // 있다(재관측으로 덮였거나 애초에 저장 실패) — 깨진 아이콘 대신 그냥 숨긴다.
                onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = 'none' }}
              />
            )}
            <div className="dataset-card-body">
              <div>{item.class_name ?? '(알 수 없음)'}</div>
              <div className="muted" style={{ fontSize: 12 }}>
                {STRATEGY_LABEL[item.grasp_strategy ?? ''] ?? item.grasp_strategy ?? '?'}
                {item.grip_level_used != null ? ` · grip ${item.grip_level_used}` : ''}
              </div>
              <div className="muted" style={{ fontSize: 12 }}>{formatTime(item.executed_at)}</div>
              <span className={`badge ${item.result === 'success' ? 'badge-g3' : 'badge-nograsp'}`}>
                {item.result === 'success' ? '성공' : `실패 (${item.failure_reason})`}
              </span>
            </div>
          </div>
        ))}
      </div>

      {!loading && items.length === 0 && <div className="empty">파지 실행 이력이 없습니다</div>}
      {loading && <div className="empty">불러오는 중…</div>}
    </div>
  )
}

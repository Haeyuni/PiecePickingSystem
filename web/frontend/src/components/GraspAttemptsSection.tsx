/** 그랩(파지) 학습 데이터 섹션 — DatasetPage의 두 번째 탭.
 *
 * pick을 실행할 때마다 execution_logs에 쌓이는 grasp_pose(실행한 자세)+result(성공/실패)
 * +point_cloud_path(graspnet_baseline이면 원본 클라우드 파일 경로, 009_point_cloud_path)를
 * 그대로 보여준다(api.ts getGraspAttempts, web/store.py query_grasp_attempts 참조) —
 * 여기서는 노출·필터·내보내기만 한다. 자세+성공 여부만으로는 GraspNet fine-tuning이
 * 안 되고 그 판정의 근거였던 클라우드가 있어야 하므로, point_cloud_path 유무가 "이 시도를
 * 재학습 데이터로 쓸 수 있는지"를 가른다.
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
              {item.point_cloud_path && (
                // fine-tuning에 쓸 원본 클라우드가 파일로 남아 있다는 표시일 뿐 — 여기서
                // 내려받지는 않는다(무겁다, JSON 내보내기에도 경로만 실린다).
                <div className="muted" style={{ fontSize: 11 }} title={item.point_cloud_path}>
                  클라우드 저장됨
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {!loading && items.length === 0 && <div className="empty">파지 실행 이력이 없습니다</div>}
      {loading && <div className="empty">불러오는 중…</div>}
    </div>
  )
}

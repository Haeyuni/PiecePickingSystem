/** 이력 화면 (화면정의서.md 3절)
 *
 * 정적 조회. 제어 화면과 상태를 공유하지 않고 자체적으로 `/api/executions`를 호출한다(1절).
 * WebSocket을 붙이지 않는 것은 의도다 — "예전에 뭘 했는가"는 수명이 긴 정보라 실시간
 * 갱신이 필요 없고, 실시간 화면과 섞으면 둘 다 읽기 어려워진다.
 */
import { Fragment, useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getExecutions } from '../api'
import type { ExecutionLog } from '../types'

const SKILL_LABEL: Record<string, string> = {
  pick: 'Pick', place_into: 'Place into', stop: '정지', home: '홈 이동',
}

const FAILURE_LABEL: Record<string, string> = {
  none: '—',
  no_contact: '접촉 없음',
  grasp_failed: '파지 실패',
  place_failed: '배치 실패',
  unreachable: '도달 불가',
  collision_expected: '충돌 예상',
}

/** 상대 시간이 아니라 절대 시각으로 보여준다 — 로봇 동작은 순서와 정확한 시각이 중요하다(4절) */
function formatTime(iso: string): string {
  const d = new Date(iso)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
}

export default function HistoryPage() {
  const [items, setItems] = useState<ExecutionLog[]>([])
  const [loading, setLoading] = useState(true)
  const [expanded, setExpanded] = useState<string | null>(null)

  const [result, setResult] = useState('')
  const [skill, setSkill] = useState('')
  const [since, setSince] = useState('')

  const load = useCallback(async () => {
    setLoading(true)
    const params: Record<string, string> = { limit: '100' }
    if (result) params.result = result
    if (skill) params.skill = skill
    // date input은 날짜만 준다 — 그날 00:00부터로 해석한다
    if (since) params.since = new Date(since).toISOString()
    try {
      setItems((await getExecutions(params)).items)
    } catch {
      setItems([])
    } finally {
      setLoading(false)
    }
  }, [result, skill, since])

  useEffect(() => { void load() }, [load])

  return (
    <div className="app">
      <div className="nav">
        <h1>이력</h1>
        <Link to="/">← 제어 화면</Link>
      </div>

      <div className="filters">
        <label className="muted" htmlFor="since">기간</label>
        <input id="since" type="date" value={since} onChange={(e) => setSince(e.target.value)} />

        <label className="muted" htmlFor="result">결과</label>
        <select id="result" value={result} onChange={(e) => setResult(e.target.value)}>
          <option value="">전체</option>
          <option value="success">성공</option>
          <option value="failure">실패</option>
        </select>

        <label className="muted" htmlFor="skill">스킬</label>
        <select id="skill" value={skill} onChange={(e) => setSkill(e.target.value)}>
          <option value="">전체</option>
          <option value="pick">Pick</option>
          <option value="place_into">Place into</option>
          <option value="stop">정지</option>
          <option value="home">홈 이동</option>
        </select>

        <button onClick={() => void load()}>새로고침</button>
      </div>

      <table>
        <thead>
          <tr>
            <th style={{ width: 130 }}>시각</th>
            <th>스킬 / 대상</th>
            <th style={{ width: 80 }}>결과</th>
            <th style={{ width: 130 }}>파지 전략</th>
            <th style={{ width: 90 }}>시각검증</th>
          </tr>
        </thead>
        <tbody>
          {items.map((item) => (
            <Fragment key={item.log_id}>
              <tr onClick={() =>
                setExpanded(expanded === item.log_id ? null : item.log_id)}>
                <td>{formatTime(item.executed_at)}</td>
                <td>
                  {SKILL_LABEL[item.skill_name] ?? item.skill_name}
                  {/* stop/home은 대상 물체가 없다 — 스킬명만 보여준다(3.2절) */}
                  {item.object_id && <span className="muted"> — {item.object_id}</span>}
                  {item.bin_id && <span className="muted"> → {item.bin_id}</span>}
                </td>
                <td className={`result-${item.result}`}>
                  {item.result === 'success' ? '성공' : '실패'}
                </td>
                <td className="muted">{item.grasp_strategy ?? '—'}</td>
                <td className="muted">
                  {item.visual_verification_passed === null
                    ? '—'
                    : item.visual_verification_passed ? '통과' : '실패'}
                </td>
              </tr>
              {expanded === item.log_id && (
                <tr className="detail">
                  <td colSpan={5}>
                    <dl>
                      <dt>trace_id</dt><dd>{item.trace_id ?? '—'}</dd>
                      <dt>request_id</dt><dd>{item.request_id ?? '—'}</dd>
                      <dt>클래스</dt><dd>{item.class_name ?? '—'}</dd>
                      <dt>프로파일</dt><dd>{item.profile_used ?? '—'}</dd>
                      <dt>목적지</dt><dd>{item.bin_id ?? '—'}</dd>
                      <dt>실패 사유</dt><dd>{FAILURE_LABEL[item.failure_reason] ?? item.failure_reason}</dd>
                      <dt>재시도</dt><dd>{item.retry_count}회</dd>
                      <dt>소요 시간</dt>
                      <dd>{item.cycle_time_ms ? `${Math.round(item.cycle_time_ms)}ms` : '—'}</dd>
                    </dl>
                  </td>
                </tr>
              )}
            </Fragment>
          ))}
        </tbody>
      </table>

      {!loading && items.length === 0 && <div className="empty">이력 없음</div>}
      {loading && <div className="empty">불러오는 중…</div>}
    </div>
  )
}

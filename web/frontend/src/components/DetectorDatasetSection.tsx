/** 검출기(YOLO11-seg) 데이터셋 섹션 — DatasetPage의 첫 번째 탭 (Roboflow 스타일).
 *
 * planner의 /internal/label-marks가 명령 실행마다 남긴 dataset_items를 web이 이미지
 * (trace_id) 단위로 묶어 보여준다. 승인/거부·내보내기는 web이 planner로 그대로 넘긴다
 * (api.ts 참조). 실제 재학습 실행(train.py)은 아직 스텁이라 이 화면에는 "학습 시작"
 * 버튼이 없다 — 다음 단계.
 */
import type { MouseEvent } from 'react'
import { useCallback, useEffect, useState } from 'react'
import {
  bulkReviewDatasetItems, DATASET_EXPORT_URL, getDatasetAnnotations, getDatasetItems,
  reviewDatasetItem,
} from '../api'
import type { AnnotationPolygon, DatasetImage } from '../types'
import AnnotationOverlay from './AnnotationOverlay'

function formatTime(iso: string): string {
  const d = new Date(iso)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
}

function objectSummary(image: DatasetImage): string {
  const names = image.objects.map((o) => o.name_ko || o.class_name || '?')
  // 같은 클래스가 여럿이면 한 번만 보여주고 개수를 붙인다 — 카드가 너무 길어지지 않게.
  const counts = new Map<string, number>()
  for (const n of names) counts.set(n, (counts.get(n) ?? 0) + 1)
  return [...counts.entries()].map(([n, c]) => (c > 1 ? `${n} x${c}` : n)).join(', ')
}

export default function DetectorDatasetSection() {
  const [images, setImages] = useState<DatasetImage[]>([])
  const [loading, setLoading] = useState(true)
  const [className, setClassName] = useState('')
  const [since, setSince] = useState('')
  const [selected, setSelected] = useState<DatasetImage | null>(null)
  const [annotations, setAnnotations] = useState<AnnotationPolygon[] | null>(null)
  const [reviewing, setReviewing] = useState(false)
  const [reviewError, setReviewError] = useState<string | null>(null)

  // 전체선택 승인/거부 — 카드 체크박스로 고른 trace_id 집합. 지금 화면에 로드된 목록
  // 기준이다(필터가 바뀌면 load()가 목록을 새로 받아오고 아래 useEffect가 선택을 비운다 —
  // 필터링돼 안 보이는 이미지가 "선택된 채로" 남아있는 걸 막는다).
  const [checked, setChecked] = useState<Set<string>>(new Set())
  const [bulkBusy, setBulkBusy] = useState(false)
  const [bulkError, setBulkError] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    const params: Record<string, string> = { limit: '200' }
    if (className) params.class_name = className
    if (since) params.since = new Date(since).toISOString()
    try {
      setImages((await getDatasetItems(params)).items)
    } catch {
      setImages([])
    } finally {
      setLoading(false)
    }
  }, [className, since])

  useEffect(() => { void load() }, [load])
  useEffect(() => { setChecked(new Set()) }, [images])

  const toggleChecked = (traceId: string, e: MouseEvent) => {
    e.stopPropagation()
    setChecked((prev) => {
      const next = new Set(prev)
      if (next.has(traceId)) next.delete(traceId)
      else next.add(traceId)
      return next
    })
  }

  const allChecked = images.length > 0 && checked.size === images.length
  const toggleAll = () => {
    setChecked(allChecked ? new Set() : new Set(images.map((i) => i.trace_id)))
  }

  const bulkReview = async (approved: boolean) => {
    if (checked.size === 0) return
    setBulkBusy(true)
    setBulkError(null)
    try {
      await bulkReviewDatasetItems([...checked], approved)
      setChecked(new Set())
      void load()
    } catch (e) {
      setBulkError((e as { message?: string }).message ?? '처리하지 못했습니다')
    } finally {
      setBulkBusy(false)
    }
  }

  const openDetail = (image: DatasetImage) => {
    setSelected(image)
    setAnnotations(null)
    setReviewError(null)
    void getDatasetAnnotations(image.trace_id)
      .then((res) => setAnnotations(res.polygons))
      .catch(() => setAnnotations([]))
  }

  const review = async (approved: boolean) => {
    if (!selected) return
    setReviewing(true)
    setReviewError(null)
    try {
      await reviewDatasetItem(selected.trace_id, approved)
      setSelected(null)
      // 승인/거부 둘 다 목록에서 그 이미지가 빠지거나 상태가 바뀌므로 다시 불러온다 —
      // 거부는 행 자체가 지워지고, 승인은 배지가 바뀐다.
      void load()
    } catch (e) {
      setReviewError((e as { message?: string }).message ?? '처리하지 못했습니다')
    } finally {
      setReviewing(false)
    }
  }

  return (
    <div>
      <div className="filters">
        <label className="muted" htmlFor="ds-class">클래스</label>
        <input
          id="ds-class" type="text" placeholder="예: toothpaste"
          value={className} onChange={(e) => setClassName(e.target.value)}
        />

        <label className="muted" htmlFor="ds-since">기간</label>
        <input id="ds-since" type="date" value={since} onChange={(e) => setSince(e.target.value)} />

        <button onClick={() => void load()}>새로고침</button>
        <a href={DATASET_EXPORT_URL}>YOLO로 내보내기 ↓</a>
      </div>

      <div className="dataset-bulk-bar">
        <label className="dataset-select-all">
          <input type="checkbox" checked={allChecked} disabled={images.length === 0}
                 onChange={toggleAll} />
          전체선택
        </label>
        <span className="muted">{checked.size > 0 ? `${checked.size}개 선택됨` : ''}</span>
        <span className="spacer" />
        <button onClick={() => void bulkReview(false)} disabled={checked.size === 0 || bulkBusy}>
          선택 거부
        </button>
        <button onClick={() => void bulkReview(true)} disabled={checked.size === 0 || bulkBusy}>
          선택 승인
        </button>
      </div>
      {bulkError && <div className="command-hint hint-error">{bulkError}</div>}

      <div className="dataset-grid">
        {images.map((image) => (
          <div className="dataset-card" key={image.trace_id} onClick={() => openDetail(image)}>
            <div className="dataset-thumb-wrap">
              <img className="dataset-thumb" src={`/api/datasets/${image.trace_id}/image`}
                   alt={objectSummary(image)} loading="lazy" />
              <input
                type="checkbox" className="dataset-card-check"
                checked={checked.has(image.trace_id)}
                onClick={(e) => toggleChecked(image.trace_id, e)}
                onChange={() => {}}
              />
            </div>
            <div className="dataset-card-body">
              <div>{objectSummary(image) || '(물체 없음)'}</div>
              <div className="muted" style={{ fontSize: 12 }}>{formatTime(image.captured_at)}</div>
              <span className={`badge ${image.reviewed ? 'badge-g3' : 'badge-confirm'}`}>
                {image.reviewed ? '승인됨' : '검토 대기'}
              </span>
            </div>
          </div>
        ))}
      </div>

      {!loading && images.length === 0 && <div className="empty">수집된 데이터가 없습니다</div>}
      {loading && <div className="empty">불러오는 중…</div>}

      {selected && (
        <div className="modal-backdrop" onClick={() => setSelected(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3>{objectSummary(selected) || '(물체 없음)'}</h3>

            <div className="dataset-detail-image-wrap">
              <img className="dataset-detail-image"
                   src={`/api/datasets/${selected.trace_id}/image`} alt="" />
              {annotations && <AnnotationOverlay polygons={annotations} />}
            </div>
            {annotations === null && <div className="muted" style={{ fontSize: 12 }}>어노테이션 불러오는 중…</div>}
            {annotations !== null && annotations.length === 0 && (
              <div className="muted" style={{ fontSize: 12 }}>
                이 이미지는 어노테이션을 표시할 수 없습니다 (이전 형식으로 수집됨)
              </div>
            )}

            <dl>
              <dt>trace_id</dt><dd>{selected.trace_id}</dd>
              <dt>물체</dt>
              <dd>{selected.objects.map((o) => o.name_ko || o.class_name || '?').join(', ')}</dd>
              <dt>수집 시각</dt><dd>{formatTime(selected.captured_at)}</dd>
              <dt>상태</dt><dd>{selected.reviewed ? '승인됨' : '검토 대기'}</dd>
            </dl>

            {reviewError && <div className="command-hint hint-error">{reviewError}</div>}

            <div className="modal-actions">
              <button onClick={() => setSelected(null)} disabled={reviewing}>닫기</button>
              <button onClick={() => void review(false)} disabled={reviewing}>거부</button>
              <button onClick={() => void review(true)} disabled={reviewing || selected.reviewed}>
                {selected.reviewed ? '이미 승인됨' : '승인'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

/** 데이터셋 화면 — "검출기"(Roboflow 스타일 수집+큐레이션+내보내기)와 "그랩 데이터"
 * (execution_logs의 pick 결과 재사용) 두 탭을 가진 얇은 껍데기.
 *
 * 이력 화면과 같은 원칙 — 정적 조회, WebSocket 없음. 실제 재학습 실행(train.py)은 아직
 * 스텁이라 이 화면에는 "학습 시작" 버튼이 없다 — 다음 단계.
 */
import { useState } from 'react'
import { Link } from 'react-router-dom'
import DetectorDatasetSection from '../components/DetectorDatasetSection'
import GraspAttemptsSection from '../components/GraspAttemptsSection'

type Tab = 'detector' | 'grasp'

const TAB_LABEL: Record<Tab, string> = { detector: '검출기 데이터셋', grasp: '그랩 데이터' }

export default function DatasetPage() {
  const [tab, setTab] = useState<Tab>('detector')

  return (
    <div className="app">
      <div className="nav">
        <h1>학습 데이터</h1>
        <Link to="/">← 제어 화면</Link>
      </div>

      <div className="scenarios">
        {(['detector', 'grasp'] as const).map((t) => (
          <button
            key={t} type="button"
            className={tab === t ? 'scenario-active' : undefined}
            onClick={() => setTab(t)}
          >
            {TAB_LABEL[t]}
          </button>
        ))}
      </div>

      {tab === 'detector' ? <DetectorDatasetSection /> : <GraspAttemptsSection />}
    </div>
  )
}

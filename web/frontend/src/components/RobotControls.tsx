/** 로봇 직접 제어 버튼 (화면정의서.md 2.2.6, 웹_인터페이스_정의서.md 2.6절) */
import { useState } from 'react'
import { homeRobot, stopRobot } from '../api'
import type { ApiError } from '../api'
import type { RobotMode } from '../types'

export default function RobotControls({ mode }: { mode: RobotMode }) {
  const [message, setMessage] = useState<string | null>(null)

  const run = async (action: () => Promise<unknown>, label: string) => {
    setMessage(null)
    try {
      await action()
      setMessage(`${label} 요청됨`)
    } catch (e) {
      setMessage((e as ApiError).message ?? `${label} 실패`)
    }
  }

  return (
    <div>
      <div className="controls">
        {/* 동작 중 홈 이동은 위험하므로 idle·error일 때만 (error도 홈으로 보내 복구한다 —
            web/routers/robot.py의 /api/robot/home 참조. busy·estopped는 계속 막는다) */}
        <button
          className={mode === 'error' ? 'btn-recover' : undefined}
          disabled={mode !== 'idle' && mode !== 'error'}
          onClick={() => run(homeRobot, mode === 'error' ? '오류 복구(홈 이동)' : '홈 이동')}
        >
          {mode === 'error' ? '오류 복구 (홈으로 이동)' : '홈 이동'}
        </button>
        {/* 정지는 mode와 무관하게 항상 활성 — busy일 때 가장 필요한 동작이다 */}
        <button className="btn-stop" onClick={() => run(stopRobot, '정지')}>
          정지
        </button>
      </div>
      {message && <div className="command-hint muted">{message}</div>}
    </div>
  )
}

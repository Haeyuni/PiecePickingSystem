/** 제어 화면 (화면정의서.md 2절)
 *
 * 실시간, WebSocket 기반. 이력 화면과 상태를 공유하지 않는다(1절).
 *
 * WebSocket에는 재전송 큐가 없으므로(웹_인터페이스_정의서 4절), 이벤트로 부분 갱신하되
 * 재연결·최초 진입 시에는 스냅샷 API(`/api/traces/{id}`, `/api/world-state`)로 다시 받는다.
 */
import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getConfirmations, getRecentTraces, getTrace, getWorldState } from '../api'
import CameraViews from '../components/CameraViews'
import CommandInput from '../components/CommandInput'
import ConfirmModal from '../components/ConfirmModal'
import ObjectList from '../components/ObjectList'
import RobotControls from '../components/RobotControls'
import SafetyBanner from '../components/SafetyBanner'
import StatusBar from '../components/StatusBar'
import TaskProgress from '../components/TaskProgress'
import { useLive } from '../hooks/useLive'
import type { DetectedObject, LiveEvent, ObjectConfirmation, RobotState, SafetyEvent, Trace } from '../types'

const INITIAL_ROBOT: RobotState = { mode: 'idle', current_skill: 'none', gripper_width_mm: 0 }

export default function ControlPage() {
  const [robot, setRobot] = useState<RobotState>(INITIAL_ROBOT)
  const [objects, setObjects] = useState<DetectedObject[]>([])
  const [trace, setTrace] = useState<Trace | null>(null)
  const [critical, setCritical] = useState<SafetyEvent | null>(null)
  const [warning, setWarning] = useState<SafetyEvent | null>(null)
  const [pending, setPending] = useState<ObjectConfirmation[]>([])
  const [modalClass, setModalClass] = useState<string | null>(null)

  const refreshWorld = useCallback(async () => {
    try {
      setObjects((await getWorldState()).objects)
    } catch {
      setObjects([])
    }
  }, [])

  const refreshPending = useCallback(async () => {
    try {
      setPending((await getConfirmations()).items)
    } catch {
      /* 확인 목록은 없어도 화면이 동작해야 한다 */
    }
  }, [])

  const refreshTrace = useCallback(async (traceId: string) => {
    try {
      setTrace(await getTrace(traceId))
    } catch {
      /* 아직 스냅샷이 없을 수 있다 */
    }
  }, [])

  const onEvent = useCallback((event: LiveEvent) => {
    switch (event.type) {
      case 'robot_state':
        setRobot({
          mode: event.mode,
          current_skill: event.current_skill,
          gripper_width_mm: event.gripper_width_mm,
        })
        // 새 mode가 idle이면 해제 조건을 만족한 배너를 내린다 (2.2.3)
        if (event.mode === 'idle') setCritical(null)
        break

      case 'safety_event':
        // 안전 이벤트는 다른 이벤트보다 먼저 눈에 띄어야 한다 (4절)
        if (event.severity === 'critical') setCritical(event)
        else setWarning(event)
        break

      case 'execution_progress':
        setTrace((prev) =>
          prev && prev.trace_id === event.trace_id
            ? {
                ...prev,
                steps: prev.steps.map((s) =>
                  s.request_id === event.request_id
                    ? { ...s, phase: event.phase, status: 'in_progress' }
                    : s,
                ),
              }
            : prev,
        )
        // 계획이 방금 생겼을 수 있다(스텝 목록이 비어 있으면 스냅샷을 당겨온다)
        setTrace((prev) => {
          if (prev && prev.trace_id === event.trace_id && prev.steps.length === 0) {
            void refreshTrace(event.trace_id)
          }
          return prev
        })
        break

      case 'execution_result':
        void refreshTrace(event.trace_id)
        break

      case 'object_confirmation_needed':
        void refreshPending()
        void refreshWorld()
        break
    }
  }, [refreshPending, refreshTrace, refreshWorld])

  const { connected } = useLive(onEvent)

  // 새로고침으로 trace_id를 잃어도 진행 중인 작업이 화면에서 사라지지 않게 한다
  const adoptRecentTrace = useCallback(async () => {
    try {
      const { items } = await getRecentTraces(1)
      if (items.length > 0) await refreshTrace(items[0].trace_id)
    } catch {
      /* 최근 작업이 없으면 빈 상태로 시작한다 */
    }
  }, [refreshTrace])

  useEffect(() => {
    void refreshWorld()
    void refreshPending()
    void adoptRecentTrace()
  }, [refreshWorld, refreshPending, adoptRecentTrace])

  // 재연결 시 놓친 이벤트를 재생하는 대신 현재 상태를 다시 받는다 (4절)
  useEffect(() => {
    if (!connected) return
    void refreshWorld()
    if (trace) void refreshTrace(trace.trace_id)
    // trace를 의존성에 넣으면 스냅샷 갱신마다 재실행된다 — 연결 전환에만 반응한다
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connected])

  const onCommandAccepted = (traceId: string, commandText: string) => {
    setTrace({
      trace_id: traceId, command_text: commandText, validation_status: null,
      validation_reason: null, sequence_id: null, steps: [],
    })
    // 계획 생성에는 LLM 왕복이 걸린다 — 잠시 뒤 스냅샷을 당겨 스텝을 채운다
    window.setTimeout(() => void refreshTrace(traceId), 1500)
  }

  const modalItem = pending.find((p) => p.class_name === modalClass) ?? null

  return (
    <div className="app">
      <div className="nav">
        <h1>피스피킹 분류 시스템</h1>
        <Link to="/history">이력 보기 →</Link>
      </div>

      <StatusBar robot={robot} connected={connected} warning={warning} />
      {critical && <SafetyBanner event={critical} />}

      <div className="grid">
        <div className="panel">
          <h2>카메라</h2>
          <CameraViews />
        </div>

        <div className="panel">
          <h2>탐지 물체 ({objects.length})</h2>
          <ObjectList objects={objects} onConfirmClick={setModalClass} />
          <RobotControls mode={robot.mode} />
        </div>
      </div>

      <div className="panel" style={{ marginTop: 12 }}>
        <h2>작업 진행</h2>
        <TaskProgress trace={trace} />
      </div>

      <div className="panel" style={{ marginTop: 12 }}>
        <CommandInput mode={robot.mode} onAccepted={onCommandAccepted} />
      </div>

      {modalItem && (
        <ConfirmModal
          item={modalItem}
          onClose={() => setModalClass(null)}
          onConfirmed={() => {
            setModalClass(null)
            void refreshPending()
            void refreshWorld()
          }}
        />
      )}
    </div>
  )
}

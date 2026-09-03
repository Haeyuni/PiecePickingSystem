/** web 백엔드 호출 (웹_인터페이스_정의서.md 2절). */
import type { ExecutionLog, ObjectConfirmation, Trace, WorldState } from './types'

export interface ApiError {
  code: string
  message: string
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  })
  const body = await response.json().catch(() => ({}))
  if (!response.ok) {
    // 백엔드는 실패도 { error: { code, message } } 형식으로 준다(6절)
    const error: ApiError = body.error ?? { code: 'UNKNOWN', message: `HTTP ${response.status}` }
    throw error
  }
  return body as T
}

export function sendCommand(commandText: string) {
  return request<{ trace_id: string; status: string }>('/api/commands', {
    method: 'POST',
    body: JSON.stringify({ schema_version: '1.0.0', command_text: commandText }),
  })
}

export interface TraceSummary {
  trace_id: string
  command_text: string
  validation_status: 'approved' | 'rejected' | null
  running: boolean
}

/** 최근 trace 목록. 새로고침으로 trace_id를 잃었을 때 화면이 스스로 복구하는 데 쓴다. */
export function getRecentTraces(limit = 5) {
  return request<{ items: TraceSummary[] }>(`/api/traces?limit=${limit}`)
}

export function getTrace(traceId: string) {
  return request<Trace>(`/api/traces/${traceId}`)
}

export function getExecutions(params: Record<string, string> = {}) {
  const query = new URLSearchParams(params).toString()
  return request<{ items: ExecutionLog[] }>(`/api/executions${query ? `?${query}` : ''}`)
}

export function getConfirmations() {
  return request<{ items: ObjectConfirmation[] }>('/api/object-confirmations')
}

export function confirmObject(className: string, corrections?: Record<string, unknown>) {
  return request<{ class_name: string; is_confirmed: boolean }>(
    `/api/object-confirmations/${className}`,
    {
      method: 'POST',
      body: JSON.stringify({ schema_version: '1.0.0', approved: true, corrections }),
    },
  )
}

export function stopRobot() {
  return request<{ stopped: boolean; cancelled_request_id: string | null }>('/api/robot/stop', {
    method: 'POST',
  })
}

export function homeRobot() {
  return request<{ status: string }>('/api/robot/home', { method: 'POST' })
}

/** 개발용 상태 확인. 아직 /world_state 전용 엔드포인트가 없어 /health가 요약을 준다. */
export function getHealth() {
  return request<{
    status: string
    mock_mode: boolean
    robot_state: { mode: string; current_skill: string; gripper_width_mm: number }
    world_state_objects: number | null
  }>('/health')
}

export function getWorldState() {
  return request<WorldState>('/api/world-state')
}

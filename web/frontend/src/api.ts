/** web 백엔드 호출 (웹_인터페이스_정의서.md 2절). */
import type {
  AnnotationPolygon, ApprovalAction, DatasetImage, Domain, ExecutionLog, GraspAttempt, Trace,
  WorldState,
} from './types'

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

export function sendCommand(commandText: string, domain: Domain = 'general') {
  return request<{ trace_id: string; status: string }>('/api/commands', {
    method: 'POST',
    body: JSON.stringify({
      schema_version: '1.0.0',
      command_text: commandText,
      domain,
    }),
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

export function stopRobot() {
  return request<{ stopped: boolean; cancelled_request_id: string | null }>('/api/robot/stop', {
    method: 'POST',
  })
}

export function homeRobot() {
  return request<{ status: string }>('/api/robot/home', { method: 'POST' })
}

export function homeAndOpenGripper() {
  return request<{ status: string }>('/api/robot/home?open_gripper=true', { method: 'POST' })
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

export function submitApproval(traceId: string, body: ApprovalAction) {
  return request<{ trace_id: string; accepted: boolean }>(
    `/api/executions/${traceId}/approval`,
    { method: 'POST', body: JSON.stringify({ schema_version: '1.0.0', ...body }) },
  )
}

export function getDatasetItems(params: Record<string, string> = {}) {
  const query = new URLSearchParams(params).toString()
  return request<{ items: DatasetImage[] }>(`/api/datasets${query ? `?${query}` : ''}`)
}

export function getDatasetAnnotations(traceId: string) {
  return request<{ polygons: AnnotationPolygon[] }>(`/api/datasets/${traceId}/annotations`)
}

export function reviewDatasetItem(traceId: string, approved: boolean) {
  return request<{ trace_id: string; approved: boolean }>(`/api/datasets/${traceId}/review`, {
    method: 'POST',
    body: JSON.stringify({ approved }),
  })
}

export function bulkReviewDatasetItems(traceIds: string[], approved: boolean) {
  return request<{ trace_ids: string[]; approved: boolean; rows_affected: number }>(
    '/api/datasets/bulk-review',
    { method: 'POST', body: JSON.stringify({ trace_ids: traceIds, approved }) },
  )
}

/** `<a href>`에 그대로 거는 URL — 클릭하면 브라우저가 zip을 네이티브로 받는다. */
export const DATASET_EXPORT_URL = '/api/datasets/export'

export function getGraspAttempts(params: Record<string, string> = {}) {
  const query = new URLSearchParams(params).toString()
  return request<{ items: GraspAttempt[] }>(`/api/grasp-attempts${query ? `?${query}` : ''}`)
}

/** `<a href>`에 그대로 거는 URL — 필터 쿼리스트링을 그대로 붙여 쓴다. */
export const GRASP_ATTEMPTS_EXPORT_URL = '/api/grasp-attempts/export'

export interface SttResult {
  recognized_text: string
  confidence: number
  low_confidence: boolean
}

/** multipart라 request()의 JSON Content-Type을 쓰지 않는다 — 브라우저가 boundary를
 * 직접 채워야 하므로 Content-Type 헤더를 아예 지정하지 않는다. */
export async function transcribeAudio(audio: Blob): Promise<SttResult> {
  const form = new FormData()
  form.append('audio', audio, 'command.webm')
  const response = await fetch('/api/stt', { method: 'POST', body: form })
  const body = await response.json().catch(() => ({}))
  if (!response.ok) {
    const error: ApiError = body.error ?? { code: 'UNKNOWN', message: `HTTP ${response.status}` }
    throw error
  }
  return body as SttResult
}

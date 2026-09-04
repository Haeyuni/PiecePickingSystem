/** web 백엔드와 주고받는 타입 (웹_인터페이스_정의서.md 2·4절).
 *
 * 서버 응답의 필드명을 그대로 쓴다 — 프론트에서 이름을 바꾸면 문서와 대조가 어려워진다.
 */

export type RobotMode = 'idle' | 'busy' | 'error' | 'estopped'
export type Profile = 'normal' | 'fragile' | 'deformable'

export interface RobotState {
  mode: RobotMode
  current_skill: string
  gripper_width_mm: number
}

export interface DetectedObject {
  object_id: string
  class_name: string
  name_ko: string
  profile: Profile
  graspable: boolean
  not_graspable_reason: string
  needs_confirmation: boolean
}

export interface WorldState {
  objects: DetectedObject[]
  needs_reobserve: string[]
}

export interface TraceStep {
  request_id: string
  skill: 'pick' | 'place_into'
  object_id: string
  bin_id: string | null
  status: 'pending' | 'in_progress' | 'success' | 'failure'
  phase: string | null
}

export interface Trace {
  trace_id: string
  command_text: string
  validation_status: 'approved' | 'rejected' | null
  validation_reason: string | null
  sequence_id: string | null
  steps: TraceStep[]
}

export interface SafetyEvent {
  event_type: string
  severity: 'warning' | 'critical'
  description: string
  requires_manual_reset: boolean
}

export interface ExecutionLog {
  log_id: string
  trace_id: string | null
  request_id: string | null
  object_id: string | null
  class_name: string | null
  skill_name: 'pick' | 'place_into' | 'stop' | 'home'
  profile_used: Profile | null
  bin_id: string | null
  grasp_strategy: string | null
  visual_verification_passed: boolean | null
  result: 'success' | 'failure'
  failure_reason: string
  retry_count: number
  cycle_time_ms: number | null
  executed_at: string
}

export interface ObjectConfirmation {
  class_name: string
  suggested_name_ko: string | null
  suggested_mass_g: number | null
  suggested_fragile: boolean | null
  suggested_deformable: boolean | null
  suggested_transparent: boolean | null
  suggested_profile: Profile | null
  suggested_by_model: string | null
  image_ref: string | null
  created_at: string
}

/** /ws/live 이벤트 (4절). 단일 채널이라 type으로 구분한다. */
export type LiveEvent =
  | ({ type: 'robot_state' } & RobotState)
  | { type: 'execution_progress'; trace_id: string; request_id: string; skill: string; phase: string }
  | { type: 'execution_result'; trace_id: string; request_id: string; success: boolean; failure_reason: string; validation_reason?: string }
  | ({ type: 'safety_event' } & SafetyEvent)
  | { type: 'object_confirmation_needed'; class_name: string }
  | ({ type: 'world_state' } & WorldState)

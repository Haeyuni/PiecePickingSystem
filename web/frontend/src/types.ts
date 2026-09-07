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

/** grasp 노드가 낸 파지 후보 하나 (sort_msgs/GraspCandidate.msg).
 *
 * Top-K가 그대로 내려온다(2026-09-07) — 화면에 숫자로 표시하지는 않지만, 후보가 몇 개
 * 도착했는지 확인하고 다음 단계(로봇 제약 기반 최종 선택)에서 쓰기 위해 타입을 둔다.
 * 실제 그리퍼 그림은 서버가 /grasp/debug_image에 그려서 내려보낸다(CameraViews).
 */
export interface GraspCandidate {
  candidate_id: string
  pose: {
    position: { x: number; y: number; z: number }
    orientation: { x: number; y: number; z: number; w: number }
  }
  score: number
  strategy: string
  gripper_width_mm: number
  grasp_depth_mm: number
}

export interface DetectedObject {
  object_id: string
  class_name: string
  name_ko: string
  profile: Profile
  graspable: boolean
  not_graspable_reason: string
  needs_confirmation: boolean
  /** 점수 내림차순. 비어 있으면 파지 후보가 없다는 뜻이다. */
  grasp_candidates?: GraspCandidate[]
}

export interface WorldState {
  objects: DetectedObject[]
  needs_reobserve: string[]
  // 온디맨드 전환(D-5) 이후 화면이 실제로 쓰는 것은 이 값뿐이다 — 탐지 물체 목록은
  // 더 이상 렌더링하지 않고, "마지막 관측이 언제였나"만 관측 패널에 보여준다.
  stamp: { sec: number; nanosec: number }
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

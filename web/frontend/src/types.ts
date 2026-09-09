/** web 백엔드와 주고받는 타입 (웹_인터페이스_정의서.md 2·4절).
 *
 * 서버 응답의 필드명을 그대로 쓴다 — 프론트에서 이름을 바꾸면 문서와 대조가 어려워진다.
 */

export type RobotMode = 'idle' | 'busy' | 'error' | 'estopped'
/** 시나리오 도메인. 어떤 명령도 기본값은 general — 도메인별 VLM 프롬프트를 쓰는
 *  시나리오 버튼(가정/약국/재활용)만 명시적인 domain을 싣는다. 가정은 special 규칙이
 *  없으므로 general과 같다. */
export type Domain = 'general' | 'pharmacy' | 'recycle'
/** 파지력 5단계: 1=가장 강하게(40N) ~ 5=가장 약하게(20N), 5N 간격. */
export type GripLevel = 1 | 2 | 3 | 4 | 5

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
grip_level: GripLevel
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
  // 작업 진행 화면에 물체 이름으로 보여주기 위한 필드 — 그 스텝의 계획 입력이 된
  // 관측(world_state)에서 가져온다. 재계획으로 물체가 사라졌으면 null일 수 있다.
  class_name: string | null
  name_ko: string | null
  bin_id: string | null
  bin_name_ko: string | null
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
  grip_level_used: GripLevel | null
  bin_id: string | null
  grasp_strategy: string | null
  visual_verification_passed: boolean | null
  result: 'success' | 'failure'
  failure_reason: string
  retry_count: number
  cycle_time_ms: number | null
  executed_at: string
}

/** 데이터셋 화면(Roboflow 스타일) 항목 하나 — dataset_items 한 행. */
export interface DatasetItem {
  item_id: string
  trace_id: string
  captured_at: string
  image_path: string
  label_path: string
  class_name: string | null
  name_ko: string | null
  attr_source: 'yaml_seed' | 'llm_suggested' | 'user_confirmed' | null
  confidence: number | null
  reviewed: boolean
}

/** 실행 전 승인 대기(명령 1건당 1회). 검증을 통과해도 이 이벤트를 받은 뒤에야 로봇이
 * 움직인다 — approve/reject/correct_label 중 하나로 응답해야 한다. */
export interface ApprovalNeededEvent {
  type: 'execution_approval_needed'
  trace_id: string
  sequence_id: string | null
  validation_status: 'approved' | 'rejected' | null
  validation_reason: string | null
  steps: TraceStep[]
  objects: DetectedObject[]
}

export type ApprovalAction =
  | { action: 'approve' }
  | { action: 'reject' }
  | { action: 'correct_label'; object_id: string; class_name?: string; name_ko?: string }

/** /ws/live 이벤트 (4절). 단일 채널이라 type으로 구분한다. */
export type LiveEvent =
  | ({ type: 'robot_state' } & RobotState)
  | { type: 'execution_progress'; trace_id: string; request_id: string; skill: string; phase: string }
  | { type: 'execution_result'; trace_id: string; request_id: string; success: boolean; failure_reason: string; validation_reason?: string }
  | ({ type: 'safety_event' } & SafetyEvent)
  | ({ type: 'world_state' } & WorldState)
  | ApprovalNeededEvent
  // 로봇 PC 마이크로 "hello rokey"를 듣는 네이티브 프로세스(tools/voice/wakeword_bridge.py)가
  // 감지 시 web에 알리면 그대로 중계된다 — 브라우저 자체 SpeechRecognition은 쓰지 않는다
  // (2026-09-09, 신뢰도 문제로 대체).
  | { type: 'wake_word_detected' }

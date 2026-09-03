-- 001_init: 초기 스키마 (시스템명세서.md 2.2절 ERD)
--
-- 재실행 가능하게 작성한다(IF NOT EXISTS) — 컨테이너 재기동마다 돌아도 안전해야 한다.
-- 적용 이력은 schema_migrations에 남긴다.
--
-- 문자열 상수는 ROS2 메시지 정의(sort_msgs)와 **같은 값**만 허용한다. 메시지 상수와
-- DB 기록값의 표기가 어긋나면 조인 쿼리가 조용히 깨지므로 CHECK로 강제한다
-- (시스템명세서 2.3절).

CREATE TABLE IF NOT EXISTS schema_migrations (
    version     text PRIMARY KEY,
    applied_at  timestamptz NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- 물체 속성 마스터. seed는 perception/config/objects.yaml (시스템명세서 2.1절)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS object_attributes (
    class_name          text PRIMARY KEY,
    name_ko             text,
    mass_g              real,
    fragile             boolean NOT NULL DEFAULT true,
    deformable          boolean NOT NULL DEFAULT false,
    transparent         boolean NOT NULL DEFAULT false,
    -- DetectedObject.msg의 PROFILE_* 상수와 동일
    profile             text NOT NULL DEFAULT 'fragile'
                        CHECK (profile IN ('normal', 'fragile', 'deformable')),
    is_confirmed        boolean NOT NULL DEFAULT false,
    -- DetectedObject.msg의 SOURCE_* 상수와 동일
    source              text NOT NULL
                        CHECK (source IN ('yaml_seed', 'llm_suggested', 'user_confirmed')),
    suggested_by_model  text,
    image_ref           text,
    created_at          timestamptz NOT NULL DEFAULT now(),
    updated_at          timestamptz NOT NULL DEFAULT now()
);

-- 확인 대기 목록 조회(GET /api/object-confirmations)가 자주 도는 경로
CREATE INDEX IF NOT EXISTS idx_object_attributes_pending
    ON object_attributes (is_confirmed) WHERE is_confirmed = false;

-- ---------------------------------------------------------------------------
-- LLM이 생성한 시퀀스와 검증 결과 이력
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS task_sequences (
    sequence_id           uuid PRIMARY KEY,
    trace_id              text NOT NULL,
    raw_command           text NOT NULL,
    world_state_snapshot  jsonb NOT NULL,
    generated_sequence    jsonb,
    validation_status     text NOT NULL
                          CHECK (validation_status IN ('approved', 'rejected')),
    validation_reason     text,
    llm_model_version     text NOT NULL,
    prompt_version        text NOT NULL,
    created_at            timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_task_sequences_trace ON task_sequences (trace_id);

-- ---------------------------------------------------------------------------
-- 실험 설정 태깅 (NFR-09 재현성)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS experiment_runs (
    experiment_tag       text PRIMARY KEY,
    stage                text,
    detection_mode       text,
    -- GraspCandidate.msg의 STRATEGY_* 상수와 동일
    grasp_strategy       text
                         CHECK (grasp_strategy IN ('heuristic_pca',
                                                   'contact_graspnet',
                                                   'graspnet_baseline')),
    torque_retry_enabled boolean,
    model_weights_path   text,
    config_snapshot      text,
    started_at           timestamptz NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- 스킬 단위 실행 결과
--
-- object_id는 런타임 인스턴스 식별자(obj_003)라 object_attributes(class_name)를
-- 참조할 수 없다. 속성 조인을 위해 class_name을 별도 컬럼으로 둔다
-- (시스템명세서 2.2절 ERD의 "object_id 참조" 표기를 이 구조로 확정).
--
-- stop/home은 trace_id·sequence_id 없이 기록된다(웹_인터페이스_정의서 2.6절).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS execution_logs (
    log_id                     uuid PRIMARY KEY,
    sequence_id                uuid REFERENCES task_sequences (sequence_id),
    trace_id                   text,
    request_id                 text,
    object_id                  text,
    class_name                 text REFERENCES object_attributes (class_name),
    -- 스킬셋: pick/place_into(FR-14) + 로봇 직접 제어(웹 인터페이스 2.6절)
    skill_name                 text NOT NULL
                               CHECK (skill_name IN ('pick', 'place_into', 'stop', 'home')),
    profile_used               text CHECK (profile_used IN ('normal', 'fragile', 'deformable')),
    bin_id                     text,
    grasp_strategy             text
                               CHECK (grasp_strategy IN ('heuristic_pca',
                                                         'contact_graspnet',
                                                         'graspnet_baseline')),
    grasp_pose                 jsonb,
    torque_trace               jsonb,
    visual_verification_passed boolean,
    result                     text NOT NULL
                               CHECK (result IN ('success', 'failure')),
    -- Pick/PlaceInto.action의 REASON_* 상수 합집합
    failure_reason             text NOT NULL DEFAULT 'none'
                               CHECK (failure_reason IN ('none', 'no_contact', 'grasp_failed',
                                                         'place_failed', 'unreachable',
                                                         'collision_expected')),
    retry_count                integer NOT NULL DEFAULT 0,
    cycle_time_ms              real,
    experiment_tag             text REFERENCES experiment_runs (experiment_tag),
    executed_at                timestamptz NOT NULL DEFAULT now()
);

-- 이력 화면(GET /api/executions)은 최신순 조회가 기본
CREATE INDEX IF NOT EXISTS idx_execution_logs_executed_at ON execution_logs (executed_at DESC);
CREATE INDEX IF NOT EXISTS idx_execution_logs_trace ON execution_logs (trace_id);

-- ---------------------------------------------------------------------------
-- hand-eye calibration 수집 포즈 (0단계)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS calibration_samples (
    sample_id             uuid PRIMARY KEY,
    joint_angles          jsonb,
    marker_pose_2d        jsonb,
    marker_pose_3d        jsonb,
    reprojection_error_mm real,
    calibration_run_id    text,
    captured_at           timestamptz NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- LLM 평가셋 채점 이력 (4단계 지표)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id            uuid PRIMARY KEY,
    prompt_version    text NOT NULL,
    llm_model_version text NOT NULL,
    dataset_version   text NOT NULL,
    run_at            timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS eval_results (
    result_id     uuid PRIMARY KEY,
    run_id        uuid NOT NULL REFERENCES eval_runs (run_id) ON DELETE CASCADE,
    test_case_id  text NOT NULL,
    category      text,
    expected_pass boolean,
    actual_pass   boolean,
    model_output  jsonb,
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_eval_results_run ON eval_results (run_id);

INSERT INTO schema_migrations (version) VALUES ('001_init')
ON CONFLICT (version) DO NOTHING;

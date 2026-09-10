-- 010_failure_reason_step2: STEP 2에서 추가된 failure_reason 값을 check constraint에 반영한다.
--
-- PlaceInto.action에 REASON_SAFE_TRANSIT_UNREACHABLE(safe_transit_unreachable),
-- Pick.action에 REASON_NO_FEASIBLE_GRASP(no_feasible_grasp)가 추가됐는데
-- execution_logs.failure_reason의 001_init.sql CHECK 제약이 갱신되지 않아,
-- 두 실패를 기록하려는 INSERT가 CheckViolation으로 죽고 있었다(2026-09-10 실물 테스트에서 발견).

ALTER TABLE execution_logs DROP CONSTRAINT IF EXISTS execution_logs_failure_reason_check;

ALTER TABLE execution_logs ADD CONSTRAINT execution_logs_failure_reason_check
    CHECK (failure_reason IN ('none', 'no_contact', 'grasp_failed',
                              'place_failed', 'unreachable',
                              'collision_expected',
                              'safe_transit_unreachable', 'no_feasible_grasp'));

INSERT INTO schema_migrations (version) VALUES ('010_failure_reason_step2')
ON CONFLICT (version) DO NOTHING;

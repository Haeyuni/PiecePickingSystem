-- 004_execution_logs_class_name_no_fk: 실행 이력을 물체 속성 마스터 수명과 분리한다.
--
-- class_name은 실행 당시의 분류 결과를 보존하는 비정규화 스냅샷이다. 신규 클래스가
-- object_attributes에 등록되기 전에도 로그가 기록되어야 하고, 향후 속성 테이블을
-- 제거해도 과거 execution_logs가 영향을 받지 않아야 한다.

ALTER TABLE execution_logs
    DROP CONSTRAINT IF EXISTS execution_logs_class_name_fkey;

INSERT INTO schema_migrations (version)
VALUES ('004_execution_logs_class_name_no_fk')
ON CONFLICT (version) DO NOTHING;

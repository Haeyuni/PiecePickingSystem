-- 010_drop_unused_experiment_calibration: 아무 코드도 쓰지 않는 두 테이블을 제거한다.
--
-- experiment_runs: INSERT/SELECT 하는 코드가 저장소 전체에 없다. execution_logs의
-- grasp_strategy 자체 컬럼(001_init)이 이미 있어 실험별 조건은 그쪽에서 바로 읽을 수
-- 있고, 이 테이블에 걸린 FK 컬럼(execution_logs.experiment_tag)도 어디서도 값을
-- 채우지 않아 항상 NULL이었다.
--
-- calibration_samples: hand-eye calibration을 실제로 수행하는
-- tools/calibration/handeye_calibration.py는 DB(psycopg)를 전혀 참조하지 않고
-- 결과를 tools/calibration/data/calibrate_data.json에 파일로만 남긴다.
--
-- object_attributes(006)와 같은 패턴 — 값이 채워진 적 없는 테이블은 스키마에 남겨둬도
-- 문서만 낡게 할 뿐이라 지운다.

ALTER TABLE execution_logs DROP COLUMN IF EXISTS experiment_tag;

DROP TABLE IF EXISTS experiment_runs;
DROP TABLE IF EXISTS calibration_samples;

INSERT INTO schema_migrations (version)
VALUES ('010_drop_unused_experiment_calibration')
ON CONFLICT (version) DO NOTHING;

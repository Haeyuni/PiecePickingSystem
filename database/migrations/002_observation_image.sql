-- 002_observation_image: execution_logs에 관측 이미지 경로 추가 (docs/on-demand-perception.md D-6)
--
-- 온디맨드 관측(D-1~D-8)이 실행되면 perception이 캡처 사진 + 마스크 + 파지 후보를
-- 합성한 이미지를 파일로 남긴다(/grasp/debug_image의 그 그림, ros_bridge.py 참조).
-- "왜 이걸 집었나"를 나중에 따지려면 그 순간의 그림이 필요하다.
--
-- 이미지 자체는 DB에 넣지 않는다 — 경로만 기록하고 파일은 data/observations/ 아래에
-- 둔다(용량이 크고, DB 백업/복제에 이미지 바이너리를 얹을 이유가 없다).
-- execution_logs 한 행이 스킬 실행 1건(pick 또는 place_into)이므로, 그 실행이 근거로
-- 삼은 관측 이미지를 같은 행에 붙인다 — task_sequences.world_state_snapshot과는
-- trace_id로 이어진다(그쪽은 계획 시점의 텍스트 스냅샷, 이쪽은 실행 시점의 그림).

ALTER TABLE execution_logs
    ADD COLUMN IF NOT EXISTS observation_image_path text;

INSERT INTO schema_migrations (version) VALUES ('002_observation_image')
ON CONFLICT (version) DO NOTHING;

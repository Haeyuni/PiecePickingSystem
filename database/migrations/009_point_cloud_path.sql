-- 009_point_cloud_path: execution_logs에 grasp 시도의 원본 포인트클라우드 경로를 남긴다.
--
-- GraspNet fine-tuning에는 grasp_pose(자세)+result(성공/실패)만으로는 부족하다 — 그
-- 판정의 근거였던 입력, 즉 GraspNet에 실제로 넣은 물체 단일 포인트클라우드(camera frame,
-- mm)가 있어야 한다. grasp 노드가 graspnet_baseline 전략일 때만 `.npz`로 저장하고
-- (data/point_clouds/<trace_id>/<observation_id>_<object_id>.npz), 그 경로가
-- GraspCandidate.point_cloud_path → world_state → planner → orchestrator를 거쳐
-- 여기 도착한다(observation_image_path와 같은 원칙 — DB엔 경로만, 실제 파일은 data/ 아래).
--
-- nullable이다 — heuristic_pca 실행이나 이 마이그레이션 이전 행에는 값이 없다.

ALTER TABLE execution_logs ADD COLUMN IF NOT EXISTS point_cloud_path text;

INSERT INTO schema_migrations (version) VALUES ('009_point_cloud_path')
ON CONFLICT (version) DO NOTHING;

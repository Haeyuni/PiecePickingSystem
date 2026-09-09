-- 007_dataset_local_class_index: 이미지별 YOLO 라벨 파일의 로컬 클래스 인덱스를 같이 남긴다.
--
-- planner의 _marks_to_yolo(app.py)는 클래스 인덱스를 **이미지 한 장 안에서 등장한 순서로만**
-- 매긴다 — 이미지마다 독립적이라, 이미지 A의 0번과 이미지 B의 0번이 다른 클래스일 수 있다.
-- 지금까지 dataset_items는 class_name(문자열)만 들고 이 로컬 인덱스를 안 남겼기 때문에
-- 여러 이미지의 라벨을 하나의 학습셋으로 합칠 때 클래스가 뒤섞이는 걸 막을 방법이 없었다.
-- 내보내기(export)가 이미지별 로컬 인덱스 → 전역 인덱스로 정확히 치환하려면 그 이미지
-- 안에서 이 class_name이 실제로 어떤 로컬 인덱스로 쓰였는지 알아야 한다 — _save_dataset_items가
-- _marks_to_yolo와 같은 계산을 공유해 이 값을 그대로 적어 넣는다(재구성이 아니라 원본).
--
-- nullable이다 — 이 마이그레이션 이전에 쌓인 행은 로컬 인덱스가 없어 NULL로 남고,
-- 내보내기는 이 값이 있는 행만 대상으로 삼는다(개수가 적어 재수집이 더 빠르다).

ALTER TABLE dataset_items ADD COLUMN IF NOT EXISTS local_class_index integer;

INSERT INTO schema_migrations (version) VALUES ('007_dataset_local_class_index')
ON CONFLICT (version) DO NOTHING;

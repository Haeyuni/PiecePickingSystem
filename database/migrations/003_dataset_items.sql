-- 003_dataset_items: 실제 명령 실행마다 나온 VLM 라벨링 결과를 데이터셋으로 남긴다
-- (Roboflow 스타일 수집+브라우징 화면의 재료, 2026-09-08).
--
-- 지금까지 VLM 결과(이미지+라벨 JSON)는 수동 테스트 스크립트(vlm_sam_test.py)에서만
-- data/samples/vlm_sam/에 쌓였고, 실제 /internal/label-marks 호출에서는 저장되지 않았다.
-- 이 표부터는 실행마다 쌓는다 — 향후 재학습용 데이터셋을 실물 운영에서 자연히 모으기
-- 위해서다. object_attributes/eval_runs와 같은 원칙으로, 이미지 자체는 DB에 넣지 않고
-- data/datasets/ 아래 경로만 기록한다.
--
-- 한 장(image_path/label_path)에 물체가 여럿이면 행도 여럿이다 — 물체별로 class_name·
-- 속성·판단 근거가 다르므로 갤러리에서 물체 단위로 걸러 볼 수 있어야 한다. is_new_class는
-- 없다 — SAM+VLM 경로가 objects.yaml 어휘를 더 이상 안 쓰면서 사라진 개념이다
-- (vlm_detect.py 상단 [속성도 VLM이 판단한다] 참조). 학습 트리거(train.py)는 이번 범위
-- 밖이다 — reviewed 플래그만 남겨 다음 단계가 큐레이션 대상을 고를 수 있게 해 둔다.

CREATE TABLE IF NOT EXISTS dataset_items (
    item_id      uuid PRIMARY KEY,
    trace_id     text NOT NULL,
    captured_at  timestamptz NOT NULL DEFAULT now(),
    image_path   text NOT NULL,
    label_path   text NOT NULL,
    class_name   text,
    name_ko      text,
    -- DetectedObject.msg의 SOURCE_* 상수와 동일 (object_attributes.source와 같은 어휘)
    attr_source  text CHECK (attr_source IN ('yaml_seed', 'llm_suggested', 'user_confirmed')),
    confidence   real,
    reasoning    text,
    reviewed     boolean NOT NULL DEFAULT false
);

CREATE INDEX IF NOT EXISTS idx_dataset_items_trace ON dataset_items (trace_id);
CREATE INDEX IF NOT EXISTS idx_dataset_items_captured ON dataset_items (captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_dataset_items_class ON dataset_items (class_name);

INSERT INTO schema_migrations (version) VALUES ('003_dataset_items')
ON CONFLICT (version) DO NOTHING;

-- 006_drop_reasoning: VLM의 "판단 근거" 한 줄(reasoning)을 완전히 없앤다.
--
-- 스키마 강제 출력에 이 필드가 있으면 VLM이 매 마크마다 답을 채워야 하는데, 그 자체는
-- 크지 않지만(한 문장), 같은 스키마의 mask_poly 필드가 output_tokens 상한을 반복해서
-- 다 채우는 사고를 일으킨 뒤(2026-09-09, vlm_detect.py 참조) 스키마에서 "안 쓰는데 굳이
-- 채우게 하는 필드"를 전부 걷어내기로 했다 — reasoning은 mask_poly와 달리 화면
-- (ApprovalModal/ObjectList/DatasetPage)에서 실제로 쓰이고 있었지만, 사용자 요청으로
-- 이 기능 자체를 없앤다.
--
-- dataset_items.reasoning은 DetectedObject.msg의 reasoning 필드, planner의
-- MarkLabel.reasoning과 짝을 이루던 컬럼이라 여기서 같이 뗀다.

ALTER TABLE dataset_items DROP COLUMN IF EXISTS reasoning;

INSERT INTO schema_migrations (version) VALUES ('006_drop_reasoning')
ON CONFLICT (version) DO NOTHING;

-- 006_drop_object_attributes: 물체 속성 마스터 테이블을 제거한다.
--
-- 이 테이블에 실제로 값이 들어간 적이 없었다 — insert는 seed.py가 기동 시 objects.yaml을
-- source='yaml_seed'/is_confirmed=true로 그대로 적재하는 것뿐이었고, "VLM이 신규
-- 클래스를 제안해 llm_suggested로 선기록"하는 planner의 vlm_client.py는 TODO 스텁으로
-- 끝까지 구현되지 않았다. 웹 쪽 확인 UI(ConfirmModal.tsx, PendingConfirmations.tsx,
-- GET/POST /api/object-confirmations)는 실제로 만들어졌지만, 위 이유로 확인 대기
-- 목록이 채워진 적이 없어 항상 빈 화면이었다.
--
-- 실행 승인 모달(ApprovalModal.tsx, POST /api/executions/{trace_id}/approval)의
-- 라벨 수정(correct_label)은 이 테이블과 무관하다 — 별개로 그대로 남는다.
--
-- detector=yolo 경로의 속성 조회(perception/attribute_db.py)는 이제 objects.yaml +
-- fallback만 본다 — DB가 없어도 동작한다는 원래 설계 그대로이므로 기능 손실은 없다.
-- detector=vlm_sam 경로는 원래부터 이 테이블을 보지 않았다.

DROP TABLE IF EXISTS object_attributes;

INSERT INTO schema_migrations (version)
VALUES ('006_drop_object_attributes')
ON CONFLICT (version) DO NOTHING;

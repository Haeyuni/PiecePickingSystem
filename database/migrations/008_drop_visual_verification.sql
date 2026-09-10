-- 008_drop_visual_verification: 시각검증 기능 자체를 없앤다 (사용자 요청).
--
-- visual_verification.py는 손목 카메라로 그리퍼 사이 물체 유무를 확인하려던 빈 스텁이었다
-- (TODO만 있고 구현 없음). 실물 경로는 visual_verification_passed를 항상 False로,
-- fake/mock 경로는 항상 True로 하드코딩해 왔을 뿐, 실제 pick 성공/실패 판정은 이미
-- RG2 'Grip detected' 비트(compliance.py/pick_server.py [PICK_VERIFY])로 하고 있었다 —
-- 이 컬럼은 판정에 쓰인 적이 없는 상시-고정값이었다.

ALTER TABLE execution_logs DROP COLUMN IF EXISTS visual_verification_passed;

INSERT INTO schema_migrations (version) VALUES ('008_drop_visual_verification')
ON CONFLICT (version) DO NOTHING;

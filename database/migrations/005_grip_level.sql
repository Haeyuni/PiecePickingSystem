-- 005_grip_level: 파지력 표현을 3단계 프로파일(normal/fragile/deformable)에서
-- 5단계 grip_level(1=가장 강하게 40N ~ 5=가장 약하게 20N, 5N 간격)로 교체한다.
--
-- 기존 프로파일은 새 단계로 이관한다 (시스템명세서.md 1.1절):
--   normal      → 3 (보통)
--   deformable  → 4 (약하게)
--   fragile     → 5 (매우 약하게, 가장 보수적)
--
-- ROS2 메시지(DetectedObject.msg GRIP_VERY_STRONG=1..GRIP_VERY_GENTLE=5,
-- Pick.action grip_level)와 **같은 값**만 허용한다 (001_init와 같은 원칙).

-- ---------------------------------------------------------------------------
-- object_attributes: profile(text) → grip_level(smallint 1..5)
-- ---------------------------------------------------------------------------
ALTER TABLE object_attributes
    ADD COLUMN IF NOT EXISTS grip_level smallint
        CHECK (grip_level BETWEEN 1 AND 5);

UPDATE object_attributes
SET grip_level = CASE profile
    WHEN 'normal'     THEN 3
    WHEN 'deformable' THEN 4
    WHEN 'fragile'    THEN 5
    ELSE 5
END
WHERE grip_level IS NULL;

ALTER TABLE object_attributes
    DROP COLUMN IF EXISTS profile;

-- 프로파일이 아니라 단계로 쿼리하므로 신규 물체도 1~5 값으로 들어가게 기본값을 둔다.
ALTER TABLE object_attributes
    ALTER COLUMN grip_level SET DEFAULT 5,
    ALTER COLUMN grip_level SET NOT NULL;

-- ---------------------------------------------------------------------------
-- execution_logs: profile_used(text) → grip_level_used(smallint 1..5)
-- ---------------------------------------------------------------------------
ALTER TABLE execution_logs
    ADD COLUMN IF NOT EXISTS grip_level_used smallint
        CHECK (grip_level_used BETWEEN 1 AND 5);

UPDATE execution_logs
SET grip_level_used = CASE profile_used
    WHEN 'normal'     THEN 3
    WHEN 'deformable' THEN 4
    WHEN 'fragile'    THEN 5
    ELSE 5
END
WHERE grip_level_used IS NULL;

ALTER TABLE execution_logs
    DROP COLUMN IF EXISTS profile_used;

INSERT INTO schema_migrations (version)
VALUES ('005_grip_level')
ON CONFLICT (version) DO NOTHING;
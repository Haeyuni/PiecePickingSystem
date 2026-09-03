"""VLM 기반 신규 물체 속성 제안 (FR-05a~FR-05c). 크롭 이미지 + 기존 클래스 목록 → 이름/재질/파손위험 1차 제안.

source='llm_suggested', is_confirmed=false로 object_attributes에 선기록.
동일 세션 내 이미 제안·확인된 클래스는 캐시된 값을 사용하고 재호출하지 않는다 (FR-05c).
"""

# TODO: 구현

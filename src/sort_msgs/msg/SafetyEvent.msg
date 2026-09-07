# control이 비상정지·충돌·제한 위반 발생 시 발행한다.
# 검증기(FR-12)의 게이트 조건으로 쓴다 — 활성 안전 이벤트가 있으면 planner가 새 시퀀스를 생성해도 검증기가 통과시키지 않는다.

string schema_version
builtin_interfaces/Time stamp

string TYPE_ESTOP=e_stop
string TYPE_COLLISION=collision
string TYPE_LIMIT_VIOLATION=limit_violation
string event_type

string SEVERITY_WARNING=warning
string SEVERITY_CRITICAL=critical
string severity

string description
bool requires_manual_reset
string trace_id                 # 진행 중이던 trace가 있었다면 그 ID, 없으면 빈 문자열
string aborted_request_id       # 진행 중이던 Pick/PlaceInto 요청이 있었다면 그 request_id, 없으면 빈 문자열

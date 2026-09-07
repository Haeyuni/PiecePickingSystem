# control이 주기적으로 발행하는 로봇 상태. 특정 명령·trace에 묶이지 않는 지속 상태라 trace_id는 없다.
# 구독자는 web이다 (planner가 아니다) — NFR-04(LLM 계획 단계는 실시간 제어 루프와 분리) 유지.

string schema_version
builtin_interfaces/Time stamp

string MODE_IDLE=idle
string MODE_BUSY=busy
string MODE_ERROR=error
string MODE_ESTOPPED=estopped
string mode

string current_skill              # pick/place_into/none — 대시보드 표시용 개략 정보
float32 gripper_width_mm
bool gripper_closed
sensor_msgs/JointState joint_state

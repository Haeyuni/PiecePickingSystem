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

# get_current_tcp가 돌려준 현재 설정된 TCP 이름 — 화면의 TCP 상태 표시용(화면정의서.md
# 2.2.2절). 좌표가 아니라 **설정 자체가 살아있는지**를 보려는 것이다: TCP(GripperDA_v1,
# 208mm)가 풀리면 이 이름이 비거나 다른 값이 되고, 그 상태로는 모든 좌표가 약 208mm씩
# 어긋난다(skill_params.yaml "info 가 비어 있으면 TCP 미선택 상태다" 주석 참조).
# 응답이 아예 없었으면(드라이버 미기동 등) 빈 문자열이다.
string tcp_name
# tcp_name이 robot_state_publisher.EXPECTED_TCP_NAME과 일치하는지 — 화면은 이 값만
# 보고 켜짐/꺼짐으로 표시한다.
bool tcp_configured

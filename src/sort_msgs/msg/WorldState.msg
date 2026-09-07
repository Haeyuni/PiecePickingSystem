# perception이 /perception/world_state_raw로 1차 발행할 때는 objects[].grasp_candidates가 빈 배열이고,
# grasp가 /world_state로 최종 발행할 때는 채워져 있다. planner/web은 /world_state만 구독하므로 항상 채워진 버전만 본다.

string schema_version
string trace_id
builtin_interfaces/Time stamp
string frame_id                # 보통 "base"

DetectedObject[] objects
string[] needs_reobserve       # graspable=false인 object_id 목록 (능동 재촬영 대상)

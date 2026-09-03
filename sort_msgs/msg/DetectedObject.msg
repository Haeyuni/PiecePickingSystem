# perception이 발행하는 물체 하나. WorldState에 중첩되므로 schema_version 없음.
# attributes는 중첩 dict를 지원하지 않는 ROS2 msg 특성상 flatten해서 필드로 둔다.

string object_id               # 예: obj_003
string class_name               # 예: glass_bottle
string name_ko                  # 예: 유리병
float32 confidence

geometry_msgs/Point position_base_m    # base 좌표계 3D 위치

float32 depth_valid_ratio      # 낮으면 투명/반사 의심 (8단계 확장에서 활용)
bool graspable
string not_graspable_reason    # graspable=false일 때만 값 있음, 아니면 빈 문자열

# --- 속성 (objects.yaml / object_attributes 테이블과 1:1 대응) ---
float32 mass_g
bool fragile
bool deformable
bool transparent

string PROFILE_NORMAL=normal
string PROFILE_FRAGILE=fragile
string PROFILE_DEFORMABLE=deformable
string profile

string SOURCE_YAML=yaml_seed
string SOURCE_LLM=llm_suggested
string SOURCE_USER=user_confirmed
string attr_source

bool needs_confirmation        # true면 신규 클래스, 사람 확인 대기 (5단계)

GraspCandidate[] grasp_candidates   # grasp 노드가 채워넣음. planner는 이미 채워진 것만 받음

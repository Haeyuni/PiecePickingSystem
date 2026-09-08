# perception이 발행하는 물체 하나. WorldState에 중첩되므로 schema_version 없음.
# attributes는 중첩 dict를 지원하지 않는 ROS2 msg 특성상 flatten해서 필드로 둔다.

string object_id               # 예: obj_003
string class_name               # 예: glass_bottle
string name_ko                  # 예: 유리병
float32 confidence

geometry_msgs/Point position_base_mm   # base 좌표계 3D 위치. 단위 mm (인터페이스_정의서 1.1절)

float32 depth_valid_ratio      # 낮으면 투명/반사 의심 (8단계 확장에서 활용)

# 물체가 자기 지지면(작업대 등) 위로 얼마나 솟아 있는지. depth로 실측한다 —
# position_base_mm.z(마스크 안쪽 median depth = 보이는 윗면)에서 마스크 바깥 링의
# 지지면 z를 뺀 값이다(perception/mask_utils.support_3d).
# 0이면 미상(링에 유효 depth가 없었거나 TCP 자세를 못 받은 경우).
#
# **쓰임**: place_into가 물체를 바구니에 놓을 z를 정할 때 쓴다. bin_pose는 빈 그리퍼로
# 티칭한 높이라, 큰 물체를 든 채 그 높이까지 내려가면 물체가 바닥에 먼저 닿아 로봇이
# 안전모드로 들어간다(2026-09-06 실물). 지지면 z = position_base_mm.z - height_mm 이므로
# 이 둘과 파지 z가 있으면 "물체가 TCP보다 얼마나 아래로 내려와 있는지"가 나온다.
float32 height_mm
# grasp가 object-only cloud의 XY convex hull로 채운다. base 좌표계 mm.
# 빈 polygon이면 footprint 미상이며 box boundary 검사를 활성화하지 않는다.
geometry_msgs/Polygon footprint_base_mm
bool graspable
string not_graspable_reason    # graspable=false일 때만 값 있음, 아니면 빈 문자열

# --- 속성 ---
# detector=yolo:    objects.yaml / object_attributes 테이블과 1:1 대응
# detector=vlm_sam: 사진을 본 VLM의 판단 (attr_source=llm_suggested, needs_confirmation=true)
#                   — docs/vlm_sam_pipeline.md의 [클래스 어휘를 주지 않는다]
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
string reasoning                # attr_source=llm_suggested일 때만 값 있음. VLM의 판단 근거 한 줄

GraspCandidate[] grasp_candidates   # grasp 노드가 채워넣음. planner는 이미 채워진 것만 받음

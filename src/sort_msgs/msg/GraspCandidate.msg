# grasp 노드가 계산한 개별 파지 후보 하나. DetectedObject에 중첩되므로 schema_version 없음.

geometry_msgs/Pose pose        # base 좌표계 기준 파지 자세; position은 프로젝트 규칙의 mm
float32 score                  # 신뢰도 0.0~1.0
float32 gripper_width_mm       # 전략이 예측한 필요 그리퍼 개폭(mm). RobotState.gripper_width_mm과
                                # 같은 단위·의미(실제 대신 목표값)다. 0이면 미상(폭을 못 낸 전략) —
                                # control은 그럴 때 기본 개폭(motion.gripper_open_m)으로 대체한다.

# 이 관측에서 이 후보를 가리키는 식별값. "<object_id>#<순위>" 형식이고 순위는 0이 1순위다.
# Top-K를 그대로 내보내므로(2026-09-07) 로그·웹·planner가 같은 후보를 가리킬 방법이 필요하다.
string candidate_id

# 전략이 "무는 지점까지 얼마나 파고드는지"로 본 값(mm). GraspNet은 예측값(depth_m x 1000)을,
# heuristic_pca는 grasp_depth_mm를 넣는다. 0이면 미상. 진단·비교용이고 control은 쓰지 않는다.
float32 grasp_depth_mm

string STRATEGY_HEURISTIC_PCA=heuristic_pca
string STRATEGY_CONTACT_GRASPNET=contact_graspnet
string STRATEGY_GRASPNET_BASELINE=graspnet_baseline
string strategy                # 어떤 방법으로 이 후보를 냈는지 (3단계 비교 실험용)

# 이 후보를 낸 관측에서, GraspNet에 실제로 넣은 물체 단일 포인트클라우드(camera frame, mm)
# 파일 경로(저장소 루트 기준 상대경로, 예: data/point_clouds/<trace_id>/<observation_id>_
# <object_id>.npz). graspnet_baseline만 채운다. 빈 문자열이면 저장 안 됐거나(heuristic_pca,
# 저장 실패) 미상 — fine-tuning 데이터로 쓸 때 grasp_pose+성공/실패만으로는 그 판정의
# 근거가 된 입력 자체가 없어 부족하다(execution_logs.point_cloud_path로 그대로 기록됨).
string point_cloud_path

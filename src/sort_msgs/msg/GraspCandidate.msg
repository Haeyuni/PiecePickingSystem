# grasp 노드가 계산한 개별 파지 후보 하나. DetectedObject에 중첩되므로 schema_version 없음.

geometry_msgs/Pose pose        # base 좌표계 기준 파지 자세; position은 프로젝트 규칙의 mm
float32 score                  # 신뢰도 0.0~1.0
float32 gripper_width_mm       # 전략이 예측한 필요 그리퍼 개폭(mm). RobotState.gripper_width_mm과
                                # 같은 단위·의미(실제 대신 목표값)다. 0이면 미상(폭을 못 낸 전략) —
                                # control은 그럴 때 기본 개폭(motion.gripper_open_m)으로 대체한다.

string STRATEGY_HEURISTIC_PCA=heuristic_pca
string STRATEGY_CONTACT_GRASPNET=contact_graspnet
string STRATEGY_GRASPNET_BASELINE=graspnet_baseline
string strategy                # 어떤 방법으로 이 후보를 냈는지 (3단계 비교 실험용)

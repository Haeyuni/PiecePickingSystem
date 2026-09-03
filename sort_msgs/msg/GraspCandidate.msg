# grasp 노드가 계산한 개별 파지 후보 하나. DetectedObject에 중첩되므로 schema_version 없음.

geometry_msgs/Pose pose        # base 좌표계 기준 파지 자세
float32 score                  # 신뢰도 0.0~1.0

string STRATEGY_HEURISTIC_PCA=heuristic_pca
string STRATEGY_CONTACT_GRASPNET=contact_graspnet
string STRATEGY_GRASPNET_BASELINE=graspnet_baseline
string strategy                # 어떤 방법으로 이 후보를 냈는지 (3단계 비교 실험용)

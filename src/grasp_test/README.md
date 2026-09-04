# Grasp Test

M0609, OnRobot RG2, RealSense에서 한 물체를 매 trial 새로 관측해 파지 모델을 비교하는 패키지입니다. 기존 WorldState, 웹, LLM, `control` 코드를 수정하지 않습니다.

## 준비

```text
~/grasp_test_assets/
  yolo/best.pt
  checkpoints/ggcnn/
  checkpoints/graspnet/
  checkpoints/contact_graspnet/
  grasp_test.local.yaml
```

`grasp_test.local.yaml`은 `config/grasp_test.example.yaml`을 복사해 모든 `/home/USER` 경로를 실제 절대 경로로 바꿉니다. `best.pt`는 반드시 사용자가 파인튜닝한 YOLO11-seg 가중치여야 합니다. 가중치와 checkpoint는 Git에 넣지 않습니다.

현재 저장소에는 실물 Move/IK action과 RG2 ROS 상태·명령 interface가 없습니다. 따라서 현재 상태는 **DRY_RUN_ONLY**이며 `execute:=true`도 로봇을 움직이지 않고 거부됩니다. 직접 Modbus 또는 관절 제어로 우회하지 않습니다.

## 사전 기동

기존 RealSense와 두산 드라이버는 별도로 실행해야 합니다. 이 패키지는 드라이버를 중복 기동하지 않습니다.

```bash
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
# 두산 M0609 드라이버와 검증된 RG2 ROS interface는 별도 터미널에서 기존 운영 절차로 기동
```

실행 전 `ros2 topic list`에서 `/camera/color/image_raw`, `/camera/aligned_depth_to_color/image_raw`, `/camera/color/camera_info`를 확인합니다. 실제 실행을 열려면 hand-eye 파일, TCP pose service, 검증된 Move/IK action, RG2 command/state topic, 작업영역·Home·approach/lift 티칭값이 모두 필요합니다.

## 실행

```bash
colcon build --merge-install
source install/setup.bash
ros2 launch grasp_test grasp_test.launch.py \
  config:=$HOME/grasp_test_assets/grasp_test.local.yaml \
  methods:=all rounds:=3 input_mode:=live reset_mode:=manual execute:=false
```

결과는 설정의 `results_dir`에 `live_physical_comparison.xlsx`, CSV, trial JSON, 모델별 로그로 저장됩니다. `best_score`는 모델별 스케일이 달라 비교하지 않으며, 실제 테스트가 가능해진 뒤 선택 기준은 성공률, 실패 유형, 전체 시간입니다.

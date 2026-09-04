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

`checkpoints/{ggcnn,graspnet,contact_graspnet}/`는 비워둬도 됩니다 — `ModelRunner`가 `assets.checkpoints` 아래에 세 폴더가 없으면 먼저 만들고(`model_runner.py`), 각 모델 컨테이너는 그 폴더가 비어 있으면 첫 실행 시 공개 checkpoint를 내려받아 채웁니다(호스트에 bind-mount되므로 이후 실행부터는 재사용됩니다). 필요한 것은 빈 폴더와 인터넷 연결뿐입니다. 다운로드가 실패하면 추측한 값으로 넘어가지 않고 해당 모델의 `<trial>_<method>.log`에 `ERROR:` 줄을 남기고 그 trial을 `ERROR`로 기록합니다. `yolo/best.pt`는 이 자동 다운로드 대상이 아니며 항상 직접 준비해야 합니다.

`robot_executor.py`는 로봇 PC에서 실제로 떠 있는 `/dsr01/motion/movej_h2r`·`/dsr01/motion/movel_h2r`(dsr_msgs2)와 `/onrobot/sendCommand`·`/onrobot/pose`(onrobot_rg_msgs)에 직접 붙습니다. `handeye_path`(hand-eye 캘리브레이션), 로봇 pose 서비스, 그리고 위 인터페이스가 모두 응답해야 `preflight()`가 통과하며, 하나라도 없으면 `execute:=true`도 `DRY_RUN_ONLY:<missing>`으로 거부됩니다. 직접 Modbus 또는 관절 제어로 우회하지 않습니다.

**참고(2026-09-04 확인):** `control_msgs/action/GripperCommand`(`/rg6_controller`)는 이 로봇에서 신뢰할 수 없는 것으로 확인됐습니다(같은 워크스페이스에서 `control` 패키지를 실물로 검증하며 발견 — `position`을 관절각으로 변환 없이 넘기고, 첫 목표 처리 시 예외로 죽음). `grasp_test`와 `control` 모두 이제 그 액션 대신 `/onrobot/sendCommand`+`/onrobot/pose`+`/onrobot_joint_states` 폴링을 씁니다.

## 사전 기동

기존 RealSense와 두산 드라이버는 별도로 실행해야 합니다. 이 패키지는 드라이버를 중복 기동하지 않습니다.

```bash
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
# 두산 M0609 드라이버와 검증된 RG2 ROS interface는 별도 터미널에서 기존 운영 절차로 기동
```

실행 전 `ros2 topic list`에서 `/camera/color/image_raw`, `/camera/aligned_depth_to_color/image_raw`, `/camera/color/camera_info`를, `ros2 action list`에서 `/dsr01/motion/movej_h2r`, `/dsr01/motion/movel_h2r`, `/rg6_controller`를 확인합니다. 실제 실행을 열려면 hand-eye 파일, TCP pose service, 위 세 액션, 그리고 `grasp_test.local.yaml`의 작업영역(workspace_mm)·Home(home_pose_deg)·approach_height_mm 값이 실제 셀에 맞게 채워져 있어야 합니다 — 이 값들은 로봇/작업대마다 다르므로 도구가 대신 정할 수 없습니다.

## 실행

```bash
colcon build --merge-install
source install/setup.bash
ros2 launch grasp_test grasp_test.launch.py \
  config:=$HOME/grasp_test_assets/grasp_test.local.yaml \
  methods:=all rounds:=3 input_mode:=live reset_mode:=manual execute:=false
```

결과는 설정의 `results_dir`에 `live_physical_comparison.xlsx`, CSV, trial JSON, 모델별 로그로 저장됩니다. `best_score`는 모델별 스케일이 달라 비교하지 않으며, 실제 테스트가 가능해진 뒤 선택 기준은 성공률, 실패 유형, 전체 시간입니다.

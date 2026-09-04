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

YOLO와 XLSX 결과 writer는 ROS 시스템 Python에 기본 포함되지 않습니다. launch 전에 워크스페이스 루트에서 실행 환경을 한 번 준비합니다.

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r src/grasp_test/requirements.txt
```

`checkpoints/{ggcnn,graspnet,contact_graspnet}/`는 비워둬도 됩니다 — `ModelRunner`가 `assets.checkpoints` 아래에 세 폴더가 없으면 먼저 만들고(`model_runner.py`), 각 모델 컨테이너는 그 폴더가 비어 있으면 첫 실행 시 공개 checkpoint를 내려받아 채웁니다(호스트에 bind-mount되므로 이후 실행부터는 재사용됩니다). 필요한 것은 빈 폴더와 인터넷 연결뿐입니다. 다운로드가 실패하면 추측한 값으로 넘어가지 않고 해당 모델의 `<trial>_<method>.log`에 `ERROR:` 줄을 남기고 그 trial을 `ERROR`로 기록합니다. `yolo/best.pt`는 이 자동 다운로드 대상이 아니며 항상 직접 준비해야 합니다.

`robot_executor.py`는 로봇 PC에서 실제로 떠 있는 `/dsr01/motion/movej_h2r`·`/dsr01/motion/movel_h2r`(dsr_msgs2)와 `/onrobot/sendCommand`·`/onrobot/pose`(onrobot_rg_msgs)에 직접 붙습니다. 명령 전 사전 IK·충돌 검증은 `motion_precheck.py`가 합니다 — MoveIt2가 아니라 Doosan `ikin` 서비스(도달 가능성)와 이 저장소가 직접 구현한 sphere-vs-AABB 스윕(등록된 table/box/jig 대비, `robot.obstacles`+`robot.gripper_radius_m`)으로 approach→grasp→lift 세 구간을 미리 확인합니다. 팔뚝/팔꿈치 같은 몸체 링크나 RG2+카메라 브래킷의 정확한 형상은 검사하지 않습니다(TCP 주변 구 하나로만 근사) — `motion_precheck.py` 모듈 docstring 참조. `robot.obstacles`가 비어 있거나 `robot.gripper_radius_m`이 없으면 `execute:=true`도 `DRY_RUN_ONLY:COLLISION_GEOMETRY_NOT_CONFIGURED`로 거부되고, IK/충돌 사전검증에서 실패하면 각각 `IK_FAILED`/`COLLISION_EXPECTED`로 기록되며 셋 다 통과해야만 실제 M0609/RG2 인터페이스를 부릅니다. 직접 Modbus 또는 관절 제어로 우회하지 않습니다.

**RG2 설정 필수:** vendor 드라이버는 action 이름을 `/rg6_controller`로 고정해 두지만, 이는 실제 그리퍼 모델명이 아닙니다. 드라이버의 기본 `gripper` 값은 `rg6`이므로, RG2 장비에서는 반드시 아래처럼 `gripper:=rg2`를 넘겨야 합니다. 이 설정이 틀리면 `/onrobot/pose`의 관절각→개폭 변환도 RG6 값으로 계산돼 실제 파지 시험을 하면 안 됩니다.

```bash
ros2 launch onrobot_rg_control bringup.launch.py ip:=192.168.1.1 port:=502 gripper:=rg2 control:=modbus
```

`grasp_test`와 `control`은 신뢰할 수 없는 `control_msgs/action/GripperCommand`(`/rg6_controller`) 대신 `/onrobot/sendCommand`+`/onrobot/pose`+`/onrobot_joint_states` 폴링을 씁니다.

## 사전 기동

기존 RealSense와 두산 드라이버는 별도로 실행해야 합니다. 이 패키지는 드라이버를 중복 기동하지 않습니다.

```bash
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
# 두산 M0609 드라이버와 검증된 RG2 ROS interface는 별도 터미널에서 기존 운영 절차로 기동
```

실행 전 `ros2 topic list`에서 `/camera/color/image_raw`, `/camera/aligned_depth_to_color/image_raw`, `/camera/color/camera_info`, `/onrobot_joint_states`를, `ros2 action list`에서 `/dsr01/motion/movej_h2r`, `/dsr01/motion/movel_h2r`를, `ros2 service list`에서 `/dsr01/dsr_controller2/motion/ikin`을 확인합니다. 또한 `ros2 param get /OnRobotRGControllerServer /onrobot/gripper`의 값이 `rg2`인지 확인합니다. 실제 실행을 열려면 hand-eye 파일, TCP pose service, RG2 command/state, **IK/충돌 사전검증**(`robot.obstacles`+`robot.gripper_radius_m`을 이 셀에 맞게 실측해 채워야 함), 그리고 `grasp_test.local.yaml`의 작업영역(workspace_mm)·Home(home_pose_deg)·approach_height_mm 값이 실제 셀에 맞게 채워져 있어야 합니다 — 이 값들은 로봇/작업대마다 다르므로 도구가 대신 정할 수 없습니다.

## 실행

```bash
colcon build --merge-install
source install/setup.bash
ros2 launch grasp_test grasp_test.launch.py \
  config:=$HOME/grasp_test_assets/grasp_test.local.yaml \
  methods:=all rounds:=3 input_mode:=live reset_mode:=manual execute:=false
```

`execute:=true`는 `/dsr01/dsr_controller2/motion/ikin`이 응답하지 않으면 여전히 `DRY_RUN_ONLY:IK_COLLISION_VALIDATION_INTERFACE_UNAVAILABLE`로 차단됩니다(IK 검사는 항상 필수).
`robot.obstacles`/`robot.gripper_radius_m`(충돌 사전검증)은 비어 있어도 더 이상 막지 않습니다 — 2026-09-04, 로봇 옆에서 직접 감독하겠다는 운영자 확인을 받고 IK만으로 실행을 허용하도록 바꿨습니다. 이 상태에서는 `motion_precheck.py`가 trial마다 "충돌 검사를 건너뛴다"는 WARN을 남기고, 사람이 눈으로 테이블/지그 충돌을 감독해야 합니다. 채워 넣으면 자동으로 충돌 사전검증이 켜집니다.

결과는 설정의 `results_dir`에 `live_physical_comparison.xlsx`, CSV, trial JSON, 모델별 로그로 저장됩니다. Contact-GraspNet은 공개 출력에 RG2 폭이 없으므로 `RG2_WIDTH_UNAVAILABLE`로 기록하고 실제 RG2 실행은 하지 않습니다. `best_score`는 모델별 스케일이 달라 비교하지 않으며, 실제 테스트가 가능해진 뒤 선택 기준은 성공률, 실패 유형, 전체 시간입니다.

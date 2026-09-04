# perception 검출 모델 테스트

`models/best.pt` = ultralytics **yolo11n-seg** (task=segment, imgsz=640, nc=6)
클래스: `0 suncream`, `1 tape`, `2 nail_product`, `3 denmark_probiotics`, `4 ecla_wet_tissue`,
`5 pigeon_spray_blue_bianca`. `config/objects.yaml`의 `model_labels`는 아직 앞의 3종만
매핑한다 — 나머지 3종은 신규 클래스로 잡혀 fragile이 강제된다(개발계획.md D-7, 2026-09-04
클래스 수 정정 참조).

## 1회 준비

```bash
python3 -m venv --system-site-packages .venv     # ROS(rclpy)를 그대로 쓰기 위해 system-site-packages
.venv/bin/pip install ultralytics
```

cv_bridge는 쓰지 않는다. venv의 opencv-python과 시스템 cv_bridge가 섞이면 깨지므로
sensor_msgs/Image ↔ ndarray 변환을 `tools/scripts/perception_capture.py:image_to_numpy`에서 직접 한다.

## 테스트 3단계

카메라가 떠 있어야 한다: `ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true`

```bash
source /opt/ros/jazzy/setup.bash

# (1) 샘플 수집 — color .png + aligned depth .npy + intrinsics .json
.venv/bin/python tools/scripts/perception_capture.py -n 5 --interval 1.0

# (2) 정지 이미지 검증 — 클래스/신뢰도/마스크 픽셀수/depth_valid_ratio/카메라 좌표 3D(mm)
.venv/bin/python tools/scripts/perception_test_image.py data/samples --conf 0.25
#   결과 오버레이: data/samples/pred/*_pred.png

# (3) 라이브 검증 — /perception/debug_image 로 오버레이 발행 (약 30 Hz, RTX 4060 기준 추론 6 ms)
.venv/bin/python tools/scripts/perception_test_live.py
ros2 run rqt_image_view rqt_image_view /perception/debug_image   # 다른 터미널
#   창을 직접 띄우려면 --show
```

## perception 노드 실행

```bash
source /opt/ros/jazzy/setup.bash && source ~/cobot2_ws/install/setup.bash
ros2 launch perception perception_launch.py        # .venv 인터프리터로 뜬다
```

launch가 `.venv/bin/python`을 prefix로 붙인다 — `ros2 run`이 만드는 스크립트는 시스템
파이썬 셔뱅이라 ultralytics를 못 찾는다. 직접 띄우려면:

```bash
DATABASE_URL=postgresql://... .venv/bin/python -m perception.node
```

**필요한 것**: 카메라(`realsense2_camera`)와 로봇 드라이버가 떠 있어야 한다. 로봇 드라이버가
필요한 이유는 카메라가 그리퍼에 붙어 있어서(eye-in-hand) base 좌표 변환에 현재 TCP 자세가
들어가기 때문이다 — `get_current_posx` 서비스를 부른다. `DATABASE_URL`은 없어도 동작하며,
그때는 `objects.yaml` seed 값을 쓴다.

주요 파라미터: `conf`(0.25), `period_s`(0.5), `min_depth_valid_ratio`(0.35),
`pose_max_age_s`(1.0), `require_robot_pose`(true).

검증: `python3 tools/scripts/check_perception.py` — 발행 내용이 인터페이스 계약을 지키는지 본다
(두 토픽의 stamp 짝, 마스크 형식·해상도, 좌표계, 속성 일치, object_id 유지).

## 아직 안 된 것

- ~~`node.py`가 스텁이다~~ → **구현 완료**. 위 세 스크립트는 여전히 모델 검증용 임시 도구다
  (노드를 띄우지 않고 가중치만 확인할 때 쓴다).
- ~~3D 좌표가 카메라 좌표계까지만 나온다~~ → **base 좌표 변환 구현**(`geometry.py`).
- **`grasp`가 아직 스텁이라 `/world_state`가 나오지 않는다.** perception은
  `/perception/world_state_raw`까지만 낸다(릴레이 구조, 인터페이스_정의서 2.0절). planner/web은
  `/world_state`를 구독하므로, grasp가 붙기 전까지 상위 계층은 `fake_world_publisher`를 본다.
- **좌표 정확도는 아직 실측되지 않았다.** 변환 수식이 캘리브레이션 코드와 같은 해석인지는
  대조했고(무작위 자세 2000개, 최대 오차 5e-13mm), 정지 물체 측정의 반복 표준편차는
  0.1~0.3mm였다. 그러나 **절대 오차**는 로봇을 두 자세로 옮겨 같은 물체가 같은 base 좌표로
  나오는지 보거나, `calibration/verify.py`로 TCP를 실제로 보내 재야 알 수 있다.
- ~~`config/objects.yaml`의 클래스가 학습된 3종과 다르다~~ → **해소(2026-09-03, D-7)**.
  `objects.yaml`이 `suncream`/`nail`/`tape` 3종을 담고 있다. 단 **모델 라벨은 `nail_product`이고
  정식 `class_name`은 `nail`**이라, `node.py`가 모델 출력을 `DetectedObject.class_name`으로
  옮길 때 `objects.yaml`의 `model_labels` 표를 반드시 지나야 한다. 안 지나면 네일이 매번
  미확인 신규품목으로 떨어지고 fragile 프로파일이 강제된다 — 오류로 보이지 않고 느려지기만 한다.

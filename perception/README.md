# perception 검출 모델 테스트

`model/best.pt` = ultralytics **yolo26n-seg** (task=segment, imgsz=640, nc=3)
클래스: `0 suncream`, `1 tape`, `2 nail_product`

## 1회 준비

```bash
python3 -m venv --system-site-packages .venv     # ROS(rclpy)를 그대로 쓰기 위해 system-site-packages
.venv/bin/pip install ultralytics
```

cv_bridge는 쓰지 않는다. venv의 opencv-python과 시스템 cv_bridge가 섞이면 깨지므로
sensor_msgs/Image ↔ ndarray 변환을 `scripts/perception_capture.py:image_to_numpy`에서 직접 한다.

## 테스트 3단계

카메라가 떠 있어야 한다: `ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true`

```bash
source /opt/ros/jazzy/setup.bash

# (1) 샘플 수집 — color .png + aligned depth .npy + intrinsics .json
.venv/bin/python scripts/perception_capture.py -n 5 --interval 1.0

# (2) 정지 이미지 검증 — 클래스/신뢰도/마스크 픽셀수/depth_valid_ratio/카메라 좌표 3D(mm)
.venv/bin/python scripts/perception_test_image.py data/samples --conf 0.25
#   결과 오버레이: data/samples/pred/*_pred.png

# (3) 라이브 검증 — /perception/debug_image 로 오버레이 발행 (약 30 Hz, RTX 4060 기준 추론 6 ms)
.venv/bin/python scripts/perception_test_live.py
ros2 run rqt_image_view rqt_image_view /perception/debug_image   # 다른 터미널
#   창을 직접 띄우려면 --show
```

## 아직 안 된 것

- `perception/perception/node.py`는 여전히 스텁이다. 위 스크립트는 **모델 검증용 임시 도구**이고
  `/perception/world_state_raw` / `/perception/instance_masks`는 발행하지 않는다.
- 3D 좌표는 **카메라 좌표계 mm**까지만 나온다. `DetectedObject.position_base_mm`으로 가려면
  `data/calibration/T_gripper2camera.npy` + 현재 TCP 자세로 base 변환을 붙여야 한다.
- ~~`config/objects.yaml`의 클래스가 학습된 3종과 다르다~~ → **해소(2026-09-03, D-7)**.
  `objects.yaml`이 `suncream`/`nail`/`tape` 3종을 담고 있다. 단 **모델 라벨은 `nail_product`이고
  정식 `class_name`은 `nail`**이라, `node.py`가 모델 출력을 `DetectedObject.class_name`으로
  옮길 때 `objects.yaml`의 `model_labels` 표를 반드시 지나야 한다. 안 지나면 네일이 매번
  미확인 신규품목으로 떨어지고 fragile 프로파일이 강제된다 — 오류로 보이지 않고 느려지기만 한다.

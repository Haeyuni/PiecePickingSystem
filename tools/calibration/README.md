# calibration

단일 카메라(eye-in-hand) hand-eye calibration. 0단계 산출물: 재투영 오차·TCP 실측 오차 검증 리포트(`docs/results/calibration_report.md`).

## 실행 순서

1. `data_recording.py`: 체커보드를 보며 로봇을 수동 이동, `q`로 이미지 + `posx` 저장 → `data/*.jpg`, `data/calibrate_data.json`
2. `handeye_calibration.py`: `cv2.calibrateCamera`로 내부 파라미터, `cv2.calibrateHandEye`(PARK)로 gripper→camera 변환 계산 → `data/calibration/T_gripper2camera.npy`
3. `verify.py`: 컬러 영상 클릭 → 뎁스로 3D 복원 → hand-eye 결과로 베이스 좌표 변환 → 해당 지점 파지·이송으로 실측 검증

## 지원 모듈

- `realsense.py`: RGB-D + `camera_info` 구독 노드(`ImgNode`)
- `onrobot.py`: OnRobot RG2 그리퍼 Modbus TCP 제어
- `modbus.py`: 그리퍼 개폐 단독 점검 스크립트

## 미구현

- `measure_bins.py`: `src/control/config/bins.yaml` 좌표 실측

## 산출물

- `data/calibration/T_gripper2camera.npy`: hand-eye 결과. 모든 서비스의 좌표 변환 기준값이라 저장소 루트의 `data/calibration/`에 두고 **버전관리한다**(`.gitignore`의 `*.npy` 규칙에 예외 등록). 경로는 `HANDEYE_PATH` 환경변수로 덮어쓸 수 있다
- `data/`: 캘리브레이션 촬영본과 포즈 기록 (로컬 전용)

단위는 mm 기준이다 — 로봇 `posx`, hand-eye 결과, 시스템 전 계층이 동일 (`docs/인터페이스_정의서.md` 1.1절).

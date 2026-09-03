# calibration

단일 카메라(eye-in-hand) hand-eye calibration. 0단계 산출물: 재투영 오차·TCP 실측 오차 검증 리포트(`docs/results/calibration_report.md`).

- `collect_calibration_poses.py`: 캘리브레이션 포즈 수집
- `hand_eye_solve.py`: `cv2.calibrateHandEye` (필요 시 `easy_handeye2`)
- `verify_tcp_touch.py`: TCP 실측 터치 검증
- `measure_bins.py`: `bins.yaml` 좌표 실측

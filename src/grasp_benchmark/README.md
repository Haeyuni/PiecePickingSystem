# Grasp Benchmark

ROS 2 시스템, WorldState, LLM, Perception, Control, M0609, RG2 동작 코드와 독립된 오프라인 파지 모델 비교 패키지입니다. 네 모델은 서로 다른 Docker 이미지에서 순차 실행되므로 PyTorch와 TensorFlow를 한 Python 환경에 함께 설치하지 않습니다.

## Scene 파일

입력은 `rgb`, `depth_m`(미터), `K`, `mask`를 포함한 `.npz` 파일입니다. 기본 명령의 `scene_001.npz`는 먼저 `~/Downloads/scene_001.npz`, 그 다음 워크스페이스의 `data/scenes/scene_001.npz`에서 찾습니다. 절대 경로도 사용할 수 있습니다.

## 실행

워크스페이스 루트에서 빌드와 환경 설정 후 실행합니다.

```bash
colcon build --merge-install
source install/setup.bash
ros2 launch grasp_benchmark compare_models.launch.py scene:=scene_001.npz
```

`results_dir`을 지정하면 결과 위치를 바꿀 수 있습니다.

```bash
ros2 launch grasp_benchmark compare_models.launch.py scene:=~/Downloads/scene_001.npz results_dir:=results
```

## 결과

`results/<scene_id>_comparison.xlsx`, `results/<scene_id>_comparison.csv`, `results/<scene_id>_comparison_preview.png`가 생성됩니다. PNG는 각 모델이 선택한 후보 중심을 원본 RGB 위에 표시해 눈으로 위치 차이를 확인할 수 있습니다. 모델별 Docker 빌드·실행 로그는 `results/<scene_id>_<model>.log`에 저장됩니다.

각 실행 전에 해당 scene의 이전 JSON, 로그, CSV, XLSX, PNG를 삭제하므로 이전 실행 결과가 섞이지 않습니다. 한 모델이 실패해도 나머지 모델과 집계는 계속 실행하며 실패 행은 `ERROR` 또는 `NOT_RUN`입니다.

## 해석 제한

- `best_score`와 `candidate_count`는 모델마다 의미가 달라 모델 간 순위 지표가 아닙니다.
- `width_validation=VALID`은 예측 폭이 RG2 설정 범위 5~100 mm에 들어간다는 뜻일 뿐, 실제 파지 성공을 보장하지 않습니다.
- Contact-GraspNet 공개 출력은 그리퍼 폭을 제공하지 않아 `width_validation=UNAVAILABLE`이 정상입니다.
- Contact-GraspNet의 `initialization_ms`(체크포인트 다운로드·네트워크 생성·가중치 로드)와 `inference_ms`(warm-up 뒤 두 번째 추론)는 분리 기록됩니다.
- 충돌 검사, IK, 실제 RG2 파지와 로봇 동작은 이 패키지 범위에 포함하지 않습니다.

## 환경 제약

GG-CNN, GraspNet-baseline, Contact-GraspNet은 NVIDIA GPU, NVIDIA Container Toolkit, Docker의 `--gpus all` 지원이 필요합니다. 각 공개 체크포인트는 첫 실행 때 네트워크로 내려받습니다. CUDA, Docker 이미지 빌드, 공개 모델 저장소 또는 체크포인트 URL 변경으로 특정 모델이 실패할 수 있으며, 상세 내용은 모델별 `.log` 파일에서 확인합니다.

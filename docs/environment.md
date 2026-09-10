# 개발/실행 환경

`시스템명세서.md` 3.4절·4.2절에서 참조하는 버전 고정 문서. 서비스별 `Dockerfile`에 반영한다.
실행 순서 자체는 저장소 `README.md` 6절을 본다.

## 1. 공통 런타임

| 항목 | 값 |
| --- | --- |
| ROS2 배포판 | Jazzy |
| Python | 3.12 |
| OS | Ubuntu 24.04 |
| Node.js | 20 (프론트엔드 빌드 전용 — `src/web/Dockerfile` 1단계) |
| DB | PostgreSQL 16 |

## 2. 워크스테이션 사양 (실측)

| 항목 | 값 | 비고 |
| --- | --- | --- |
| GPU | NVIDIA GeForce RTX 4060, VRAM 8GB | 드라이버 595.84 |
| CPU | *(미기재)* | TODO |
| RAM | *(미기재)* | TODO |
| 저장공간 | *(미기재)* | TODO |

8GB VRAM 대응 전략은 `시스템명세서.md` 4.2절 참조 (`perception`/`grasp` 순차 추론, YOLO11 `n`/`s` 사이즈 우선, fp16/bf16 혼합정밀도 등).

## 3. CUDA / PyTorch 조합

**perception 확정 (2026-09-04 실측)**: 드라이버 595.84가 CUDA 13.2까지 지원한다. `pip install
ultralytics`가 그대로 torch 2.14.0(cu13 빌드)·torchvision 0.29.0·opencv-python 5.0.0.93·
numpy 2.5.2를 받아 오고, `nvidia-smi`/`torch.cuda.is_available()` 양쪽에서 GPU를 잡는 것까지
확인했다(호스트 `.venv`, Docker 컨테이너 둘 다). 고정값은 `src/perception/requirements.txt`.
베이스 이미지에는 CUDA 툴킷을 넣지 않는다 — torch 휠이 필요한 런타임을 pip 의존성으로
가져오고, GPU 접근은 호스트 드라이버 + nvidia-container-toolkit이 맡는다
(`docker-compose.yml`의 `deploy.resources.reservations.devices`).

`detector=vlm_sam`이 쓰는 SAM 가중치도 같은 ultralytics 런타임을 탄다 — 2026-09-08에
mobile_sam에서 **SAM2 base**(`models/sam2_b.pt`, 약 310MB)로 교체했다(정확도 실측 근거는
`vlm_sam_pipeline.md`). `sam_marks.segment_everything`이 파일명으로 SAM/SAM2/SAM3를 갈라
쓰므로 가중치 파일만 바꾸면 코드 수정이 필요 없다.

**grasp 확정 (2026-09-07)**: `grasp` 노드 자체는 PyTorch를 쓰지 않는다 — 포인트클라우드
계산에 NumPy만 쓰고, 학습 기반 전략은 HTTP로 넘긴다. 그래서 노드와 추론 환경의 CUDA/torch
조합을 맞출 필요가 없다.

- `heuristic_pca`: 추가 의존 없음
- `graspnet_baseline`: **별도 이미지**(`src/grasp/docker/graspnet_baseline/Dockerfile`)가
  GraspNet-baseline과 그 CUDA 확장·PyTorch를 담고, compose의 `graspnet` 서비스로 상주하며
  `:8200`에서 HTTP로 받는다. 체크포인트는 저장소 밖 assets에서 읽기 전용 마운트한다.
- `contact_graspnet`: 미해당 — 가중치와 별도 CUDA 환경이 준비되기 전까지 레지스트리에
  등록하지 않는다(`시스템명세서.md` 3.2절).

## 4. 서비스별 언어/런타임

`시스템명세서.md` 3.1절 표 참조 — 서비스별 Dockerfile에 반영.

| 서비스 | 베이스 이미지 |
| --- | --- |
| `perception`·`grasp`·`control`·`web` | `ros:jazzy-ros-base` (+ `rmw_cyclonedds_cpp`) |
| `planner` | `python:3.12-slim` (ROS2 없음) |
| `graspnet` | 자체 CUDA/PyTorch 이미지 |
| `db` | `postgres:16` |

**RMW는 시스템 전체가 같아야 한다.** `ros:jazzy-ros-base`의 기본 RMW는 FastDDS인데
워크스테이션의 노드는 CycloneDDS로 뜬다. 서로 다르면 `ros2 topic list`에는 토픽이 보이는데
**구독 콜백이 한 번도 불리지 않는다** — B4 검증에서 실제로 겪은 증상이다. 컨테이너에
`rmw_cyclonedds_cpp`를 넣고 호스트의 `~/.config/cyclonedds` 설정을 읽기 전용으로 마운트해
인터페이스 선택까지 맞춘다.

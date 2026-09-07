# 개발/실행 환경

`시스템명세서.md` 3.4절·4.2절에서 참조하는 버전 고정 문서. 서비스별 `Dockerfile`에 반영한다.

## 1. 공통 런타임

| 항목 | 값 |
| --- | --- |
| ROS2 배포판 | Jazzy |
| Python | 3.12 |
| OS | Ubuntu 24.04 |

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

`grasp`는 아직 미해당 — heuristic_pca(numpy만 필요)만 구현되어 있어 Open3D/PyTorch/
Contact-GraspNet/GraspNet-baseline이 실제로 붙는 시점에 다시 실측한다.

## 4. 서비스별 언어/런타임

`시스템명세서.md` 3.1절 표 참조 — 서비스별 Dockerfile에 반영.

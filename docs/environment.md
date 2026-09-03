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

TODO: `perception`(YOLO11-seg)과 `grasp`(Open3D, PyTorch, Contact-GraspNet, GraspNet-baseline)는 CUDA 버전을 분리한다 (시스템명세서 3.1절). 실측 후 조합 확정.

## 4. 서비스별 언어/런타임

`시스템명세서.md` 3.1절 표 참조 — 서비스별 Dockerfile에 반영.

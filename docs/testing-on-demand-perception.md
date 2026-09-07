# 온디맨드 인지(SAM+VLM) 테스트 가이드

`docs/on-demand-perception.md`에서 구현한 변경을 실제로 돌려보기 위한 안내. 설계·결정 근거는
그 문서를 보고, 여기는 "무엇을 어떻게 실행하는가"만 다룬다.

## 1. 바뀐 것 요약

- `perception`이 0.5초 주기 발행에서 **명령을 받을 때만 관측하는 액션 서버**로 전환됨
  (`Observe.action`, `MODE_FULL`/`MODE_REPROMPT`).
- YOLO 옆에 **SAM+VLM 검출기**가 detector 스위치로 추가됨 — 학습한 7클래스 밖의 물건도
  인지 가능. 기본값은 여전히 `yolo`이고, `PERCEPTION_DETECTOR=vlm_sam`으로 전환한다.
- planner에 `POST /internal/label-marks` 신설 — SAM이 그린 번호에 VLM이 이름만 붙인다.
- web: `executor.observe()` 추가, UI의 탐지 물체 목록이 확인 대기 패널로 교체됨.
- grasp 노드는 **무수정** — `world_state_raw`/`instance_masks` 계약을 그대로 지킨다.

`vlm_sam`은 **하드웨어(로봇·카메라·GPU)로 아직 검증되지 않았다**
(`docs/on-demand-perception.md` 6절). 아래 1~3단계는 하드웨어 없이 확인 가능하고,
4단계부터 하드웨어가 필요하다.

## 2. 단위 테스트 (비용 없음, 가장 먼저 돌려볼 것)

```bash
cd services/planner && python -m pytest tests/ -v        # label-marks, validator 등 27건
cd src/perception && python -m pytest test/ -v            # detectors, sam_marks, sam_reprompt, mask_utils
cd src/perception_common && python -m pytest test/ -v     # geometry 역투영 왕복 테스트
cd web/frontend && npm run build                          # UI 빌드 확인
```

## 3. SAM+VLM 파이프라인 단독 확인 (ROS 불필요, 정지 이미지 1장)

`OPENAI_API_KEY`가 있어야 하고, 호출당 API 비용이 든다.

```bash
.venv/bin/python tools/scripts/vlm_sam_test.py services/planner/image.png \
    --command "치약 왼쪽으로"
```

- 결과는 `data/samples/vlm_sam/`에 마스크 오버레이(`*_marks.png`), 박스(`*_boxes.png`),
  VLM 원본 응답(`*_scene.json`) 등으로 저장된다.
- `--reuse-marks`: SAM 전체 분할(CPU 26초)만 재사용하고 지시만 바꿔서 반복.
- `--from-json data/samples/vlm_sam/image_scene.json`: VLM 호출까지 통째로 재사용해
  API 비용 없이 뒷단만 반복.
- `--mode box`: VLM이 박스를 먼저 주고 SAM 프롬프트로 쓰는 원래 설계 경로(비교용). 기본값
  `som`(SAM 먼저 → VLM이 번호에 이름만)이 gpt-4o에서 훨씬 정확하다(`docs/vlm_sam_pipeline.md`).

## 4. MOCK_MODE 전체 E2E (로봇/GPU 없이 전체 흐름)

```bash
cp .env.example .env   # MOCK_MODE=1이 기본값
docker compose up --build
python3 tools/scripts/check_web_e2e.py --url http://localhost:8000
```

명령 전송 → WebSocket 진행 상황 수신 → 실행 결과 → 이력 조회까지 한 바퀴 돈다. planner를
실제로 호출하므로 LLM 비용이 명령 1건만큼 든다.

## 5. 실제 SAM+VLM 검출기로 구동 (하드웨어 필요)

```bash
# .env에 추가
PERCEPTION_DETECTOR=vlm_sam

docker compose --profile ros up web_ros
```

주의:
- `colcon build`/`ros2 action list`로 `Observe.action`이 실제 ROS2 환경에서 빌드·구동되는지
  아직 미검증 (개발 세션이 rclpy 없는 환경이라 `py_compile`/`pyflakes`로만 확인함).
- GPU에서의 SAM everything 소요시간이 미실측 — CPU 26초만 잰 상태. `orchestrator.py`의
  `OBSERVE_FULL_TIMEOUT_S`(60초)/`OBSERVE_REPROMPT_TIMEOUT_S`(10초)는 그 위에 여유를 얹은
  추정값이라 GPU 실측 후 낮출 필요가 있음.
- 재투영 프롬프트 정확도(물체가 밀렸을 때 60mm 안전망이 적절한지), YOLO 대비 마스크 IoU도
  아직 수치로 확인되지 않음.

이 단계까지 검증됐는지는 `docs/on-demand-perception.md` "6. 아직 실측되지 않은 것"에
계속 갱신할 것.

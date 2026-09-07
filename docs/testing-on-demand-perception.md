# 온디맨드 인지(SAM+VLM) 테스트 가이드

`docs/on-demand-perception.md`에서 구현한 변경을 실제 로봇으로 검증하기 위한 안내. 설계·결정
근거는 그 문서를 보고, 여기는 "무엇을 어떻게 실행하는가"만 다룬다.

**`vlm_sam`은 하드웨어로 아직 검증된 적이 없다** (`docs/on-demand-perception.md` 6절) —
아래 순서로 처음 켜볼 때 이 문서의 "6. 흔한 실패 지점"을 먼저 훑어두는 것을 권장한다.

## 1. 바뀐 것 요약

- `perception`이 0.5초 주기 발행에서 **명령을 받을 때만 관측하는 액션 서버**로 전환됨
  (`Observe.action`, `MODE_FULL`=전체 스캔/`MODE_REPROMPT`=재관측).
- YOLO 옆에 **SAM+VLM 검출기**가 detector 스위치로 추가됨 — 학습한 7클래스 밖의 물건도
  인지 가능. 기본값은 여전히 `yolo`이고, `PERCEPTION_DETECTOR=vlm_sam`으로 전환한다.
- planner에 `POST /internal/label-marks` 신설 — SAM이 그린 번호에 VLM이 이름만 붙인다.
- web: `executor.observe()` 추가, UI의 탐지 물체 목록이 확인 대기 패널로 교체됨.
- grasp 노드는 **무수정** — `world_state_raw`/`instance_masks` 계약을 그대로 지킨다.

## 2. 사전 준비물

| 항목 | 확인 |
| --- | --- |
| 로봇 | Doosan cobot, `dsr_control2`/`doosan-robot2` 드라이버 기동 중 |
| 카메라 | eye-in-hand RealSense, `realsense2_camera` 노드로 `/camera/color/image_raw`·`/camera/aligned_depth_to_color/image_raw`·`/camera/color/camera_info` 게시 중 |
| 캘리브레이션 | `T_gripper2camera.npy` 존재 (hand-eye 결과, `.env`의 `HANDEYE_PATH`가 가리키는 경로) |
| GPU | CUDA 인식되는 GPU (`docs/environment.md` — RTX 4060/8GB 기준 실측). CPU로는 SAM everything이 ~26~40초 걸려 `OBSERVE_FULL_TIMEOUT_S`(60초)에 위험하게 붙는다 |
| API 키 | `.env`에 `OPENAI_API_KEY` (VLM 라벨링 + planner 둘 다 사용) |
| ROS 도메인 | 호스트 `echo $ROS_DOMAIN_ID`와 `.env`의 `ROS_DOMAIN_ID`가 같은 값 |

## 3. 실제 로봇으로 바로 켜보기

```bash
cp .env.example .env
```

`.env`에서 아래 값을 실제 환경에 맞게 채운다:

```bash
MOCK_MODE=0
PERCEPTION_DETECTOR=vlm_sam
OPENAI_API_KEY=sk-...
HANDEYE_PATH=/data/calibration/T_gripper2camera.npy   # 컨테이너 내부 경로
ROS_DOMAIN_ID=0                                        # 호스트와 동일하게
#DOOSAN_WS_DIR=/home/사용자/cobot2_ws                  # 기본값 $HOME/cobot2_ws 아니면 지정
#CYCLONEDDS_CONFIG_DIR=/home/사용자/.config/cyclonedds  # 기본값 $HOME/.config/cyclonedds 아니면 지정
```

기동 순서 (perception/grasp/control은 host 네트워크로 떠서 호스트의 로봇 드라이버·카메라와
DDS로 직접 통신한다 — 먼저 로봇 드라이버와 카메라 노드가 돌고 있어야 한다):

```bash
# 0) 로봇 드라이버 + RealSense가 이미 호스트(또는 별도 컨테이너)에서 기동 중이어야 함
#    ros2 topic list 로 /camera/... 토픽과 get_current_posx 서비스가 보이는지 먼저 확인

# 1) DB + planner + perception/grasp/control
docker compose up --build -d db planner perception grasp control

# 2) 새로 뜬 perception이 vlm_sam으로 뜬 것 확인
docker compose logs perception | grep -i detector

# 3) web을 ROS 모드로 (mock web과 포트 8000이 겹치니 동시에 띄우지 않는다)
docker compose --profile ros up --build web_ros
```

명령 실행:

```bash
# UI: web/frontend 접속 후 명령창에 입력
# 또는 curl로 직접
curl -X POST http://localhost:8000/api/commands \
  -H "Content-Type: application/json" \
  -d '{"schema_version":"1.0.0","command_text":"치약 왼쪽 박스로 옮겨줘"}'
```

확인할 것:
- 명령 진입 시 관측 A(`MODE_FULL`) 트리거 → home 이동 → SAM everything(관측 이미지가
  `data/observations/<trace_id>/`에 저장됨) → VLM 라벨링 → `/world_state` 발행까지 걸리는
  시간. `docs/on-demand-perception.md` D-4/D-8 참조 — CPU 실측 기준 ~10초대였고, GPU
  실측치는 아직 없다(이번이 그 첫 실측이 될 수 있음. 재면 6절 문서에 반영해줄 것).
- pick → place_into 사이 관측 B(`MODE_REPROMPT`, 재투영 박스 프롬프트)가 CPU 대비
  체감상 즉시 끝나는지(설계상 0.2초 근방).
- `/grasp/debug_image`로 중계되는 카메라 뷰에 마스크 윤곽 + 그리퍼 팜/손가락이 겹쳐 그려지는지
  (UI의 탐지 물체 목록은 없어졌고, 이 뷰 + 확인 대기 패널로 대체됨 — D-5).
- 매칭 안 되는 새 마스크(가려졌다 드러난 물체 등)가 해당 명령 시퀀스에서 제외되고
  다음 명령의 관측 A에서만 다시 잡히는지(D-3).

## 4. 실패 시 먼저 볼 로그

```bash
docker compose logs perception -f     # Observe 액션 phase, SAM/VLM 타임아웃
docker compose logs planner -f        # /internal/label-marks 응답, VLM 호출 에러
docker compose logs web_ros -f        # orchestrator의 관측 트리거/타임아웃
```

`ros2 action list`로 `/observe` 액션이 떠 있는지, `ros2 topic echo /world_state`로 관측이
실제로 발행되는지도 확인 가능 (perception 컨테이너 안에서 `docker exec`로).

## 5. 흔한 실패 지점 (하드웨어 미검증이라 미리 알아둘 것)

- **`Observe.action`이 실제로 빌드되는지 미검증.** 개발 세션이 rclpy 없는 환경이라
  `py_compile`/`pyflakes`로만 확인했다. `colcon build` 시 `sort_msgs`의 액션 인터페이스
  생성 실패가 있으면 여기부터 의심.
- **GPU에서의 SAM everything 시간 미실측.** `OBSERVE_FULL_TIMEOUT_S`(60초)/
  `OBSERVE_REPROMPT_TIMEOUT_S`(10초, `orchestrator.py`)는 CPU 실측(26~40초/0.2초)에 여유를
  얹은 추정값이다. 타임아웃으로 관측이 조용히 실패하면 이 값부터 늘려서 재현되는지 확인.
- **재투영 프롬프트가 빗나가는 경우.** 물체가 사람 손이나 다른 원인으로 크게 밀린 경우
  60mm 안전망(D-2, tracker의 3D 거리 검사)이 적절한지 아직 실측 안 됨 — 그 물체가
  "사라짐"으로 잘못 판정되면 이 임계값을 의심.
- **`debug_color_max_age_s=15.0`(grasp).** 관측 A가 10초 이상 걸리면 grasp의 컬러 프레임
  버퍼가 비어 debug 이미지가 안 그려질 수 있다 — D-7 참조.
- **`execution_logs.observation_image_path`는 아직 실제로 채워지지 않는다.** 컬럼만 있고
  배선은 미완성 — DB에서 관측 사진 경로를 찾아도 비어 있는 게 정상.

## 6. 하드웨어 없이 먼저 확인하고 싶다면

로봇을 붙이기 전에 오프라인으로 돌려볼 수 있는 것들.

```bash
# 단위 테스트
cd services/planner && python -m pytest tests/ -v
cd src/perception && python -m pytest test/ -v
cd src/perception_common && python -m pytest test/ -v
cd web/frontend && npm run build

# SAM+VLM 파이프라인만 정지 이미지로 (OPENAI_API_KEY 필요, 호출당 비용 발생)
.venv/bin/python tools/scripts/vlm_sam_test.py services/planner/image.png \
    --command "치약 왼쪽으로"

# MOCK_MODE 전체 E2E (로봇/GPU 없이 web 흐름 전체)
cp .env.example .env   # MOCK_MODE=1이 기본값
docker compose up --build
python3 tools/scripts/check_web_e2e.py --url http://localhost:8000
```

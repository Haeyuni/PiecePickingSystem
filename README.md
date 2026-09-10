# PiecePickingSystem — 자연어 지시 기반 다품종 분류·피스피킹 시스템

## 0. 프로젝트 개요

- **목표**: 품목이 바뀔 때마다 재티칭·재프로그래밍하는 비용을 없애는 것. 사람이 말이나
  글로 내린 지시를 로봇이 그대로 해석해 물체를 골라 집고 지정한 곳에 놓는다
- **주요 기능**: 도메인(가정·약국·재활용)을 고른 뒤 명령을 넣으면 관측 → LLM 계획 →
  결정적 검증 → 사람 승인 → `pick` → `place_into`가 이어진다. 등록 어휘가 없는 물체도
  SAM+VLM 경로로 인지하고, 실행 결과와 라벨은 데이터셋·실행 로그로 쌓인다
- **사용 장비**: Doosan M0609 (6축 협동로봇), OnRobot RG2 그리퍼, RealSense RGB-D
  (eye-in-hand)
- **개발 환경**: Ubuntu 24.04 LTS, ROS 2 Jazzy, Docker Compose
- **주요 기술 스택**: ROS 2, FastAPI + rclpy 브리지, React (Vite/TypeScript),
  PostgreSQL 16, OpenAI API (LLM/VLM), YOLO11-seg · SAM2, GraspNet-baseline
- **기간**: 2026.09.01 ~ 2026.09.11

## 1. 시스템 설계

![시스템 아키텍처](https://raw.githubusercontent.com/Haeyuni/PiecePickingSystem/main/docs/diagrams/system_architecture.png)

`perception`이 `grasp_candidates`가 빈 `/perception/world_state_raw`를 1차 발행하고,
`grasp`가 `instance_masks`와 짝지어 후보를 채운 뒤 `/world_state`로 최종 발행한다.
**`/world_state`의 발행자는 `grasp` 하나뿐이다** — 두 노드가 같은 토픽에 쓰면 구독자가
"빈 버전"과 "완성 버전"을 구분할 수 없다.

### 1.1 통신 인터페이스

```text
Topic     /perception/world_state_raw (WorldState) · /perception/instance_masks (InstanceMasks)
          /world_state (WorldState — 발행자는 grasp 하나)
          /control/robot_state (RobotState) · /control/safety_events (SafetyEvent)
          /control/selected_grasp (SelectedGrasp, transient local)
          /perception/debug_image · /grasp/debug_image (sensor_msgs/Image)
Action    /observe (Observe, perception)
          /pick (Pick) · /place_into (PlaceInto) · /home (Home) — 전부 control
HTTP      web      :8000  /api/* · /ws/live
          planner  :8100  /internal/plan · /internal/label-marks · /internal/datasets/*
          graspnet :8200  /health · 추론 엔드포인트
```

- 이 저장소가 **정의하는 ROS2 서비스는 없다.** 서비스 호출은 전부 외부 드라이버 쪽이다
  (`get_current_posx`·`ikin`·`fkin` 등 dsr, `/onrobot/sendCommand`·`/onrobot/pose`).
- 외부 의존 토픽: `realsense2_camera`의 `/camera/color/image_raw`,
  `/camera/aligned_depth_to_color/image_raw`, `/camera/color/camera_info`.
  depth는 color와 같은 픽셀 grid·camera frame의 `16UC1`(mm)이어야 한다.
- 길이는 계층 경계 어디서나 **mm**다. 두산 API(`get_current_posx`·`ikin`)도 mm라 지금은
  환산이 일어나는 곳이 없고, ROS 관례(m)로 바꿔야 할 스택이 붙으면 그 환산은
  `src/control/control/units.py` 한 곳에서만 한다.

### 1.2 네트워크 구성

웹·planner·ROS 노드·DB가 **로봇 PC 한 대**에서 전부 돈다.

![네트워크 구성](https://raw.githubusercontent.com/Haeyuni/PiecePickingSystem/main/docs/diagrams/network.png)

- `perception`/`grasp`/`control`/`web_ros`는 `network_mode: host`다. DDS 디스커버리가
  멀티캐스트에 기대므로 브리지 네트워크에서는 호스트의 드라이버·카메라 노드를 못 찾는다.
- host 네트워크에는 서비스 이름 DNS가 없어 `planner`/`db`/`graspnet`을 게시된 포트로
  `localhost` 호출한다.
- 외부로 나가는 통신은 OpenAI API(planner) 하나뿐이다. 키는 `.env`에만 두고 이미지에
  굽지 않는다.

## 2. 플로우 차트

### 2.1 명령 처리 시퀀스

![명령 처리 시퀀스](https://raw.githubusercontent.com/Haeyuni/PiecePickingSystem/main/docs/diagrams/command_flow.png)

| 단계 | 하는 일 | 담당 |
|---|---|---|
| 명령 접수 | 텍스트 또는 음성(STT). `robot_state.mode`가 idle이 아니면 즉시 거부 | web |
| 관측 A | `Observe` `MODE_FULL` — SAM everything + VLM 라벨링으로 "무엇이 있는가"를 새로 안다 | perception → planner |
| 계획 | 관측된 물체·목적지 목록 안에서만 시퀀스를 생성한다(그라운딩) | planner |
| 검증 | 결정적 검증기 — 가반하중·작업반경·파지 가능 여부·안전 이벤트를 본다 | planner |
| 실행 승인 | 계획된 스텝과 판단 근거를 보여주고 승인/거부/라벨 수정을 받는다 | 브라우저 |
| 실행 | `pick` → `place_into`를 순서대로 액션 호출 | web → control |

- **검증을 통과해도 곧바로 실행하지 않는다.** SAM+VLM 경로가 어휘 없이 물체 속성
  (파지력에 직결되는 `grip_level` 포함)까지 스스로 판단하므로, 명령 1건당 한 번은 사람이
  본다. 승인 응답은 `approve` / `reject` / `correct_label` 셋이고, `correct_label`은
  라벨을 고쳐 재계획한 뒤 같은 승인을 다시 요청한다.
- 정지(`/api/robot/stop`)와 홈복귀(`/api/robot/home`)는 자연어 해석이 필요 없는 동작이라
  planner를 거치지 않고 web이 control을 직접 부른다. 특히 정지는 `mode`와 무관하게 항상
  받는다.
- 관측 B(`MODE_REPROMPT`)는 VLM을 부르지 않고 직전 관측 물체가 아직 있는지만 재투영
  박스로 확인한다. 쓰는 곳은 두 군데다 — 파지 후보가 없어 거부됐을 때의 재시도,
  `place_into`를 끝낸 뒤 화면·최신 상태 갱신.

### 2.2 동작 순서도

![동작 순서도](https://raw.githubusercontent.com/Haeyuni/PiecePickingSystem/main/docs/diagrams/operation_flow.png)

실패했을 때의 갈래는 두 가지이며 서로 다르게 처리한다.

| 실패 지점 | 처리 | 상한 |
|---|---|---|
| `pick` 계열 | home 복귀 → 관측 A 다시 → 재계획 | 재계획 2회 |
| `place_into` | 같은 스텝을 새 `request_id`로 재전송(로봇이 이미 물체를 쥐고 있어 재계획할 것이 없다) | 재시도 2회 |

- 후보 선택의 최종 결정권은 `control`에 있다. `planner`는 점수 내림차순 후보 전체를
  실려 보내고, 개폭 유효성·IK·관절 한계·접근 적합도는 로봇에 붙어 있어야 답할 수 있으므로
  `control/grasp_selection.py`가 고른다.
- 파지 성공 판정은 개폭 추정이 아니라 RG2 컨트롤러의 `Grip detected` 비트를 쓴다.
  변형체를 제대로 물어도 개폭이 끝까지 닫히는 경우가 있기 때문이다.

## 3. 운영체제 환경

| 항목 | 값 |
|---|---|
| OS | Ubuntu 24.04 LTS |
| ROS 배포판 | ROS 2 Jazzy |
| Python | 3.12 |
| 빌드 | colcon (ROS2 패키지) · Docker Compose (서비스) |
| Node.js | 20 (Vite 7 / TypeScript 5.9 / React 19) |
| DB | PostgreSQL 16 |
| GPU | NVIDIA RTX 4060 8GB, 드라이버 595.84 (실측 기준) |

버전 고정의 근거와 CUDA/PyTorch 조합은 `docs/environment.md`에 있다.

### 워크스페이스 배치

이 저장소는 **두산 드라이버 워크스페이스와 별도**다. 드라이버(`dsr_*`, `onrobot_*`)는
`~/cobot2_ws`에 있고, 컨테이너는 거기서 메시지 패키지만 읽기 전용으로 마운트해 쓴다
(`DOOSAN_WS_DIR`).

```text
PiecePickingSystem/          # ROS2 컨테이너 안에서는 /ros2_ws
├── src/                     # ROS2 패키지
│   ├── sort_msgs/           # msg · action 정의 (ament_cmake)
│   ├── perception/          # 온디맨드 관측 (YOLO11-seg / SAM+VLM)
│   ├── grasp/               # 파지 후보 계산 (2전략)
│   ├── control/             # pick · place_into · home 액션 서버
│   ├── perception_common/   # 좌표 변환·이미지 유틸 공용 패키지
│   └── web/                 # FastAPI + rclpy 브리지
├── services/planner/        # ROS2와 무관한 계획 API (COLCON_IGNORE)
├── web/frontend/            # Vite + React UI
├── database/migrations/     # PostgreSQL 스키마
├── data/                    # 캘리브레이션 결과·mock 픽스처·관측/데이터셋 산출물
├── models/                  # 로컬 모델 가중치 (git 밖)
├── tools/                   # calibration · training · scripts · voice
└── docs/
```

`services/planner`, `tools/{calibration,training,scripts}`, `data/`에는 `COLCON_IGNORE`가
있어 `colcon build`가 그 폴더를 스캔조차 하지 않는다. `planner`가 ROS2 패키지가 아닌 이유는
HTTP·LLM·DB 어디에도 실시간 토픽이 필요 없기 때문이다 — `control`로는 web을 거쳐서만
도달한다.

## 4. 사용한 장비 목록

### 4.1 로봇 · 제어

![장비 구성](https://raw.githubusercontent.com/Haeyuni/PiecePickingSystem/main/docs/diagrams/hardware_stack.png)

| 장비 | 모델 | 비고 |
|---|---|---|
| 협동로봇 | Doosan M0609 (6축) | 가반하중 6kg, 작업반경 900mm, 관절 토크센서 내장 |
| 그리퍼 | OnRobot RG2 | `/onrobot/sendCommand`(개폭·힘), `/onrobot/status`의 `Grip detected` 비트로 파지 판정 |
| 비전 | RealSense RGB-D | 손목 장착(eye-in-hand). `realsense2_camera`가 color/aligned depth/CameraInfo 발행 |
| 제어박스 | 두산 표준 제어박스 | 비상정지 하드와이어 직결 |
| 로봇 PC | Ubuntu 24.04 · RTX 4060 8GB | 웹·planner·ROS 노드·DB를 전부 구동 |
| 마이크 | 로봇 PC 연결 USB 마이크 | "hello rokey" 웨이크워드 감지용(`tools/voice/wakeword_bridge.py`) |

- 카메라가 그리퍼에 붙어 있어 좌표 변환이 매 프레임 TCP 자세에 따라 달라진다:
  `T_base_camera = posx_to_matrix(get_current_posx()) @ T_gripper2camera`.
- `T_gripper2camera`는 **컨트롤러에 TCP(`GripperDA_v1`)가 선택돼 있는 상태**로 풀려 있다.
  TCP가 풀리면 좌표 전체가 조용히 약 208mm 어긋난다 — 상태바의 TCP 표시가 그것을 본다.
  전말은 `docs/problem/2026-09-07-grasp-coordinate-offset.md`에 있다.

### 4.2 작업대 · 작업물

![작업 셀 배치](https://raw.githubusercontent.com/Haeyuni/PiecePickingSystem/main/docs/diagrams/cell_layout.png)

| 구성 | 값 | 비고 |
|---|---|---|
| 목적지 | `left_box`(왼쪽 박스) · `right_box`(오른쪽 박스) | 좌표·상단 내부 모서리 4점·바닥점을 실측해 `src/control/config/bins.yaml`에 고정 |
| 작업물 (YOLO 경로) | 치약 · 물티슈 · 선크림 · 토끼 인형 · 접이 우산 · 섬유탈취제 · 젤네일 | `models/best.pt`(yolo11n-seg, 7클래스)와 `src/perception/config/objects.yaml`이 1:1 |
| 작업물 (SAM+VLM 경로) | 제한 없음 | 등록 어휘를 보지 않고 사진을 본 VLM이 이름·속성·파지 단계를 함께 답한다 |

파지력은 힘(N)이 아니라 **5단계 `grip_level`**로 전달한다(1=가장 강하게 40N ~
5=가장 약하게 20N). 단계→힘 매핑은 `objects.yaml`과 `skill_params.yaml`이 소유하고,
LLM·VLM은 단계 번호만 고른다. 미확인 신규 클래스는 5로 강제한다.

## 5. 의존성

### 5.1 ROS 2 (rosdep)

각 패키지 `package.xml`에 선언돼 있다. 컨테이너로 돌리면 이미지가 알아서 챙기고,
호스트에서 직접 빌드할 때만 필요하다.

```bash
rosdep install --from-paths src --ignore-src -r -y
```

| 패키지 | 주요 의존 |
|---|---|
| `sort_msgs` | `rosidl_default_generators`, `std_msgs`, `geometry_msgs`, `sensor_msgs`, `builtin_interfaces` |
| `perception` | `rclpy`, `sensor_msgs`, `geometry_msgs`, `sort_msgs`, `perception_common` (+ ultralytics·torch는 pip) |
| `grasp` | `rclpy`, `sensor_msgs`, `geometry_msgs`, `sort_msgs`, `perception_common` |
| `control` | `rclpy`, `sensor_msgs`, `geometry_msgs`, `sort_msgs`, `perception_common`, `dsr_msgs2`, `onrobot_rg_msgs` |
| `web` | `rclpy`, `sort_msgs` (`MOCK_MODE=0`일 때만 로드) |
| `perception_common` | `sensor_msgs`, `dsr_msgs2` |

외부 워크스페이스(`~/cobot2_ws`)에서 온다: `dsr_msgs2`(`get_current_posx`·`ikin`·
`MovejH2r`/`MovelH2r` 등)와 `onrobot_rg_msgs`(`SetCommand`·`GripperPose`)는 이미지에
굽지 않고 `/doosan_ws`·`/onrobot_ws`로 읽기 전용 마운트한다 — 드라이버 버전이 올라가도
이미지를 다시 만들 이유가 없게. 드라이버 자체(`m0609_rg2_bringup`)는 컨테이너가 아니라
호스트에서 띄운다(6절 1단계).

### 5.2 서비스 — pip

<details>
<summary><code>services/planner/requirements.txt</code> · <code>src/web/requirements.txt</code> · <code>src/perception/requirements.txt</code></summary>

```text
# planner — ROS2 설치가 필요 없다
fastapi · uvicorn · openai · pydantic · python-multipart
psycopg[binary] · pyyaml · opencv-python-headless · numpy

# web — rclpy는 여기 없다(ros:jazzy 베이스가 제공)
fastapi · uvicorn[standard] · httpx · psycopg[binary]
opencv-python-headless · numpy · faster-whisper · python-multipart · PyYAML

# perception — 호스트 .venv 실측 조합을 그대로 고정
ultralytics==8.4.138 · torch==2.14.0 · torchvision==0.29.0
opencv-python==5.0.0.93 · numpy==2.5.2 · PyYAML==6.0.1 · httpx · psycopg[binary]
```

</details>

> `planner`는 rclpy 노드가 아니라 **HTTP 서비스**다. web을 거쳐서만 control에 도달하므로
> 백엔드 컨테이너에 ROS 2 설치가 필요 없다. `web`은 반대로 `MOCK_MODE=0`일 때만
> `ros_bridge.py`에서 rclpy를 import한다 — `main.py`와 `routers/`는 ROS2를 모른다.

### 5.3 프론트엔드 — `web/frontend/package.json`

```text
react 19.2 · react-dom 19.2 · react-router-dom 7.9
vite 7.1 · typescript 5.9 · @vitejs/plugin-react 5.0
```

## 6. 실행 순서

로봇 드라이버와 카메라는 **컨테이너 밖(호스트)에서 먼저** 떠 있어야 한다. `perception`·
`grasp`·`control`이 host 네트워크로 붙어 그 노드들과 DDS로 직접 통신한다.

### 0. 환경 파일 (최초 1회)

```bash
cp .env.example .env
```

| 값 | 설명 |
|---|---|
| `OPENAI_API_KEY` | LLM 계획 + VLM 라벨링 양쪽이 쓴다 |
| `POSTGRES_PASSWORD` | 로컬 생성: `openssl rand -base64 24` |
| `MOCK_MODE` | `1`이면 web이 ROS2 대신 `data/mock` 픽스처로 돈다 |
| `ROS_DOMAIN_ID` | 호스트 셸의 `echo $ROS_DOMAIN_ID`와 같아야 서로 보인다 |
| `PERCEPTION_DETECTOR` | `yolo`(기본) 또는 `vlm_sam` |
| `DOOSAN_WS_DIR` · `CYCLONEDDS_CONFIG_DIR` | 기본값이 아니면 **절대경로로** 지정 |

### 1. 로봇 드라이버 — 터미널 A

```bash
tools/scripts/run_bringup.sh start mode:=real host:=192.168.1.100
tools/scripts/run_bringup.sh status     # SCHED_FIFO 적용 여부까지 본다
```

`ros2 launch m0609_rg2_bringup bringup.launch.py`를 직접 치는 것과 같은 일을 하되,
`setsid`로 별도 세션에 띄워 **터미널의 Ctrl+C나 창 닫기로 드라이버가 죽지 않게** 한다.
끄는 것은 `run_bringup.sh stop`이다.

### 2. RealSense 카메라 — 터미널 B

```bash
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
```

`align_depth`는 필수다 — `grasp`가 color 픽셀 grid에 정렬된 depth로 포인트클라우드를
재구성한다.

### 3. 서비스 스택 — 터미널 C

```bash
FAKE_ROBOT=0 docker compose up -d db planner perception graspnet grasp control
```

`FAKE_ROBOT`은 `control`이 읽는다 — 기본값 `1`은 로봇 없이 도는 모의 실행이라, **실물에
붙일 때 `0`을 빠뜨리면 로봇이 움직이지 않는데 성공만 돌아온다.**

| 서비스 | 포트 | 비고 |
|---|---|---|
| `db` | 5432 | host 네트워크 컨테이너가 붙어야 해서 포트를 게시한다 |
| `planner` | 8100 | 헬스체크 통과 후 web이 뜬다 |
| `perception` | — | GPU. `PERCEPTION_DETECTOR`로 검출기를 고른다 |
| `graspnet` | 8200 | GPU. 모델 로드에 최대 180초 — `/health`가 200이면 준비 완료 |
| `grasp` | — | GPU. `GRASP_STRATEGY`(기본 `graspnet_baseline`)로 전략을 고른다 |
| `control` | — | 실물은 `FAKE_ROBOT=0` |

`graspnet`을 띄우지 않아도 `grasp`는 뜬다 — 그 관측만 후보 없이 건너뛰고 로그를 남긴다.
`heuristic_pca`로 돌릴 때는 아예 필요 없다.

### 4. 웹 — 터미널 D

```bash
docker compose --profile ros up -d web_ros
```

`web_ros`는 mock `web`과 8000 포트가 겹치므로 **둘을 동시에 띄우지 않는다.** 그래서
`profile`로 기본 기동에서 빼 두었다. 이미지가 프론트엔드 빌드 산출물을 함께 담고 있어
`http://localhost:8000`이 곧 UI다.

| 화면 | 주소 |
|---|---|
| 제어 화면 | http://localhost:8000/ |
| 이력 화면 | http://localhost:8000/history |
| 학습 데이터 화면 | http://localhost:8000/datasets |
| 헬스체크 | http://localhost:8000/health · http://localhost:8100/health |

### 5. (선택) 웨이크워드 브리지 — 터미널 E

```bash
python3 tools/voice/wakeword_bridge.py
```

로봇 PC 마이크로 "hello rokey"를 듣고 web에 알린다. 브라우저 SpeechRecognition은
사전에 없는 단어를 신뢰성 있게 못 잡아 이 방식으로 대체했다. 드라이버처럼 계속 떠 있는
프로세스이고 사용자가 직접 띄우고 끈다 — 참조하는 openwakeword 모델은 이 저장소가
관리하지 않는 로봇 PC 로컬 자산이라 자동 기동에 넣지 않았다.

### 실행 순서 요약

```text
① run_bringup.sh start (로봇 드라이버)  →  ② realsense2_camera
   →  ③ docker compose up -d db planner perception graspnet grasp control
   →  ④ docker compose --profile ros up -d web_ros
   →  ⑤ (선택) wakeword_bridge.py
```

<details>
<summary>로봇·GPU 없이 확인하기 (mock 모드)</summary>

```bash
cp .env.example .env                            # MOCK_MODE=1이 기본값
docker compose up -d --build db planner web     # GPU·ROS2가 필요한 서비스는 빼고 띄운다
python3 tools/scripts/check_web_e2e.py --url http://localhost:8000
```

서비스를 나열하지 않으면 `perception`/`grasp`/`graspnet`까지 함께 뜬다 — GPU도 로봇도 없는
자리에서는 셋 다 필요 없다.

프론트엔드만 고칠 때는 컨테이너 재빌드 없이 dev 서버를 쓴다 — `/api`·`/ws`는
`http://localhost:8000`으로 프록시된다.

```bash
cd web/frontend && npm install && npm run dev    # http://localhost:5173
```

단위 테스트는 계층마다 도는 자리가 다르다. `planner`는 런타임 이미지에 테스트 의존성을
더하지 않으려고 stdlib `unittest`만 쓰고, `control`/`grasp`는 rclpy와 `sort_msgs`가 필요해
컨테이너 안에서 돈다.

```bash
# planner — 컨테이너 안에 의존성이 이미 다 있다
docker compose exec planner python -m unittest discover -s tests -t .

# perception / perception_common — 호스트 .venv(pytest 포함)에서
cd src/perception && python -m pytest test -q
cd src/perception_common && python -m pytest test -q

# grasp / control — rclpy + sort_msgs가 갖춰진 컨테이너 안에서
docker exec ros2_ws-grasp-1 bash -c \
  'source /opt/ros/jazzy/setup.bash && source /ros2_ws/install/setup.bash &&
   cd /ros2_ws && python3 -m pytest src/grasp/test -q'

# 프론트엔드 — 타입체크까지 함께 돈다
cd web/frontend && npm run build
```

</details>

<details>
<summary>설정을 고쳤을 때 무엇을 다시 띄우는가</summary>

| 고친 것 | 필요한 조치 |
|---|---|
| `grasp_params.yaml` · `skill_params.yaml` · `bins.yaml` | 마운트되어 있다 — `docker compose restart grasp` 또는 `control` |
| `objects.yaml` | `planner`는 마운트라 `restart`면 되고, `perception`은 이미지에 굽혀 있어 `up -d --build perception` |
| `.py` | `docker compose up -d --build <서비스>` |
| `sort_msgs`의 `msg`/`action` | 해당 인터페이스를 쓰는 서비스 전부 재빌드 |
| `database/migrations/*.sql` | `planner`가 기동 시 적용한다 — `docker compose restart planner` |

</details>

## 7. 문서

| 문서 | 내용 |
|---|---|
| [`docs/BR(Business Requirement).md`](docs/BR%28Business%20Requirement%29.md) | 요구사항 정의 — 범위·FR/NFR·단계 로드맵 |
| [`docs/시스템명세서.md`](docs/시스템명세서.md) | DB 설계·기술 스택·하드웨어·배포 구성 |
| [`docs/인터페이스_정의서.md`](docs/인터페이스_정의서.md) | `sort_msgs` 메시지·액션·설정 파일 스키마 |
| [`docs/웹_인터페이스_정의서.md`](docs/웹_인터페이스_정의서.md) | 브라우저↔web↔planner의 HTTP/WebSocket 계약 |
| [`docs/화면정의서.md`](docs/화면정의서.md) | 세 화면의 구성 요소와 데이터 출처 |
| [`docs/개발계획.md`](docs/개발계획.md) | 개발 순서·결정 기록(D-1~D-8)·마일스톤 |
| [`docs/on-demand-perception.md`](docs/on-demand-perception.md) | 주기 발행 → 온디맨드 관측 전환 설계 |
| [`docs/vlm_sam_pipeline.md`](docs/vlm_sam_pipeline.md) | SAM+VLM 인지 파이프라인과 실측 결과 |
| [`docs/environment.md`](docs/environment.md) | 런타임·CUDA/PyTorch 버전 고정 근거 |
| [`docs/problem/`](docs/problem) | 실물에서 겪은 문제의 원인 분석 |
| [`docs/results/`](docs/results) | 단계별 실험 리포트 |
| [`docs/diagrams/`](docs/diagrams) | 위 그림 6장. `tools/scripts/make_diagrams.py`가 만든다 — 손으로 그리지 않는다 |

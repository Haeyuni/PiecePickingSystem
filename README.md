# PiecePickingSystem — 자연어 지시 기반 다품종 분류·피스피킹 시스템

## 0. 프로젝트 개요

- **목표**: 품목이 바뀔 때마다 재티칭·재프로그래밍하는 비용을 없애는 것.  
사람이 말이나 글로 내린 지시를 로봇이 그대로 해석해 물체를 골라 집고 지정한 곳에 놓는다.
- **주요 기능**: 도메인(가정·약국·재활용)을 고른 뒤 명령을 넣으면 관측 → LLM 계획 →
  검증 → 사용자 승인 → `pick` → `place_into`가 이어진다.  
  등록 어휘가 없는 물체도 SAM+VLM 경로로 인지하고 실행 결과와 라벨은 데이터셋·실행 로그로 쌓인다
- **사용 장비**: Doosan M0609 (6축 협동로봇), OnRobot RG2 그리퍼, RealSense RGB-D (eye-in-hand)
- **개발 환경**: Ubuntu 24.04 LTS, ROS 2 Jazzy, Docker Compose
- **주요 기술 스택**: ROS 2, FastAPI + rclpy 브리지, React (Vite/TypeScript), PostgreSQL 16, OpenAI API (LLM/VLM), YOLO11-seg · SAM2, GraspNet-baseline
- **기간**: 2026.09.01 ~ 2026.09.11

## 1. 시스템 설계

![시스템 아키텍처](https://raw.githubusercontent.com/Haeyuni/PiecePickingSystem/main/docs/diagrams/system_architecture.png?v=4)

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

- 외부 의존 토픽: `realsense2_camera`의 `/camera/color/image_raw`, `/camera/aligned_depth_to_color/image_raw`, `/camera/color/camera_info`

### 1.2 네트워크 구성

![네트워크 구성](https://raw.githubusercontent.com/Haeyuni/PiecePickingSystem/main/docs/diagrams/network.png?v=4)

웹·planner·ROS 노드·DB가 **로봇 PC 한 대**에서 전부 돈다.

### 1.3 물체 인지 방식

| 방식 | 어떻게 물체를 찾는가 | 한계 |
|---|---|---|
| `vlm_sam` | **기본값.** **SAM2**가 장면 전체를 조각내 번호를 붙이고, 그 프레임을 **VLM**(planner 경유)에 보내 번호마다 이름과 속성(무게·파손위험·변형·투명·파지 단계)을 받는다. 등록 어휘를 주지 않아 **처음 보는 물체도 인지**한다. 재관측(관측 B)은 VLM 없이 직전 위치를 박스 프롬프트로 SAM만 다시 돌린다 | 관측 한 번에 10초 안팎 걸리고 OpenAI API가 필요하다. 파지력을 정하는 속성이 모델 출력이라 실행 전 사용자 승인을 거친다 |
| `yolo` | **선택.** 학습한 **YOLO11-seg**(`models/best.pt`, 7클래스)가 컬러 프레임에서 인스턴스 마스크와 클래스를 낸다. 무게·파손위험·파지 단계는 클래스 이름으로 `objects.yaml`/DB에서 조회한다. 학습은 `tools/training/detector/train_yolo11n_seg.ipynb` | 학습한 7클래스 밖의 물체는 못 찾는다. 등록되지 않은 클래스는 가장 약한 파지(grip_level 5)로 강제된다 |

**파라미터 설정으로 인지 방식 변경 가능(default: `vlm_sam`)**

```bash
# 1) launch 인자
ros2 launch perception perception_launch.py detector:=yolo

# 2) 환경변수 (compose) — .env에 PERCEPTION_DETECTOR=yolo 로 둬도 된다
PERCEPTION_DETECTOR=yolo docker compose up -d perception
```

### 1.4 파지점 계산 전략

| 전략 | 어떻게 파지점을 찾는가 | 한계 |
|---|---|---|
| `graspnet_baseline` | **기본 전략.** 세그된 **물체 포인트클라우드**(마스크 영역만)를 상주 GPU 추론 서버(`graspnet` 서비스)에 HTTP로 보내 **파지 자세(접근·닫힘축)** 를 받는다. **위치와 개폭은 같은 클라우드로 실측해 다시 잡는다**. 자세는 `T_base_camera → T_camera_graspnet → T_graspnet_tcp`를 거쳐 base-mm TCP Pose로 한 번에 변환한다 | 장면 전체를 넘기면 GPU 사용량이 커져서 물체만 잘라 넣는다. 그래서 위치·폭 예측은 믿지 않고 자세만 쓰며 후보가 1~2곳으로 뭉친다. GPU와 checkpoint가 필요하다 |
| `heuristic_pca` | 기하 베이스라인. 물체의 **윗면 점들을 XY로 투영해 PCA**를 돌리고 **짧은 축 방향으로 손가락을 닫는다.** 긴 축을 가로질러 잡으면 개폭을 넘기거나 접촉면이 얕아 미끄러진다. 접근은 위에서 수직(base −Z) | 윗면이 수평이라고 가정하고 접근 방향을 −Z로 고정한다. 기울어진 물체와 클러터를 못 다룬다. |

**파라미터 설정으로 파지점 전략 변경 가능(default: `graspnet_baseline`)**

```bash
# 1) launch 인자
ros2 launch grasp grasp_launch.py strategy:=heuristic_pca

# 2) 환경변수 (compose)
GRASP_STRATEGY=heuristic_pca docker compose up -d grasp
```

```yaml
# 3) src/grasp/config/grasp_params.yaml — 기본값
strategy:
  name: graspnet_baseline     # heuristic_pca | graspnet_baseline
```

## 2. 플로우 차트

### 2.1 명령 처리 시퀀스

![명령 처리 시퀀스](https://raw.githubusercontent.com/Haeyuni/PiecePickingSystem/main/docs/diagrams/command_flow.png?v=4)

### 2.2 동작 순서도

![동작 순서도](https://raw.githubusercontent.com/Haeyuni/PiecePickingSystem/main/docs/diagrams/operation_flow.png?v=4)

실패했을 때의 갈래는 두 가지이며 서로 다르게 처리한다.

| 실패 지점 | 처리 | 상한 |
|---|---|---|
| `pick` 계열 | home 복귀 → 관측 A 다시 → 재계획 | 재계획 2회 |
| `place_into` | 같은 스텝을 새 `request_id`로 재전송(로봇이 이미 물체를 쥐고 있어 재계획할 것이 없다) | 재시도 2회 |

- 후보 선택의 최종 결정권은 `control`에 있다. `planner`는 점수 내림차순 후보 전체를
  실려 보내고 개폭 유효성·IK·관절 한계·접근 적합도는 로봇에 붙어 있어야 답할 수 있으므로
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
`~/cobot2_ws`에 있고 컨테이너는 거기서 메시지 패키지만 읽기 전용으로 마운트해 쓴다
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
├── models/                  # 로컬 모델 가중치 (git 밖) — sam2_b.pt · best.pt
├── tools/
│   ├── calibration/         # 핸드아이·TCP·박스 좌표 실측
│   ├── training/
│   │   ├── detector/        # YOLO11-seg 학습 — train_yolo11n_seg.ipynb (Colab, T4)
│   │   ├── auto_labeling/   # SAM 보조 라벨링·라벨 검증
│   │   └── grasp/           # GraspNet fine-tuning (예정)
│   ├── scripts/             # 점검·다이어그램 생성 스크립트
│   └── voice/               # 웨이크워드 브리지
└── docs/
```

`planner`가 ROS2 패키지가 아닌 이유는 HTTP·LLM·DB 어디에도 실시간 토픽이 필요 없기 때문이다.

## 4. 사용한 장비 목록

### 4.1 로봇 · 제어

![장비 구성](https://raw.githubusercontent.com/Haeyuni/PiecePickingSystem/main/docs/diagrams/hardware_stack.png?v=4)

| 장비 | 모델 | 비고 |
|---|---|---|
| 협동로봇 | Doosan M0609 (6축) | 가반하중 6kg, 작업반경 900mm, 관절 토크센서 내장 |
| 그리퍼 | OnRobot RG2 | `/onrobot/sendCommand`(개폭·힘), `/onrobot/status`의 `Grip detected` 비트로 파지 판정 |
| 비전 | RealSense RGB-D | 손목 장착(eye-in-hand). `realsense2_camera`가 color/aligned depth/CameraInfo 발행 |
| 제어박스 | 두산 표준 제어박스 | 비상정지 하드와이어 직결 |
| 로봇 PC | Ubuntu 24.04 · RTX 4060 8GB | 웹·planner·ROS 노드·DB를 전부 구동 |
| 마이크 | 노트북 내장 마이크 | "hello rokey" 웨이크워드 감지용(`tools/voice/wakeword_bridge.py`) |

- TCP가 풀리면 좌표 전체가 약 208mm 어긋난다(RG-2 그리퍼 기준)

### 4.2 작업대 · 작업물

![작업 셀 배치](https://raw.githubusercontent.com/Haeyuni/PiecePickingSystem/main/docs/diagrams/cell_layout.png?v=4)

## 5. 의존성

### 5.1 ROS 2 (rosdep)

각 패키지 `package.xml`에 선언돼 있다. 컨테이너로 돌리면 이미지가 알아서 챙기고
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
| `PERCEPTION_DETECTOR` | `vlm_sam`(기본) 또는 `yolo` |
| `DOOSAN_WS_DIR` · `CYCLONEDDS_CONFIG_DIR` | 기본값이 아니면 **절대경로로** 지정 |

모델 가중치는 git에 없으므로 직접 넣어 둔다.

| 파일 | 쓰는 곳 |
|---|---|
| `models/sam2_b.pt` | `perception` — `vlm_sam`(기본). planner와 `OPENAI_API_KEY`도 함께 필요하다 |
| `models/best.pt` | `perception` — `yolo`를 고를 때만 |
| `${PIECE_PICKING_ASSETS_DIR:-~/piece_picking_assets}/models/graspnet/checkpoint.tar` | `graspnet` — `graspnet_baseline`(기본) |

> SAM 가중치는 노드가 뜰 때가 아니라 **첫 관측 때** 올라간다. `sam2_b.pt`가 없어도 `perception`은
> 정상으로 떠 보이고 첫 명령에서 실패하므로, 새 환경에서는 관측 한 번을 먼저 확인한다.

### 1. 로봇 드라이버 — 터미널 A

```bash
tools/scripts/run_bringup.sh start mode:=real host:=192.168.1.100
tools/scripts/run_bringup.sh status     # SCHED_FIFO 적용 여부까지 본다
```

`ros2 launch m0609_rg2_bringup bringup.launch.py`를 직접 치는 것과 같은 일을 하되
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

`FAKE_ROBOT`은 `control`이 읽는다 — 기본값 `1`은 로봇 없이 도는 모의 실행이라 **실물에
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
프로세스이고 사용자가 직접 띄우고 끈다.

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
더하지 않으려고 stdlib `unittest`만 쓰고 `control`/`grasp`는 rclpy와 `sort_msgs`가 필요해
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
| [`docs/diagrams/`](docs/diagrams) | 위 그림 6장. `tools/scripts/make_diagrams.py`가 만든다 |

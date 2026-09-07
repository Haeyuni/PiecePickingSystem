# PiecePickingSystem
자연어 기반 다품종 분류 피스피킹 시스템

## Workspace layout

`/ros2_ws`에 복제하는 것을 기준으로 구성한다.

```text
/ros2_ws
├── src/                  # ROS2 패키지
│   ├── control/
│   ├── grasp/
│   ├── perception/
│   └── sort_msgs/
│   └── web/              # FastAPI 및 ROS2 브리지
├── web/
│   └── frontend/         # Vite UI
├── services/planner/     # ROS2와 분리된 계획 API
├── database/migrations/  # PostgreSQL 스키마
├── data/                 # 공용 런타임 데이터와 픽스처
├── models/               # 로컬 모델 가중치
├── tools/                # calibration, scripts, training
└── docs/
```

ROS2 패키지는 모두 `src/` 아래에 있다.

```bash
cd /ros2_ws
colcon build
```

컨테이너 서비스는 저장소 루트에서 실행한다.

```bash
docker compose up --build
```

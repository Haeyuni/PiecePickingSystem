# grasp

`world_state_raw` + `instance_masks` + depth → 물체별 포인트클라우드 → 파지 후보를 채워
`/world_state`로 발행한다. `/world_state`의 발행자는 이 노드 하나다(인터페이스_정의서 2.0절).

## 실행

compose로 띄운다. 전략은 환경변수로 고르며, 이미지 재빌드가 필요 없다.

```bash
FAKE_ROBOT=0 docker compose --profile ros up -d db planner perception graspnet grasp control web_ros
```

```bash
GRASP_STRATEGY=heuristic_pca docker compose up -d grasp   # 기본값은 graspnet_baseline
```

## 전략

| 이름 | 필요한 것 |
|---|---|
| `heuristic_pca` | 없음. 포인트클라우드만으로 계산한다 |
| `graspnet_baseline` | compose의 `graspnet` 서비스(GPU) + checkpoint |

`graspnet_baseline`은 **상주 추론 서버**에 HTTP로 보낸다 — `grasp_params.yaml`의
`graspnet_baseline.endpoint`(기본 `http://localhost:8200`)가 compose의 `graspnet` 서비스다.
grasp는 `network_mode: host`라 서비스 이름 DNS 대신 localhost로 부른다.

checkpoint는 저장소 밖 assets에 둔다. compose가 이 경로를 컨테이너의 `/checkpoint.tar`로
read-only 마운트한다(`docker-compose.yml`의 `graspnet` 서비스):

```text
${PIECE_PICKING_ASSETS_DIR:-$HOME/piece_picking_assets}/models/graspnet/checkpoint.tar
```

## 설정

`config/grasp_params.yaml`은 **마운트된다** — 값을 고치고 재빌드 없이 반영한다.

```bash
docker compose restart grasp
```

`.py`를 고쳤을 때만 `docker compose up -d --build grasp`.

주요 값은 파일 안 주석에 근거와 이력이 함께 있다. 특히:

* `refine_grasp_depth_mm` — 파지 깊이의 유일한 손잡이. 얕게 물면 여기를 올린다.
* `approach_angle_hard_max_deg` — 이 각도를 넘는 후보만 버린다(2026-09-08 재설계).
  **예전의 `approach_angle_threshold_deg`/`step`/`max_deg` 3종 확장 필터는 없앴다** —
  15도부터 넓히다 처음 통과자에서 멈추는 구조라 실제 상한이 30도가 아니라 "후보가
  하나라도 있는 가장 좁은 5도 구간"이었고, Top-K를 남기려는 의도와 어긋났다.
  기울어진 후보를 실제로 쓸지는 control이 정한다(개폭·IK·관절·안전 + 접근 적합도
  랭킹, `control/grasp_selection.py`). 여기를 좁힐수록 자세는 좋아지지만 후보가
  전멸하는 물체가 늘어난다.
* `camera_frame_offset_mm` — **0으로 두는 것이 맞다.** 올리기 전에 그 주석을 읽을 것.

좌표계·오프셋의 전체 그림은 `docs/problem/2026-09-07-grasp-coordinate-offset.md`에 있다.

## 좌표 변환

카메라가 그리퍼에 붙어 있어(eye-in-hand) 변환은 매 프레임 TCP 자세에 따라 달라진다.

```text
T_base_camera = posx_to_matrix(get_current_posx()) @ T_gripper2camera
```

`T_gripper2camera`(`data/calibration/T_gripper2camera.npy`)는 **`get_current_posx()`가
보고하는 TCP 정의**로 풀려 있다. 컨트롤러의 TCP가 바뀌거나 선택 해제되면 좌표 전체가
조용히 틀어진다 — 위 문서의 "원인 0" 참조.

`graspnet_baseline.T_graspnet_tcp_mm`은 GraspNet의 축 규약(X=접근, Y=닫힘)을 이 프로젝트의
TCP 규약(Z=접근, X=닫힘)으로 바꾸는 회전이다. 최종 후보는 다음으로 계산된다.

```text
T_base_tcp = T_base_camera @ T_camera_graspnet @ T_graspnet_tcp
```

## 진단 도구

`tools/calibration/`이 컨테이너에 `/tools`로 마운트되어 있다. rclpy + sort_msgs +
dsr_msgs2 + perception_common이 모두 갖춰진 유일한 곳이라 그대로 돈다.

```bash
docker exec ros2_ws-grasp-1 bash -c '
  source /opt/ros/jazzy/setup.bash && source /ros2_ws/install/setup.bash &&
  source /doosan_ws/share/dsr_msgs2/local_setup.bash &&
  python3 /tools/measure_tcp_offset.py'
```

| 도구 | 무엇을 재는가 |
|---|---|
| `measure_tcp_offset.py` | TCP 원점 ↔ 실제 파지점 (로봇 좌표만 사용) |
| `check_handeye_consistency.py` | hand-eye 이동 성분 보정량 |
| `touch_check.py` | perception 좌표 vs 실제 손끝 |

## 테스트

```bash
docker exec ros2_ws-grasp-1 bash -c \
  'source /opt/ros/jazzy/setup.bash && source /ros2_ws/install/setup.bash &&
   cd /ros2_ws && python3 -m pytest src/grasp/test -q'
```

# 온디맨드 인지 전환 설계안

주기 발행하던 `perception`을 **명령을 받은 순간에만 도는 구조**로 바꾸고, 검출기를 YOLO에서
SAM+VLM으로 교체한다. 파이프라인 자체의 실측 근거는 `vlm_sam_pipeline.md`에 있고, 이 문서는
그것을 런타임에 붙일 때 정해야 했던 것들을 남긴다.

## 1. 무엇이 바뀌고 무엇이 그대로인가

`perception` 노드는 **없어지지 않는다.** 없어지는 것은 셋뿐이다.

- YOLO 추론 (`models/best.pt`)
- `period_s=0.5` 타이머 (`create_timer`)
- everything 모드를 매 관측마다 도는 것 (→ 3절)

나머지는 그대로 남고, 남는 쪽이 훨씬 많다.

| 남는 것 | 없으면 안 되는 이유 |
| --- | --- |
| hand-eye 변환 + `get_current_posx` | **eye-in-hand다.** 카메라가 그리퍼에 붙어 있어 base 좌표를 만드는 유일한 경로 |
| mask → median depth → camera 3D → base 3D | `geometry.py`. SAM은 2D 마스크까지만 준다 |
| `min_depth_valid_ratio`(0.35) / `min_mask_pixels`(240) | 투명·반사 물체가 여기 걸려 `graspable=false`가 된다 |
| `attribute_db` 조회 + FR-05b fragile 강제 | 신규 클래스 보수적 처리 |
| `ObjectTracker` | 없애는 게 아니라 **이 설계의 핵심이 된다** (→ 3절) |
| `world_state_raw` + `instance_masks` 발행 계약 | 유지하면 **grasp 노드를 전혀 안 건드린다** |
| unknown crop 저장 | FR-05a |

즉 트리거와 검출기만 갈아끼우는 일이다.

## 2. 흐름

```text
명령 수신
  → home 이동 (시야 확보)
  → [관측 A: 전체 스캔]  SAM everything → 마크 → VLM 라벨링   ← VLM은 여기 1회뿐
  → 마스크 + depth + TCP 자세 → base 3D → world_state_raw
  → grasp: 파지 후보 → /world_state
  → planner: 물체 목록 + 지시 → 스킬 시퀀스 (없는 물체면 거부)
  → 스텝 실행
      ├ pick → place_into → home 자동 복귀
      └ [관측 B: 재투영 스캔]  이전 물체를 프롬프트로 SAM 1패스   ← VLM 안 부름
  → 남은 스텝 반복
```

**관측 A와 B는 다른 일이다.** A는 "무엇이 있는지 모르는 상태"에서 장면 전체를 훑고, B는
"직전에 무엇이 어디 있었는지 아는 상태"에서 그것들이 아직 있는지 확인한다. 아는 것이 다르니
같은 방법을 쓸 이유가 없다.

## 3. 결정 기록

### D-1. 재관측은 everything이 아니라 재투영 **박스** 프롬프트로 한다

관측 B에서 SAM everything을 다시 돌리지 않는다. 대신 **직전 관측이 남긴 물체의 3D 범위를
현재 TCP 자세로 카메라 평면에 재투영해 박스 프롬프트**로 준다. SAM은 이미지 임베딩 1회로
끝난다 — CPU 실측 26.6초 → **0.2초**.

**점이 아니라 박스인 것이 중요하다** (2026-09-07 실측). 마스크 중심을 점 프롬프트로 주면
SAM이 물체가 아니라 그 위의 무늬 조각을 잡는다:

| 프롬프트 | 물티슈 | 섬유탈취제 | 치약 |
| --- | --- | --- | --- |
| 중심점 1개 | 0.073 | 0.028 | 0.885 |
| 마스크 내부 9점 | 0.037 | 0.979 | 0.873 |
| **박스** | **0.988** | **0.978** | **0.886** |

(직전 마스크와의 IoU. 박스는 물체당 0.05초.) 점 하나는 "이 부분"인지 "이 물체"인지 모호하다.
그래서 직전 관측이 중심 좌표만이 아니라 **마스크 테두리를 각자의 depth로 역투영한 base 점들**을
남긴다(`mask_utils.mask_extent_3d`) — 그것을 되돌리면 박스가 된다. 범위를 못 남긴 물체는
중심점으로 대신하지 않고 이번 관측에서 뺀다.

everything을 다시 돌리지 않는 이유는 셋이다.

1. **속도.** everything 모드는 CPU에서 26초였다(`vlm_sam_pipeline.md`). GPU 실측은 아직
   없지만, `orchestrator._wait_for_fresh_observation`의 타임아웃이 5.0초이고 재계획
   경로(`orchestrator.py:149`)는 1.0초다. everything이 그보다 느리면 재관측이 통째로
   타임아웃된다.
2. **매칭이 공짜가 된다.** 프롬프트가 곧 "이 물체"이므로 새 마스크와 옛 라벨을 이어붙이는
   문제가 애초에 생기지 않는다 (D-2가 이것 때문에 사라진다).
3. **eye-in-hand에 강하다.** 재투영은 현재 TCP 자세를 쓰므로 시점이 바뀌어도 성립한다.
   2D IoU 매칭은 home 자세에서만 유효한데, 관측이 항상 home인 것은 아니다
   (`orchestrator.py:218`은 home 복귀와 무관한 신선도 대기다).

### D-2. `ObjectTracker`의 `class_name` 매칭 조건

원래 문제: `assign()`은 클래스가 같아야 매칭하는데(`tracker.py`), 재관측 검출에는 클래스가
없다 — 라벨을 알아야 매칭되고 매칭돼야 라벨을 아는 순환.

**D-1을 채택하면 이 순환은 사라진다.** 프롬프트를 어느 `object_id`로 만들었는지 알고 있으므로
매칭 결과가 이미 정해져 있다. 그래도 tracker는 남긴다 — 재투영이 빗나가거나(물체가 밀렸다)
프롬프트가 엉뚱한 것을 잡았을 때 **3D 60mm 검사로 걸러내는 안전망**으로 쓴다. 거리가
임계값을 넘으면 같은 물체로 잇지 않고 사라진 것으로 본다.

### D-3. 매칭 안 되는 새 마스크는 시퀀스 대상에서 뺀다

관측 B에서 프롬프트에 대응하지 않는 영역이 생길 수 있다 — 가려졌다 드러난 물체, 떨어뜨린
물체. 라벨이 없다.

**미확인으로 두고 그 명령의 시퀀스 대상에서 제외한다.** 다음 명령 때 관측 A가 본다.
지시하지 않은 물체를 옮기는 것이 못 옮기는 것보다 나쁘다는 원칙(`llm_client` 프롬프트 7-1,
`vlm_detect` 프롬프트 2번)과 같은 판단이다.

### D-4. 트리거는 서비스가 아니라 액션

관측 A는 SAM + API 왕복으로 명령당 10초 안팎이다. 서비스 호출로는 진행 표시도 취소도 안 된다.
액션으로 만들고, `_wait_for_fresh_observation`의 5.0초 / 1.0초 타임아웃은 관측 A/B의
실측값에 맞춰 다시 잡는다.

### D-5. UI는 탐지 물체 목록 대신 캡처 사진 + 마스크 + 그립 모양을 보여준다

**새로 만들 것이 거의 없다.** `/grasp/debug_image`가 이미 그 셋을 그린다 —
캡처된 컬러 프레임 위에 마스크 윤곽(`grasp/node.py:504`), 그리퍼 팜과 양 손가락
(`:562~566`), 전략·점수 라벨. base→카메라 역투영도 `_draw_grasp_candidates`에 구현돼 있다.

| | 할 일 |
| --- | --- |
| 없앤다 | `ros_bridge._on_world_state`의 `on_event` 푸시(`:196`)와 프론트의 탐지 물체 목록 |
| 유지 | `latest_world_state` 자체 — orchestrator/planner가 계획에 쓴다. UI에만 안 보낸다 |
| 추가 | web이 `/grasp/debug_image`를 JPEG로 중계 (카메라 뷰에서 이미 하는 일) |

`InstanceMasks`의 "grasp만 구독하는 내부용" 계약은 그대로 지켜진다 — web이 받는 것은
마스크가 아니라 **이미 그려진 이미지**다.

화면에는 최신 한 장만 띄우고, 스텝별로 되짚는 것은 이력 화면에서 한다.

### D-6. 캡처 사진은 파일로 남기고 경로만 DB에 넣는다

"왜 이걸 집었나"를 나중에 따지려면 그 순간의 그림이 필요하다. 이미지를 DB에 넣지 않는다 —
`data/observations/<trace_id>/<stamp>.png`에 파일로 두고 경로만 기록한다.

`execution_logs`에는 지금 그런 컬럼이 없다(`001_init.sql`). 마이그레이션 `002`에
`observation_image_path text`를 추가한다. `task_sequences.world_state_snapshot`과는
`trace_id`로 이어진다.

### D-7. `grasp`의 `debug_color_max_age_s=15.0`

주기 발행을 전제로 잡힌 값이라 온디맨드에서 다시 봐야 한다. 관측 A는 SAM+VLM에 10초가
걸리므로, 그 사이 컬러 프레임 버퍼가 비면 debug 이미지가 안 그려진다.

### D-8. 라벨링이 끝나면 다시 찍는다

perception은 `world.stamp`에 **촬영 시각**을 박고, grasp는 그 stamp로 depth 프레임을 찾는다
(`max_depth_age_s=0.5`, `DEPTH_BUFFER_SIZE=60` ≈ 2초). SAM+VLM이 10초 걸리면 발행 시점에
stamp가 이미 10초 전이라 짝이 되는 depth가 버퍼에서 밀려나 있다 — `_blockers`가 막고 경고
로그만 남긴 채 `/world_state`가 안 나간다. **온디맨드 관측이 전부 조용히 버려진다.**

그래서 관측 A는 두 프레임을 쓴다.

```text
프레임 A 촬영 → everything → 마크 → VLM 라벨링 (10초)
  → 프레임 B 새로 촬영 → 라벨별 재투영 박스로 SAM 1패스 (0.2초)
  → B의 stamp로 발행
```

로봇은 home에서 멈춰 있어 장면이 같으므로 라벨이 그대로 옮겨간다. 발행되는 관측이 항상
0.5초 이내가 되어 **grasp는 무수정**이고, 두 번째 패스는 D-1의 재관측 코드를 그대로 쓴다.

## 4. 구현 현황 (2026-09-07)

D-1~D-8 전부 코드로 들어갔다. **하드웨어(로봇·카메라·GPU)가 없어 오프라인으로 검증
가능한 것까지만 확인했다** — 개발 계획의 단계 구분과 검증 방법은
`services/planner`·`tools/scripts/vlm_sam_test.py`·이 파일의 커밋 이력을 참조.

| 파일 | 실제로 한 일 | 검증 |
| --- | --- | --- |
| `src/perception/perception/detectors/{base,yolo,sam_vlm,sam_reprompt}.py` | 검출기 경계 신설. `sam_vlm`=관측 A(everything+VLM), `sam_reprompt`=관측 B(재투영 박스 SAM 1패스) | pytest 27건, `vlm_sam_test.py` 리팩터 전후 결과 동일 |
| `src/perception/perception/sam_marks.py` | SAM 관련 로직 승격(everything/박스 프롬프트/조각 합치기). `vlm_sam_test.py`와 노드가 같은 코드를 쓴다 | 위와 동일 |
| `src/perception/perception/node.py` | `create_timer` 제거 → `Observe` 액션 서버. D-8(관측 A 후 재촬영), `_save_observation_image`(D-6) | py_compile + pyflakes만 (rclpy 없음 — ⚠️ `colcon build` 미검증) |
| `src/perception_common/perception_common/geometry.py` | `camera_from_base`/`project_to_pixel` 추가 (역투영) | 왕복 오차 무작위 자세 2000개, <1e-6mm |
| `src/perception/perception/tracker.py` | **변경 없음.** 애초 계획은 `class_name` 매칭 조건을 완화할 생각이었지만, 실제 구현(재투영 박스가 라벨을 그대로 상속)에서는 재관측 검출도 항상 class_name을 채워서 나온다 — tracker의 기존 매칭 조건이 그대로 성립해 순환 자체가 생기지 않았다 |  |
| `src/sort_msgs/action/Observe.action` | 신설 (`MODE_FULL`/`MODE_REPROMPT`, phase 5종) | `CMakeLists.txt` 등록만 — ⚠️ 빌드 미검증 |
| `services/planner/src/app.py` | `POST /internal/label-marks` 신설. 구현은 `vlm_detect.py`(⚠️ `vlm_client.py`와 다른 파일 — FR-05a용 별개 스텁) | 단위 테스트 8건 + 실제 VLM 호출 1회(스모크) |
| `src/web/web/executor.py` / `executor_mock.py` / `ros_bridge.py` | `observe(trace_id, mode)` 추가. mock은 stamp만 갱신, ROS는 `Observe` 액션 클라이언트(`_send_goal` 공용화) | `check_web_e2e.py`로 전체 흐름 확인(아래) |
| `src/web/web/orchestrator.py` | `_wait_for_fresh_observation`을 수동 폴링 → 능동 트리거로. 명령 진입 시 `mode="full"`, grasp-후보-재시도와 스텝 사이는 `mode="reprompt"` | 동일 |
| `web/frontend/` | `ObjectList.tsx` 삭제, `PendingConfirmations.tsx` 신설(확인 모달 진입점 이관), `CameraViews.tsx`에 "n초 전 관측" 표시 | `npm run build` 통과 |
| `database/migrations/002_observation_image.sql` | `execution_logs.observation_image_path` 컬럼 추가 | — |
| `src/grasp/grasp/node.py` | **변경 없음, 계획대로.** `debug_color_max_age_s`도 안 건드렸다 — D-8이 world.stamp를 항상 신선하게 유지하므로 grasp 자체 컬러 버퍼(15초, ~30fps)가 항상 그보다 최근 프레임을 갖고 있어 여유를 늘릴 이유가 없었다 |  |

**바로잡은 것 — `_object_bottom_offset_mm`은 손대지 않는다.** 처음엔 스텝 사이
재관측(mode="reprompt")의 결과로 `_execute_steps`의 `world_state`/`class_map`을 갱신할
생각이었다. 그런데 그 함수가 쓰는 `grasp_pose`는 이미 planner 응답 시점에
`trace["steps"]`에 고정돼 있고, `_object_bottom_offset_mm`도 world_state가 아니라
**직전 pick 스텝의 기록**에서 파지 z를 가져온다 — 이 계산이 필요로 하는 것은 물체가
집히기 **전**(테이블 위)의 높이/위치이므로, 재관측(이미 집힌 뒤라 테이블에 없다)으로
world_state를 갈아끼우면 오히려 정보를 잃는다. 그래서 스텝 사이 트리거는 D-5/D-6의
관측 이미지·`executor.get_latest_world_state()` 갱신용으로만 쓰고, 그 결과로
`world_state`/`class_map`을 재대입하지 않는다(`orchestrator.py`의 해당 주석 참조).

**아직 안 한 것**:
- `execution_logs.observation_image_path`를 실제로 채우는 배선. 컬럼만 만들었다 — 한
  관측이 여러 스킬 실행(pick+place)에 걸쳐 재사용되는 경우가 있어 "어느 로그 행에
  어느 사진을 붙일지"를 먼저 정해야 한다.
- `perception_launch.py`/`docker-compose.yml`에 `detector`/`SAM_MODEL`/`PLANNER_URL`
  파라미터를 연결했다(완료). 처음에는 기본값을 `yolo`로 남겨 뒀지만, 2026-09-10에
  실제 운영에 맞춰 **기본값을 `vlm_sam`으로 바꿨다** — `yolo`는 `PERCEPTION_DETECTOR=yolo`로
  고르는 선택 경로다.

## 5. 같이 고쳐야 할 문서

- ~~화면정의서 2.2.4~~ / ~~2.2.5~~ — **완료.** 2.2.4는 관측 뷰(마스크+파지 후보+관측
  시각)로 개정, 2.2.5(탐지 물체 목록)는 삭제하고 확인 모달 진입점 이관을 명시했다.
- ~~인터페이스_정의서 2.0절~~ — **완료.** 발행 계기가 타이머→액션으로 바뀌었다고 적고,
  4.4절에 `Observe.action` 전체를 문서화했다.
- ~~웹_인터페이스_정의서~~ — **완료.** `world_state` 이벤트를 4절에 명시(계속 쓴다 —
  객체 목록이 아니라 stamp만), `/internal/label-marks`를 5.4절에 추가, 아키텍처
  다이어그램(1절)에 `observe` 액션 흐름 추가.
- **시스템명세서** — perception의 검출기·주기 관련 서술은 아직 안 건드렸다. ⚠️ 남음.

## 6. 아직 실측되지 않은 것

- **GPU에서의 SAM everything 시간.** CPU 26초만 재 봤다. D-4의 타임아웃을 정하려면 필요하다.
- **재투영 프롬프트의 정확도.** 물체가 밀렸을 때 프롬프트가 얼마나 빗나가는지, D-2의 60mm
  안전망이 적절한 값인지.
- **YOLO와의 마스크 IoU.** "YOLO를 대체할 수 있는가"를 수치로 말하려면 같은 프레임 비교가
  필요하다.
- **오염되지 않은 프레임에서의 재측정** (`vlm_sam_pipeline.md`의 "테스트 사진의 오염").
- **`orchestrator.py`의 `OBSERVE_FULL_TIMEOUT_S`(60초)/`OBSERVE_REPROMPT_TIMEOUT_S`(10초)는
  추정값이다.** CPU 실측(everything 26~40초, 재투영 박스 0.2초)에 여유를 얹었을 뿐 GPU
  실측이 아니다 — 위 항목이 채워지면 같이 낮춘다.
- **`Observe.action`이 실제 ROS2 환경에서 빌드·구동되는지.** 이 세션은 rclpy가 없는
  환경이라 `py_compile`/`pyflakes`로만 확인했다. `docker compose build perception` +
  `ros2 action list`가 남아 있다.

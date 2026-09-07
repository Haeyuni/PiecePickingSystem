# movel 간헐 정지 · 사이클마다 느려짐 · heartbeat 알람 — H2R goal 스레드 누수

- 날짜: 2026-09-06
- 영향 범위: `dsr_controller2`(벤더 드라이버 포크), `src/control/control/dsr_motion.py`
- 상태: 해결. 실물 4사이클 연속 정상 동작으로 검증.

---

## 1. 증상

며칠에 걸쳐 따로따로 조사하던 세 가지가 사실 **하나의 원인**이었다.

1. **간헐적 movel 정지.** pick/place의 어느 단계에서든 무작위로, 로봇이 이동 도중
   멈칫하거나 명령을 받고도 아예 안 움직였다. 재현 조건을 못 찾았다.
2. **실행할수록 느려짐.** 드라이버를 재시작한 직후부터 사이클마다 거의 규칙적으로
   증가했다 — pick 14.9→22.6→31.3초, place 45.1→58.3→59.5초. control 컨테이너를
   재시작해도 안 줄었고, **드라이버를 재시작해야만** 초기화됐다.
3. **컨트롤러 알람** `A heartbeat packet was not received for 5 seconds`
   (DRFL 알람 index 1041). 한 세션에서 **77번** 찍혔다.

곁다리로 같이 관측되던 것들 — 전부 같은 원인이었다.

- `controller_manager: Overrun detected!` 가 계속 뜨고, 그 안의 `Read time` 이
  **정확히 10ms 또는 20ms** 였다(제어 주기 10ms의 정수배).
- `get_current_posx` 가 movel 직후 10~20초씩 무응답. perception/grasp가 robot_pose를
  못 받아 파이프라인이 멈췄다.
- 액션 result 통지가 사실상 매번 유실돼서, control이 `get_current_posx` 로 도착을
  직접 확인하는 우회 경로(`on_timeout_verify`)로만 성공 처리되고 있었다.

---

## 2. 결정적 증거

로봇이 **20분째 완전히 정지**해 있고 control이 아무 goal도 보내지 않은 상태에서:

```
$ ros2 topic hz /dsr01/motion/movel_h2r/_action/feedback
average rate: 100.011
average rate:  99.999
```

15:20에 보낸 movel goal 하나의 실행 스레드가 **11분 뒤에도 100Hz로 돌고 있었다.**
`/proc/<pid>/task` 로 확인하니 그 시각에 생성된 스레드가 그대로 살아 CPU를 쓰고 있었다.

과거 로그의 goal 종결 집계도 같은 이야기를 한다.

| 세션 | 기간 | Movel 수락 | 도착 | 취소 | **종결 안 됨(누수)** |
|---|---|---|---|---|---|
| `822823` (수정 전) | 409초 | 31 | 5 | 0 | **26** |
| `828299` (수정 전) | 2225초 | 2 | 0 | 1 | **1** |

`822823` 세션은 3사이클 남짓 도는 동안 **26개의 스레드를 영구히 남겼고**, 같은 세션에
heartbeat 알람(index 1041)이 **77번** 찍혔다.

---

## 3. 근본 원인

### 3.1 드라이버: 실행 스레드가 빠져나갈 길이 두 개뿐

`dsr_controller2.cpp` 의 `handle_accepted_movel_h2r` / `handle_accepted_movej_h2r` 는
goal 하나마다 `std::thread{...}.detach()` 로 실행 스레드를 띄운다. 그 100Hz 루프는

- `goal_handle->is_canceling()` 이거나
- 현재 위치가 목표의 허용오차(movel 0.3, movej 0.1) 안에 들어왔을 때

**만** `return` 한다. 클라이언트가 결과를 안 기다리고 떠나면 스레드는 **프로세스가 죽을
때까지** 돈다. 벤더 코드에 goal 중복 검사도, 타임아웃도, `is_active()` 검사도 없었다.

### 3.2 control: 도착을 확인하면 취소 없이 그냥 떠났다

`dsr_motion.py` 의 `call_action_blocking` 은 이랬다.

```python
if on_timeout_verify():
    logger.info("on_timeout_verify로 도착 확인, 성공 처리")
    return True, None          # ← remote_handle을 취소하지 않고 반환
```

이 환경은 액션 result 통지가 거의 매번 유실돼서 **이 경로가 사실상 모든 이동의 정상
종료 경로**였다. 즉 **이동 한 번마다 드라이버에 100Hz 스레드가 하나씩 영구히 쌓였다.**
pick 3회 + place 5회 = 사이클당 약 8개. 관측된 선형 증가와 정확히 맞는다.

### 3.3 남은 스레드가 실제로 하는 일

```cpp
{ lock(get_drfl_mutex()); cur_pos = Drfl->get_current_pose(...); }   // 10ms마다
{ lock(get_drfl_mutex()); Drfl->hold2run(); }                        // 10ms마다
```

`get_drfl_mutex()` 는 **하나뿐인** 전역 뮤텍스다. 다음 셋이 같이 쓴다.

- `DRHWInterface::read()` — controller_manager의 **100Hz RT 루프**
- `get_current_posx_cb` 등 aux_control 서비스 — perception/grasp/control이 호출
- H2R 실행 스레드들

그래서 **누수된 스레드 하나마다 초당 200번의 DRFL 왕복이 그 뮤텍스 위에 얹힌다.**
여기서 세 증상이 전부 나온다.

- `read()` 가 뮤텍스에서 대기 → `Read time` 이 DRFL 왕복 주기의 정수배(10/20ms) → Overrun
- `get_current_posx` 가 뒤로 밀림 → 10~20초 무응답 → perception/grasp 정지
- DRFL 명령 채널 포화 → 컨트롤러가 heartbeat를 제때 못 받음 → **알람 1041**
- 스레드가 사이클마다 쌓임 → **실행할수록 느려짐**

### 3.4 그리고 "간헐적 movel 정지"

남은 스레드는 **자기 옛 목표**와 현재 위치를 계속 비교한다. 팔이 나중에 그 지점
근처(0.3 이내)를 지나가면 `is_arrived` 가 참이 되어

```cpp
{ lock(get_drfl_mutex()); Drfl->stop(STOP_TYPE_QUICK); }
```

를 **진행 중인 다른 모션 한가운데서** 쏜다. pick/place는 접근 지점을 몇 번씩 되짚어
지나가므로, 단계와 무관하게 무작위로 터졌다. 재현 조건을 못 찾았던 이유가 이것이다 —
"이번 이동"이 아니라 **몇 분 전에 버려진 goal**이 원인이었다.

---

## 4. 수정 내용

### 4.1 `src/control/control/dsr_motion.py` — 누수의 발생 지점

`_release_remote_goal()` 를 추가하고, 결과를 기다리지 않고 빠져나가는 모든 경로에서
원격 goal을 **반드시 취소**하게 했다.

- `on_timeout_verify` 성공 경로 (거의 모든 이동의 정상 종료 경로)
- 수락 응답이 늦게 도착해 handle이 생긴 경로

취소 확인 대기는 **1.5초로 짧게** 뒀다. 이 함수는 매 이동마다 도는 정상 경로라
여기서 오래 기다리면 그게 곧 사이클 시간이 된다. 중요한 것은 취소를 *보내는* 것이지
확인을 *받는* 것이 아니다 — 확인이 안 와도 드라이버가 다음 goal에서 정리한다(4.2).

### 4.2 `dsr_controller2.cpp` / `.hpp` — 구조적으로 못 쌓이게

H2R 모션이 **한 번에 하나만** 살아 있도록 만들었다. movej/movel이 같은 팔을 움직이므로
둘이 공유하고, 듀얼암에서 서로를 선점하지 않도록 컨트롤러 인스턴스 멤버로 뒀다.

| 장치 | 역할 |
|---|---|
| `h2r_generation_` (`std::atomic<uint64_t>`) | 새 H2R goal마다 증가. 자기 세대가 아닌 스레드는 모션을 멈추고 abort. 클라이언트가 취소를 안 보내도, 취소가 유실돼도 **다음 goal이 반드시 앞 스레드를 정리한다.** |
| `h2r_exec_mutex_` | 앞 스레드가 `stop()` 까지 마치고 빠져나온 뒤에야 다음 모션이 시작된다. 두 모션이 겹쳐 나가지 않게. |
| `!goal_handle->is_active()` 검사 | 이 스레드 밖에서 종료된 goal이면 즉시 정리. |
| watchdog 180초 (`kH2rWatchdogSec`) | 위 셋이 다 안 걸려도 스레드가 무한정 남지 않게 하는 최후의 상한. |

추가로 feedback 발행을 100Hz → **20Hz** (`kH2rFeedbackDiv = 5`)로 낮췄다. 소비자는
control의 도착 확인 하나뿐이고 그쪽 판정 주기가 1초라 100Hz가 필요 없다. 이 셀은
loopback DDS 포화가 액션 응답 유실의 원인으로 이미 지목돼 있다.

### 4.3 곁다리 — SCHED_FIFO

`chrt` prefix는 **애초에 필요 없었다.** `ros2_control_node` 가 자기 RT 스레드에 이미
스스로 SCHED_FIFO 50을 시도하고, 실패하면 경고만 남기고 정상 기동한다.

```
Spawning controller_manager RT thread with scheduler priority: 50
Could not enable FIFO RT scheduling policy: ... (Operation not permitted)
```

막고 있던 건 프로세스의 `RLIMIT_RTPRIO` 뿐이다. `/etc/security/limits.d/90-dsr-control-rt.conf`
(`rokey - rtprio 80`)는 이미 있고 `pam_limits` 도 `/etc/pam.d/gdm-password` 에 있으므로,
**로그아웃 후 다시 로그인하면** 적용된다. 확인은 새 터미널에서 `ulimit -r` 이 80인지 보면 된다.

과거에 launch에 `chrt -f` prefix를 넣었다가 되돌린 이력이 있는데, 권한이 없을 때 chrt가
실패하면서 **드라이버가 통째로 안 뜨는** 훨씬 나쁜 결과가 났기 때문이다. 다시 넣지 말 것.

> 참고: 이번 Overrun은 스케줄러 선점이 아니라 **뮤텍스 대기**가 원인이었다. RT 우선순위는
> 이 문제의 해결책이 아니다.

---

## 5. 검증 결과 (실물 4사이클)

드라이버 세션 `875258`, 2026-09-06 16:05~16:16.

### 사이클 시간 — 증가가 사라졌다

| 사이클 | pick | place |
|---|---|---|
| 1 | 15.43초 | 31.70초 |
| 2 | 13.42초 | 28.99초 |
| 3 | 13.22초 | 28.56초 |
| 4 | 14.41초 | 30.52초 |

수정 전 pick 14.9→22.6→31.3 / place 45.1→58.3→59.5 와 비교. **단조 증가가 없고**
±1초 노이즈만 남았다. place는 절대값도 45~60초 → 29~32초로 줄었다.

### goal 종결 — 누수 0

| 항목 | 수정 전 (`822823`, 3사이클) | 수정 후 (`875258`, 4사이클) |
|---|---|---|
| Movel 수락 | 31 | 40 |
| └ 도착 | 5 | 2 |
| └ 취소 | 0 | 38 |
| └ **종결 안 됨** | **26** | **0** |
| Movej 수락 / 도착 | 4 / 4 | 4 / 4 |
| Overrun | 173 | 54 |
| Skip dt | 525 | 74 |
| **알람 1041 (heartbeat)** | **77** | **0** |

수정 후 알람은 index 3205/3206(특이점 영역 진입/이탈 알림, level 1) 16건뿐이다 —
movel의 정상적인 경로 알림이고 heartbeat와 무관하다.

`superseded` 0건, watchdog 0건. control 쪽 취소가 항상 먼저 도착해서 드라이버의
선점 장치는 한 번도 발동할 필요가 없었다 — 의도한 대로 순수 보험으로 남았다.

### 부수 지표

- 4사이클 완료 후 로봇 정지 상태에서 `ros2 topic hz .../movel_h2r/_action/feedback` → **출력 없음** (누수 스레드 0)
- control 로그: 실패·타임아웃·`get_current_posx 무응답` **0건**
- perception/grasp: 드라이버 기동 이후 `get_current_posx 응답이 없어` **0건**
- 취소 확인이 1.5초를 넘긴 경우 **0건**

---

## 6. 확인 방법 (다음에 의심될 때)

로봇이 **멈춰 있는 상태**에서:

```bash
# 1) 누수 스레드 — 아무것도 안 나와야 정상
ros2 topic hz /dsr01/motion/movel_h2r/_action/feedback

# 2) goal 종결 집계 — 수락 == 도착 + 취소 여야 정상
L=$(ls -t ~/.ros/log/ros2_control_node_*.log | head -1)
echo "수락 $(grep -c 'Received goal request for MovelH2r' $L)" \
     "도착 $(grep -c 'MovelH2r goal arrived' $L)" \
     "취소 $(grep -c 'MovelH2r goal canceled' $L)"

# 3) heartbeat 알람 — 1041 이 있으면 재발
grep -A3 'callback OnLogAlarm' $L | grep 'index :' | grep -o '[0-9]*$' | sort | uniq -c
```

`tools/scripts/run_bringup.sh status` 로도 볼 수 있다.

---

## 7. 함께 발견한 별개 문제 — bringup 중복 기동

증상 조사 도중, **bringup이 두 개 떠 있으면** 아래가 일어나는 것을 실물로 확인했다.
위 스레드 누수와는 **별개 원인**이므로 헷갈리지 말 것.

- `ros2_control_node` 둘이 같은 `/dsr01` 네임스페이스로 같은 실 컨트롤러에 붙는다.
- 같은 이름의 서비스 서버가 둘이 되어 `get_current_posx` 요청이 갈라지고 **응답이 아예
  안 온다.** perception/grasp가 robot_pose를 못 받아 정지 → web이
  `월드 상태가 오래되었습니다 (31.5s 전, perception 확인 필요)` 로 명령을 거부한다.
- DRCF는 RT control 채널과 access control을 한 클라이언트에게만 준다. 두 번째가
  빼앗고 그게 죽으면 살아남은 쪽은 채널이 끊긴 채 남는다 — `joint_states` 조차
  발행이 멈추고 **SIGINT도 SIGTERM도 무시**해서 SIGKILL로만 정리됐다.
- 드라이버 로그에 남는 흔적:
  `ResourceManager has already loaded a urdf`,
  `Controller 'dsr_controller2' can not be configured from 'active' state`.

로그만 보면 "서비스가 그냥 죽은 것"처럼 보여서 알아채기 어렵다. 위 두 줄이 보이면
`pgrep -f '^[^ ]*/controller_manager/ros2_control_node'` 로 **개수부터** 세어볼 것.

launch 파일에 중복 기동을 거부하는 검사를 넣었다가, 과하다는 판단으로 제거했다.
**드라이버는 사람이 직접 하나만 띄우는 것을 규칙으로 한다.**

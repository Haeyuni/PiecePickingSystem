#!/usr/bin/env bash
# M0609 로봇 드라이버(bringup) 기동/중지 도우미.
#
# 손으로 `ros2 launch m0609_rg2_bringup bringup.launch.py ...` 를 치는 것과 같은 일을
# 하지만, 실물 운용에서 반복해서 문제가 됐던 두 가지를 함께 처리한다.
#
# 1) **실시간 우선순위(SCHED_FIFO)**
#    controller_manager 의 ros2_control_node 는 자기 RT 스레드에 **이미 스스로**
#    SCHED_FIFO 50 을 걸려고 시도한다(로그: "Spawning controller_manager RT thread with
#    scheduler priority: 50"). 실패하면 경고만 남기고 SCHED_OTHER 로 그대로 뜬다.
#    즉 `chrt -f` 같은 prefix 는 필요 없다 — 2026-09-06 에 launch 에 chrt prefix 를
#    넣었다가, 권한이 없을 때 chrt 가 그냥 실패해서 **드라이버가 통째로 안 뜨는** 훨씬
#    나쁜 결과를 봤다(그래서 되돌렸다).
#    실제로 막고 있던 건 프로세스의 RLIMIT_RTPRIO 였다. /etc/security/limits.d/
#    90-dsr-control-rt.conf 에 `rokey - rtprio 80` 이 들어 있고 pam_limits 도 켜져
#    있으므로(/etc/pam.d/gdm-password), **한 번 로그아웃했다 로그인하면** 그 세션의
#    모든 터미널에서 `ulimit -r` 이 80 이 되고 위 내장 시도가 그냥 성공한다.
#    이 스크립트는 기동 전에 그 상태를 확인해서 알려주고, 안 돼 있어도 **평소대로**
#    (SCHED_OTHER 로) 기동한다 — 우선순위 때문에 로봇이 안 뜨는 일은 다시 없어야 한다.
#
# 2) **터미널 Ctrl+C 로부터 격리**
#    bringup 을 터미널의 포그라운드 잡으로 띄우면 그 창에 Ctrl+C 가 한 번 들어가는
#    순간 드라이버가 통째로 죽는다(2026-09-06 15:05 에 실제로 발생 — launch 로그에
#    "user interrupted with ctrl-c (SIGINT)" 가 남았고 로봇 연결이 끊겼다).
#    여기서는 setsid 로 **별도 세션/프로세스 그룹**에 띄우므로 그 창에서 무슨 키를
#    눌러도, 창을 닫아도 드라이버는 살아 있다. 끄려면 `run_bringup.sh stop`.
#
# 사용법:
#   tools/scripts/run_bringup.sh start [launch 인자...]   # 기본: mode:=real host:=192.168.1.100
#   tools/scripts/run_bringup.sh stop
#   tools/scripts/run_bringup.sh status
#   tools/scripts/run_bringup.sh logs
set -uo pipefail

COBOT_WS="${COBOT_WS:-$HOME/cobot2_ws}"
ROS_DISTRO_SETUP="${ROS_DISTRO_SETUP:-/opt/ros/jazzy/setup.bash}"
RUN_DIR="${RUN_DIR:-$HOME/.dsr_bringup}"
PID_FILE="$RUN_DIR/bringup.pid"
# start 가 띄운 ros2_control_node 의 PID. stop 이 자기 것만 정리하려고 남긴다.
CM_PID_FILE="$RUN_DIR/control_node.pid"
LOG_FILE="$RUN_DIR/bringup.log"
DEFAULT_ARGS=(mode:=real host:=192.168.1.100)

# RT 우선순위 요구치. controller_manager 가 내부적으로 거는 값과 같다 — 이보다 낮으면
# 어차피 setscheduler 가 EPERM 이라 굳이 올릴 이유가 없다.
REQUIRED_RTPRIO=50

msg()  { printf '%s\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*" >&2; }
err()  { printf '\033[31m%s\033[0m\n' "$*" >&2; }
ok()   { printf '\033[32m%s\033[0m\n' "$*"; }

running_pid() {
    [ -f "$PID_FILE" ] || return 1
    local pid; pid=$(cat "$PID_FILE" 2>/dev/null) || return 1
    [ -n "$pid" ] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    printf '%s' "$pid"
}

# rtprio soft 한도를 hard 한도까지 올린다. 올릴 수 없어도 실패로 취급하지 않는다 —
# 그 경우 ros2_control_node 는 경고 한 줄 남기고 SCHED_OTHER 로 정상 기동한다.
prepare_rtprio() {
    local soft hard
    soft=$(ulimit -S -r 2>/dev/null || echo 0)
    hard=$(ulimit -H -r 2>/dev/null || echo 0)
    [ "$hard" = "unlimited" ] && hard=99
    [ "$soft" = "unlimited" ] && soft=99

    if [ "${hard:-0}" -ge "$REQUIRED_RTPRIO" ] 2>/dev/null; then
        if [ "${soft:-0}" -lt "${hard:-0}" ] 2>/dev/null; then
            ulimit -S -r "$hard" 2>/dev/null || true
        fi
        ok "rtprio 한도 $(ulimit -S -r) — controller_manager 가 SCHED_FIFO 50 을 잡을 수 있다."
        return
    fi

    warn "rtprio 한도가 ${soft} 이라 SCHED_FIFO 를 못 쓴다 — SCHED_OTHER 로 기동한다(동작에는 지장 없음)."
    warn "  고치는 법: /etc/security/limits.d/90-dsr-control-rt.conf 는 이미 있다(rokey - rtprio 80)."
    warn "            pam_limits 가 로그인할 때만 적용되므로 **로그아웃 후 다시 로그인**하면 된다."
    warn "            확인: 새 터미널에서 'ulimit -r' 이 80 으로 나오면 성공."
}

cmd_start() {
    local pid
    if pid=$(running_pid); then
        err "이미 떠 있다 (PID $pid). 다시 띄우려면 먼저 '$0 stop'."
        return 1
    fi
    if [ ! -f "$COBOT_WS/install/setup.bash" ]; then
        err "$COBOT_WS/install/setup.bash 가 없다 — COBOT_WS 를 확인할 것."
        return 1
    fi

    mkdir -p "$RUN_DIR"
    prepare_rtprio

    local args=("$@")
    [ ${#args[@]} -eq 0 ] && args=("${DEFAULT_ARGS[@]}")

    msg "기동: ros2 launch m0609_rg2_bringup bringup.launch.py ${args[*]}"
    # setsid: 이 터미널의 프로세스 그룹 밖에 둔다 → 창의 Ctrl+C 나 창 닫기로 안 죽는다.
    setsid bash -c "
        source '$ROS_DISTRO_SETUP'
        source '$COBOT_WS/install/setup.bash'
        exec ros2 launch m0609_rg2_bringup bringup.launch.py ${args[*]}
    " >"$LOG_FILE" 2>&1 &
    local launcher_pid=$!
    printf '%s' "$launcher_pid" > "$PID_FILE"
    disown "$launcher_pid" 2>/dev/null || true

    sleep 3
    if ! running_pid >/dev/null; then
        err "기동 직후 죽었다. 로그: $LOG_FILE"
        tail -20 "$LOG_FILE" >&2
        return 1
    fi
    # stop 이 고아를 자기 것만 정리할 수 있도록 자식 ros2_control_node 의 PID 를 남긴다.
    pgrep -P "$launcher_pid" -f 'ros2_control_node' > "$CM_PID_FILE" 2>/dev/null || rm -f "$CM_PID_FILE"
    ok "기동됨 (PID $launcher_pid). 로그: $LOG_FILE"
    msg "  로그 보기: $0 logs        중지: $0 stop"
}

cmd_stop() {
    local pid
    if ! pid=$(running_pid); then
        msg "떠 있지 않다."
        rm -f "$PID_FILE"
        reap_own_control_node
        return 0
    fi
    msg "중지 중 (PID $pid)..."
    # ros2 launch 는 SIGINT 로 자식들을 순서대로 정리한다. 프로세스 그룹 전체에 보낸다.
    kill -INT -- "-$pid" 2>/dev/null || kill -INT "$pid" 2>/dev/null
    for _ in $(seq 1 20); do
        running_pid >/dev/null || { rm -f "$PID_FILE"; reap_own_control_node; ok "중지됨."; return 0; }
        sleep 1
    done
    warn "SIGINT 로 안 끝났다 — SIGTERM."
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
    sleep 5
    running_pid >/dev/null && { kill -KILL -- "-$pid" 2>/dev/null; sleep 1; }
    rm -f "$PID_FILE"
    reap_own_control_node
    ok "중지됨."
}

# launch 가 죽어도 그 아래 ros2_control_node 가 살아남는 경우가 있다 — DRFL 연결이
# 꼬이면 SIGINT 는 물론 SIGTERM 까지 무시한다(2026-09-06 실물 확인: SIGKILL 로만
# 정리됐다). 그 고아가 남으면 로봇 연결을 계속 붙들고 있으므로 stop 은 정말 없어진
# 것까지 확인하고 끝나야 한다.
#
# **우리가 start 로 띄운 PID 만 건드린다.** 이름으로 훑어서 죽이면 사람이 따로 띄운
# 드라이버까지 같이 죽는다 — 이 스크립트는 자기가 띄운 것만 책임진다.
reap_own_control_node() {
    [ -f "$CM_PID_FILE" ] || return 0
    local cm; cm=$(cat "$CM_PID_FILE" 2>/dev/null)
    rm -f "$CM_PID_FILE"
    [ -n "$cm" ] || return 0
    kill -0 "$cm" 2>/dev/null || return 0
    warn "launch 는 끝났는데 우리가 띄운 ros2_control_node(PID $cm)가 남아 있다 — 정리한다."
    kill -TERM "$cm" 2>/dev/null
    for _ in $(seq 1 10); do
        kill -0 "$cm" 2>/dev/null || { ok "정리됨."; return 0; }
        sleep 1
    done
    warn "SIGTERM 무시 — SIGKILL (PID $cm)"
    kill -KILL "$cm" 2>/dev/null
    sleep 2
    kill -0 "$cm" 2>/dev/null && err "여전히 남아 있다 — 수동 확인 필요." || ok "정리됨."
}

cmd_status() {
    local pid
    if ! pid=$(running_pid); then
        msg "bringup(이 스크립트가 띄운 것): 떠 있지 않다."
    else
        msg "bringup: PID $pid, 기동 후 $(ps -o etime= -p "$pid" | tr -d ' ')"
    fi

    local cm
    cm=$(pgrep -f '^[^ ]*/controller_manager/ros2_control_node' | head -1)
    if [ -z "$cm" ]; then
        msg "ros2_control_node: 없음"
        return 0
    fi
    msg "ros2_control_node: PID $cm, 스레드 $(ls /proc/$cm/task 2>/dev/null | wc -l)개"
    # RT 스레드가 실제로 FIFO 로 떴는지. 하나라도 FIFO 면 성공이다.
    if ps -L -o cls= -p "$cm" 2>/dev/null | grep -q FF; then
        ok "  스케줄링: SCHED_FIFO 적용됨"
    else
        warn "  스케줄링: 전부 SCHED_OTHER (rtprio 한도 미적용 — 재로그인 필요)"
    fi
    # H2R 실행 스레드가 남아 있는지 보는 가장 확실한 방법: 로봇이 멈춰 있는데도 액션
    # feedback 이 계속 나오면 실행 스레드가 아직 도는 것이다(정상이면 모션 중에만 나온다).
    msg "  실행 스레드가 남았는지 확인(로봇이 멈춰 있을 때만 의미 있음):"
    msg "    ros2 topic hz /dsr01/motion/movel_h2r/_action/feedback   # 아무것도 안 나와야 정상"
}

case "${1:-}" in
    start)  shift; cmd_start "$@" ;;
    stop)   cmd_stop ;;
    status) cmd_status ;;
    logs)   tail -f "$LOG_FILE" ;;
    *)      msg "사용법: $0 {start [launch 인자...]|stop|status|logs}"; exit 1 ;;
esac

#!/usr/bin/env python3
"""로봇만으로 **TCP 원점 ↔ 실제 파지점**의 차이를 잰다 (툴 좌표계 mm).

**무엇을 재는가.** 두산 컨트롤러가 `get_current_posx`로 보고하고 `movel`로 명령하는
TCP 원점은, RG2 손가락이 실제로 물체를 무는 지점과 다르다. 파이프라인은 "손가락이 여기서
만나라"를 계산해 놓고 그 자리에 TCP 원점을 보내므로, 이 차이만큼 손끝이 빗나간다.
2026-09-07 실물에서 이 값은 접근축으로 약 18mm였고, 그래서 그리퍼가 물체 위 허공에서
닫혔다.

**왜 perception을 안 쓰는가.** 물체 좌표와 대보는 방식(touch_check.py)은 "TCP 오프셋"과
"perception 오차"가 **한 숫자에 섞여** 나온다 — 둘 다 툴 좌표계 상수라 한 자세에서는
구분되지 않는다. 여기서는 로봇의 자기 좌표만 쓴다. 카메라도, 물체 인식도, 티칭한
좌표값도 필요 없다. 답은 순수한 기구 상수 하나다.

**원리 (산업 로봇의 4점 TCP 캘리브레이션과 같다).** 공간에 고정된 점 하나를 서로 다른
자세로 여러 번 짚는다. 손끝이 매번 같은 점 P에 있으므로, 자세 i마다

    R_i @ g + t_i = P          (g = 툴 좌표계 오프셋, R_i·t_i = 그때의 TCP 자세/위치)

가 성립한다. 미지수는 g(3) + P(3) = 6개, 자세 하나가 식 3개를 주므로 **자세를 2개만
넘겨도 풀린다**(3~4개면 잔차로 신뢰도까지 본다). 자세들의 **회전이 서로 충분히 달라야**
한다 — 위치만 옮기고 각도가 같으면 R_i가 전부 같아 g와 P가 분리되지 않는다.

**쓰는 법.**
  1) 작업대에 **뾰족하고 고정된 기준점**을 둔다(볼트 머리, 각진 블록 모서리 등).
     움직이지 않기만 하면 되고, 그 점의 좌표는 몰라도 된다.
  2) 그리퍼를 **완전히 닫는다**. 닫힌 두 손끝이 만나는 그 지점이 곧 파지점이다.
  3) 닫힌 손끝을 기준점에 살짝 댄다. 그 상태로 가만히 있으면 표본이 하나 기록된다.
  4) **자세(각도)를 바꿔** 같은 점을 다시 짚는다. 3~4번 반복한다.
     기울기를 20~40도씩 바꿀수록 결과가 정확해진다.

    python3 measure_tcp_offset.py                 # 자세가 안정되면 자동 기록
    python3 measure_tcp_offset.py --samples 5
    python3 measure_tcp_offset.py --interactive   # Enter를 칠 때만 기록 (docker exec -it 필요)

**이 스크립트는 로봇에 아무 명령도 보내지 않는다.** posx를 읽기만 한다. 조그는 사람이 한다.

**결과를 어디에 넣는가.** `src/control/config/skill_params.yaml`의

    tool:
      grasp_center_offset_mm: {x: .., y: .., z: ..}

에 그대로 옮기고 `docker compose restart control`. 재빌드는 필요 없다(설정은 마운트된다).
"""
import argparse
import math
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node

from dsr_msgs2.srv import GetCurrentPosx

# perception_common과 **같은** ZYZ 해석을 써야 한다. 여기서 각도를 다르게 읽으면 결과가
# hand-eye 캘리브레이션과 다른 좌표계의 값이 되어, 넣는 순간 조용히 틀어진다.
from perception_common.geometry import posx_to_matrix

POSX_SERVICE = "/dsr01/dsr_controller2/aux_control/get_current_posx"


def solve_offset(poses: list[list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """자세 목록 → (g, P, 자세별 잔차). 최소제곱.

        R_i g - P = -t_i     →     [R_i | -I] [g; P] = -t_i

    를 자세마다 3줄씩 쌓아서 푼다.
    """
    rows, rhs = [], []
    for posx in poses:
        matrix = posx_to_matrix(posx)
        rows.append(np.hstack([matrix[:3, :3], -np.eye(3)]))
        rhs.append(-matrix[:3, 3])
    A = np.vstack(rows)
    b = np.hstack(rhs)
    solution, *_ = np.linalg.lstsq(A, b, rcond=None)
    g, point = solution[:3], solution[3:]
    residuals = np.array([
        np.linalg.norm(posx_to_matrix(posx)[:3, :3] @ g + posx_to_matrix(posx)[:3, 3] - point)
        for posx in poses])
    return g, point, residuals


def solve_axis_only(poses: list[list[float]]) -> tuple[float, np.ndarray]:
    """툴 z축 성분 하나만 미지수로 두고 푼다 (g = (0, 0, gz)).

    두 손가락이 대칭으로 닫히는 그리퍼라면 파지점은 **툴 축 위**에 있으므로 x/y는 0이어야
    한다. 자유해의 x/y가 크게 나오면 짚은 지점이 손끝 중앙이 아니었거나 자세 변화가
    부족했다는 신호다 — 이 제약해와 견줘 보라고 함께 낸다.
    """
    rows, rhs = [], []
    for posx in poses:
        matrix = posx_to_matrix(posx)
        rows.append(np.hstack([matrix[:3, 2:3], -np.eye(3)]))
        rhs.append(-matrix[:3, 3])
    solution, *_ = np.linalg.lstsq(np.vstack(rows), np.hstack(rhs), rcond=None)
    return float(solution[0]), solution[1:]


def approach_spread_deg(poses: list[list[float]]) -> float:
    """자세들의 **접근축(툴 z)** 이 서로 얼마나 벌어져 있는지(도).

    **회전차만 봐서는 안 되는 이유.** 손목을 툴 축 중심으로 스핀만 하면 회전행렬은 크게
    달라지지만 접근축은 그대로다. 그런데 찾으려는 오프셋은 (대칭 그리퍼라면) 그 축 위에
    있어서, `R_i @ g`가 전혀 변하지 않는다 — z 성분이 방정식에서 통째로 사라진다.
    즉 스핀 4번은 회전차 100도가 넘어도 **아무것도 못 푼다**. 축이 벌어져야 한다.
    """
    worst = 0.0
    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            za = posx_to_matrix(poses[i])[:3, 2]
            zb = posx_to_matrix(poses[j])[:3, 2]
            cos = max(-1.0, min(1.0, float(np.dot(za, zb))))
            worst = max(worst, math.degrees(math.acos(cos)))
    return worst


def conditioning(poses: list[list[float]]) -> float:
    """최소제곱 행렬의 최소 특이값. 0에 가까우면 해가 정해지지 않는다(퇴화).

    위 접근축 검사보다 일반적이다 — 어떤 이유로든 미지수가 분리되지 않으면 여기 걸린다.
    """
    rows = [np.hstack([posx_to_matrix(p)[:3, :3], -np.eye(3)]) for p in poses]
    return float(np.linalg.svd(np.vstack(rows), compute_uv=False)[-1])


def rotation_spread_deg(poses: list[list[float]]) -> float:
    """자세들 사이의 최대 회전 차이(도)."""
    worst = 0.0
    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            ra = posx_to_matrix(poses[i])[:3, :3]
            rb = posx_to_matrix(poses[j])[:3, :3]
            cos = max(-1.0, min(1.0, (float((ra.T @ rb).trace()) - 1.0) / 2.0))
            worst = max(worst, math.degrees(math.acos(cos)))
    return worst


class Reader(Node):
    def __init__(self):
        super().__init__("measure_tcp_offset")
        self._client = self.create_client(GetCurrentPosx, POSX_SERVICE)

    def posx(self, timeout_s: float = 3.0) -> list[float] | None:
        if not self._client.wait_for_service(timeout_sec=timeout_s):
            return None
        future = self._client.call_async(GetCurrentPosx.Request(ref=0))
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
        result = future.result()
        if result is None or not result.success or not result.task_pos_info:
            return None
        return [float(v) for v in result.task_pos_info[0].data[:6]]


def different_enough(pose: list[float], seen: list[list[float]], min_deg: float) -> bool:
    """**접근축**이 충분히 벌어졌을 때만 새 표본으로 본다 (스핀은 표본이 아니다)."""
    return all(approach_spread_deg([pose, old]) >= min_deg for old in seen)


def collect_auto(node: Reader, target: int, min_deg: float, still_s: float,
                 timeout_s: float) -> list[list[float]]:
    """TCP가 멈춰 있고 이전 표본들과 자세가 충분히 다르면 한 표본으로 기록한다.

    Enter 입력을 쓰지 않는 이유는 `docker exec`가 TTY 없이 불리는 경우가 많아서다
    (프로젝트의 다른 진단 스크립트도 그렇게 실행한다).
    """
    poses: list[list[float]] = []
    still_since: float | None = None
    previous: list[float] | None = None
    deadline = time.monotonic() + timeout_s
    announced = False

    while len(poses) < target and time.monotonic() < deadline:
        pose = node.posx()
        if pose is None:
            time.sleep(0.3)
            continue
        moved = 9e9 if previous is None else float(np.linalg.norm(
            np.array(pose[:3]) - np.array(previous[:3])))
        turned = 9e9 if previous is None else rotation_spread_deg([pose, previous])
        previous = pose

        if moved > 0.3 or turned > 0.2:      # 아직 움직이는 중
            still_since = None
            announced = False
            time.sleep(0.25)
            continue

        if not different_enough(pose, poses, min_deg):
            if not announced and poses:
                print(f"  (툴을 {min_deg:.0f}도 이상 **기울여** 주세요 — 손목 스핀은 "
                      "접근축이 그대로라 표본이 되지 않습니다)", flush=True)
                announced = True
            time.sleep(0.4)
            continue

        if still_since is None:
            still_since = time.monotonic()
            print(f"  정지 감지 — {still_s:.0f}초 유지하면 기록합니다...", flush=True)
        elif time.monotonic() - still_since >= still_s:
            poses.append(pose)
            print(f"  [{len(poses)}/{target}] 기록: "
                  f"({pose[0]:.1f}, {pose[1]:.1f}, {pose[2]:.1f}) "
                  f"ZYZ=({pose[3]:.1f}, {pose[4]:.1f}, {pose[5]:.1f})", flush=True)
            still_since = None
            announced = False
            print("  다음 자세로 옮겨 같은 점을 다시 짚어 주세요.", flush=True)
        time.sleep(0.3)
    return poses


def collect_interactive(node: Reader, target: int) -> list[list[float]]:
    poses: list[list[float]] = []
    while len(poses) < target:
        try:
            answer = input(f"[{len(poses) + 1}/{target}] 손끝을 기준점에 댄 뒤 Enter "
                           "(q=중단): ")
        except EOFError:
            print("\n표준입력이 없습니다 — --interactive 없이(자동 모드로) 다시 실행하거나 "
                  "docker exec에 -it를 주세요.")
            break
        if answer.strip().lower() == "q":
            break
        pose = node.posx()
        if pose is None:
            print("  posx를 못 읽었습니다. 로봇 드라이버를 확인하고 다시 시도하세요.")
            continue
        poses.append(pose)
        print(f"  기록: ({pose[0]:.1f}, {pose[1]:.1f}, {pose[2]:.1f}) "
              f"ZYZ=({pose[3]:.1f}, {pose[4]:.1f}, {pose[5]:.1f})")
    return poses


def report(poses: list[list[float]]) -> int:
    print(f"\n표본 {len(poses)}개")
    for i, pose in enumerate(poses):
        print(f"  자세{i}: ({pose[0]:8.1f}, {pose[1]:8.1f}, {pose[2]:8.1f}) "
              f"ZYZ=({pose[3]:7.1f}, {pose[4]:7.1f}, {pose[5]:7.1f})")

    spread = rotation_spread_deg(poses)
    tilt = approach_spread_deg(poses)
    smallest = conditioning(poses)
    print(f"\n자세 간 최대 회전차   = {spread:.1f}도")
    print(f"접근축(툴 z) 최대 벌어짐 = {tilt:.1f}도   <- 이쪽이 핵심이다")
    print(f"최소제곱 최소 특이값     = {smallest:.3f}")
    if tilt < 10.0:
        print("\n**접근축이 거의 안 벌어졌다 — 결과가 무의미하다.**")
        print("손목을 축 중심으로 돌리기만(스핀) 하면 회전차는 커져도 접근축은 그대로라,")
        print("툴 축 방향 오프셋이 방정식에서 사라진다. **툴을 기울여** 다시 재라")
        print("(수직 자세에서 ry를 180도에서 20~40도 벗어나게).")
        return 1
    if smallest < 0.2:
        print("\n**해가 잘 정해지지 않는다(퇴화에 가깝다).** 자세를 더 다양하게 "
              "— 특히 접근축 방향을 더 크게 벌려 다시 재라.")
        return 1

    g, point, residuals = solve_offset(poses)
    gz, point_axis = solve_axis_only(poses)

    print(f"\n기준점 P (base, mm)   = ({point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f})")
    print(f"자세별 잔차 (mm)       = {np.array2string(residuals, precision=2)}")
    print(f"  최대 {residuals.max():.2f}mm / RMS {float(np.sqrt((residuals ** 2).mean())):.2f}mm")

    print("\n" + "=" * 78)
    print("자유해 (x/y/z 모두 미지수):")
    print(f"  grasp_center_offset_mm = {{x: {g[0]:.1f}, y: {g[1]:.1f}, z: {g[2]:.1f}}}")
    print("툴 축 제약해 (대칭 그리퍼라면 이쪽이 물리적으로 맞다):")
    print(f"  grasp_center_offset_mm = {{x: 0.0, y: 0.0, z: {gz:.1f}}}")
    print("=" * 78)

    lateral = float(np.hypot(g[0], g[1]))
    if residuals.max() > 3.0:
        print(f"\n잔차 {residuals.max():.1f}mm — 매번 **같은 점**을 짚지 못했을 가능성이 크다.")
        print("기준점을 더 뾰족한 것으로 바꾸고, 손끝이 닿는 순간에서 멈춰 다시 재라.")
    elif lateral > 4.0:
        print(f"\n자유해의 가로 성분이 {lateral:.1f}mm다. 대칭 그리퍼의 파지점은 툴 축 위에 "
              "있어야 하므로,")
        print("짚은 지점이 손끝 중앙에서 벗어났을 수 있다. 두 해가 크게 다르면 제약해를 쓰라.")
    else:
        print(f"\n두 해가 잘 맞는다(가로 성분 {lateral:.1f}mm, 잔차 최대 "
              f"{residuals.max():.1f}mm). 제약해를 그대로 쓰면 된다.")

    print("\n넣을 곳: src/control/config/skill_params.yaml 의 tool.grasp_center_offset_mm")
    print("적용:   docker compose restart control   (설정은 마운트되어 재빌드 불필요)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=4, help="기록할 자세 수 (최소 2, 권장 3~4)")
    ap.add_argument("--min-deg", type=float, default=15.0,
                    help="이전 표본들과 이 각도 이상 달라야 새 표본으로 인정")
    ap.add_argument("--still-s", type=float, default=2.0, help="이만큼 멈춰 있으면 기록")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--interactive", action="store_true", help="Enter를 칠 때만 기록")
    args = ap.parse_args()

    rclpy.init()
    node = Reader()
    print(__doc__.split("**쓰는 법.**")[1].split("**이 스크립트는")[0].strip())
    print()
    try:
        if args.interactive:
            poses = collect_interactive(node, args.samples)
        else:
            poses = collect_auto(node, args.samples, args.min_deg, args.still_s, args.timeout)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if len(poses) < 2:
        print(f"\n표본이 {len(poses)}개뿐이라 풀 수 없다. 서로 다른 자세로 최소 2개가 필요하다.")
        return 1
    return report(poses)


if __name__ == "__main__":
    sys.exit(main())

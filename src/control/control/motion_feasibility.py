"""모션 실행 가능성 — 끝점 IK 하나가 아니라 **실제로 이어지는 경로**를 본다 (STEP 2).

지금까지는 후보 하나를 현재 solution space에서 한 번 ikin으로 물어보고, 안 되면 그 후보를
버렸다. 그런데 "이 자세로 못 간다"의 실제 뜻은 대부분 **"이 관절 자세로는 못 간다"**이지
"이 파지가 불가능하다"가 아니다. 같은 파지를 다른 관절 configuration으로, 또는 다른 경로로
가면 되는 경우가 많다.

여기서 제공하는 것:

1. `solution_space_order` — 현재 space 우선, 그다음 드라이버가 지원하는 나머지
   (dsr_msgs2/Ikin.srv 실측: `int8 sol_space  # solution space : 0 ~ 7`).
2. `find_ik` — (자세 표현 × solution space) 조합을 순서대로 보고 **첫 성공**을 돌려준다.
   `find_ik_group`은 여러 지점(접근+파지)을 **같은 space에서 함께** 푸는 조합을 찾는다.
3. `path_feasible` — 두 끝점 사이를 몇 개 지점으로 나눠 IK·관절 연속성·명백한 충돌을 본다.
   막힌 곳이 끝점인지 중간인지를 `endpoint_failure`로 구분해 돌려준다 — 중간이면 우회
   경유점으로 풀 수 있고, 끝점이면 그 자세 자체에 해가 없어 경유점으로도 못 푼다.
4. `routes` — direct 우선, 안 되면 소수의 우회 경로(상승→회전→이동 등). pick의 "현재 →
   접근" 구간과 place의 안전고도 이송이 **같은 함수**를 쓴다.
5. `ik_cache` — 한 번의 계획 안에서 같은 (posx, space) 질의를 재사용한다.

**MoveIt을 쓰지 않는다.** 여기서 하는 것은 "명백히 안 되는 것을 거른다"이지 경로 계획이
아니다. 애매하면 **막지 않는다** — 정보가 없어서 못 여는 문은 여기서 닫지 않는다.

이 모듈은 ROS2를 import하지 않는다. IK 질의는 호출부가 콜러블로 넘긴다
(grasp_selection.py와 같은 이유 — 로봇 없이 규칙 전체를 테스트할 수 있어야 한다).
"""
import math
from dataclasses import dataclass, field

# dsr_msgs2/srv/Ikin.srv에서 직접 확인한 값이다(설치본 /doosan_ws):
#     int8 sol_space   # solution space : 0 ~ 7
# 추측으로 범위를 만들지 않는다 — 드라이버가 바뀌면 여기가 아니라 그 파일을 다시 본다.
SOLUTION_SPACES = tuple(range(8))

# 경로를 몇 개 지점으로 나눠 볼지(양 끝 포함). 3이면 시작·중간·끝이다. 늘리면 촘촘해지지만
# ikin 호출이 그만큼 늘어난다(실측 약 31ms/회).
DEFAULT_PATH_SAMPLES = 3
# 인접 지점 사이 관절이 이보다 크게 튀면 그 구간에서 configuration이 갈린 것으로 본다.
# **근거 없는 초기값이다** — 실물에서 멀쩡한 경로가 막히면 올리고, 팔이 크게 휘돌면 내린다.
DEFAULT_MAX_JOINT_JUMP_DEG = 90.0


@dataclass(frozen=True)
class IkOption:
    """찾아낸 실행 가능한 관절 해 하나."""

    posx: list                 # 실제로 명령할 TCP [x,y,z,rx,ry,rz]
    sol_space: int
    verdict: object            # dsr_motion.IkVerdict
    representation: str = "original"   # 어느 자세 표현이었는지 (로그용)

    @property
    def posj(self):
        return getattr(self.verdict, "posj", None)


@dataclass
class PathCheck:
    """경로 하나의 검사 결과."""

    feasible: bool
    reason: str = ""
    checked: int = 0
    max_joint_jump_deg: float | None = None
    unknown: int = 0           # ikin 무응답으로 판단 못 한 지점 수
    # **끝점이 막힌 것인지 중간이 막힌 것인지.** 끝점이 막혔으면 우회 경유점으로도 못
    # 푼다(그 자세 자체에 해가 없다). 중간이 막힌 것이라면 돌아서 가면 되므로 호출부가
    # 후보를 버리는 대신 경유점을 넣어 다시 본다 — 이 구분이 없으면 둘 다 똑같이
    # "이 후보는 안 된다"가 돼 멀쩡한 파지가 사라진다.
    endpoint_failure: bool = False
    # 끝점은 멀쩡한데 **중간 보간 지점**의 IK만 안 풀린 횟수. strict=False일 때만 채워지고,
    # 그때는 실패로 치지 않는다 — 아래 path_feasible의 strict 설명 참조.
    soft_failures: int = 0
    options: list = field(default_factory=list)   # 지점별 IkOption(있을 때만)


def solution_space_order(current: int | None) -> list[int]:
    """검사할 solution space 순서. **현재 space가 항상 먼저다.**

    movel은 현재 space를 유지하므로, 지금 자세에서 그대로 갈 수 있는 해가 있으면 그게 가장
    자연스럽다. 그다음에야 다른 space를 본다 — 다른 space의 해는 관절 configuration이
    달라 로봇이 크게 자세를 바꿔야 갈 수 있기 때문이다.
    """
    order = [space for space in SOLUTION_SPACES]
    if current is None:
        return order
    current = int(current)
    return ([current] + [space for space in order if space != current]
            if current in order else order)


def find_ik(posx_options, ik_verdict_of, spaces) -> IkOption | None:
    """(자세 표현 × solution space)를 순서대로 물어 **첫 성공**을 돌려준다. 없으면 None.

    `posx_options`는 `(posx, representation)` 목록이다 — 같은 물리적 파지를 나타내는 여러
    표현(예: RG2 180도 뒤집기)을 순서대로 넣는다. 앞쪽이 더 선호되는 표현이다.

    **무응답(IK_UNKNOWN)은 성공으로 치지 않는다.** 여기서 찾는 것은 "확실히 갈 수 있는
    해"다 — 못 물어본 것을 있다고 하면 실행에서 그대로 실패한다. 다만 호출부가 "전부
    무응답이었나"를 구분할 수 있게 아무 성공도 없으면 None만 돌려준다(무응답 여부는
    호출부가 verdict로 따로 본다).
    """
    for posx, representation in posx_options:
        for space in spaces:
            verdict = ik_verdict_of(posx, space)
            if getattr(verdict, "ok", False):
                return IkOption(posx=list(posx), sol_space=int(space), verdict=verdict,
                                representation=representation)
    return None


@dataclass(frozen=True)
class IkGroup:
    """여러 지점을 **같은 solution space에서 함께** 푼 결과 (2026-09-10 보완)."""

    sol_space: int
    representation: str
    options: list          # 입력 순서 그대로의 IkOption 목록

    @property
    def posj_list(self):
        return [option.posj for option in self.options]


def find_ik_group(groups, ik_verdict_of, spaces) -> IkGroup | None:
    """여러 지점이 **한 solution space에서 동시에** 풀리는 첫 조합을 찾는다. 없으면 None.

    **왜 지점마다 따로 풀면 안 되는가.** movel은 한 이동 안에서 solution space를 바꾸지
    못한다. 접근점과 파지점을 각각 `find_ik`로 물으면 접근이 space 3, 파지가 space 5에서
    먼저 풀리는 일이 생기는데, 그 둘은 한 번의 하강으로 이어지지 않는다 — 그러면 멀쩡한
    파지가 "configuration이 다르다"는 이유로 버려진다. 두 점은 접근축을 따라 몇 cm 떨어져
    있을 뿐이라 대개 **같은 space에도 둘 다 해가 있는데**, 따로 물었기 때문에 그 space를
    못 보고 지나친 것뿐이다.

    `groups`는 `(representation, [posx, ...])` 목록이다 — 같은 물리적 파지를 나타내는 여러
    표현(예: RG2 180도 뒤집기)을 선호 순서대로 넣는다. 한 표현 안의 모든 지점이 같은
    space에서 풀려야 그 조합을 채택한다.
    """
    for representation, posx_list in groups:
        if not posx_list:
            continue
        for space in spaces:
            options = []
            for posx in posx_list:
                verdict = ik_verdict_of(posx, space)
                if not getattr(verdict, "ok", False):
                    break
                options.append(IkOption(posx=list(posx), sol_space=int(space),
                                        verdict=verdict, representation=representation))
            if len(options) == len(posx_list):
                return IkGroup(sol_space=int(space), representation=representation,
                               options=options)
    return None


def ik_cache(ik_verdict_of, digits: int = 1):
    """같은 (posx, space) 질의를 두 번 하지 않게 감싼다.

    한 번의 place 이송 계획에서 경로 3종 x space 8개를 2패스로 훑으면 같은 지점을 반복해서
    묻게 된다 — ikin은 실측 약 31ms라 그대로 두면 계획에만 몇 초가 든다. 로봇은 그동안
    움직이지 않으므로 같은 질문의 답은 같다(캐시는 한 번의 계획 안에서만 산다).

    0.1mm/0.1도까지만 보고 키를 만든다 — 부동소수 비교로 같은 값을 놓치지 않으면서 서로
    다른 자세를 뭉치지도 않는 해상도다.
    """
    cache: dict = {}

    def cached(posx, space):
        key = (tuple(round(float(v), digits) for v in list(posx)[:6]), int(space))
        if key not in cache:
            cache[key] = ik_verdict_of(posx, space)
        return cache[key]

    return cached


def joint_jump_deg(first, second) -> float | None:
    """두 관절해 사이 최대 성분 차(도). 하나라도 없으면 None."""
    if not first or not second:
        return None
    return max(abs(float(a) - float(b)) for a, b in zip(first, second))


def path_samples(start_posx, end_posx, count: int = DEFAULT_PATH_SAMPLES) -> list[list]:
    """두 끝점 사이를 `count`개 지점으로 나눈다(양 끝 포함).

    **위치만 보간하고 회전은 끝점 것을 유지한다.** 이 시스템의 movel은 한 이동 안에서
    회전을 그대로 두고 위치만 바꾸는 방식으로 쓰이고(pick_server/place_server의
    `next_target`), ZYZ 파라미터를 중간값으로 보간하면 ry가 180도 근처에서 엉뚱한 자세가
    나온다(dsr_motion 모듈 docstring). 실제 실행과 같은 모양으로 재는 것이 목적이다.
    """
    count = max(2, int(count))
    start, end = list(start_posx), list(end_posx)
    samples = []
    for index in range(count):
        ratio = index / (count - 1)
        samples.append([start[i] + (end[i] - start[i]) * ratio for i in range(3)]
                       + list(end[3:6]))
    return samples


def path_feasible(start_posx, end_posx, ik_verdict_of, sol_space: int, *,
                  samples: int = DEFAULT_PATH_SAMPLES,
                  max_joint_jump_deg: float = DEFAULT_MAX_JOINT_JUMP_DEG,
                  obstacle=None, strict: bool = True) -> PathCheck:
    """한 구간이 실제로 이어질 수 있는지. 끝점만이 아니라 중간 지점도 본다.

    보는 것 셋:
      1. 각 지점에 IK 해가 있는가(`sol_space` 고정 — 한 movel 안에서 space는 안 바뀐다)
      2. 인접 지점 사이 관절이 크게 튀지 않는가(튀면 그 사이에서 configuration이 갈린다)
      3. `obstacle(posx) -> str`가 주어지면 그 지점이 명백한 충돌인가

    **무응답은 실패로 치지 않는다.** ikin이 죽었다는 이유로 모든 경로를 막으면 로봇이
    통째로 멈춘다 — grasp_selection이 IK 무응답을 탈락으로 다루지 않는 것과 같은 이유다.
    대신 `unknown`에 세어 두고 호출부가 로그로 구분할 수 있게 한다.

    **strict=False면 중간 보간 지점의 IK 실패를 실패로 치지 않는다** (2026-09-10 실물).
    그날 place가 경로 3개 x space 8개를 전부 "경로 중간 지점의 IK 실패(unreachable)"로
    버리고 로봇이 아예 안 움직였다. 중간 지점은 **직선 보간이라는 가정** 위에서, 그것도
    space를 강제로 고정해 물어본 값이다 — ikin은 해가 없어도 값을 채워 돌려주고 경계
    근처에서는 널뛴다고 이 저장소가 이미 적어 둔 그 질의다(grasp_selection._has_real_solution).
    끝점이 확실하다면 그것만으로 예전(릴리스) 동작과 같고, 중간 지점 판정은 "더 깨끗한
    경로가 있으면 그걸 고르자"는 **선호**로 쓰는 것이 맞다. 끝점 IK 실패와 장애물은
    strict와 무관하게 언제나 실패다.
    """
    check = PathCheck(feasible=True)
    previous_posj = None
    biggest_jump = None
    path = path_samples(start_posx, end_posx, samples)
    last_index = len(path) - 1
    for index, posx in enumerate(path):
        if obstacle is not None:
            hit = obstacle(posx)
            if hit:
                return PathCheck(feasible=False, reason=hit, checked=check.checked,
                                 max_joint_jump_deg=biggest_jump, unknown=check.unknown)
        verdict = ik_verdict_of(posx, sol_space)
        check.checked += 1
        if not getattr(verdict, "known", False):
            check.unknown += 1
            previous_posj = None       # 연속성을 이어서 볼 수 없다
            continue
        if not getattr(verdict, "ok", False):
            endpoint = index in (0, last_index)
            if strict or endpoint:
                return PathCheck(
                    feasible=False,
                    reason=(f"{'끝점' if endpoint else '경로 중간 지점'}의 IK "
                            f"실패({getattr(verdict, 'status', '?')})"),
                    checked=check.checked, max_joint_jump_deg=biggest_jump,
                    unknown=check.unknown, soft_failures=check.soft_failures,
                    endpoint_failure=endpoint)
            check.soft_failures += 1
            previous_posj = None       # 여기서 끊겼으니 연속성도 이어서 못 본다
            continue
        posj = getattr(verdict, "posj", None)
        jump = joint_jump_deg(previous_posj, posj)
        if jump is not None:
            biggest_jump = jump if biggest_jump is None else max(biggest_jump, jump)
            if jump > float(max_joint_jump_deg):
                return PathCheck(
                    feasible=False,
                    reason=f"경로 중간에서 관절이 {jump:.0f}도 튄다(한계 {max_joint_jump_deg:.0f}도)",
                    checked=check.checked, max_joint_jump_deg=biggest_jump,
                    unknown=check.unknown)
        previous_posj = posj
    check.max_joint_jump_deg = biggest_jump
    return check


def config_switch_feasible(before_posj, after_posj, fkin_of, *, obstacle=None,
                           samples: int = 5) -> PathCheck:
    """movej로 solution space를 바꾸는 구간이 실제로 안전한지, **관절 이동량이 아니라
    그 경로가 실제로 지나는 TCP 지점**으로 확인한다 (STEP 3, 2026-09-10).

    같은 TCP 목표를 다른 solution space로 푸는 movej는 관절 공간에서 보간되므로 직선이
    아니다 — 관절이 크게 움직인다고 팔이 반드시 위험한 곳을 지나는 것도, 조금 움직인다고
    반드시 안전한 것도 아니다. 이 모듈 docstring의 "애매하면 막지 않는다"와 같은 원칙으로,
    보간 중간 지점의 FK를 실제로 구해 **명백한 충돌만** 본다.

    2026-09-10 실물: 이 구간을 관절 이동량(고정 90도 한계)만으로 걸렀더니, 실제로
    필요했던 168~257도 전환이 8개 solution space 중 사실상 전부 막혀 place가 매번
    "안전고도에서 바구니 상공까지 갈 방법이 없다"로 끝났다 — 각도 크기 자체는 그
    전환이 위험하다는 근거가 아니었다(SOLUTION_SPACES 조합상 팔/팔꿈치/손목 중 하나가
    큰 폭으로 바뀌는 것이 정상이다).

    `fkin_of(joint_deg) -> posx | None` — 무응답이면 그 지점은 `soft_failures`로 세고
    건너뛴다(모르는 지점을 위험하다고 단정하지 않는다). `obstacle`이 없으면 FK만
    확인하고 항상 feasible이다 — 호출부에 작업대 높이 정보가 없을 때(예: pick)도
    "모른다"를 "위험하다"로 바꾸지 않기 위해서다.
    """
    if not before_posj or not after_posj:
        return PathCheck(False, "관절값을 몰라 전환 경로를 확인할 수 없다")
    samples = max(3, int(samples))
    soft = 0
    for index in range(1, samples - 1):
        ratio = index / (samples - 1)
        joint = [b + (a - b) * ratio for b, a in zip(before_posj, after_posj)]
        if obstacle is None:
            continue
        posx = fkin_of(joint)
        if posx is None:
            soft += 1
            continue
        hit = obstacle(posx)
        if hit:
            return PathCheck(False, f"configuration 전환 중간 지점이 {hit}",
                             checked=index, soft_failures=soft)
    return PathCheck(True, checked=samples, soft_failures=soft)


def routes(start_posx, end_posx, transit_z: float) -> list[tuple[str, list]]:
    """시도할 경로 몇 개. **direct가 항상 먼저다.**

    반환: `(이름, [경유 posx, ...])` 목록. 경유점 목록은 시작점을 포함하지 않는다 —
    호출부가 시작점에서 순서대로 이동한다.

    우회 경로는 `transit_z`(안전 높이)를 쓴다. **이 높이는 낮추지 않는다** — 안전을 위한
    높이를 IK 때문에 낮추면 그 높이를 둔 이유가 사라진다(STEP 2 §6). 대신 같은 높이에서
    회전/이동 순서를 바꾼 경로를 후보로 둔다.

    고정 waypoint 하나를 모든 이동에 강제하지 않는다 — 여기서 만드는 것은 시작점과
    목표점에서 파생된 소수의 경로뿐이고, 전부 실패하면 호출부가 실패로 보고한다.
    """
    start, end = list(start_posx), list(end_posx)
    rise = [start[0], start[1], transit_z, *start[3:6]]           # 제자리 수직 상승
    rise_turned = [start[0], start[1], transit_z, *end[3:6]]      # 그 자리에서 목표 자세로 회전
    over_target = [end[0], end[1], transit_z, *end[3:6]]          # 목표 상공(목표 자세)
    over_target_keep = [end[0], end[1], transit_z, *start[3:6]]   # 목표 상공(시작 자세 유지)
    # 같은 지점이 연달아 나오면 하나로 줄인다. 호출부가 목표를 이미 안전고도에 두고
    # 부르면(place_server의 over_target) 마지막 두 경유점이 같은 자리가 되는데, 그대로
    # 두면 길이 0짜리 구간을 IK로 또 검사한다 — 한 space당 ikin 3회(약 90ms)가 순수 낭비고,
    # 실행에서도 제자리 movel이 한 번 더 나간다.
    def trimmed(waypoints):
        result = []
        for waypoint in waypoints:
            if not result or [round(v, 6) for v in waypoint] != [round(v, 6) for v in result[-1]]:
                result.append(waypoint)
        return result

    return [
        ("direct", [end]),
        ("rise_rotate_traverse", trimmed([rise, rise_turned, over_target, end])),
        ("rise_traverse_rotate", trimmed([rise, over_target_keep, over_target, end])),
    ]


def support_plane_obstacle(floor_z: float, tolerance_mm: float = 0.0):
    """작업대/바구니 바닥을 뚫는 지점을 막는 `obstacle` 콜러블을 만든다.

    **명백한 것만 막는다.** floor_z를 모르면(None) 이 검사는 아예 만들지 않는다 —
    정보가 없을 때 "일단 충돌"로 처리하면 멀쩡한 경로가 통째로 죽는다(STEP 2 §4).
    """
    if floor_z is None:
        return None
    limit = float(floor_z) - float(tolerance_mm)

    def obstacle(posx) -> str:
        if float(posx[2]) < limit:
            return (f"경로 지점 z {float(posx[2]):.1f}mm가 지지면 {float(floor_z):.1f}mm "
                    f"아래(허용 {tolerance_mm:.1f}mm)")
        return ""

    return obstacle


def combine_obstacles(*obstacles):
    """여러 `obstacle` 콜러블을 하나로. None은 건너뛴다."""
    active = [obstacle for obstacle in obstacles if obstacle is not None]
    if not active:
        return None

    def obstacle(posx) -> str:
        for check in active:
            hit = check(posx)
            if hit:
                return hit
        return ""

    return obstacle


def rim_crossing_obstacle(corners_xy, rim_z: float, margin_mm: float = 0.0):
    """테두리 높이 아래에서 바구니 경계선을 **넘나드는** 이동을 막는 검사기.

    지점 하나만으로는 판단할 수 없어(안/밖 자체는 정상) 인접 두 지점을 함께 본다 —
    한쪽은 안, 한쪽은 밖이면서 둘 다 테두리보다 낮으면 벽을 통과하는 이동이다.
    """
    if not corners_xy or rim_z is None:
        return None
    xs = [float(x) for x, _ in corners_xy]
    ys = [float(y) for _, y in corners_xy]
    x_lo, x_hi = min(xs) - float(margin_mm), max(xs) + float(margin_mm)
    y_lo, y_hi = min(ys) - float(margin_mm), max(ys) + float(margin_mm)

    def inside(posx) -> bool:
        return x_lo <= float(posx[0]) <= x_hi and y_lo <= float(posx[1]) <= y_hi

    def crosses(first, second) -> str:
        if float(first[2]) >= float(rim_z) and float(second[2]) >= float(rim_z):
            return ""
        if inside(first) != inside(second):
            return (f"테두리 z {float(rim_z):.1f}mm 아래에서 바구니 벽을 가로지른다 "
                    f"({'안→밖' if inside(first) else '밖→안'})")
        return ""

    return crosses


def segment_obstacle_hit(samples, crosses) -> str:
    """인접 지점 쌍에 `crosses`를 적용해 첫 위반을 돌려준다. 없으면 빈 문자열."""
    if crosses is None:
        return ""
    for first, second in zip(samples, samples[1:]):
        hit = crosses(first, second)
        if hit:
            return hit
    return ""


def rotation_span_deg(a_zyz, b_zyz, rotation_diff) -> float | None:
    """두 자세 사이 회전량(도). 비교 함수는 호출부가 준다(ZYZ 특이점 때문에 성분 차 금지)."""
    if a_zyz is None or b_zyz is None or rotation_diff is None:
        return None
    return float(rotation_diff(a_zyz, b_zyz))


def travel_mm(a_posx, b_posx) -> float:
    return math.dist([float(v) for v in a_posx[:3]], [float(v) for v in b_posx[:3]])

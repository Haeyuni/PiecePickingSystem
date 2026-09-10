#!/usr/bin/env python3
"""README에 들어가는 다이어그램 6종을 SVG로 그려 PNG로 굽는다.

**그림을 손으로 그리지 않고 코드로 두는 이유**: 예전에 docs/visual_material/ERD.png가
스키마보다 뒤처진 채 남아 있다가(제거된 테이블이 그대로 그려져 있었다) 그냥 지워졌다.
그림도 문서와 같이 낡는데, 원본이 없으면 고치는 대신 지우게 된다. 여기 좌표와 문구를
고치고 다시 돌리면 6장이 한 번에 갱신된다.

값의 출처(bins.yaml 실측 좌표, 포트, 토픽·액션 이름)는 저장소 안에 있고, 이 파일은 그
값을 옮겨 적는다 — 새로 지어내지 않는다.

실행:
    python3 tools/scripts/make_diagrams.py          # docs/diagrams/*.png 갱신

PNG 변환은 Chrome 헤드리스를 쓴다(macOS 기본 도구에 SVG 래스터라이저가 없다).
--force-device-scale-factor=2 라서 논리 좌표의 2배 해상도로 나온다.
"""
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "docs" / "diagrams"

FONT = ("'Apple SD Gothic Neo','Pretendard','Noto Sans KR',"
        "-apple-system,'Segoe UI','Helvetica Neue',sans-serif")
MONO = "'SFMono-Regular',Menlo,Consolas,monospace"

BG = "#ffffff"
INK = "#12161f"
MUTED = "#5b6673"
LINE = "#b9c1cc"
FAINT = "#dde2e9"

# (테두리, 채우기) — 계층별 색은 6장 전부에서 같은 뜻으로 쓴다.
PERCEPTION = ("#2f6fd0", "#eaf1fc")
GRASP = ("#1f8a63", "#e7f5ef")
PLANNER = ("#b8791a", "#fdf3e1")
CONTROL = ("#7a5bc4", "#f0ebfb")
WEB = ("#37475c", "#ebeef2")
DB = ("#68727f", "#eef0f3")
HW = ("#3f4753", "#f1f3f6")
EXT = ("#8a919b", "#f6f7f9")
DANGER = ("#bf3b2c", "#fdecea")
NEUTRAL = ("#9aa2ad", "#fbfcfd")


# --- SVG 조립 ---------------------------------------------------------------

def esc(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def text_w(text: str, size: float) -> float:
    """문자열의 대략적인 렌더 폭(px). 라벨 뒤에 깔 흰 판의 크기를 잡는 용도라
    정확할 필요는 없고, 한글이 라틴 문자의 약 1.8배라는 것만 반영한다."""
    w = 0.0
    for ch in text:
        w += size * (0.98 if ord(ch) > 0x2000 else 0.54)
    return w


class Svg:
    def __init__(self, width: int, height: int):
        self.w, self.h = width, height
        self.parts: list[str] = []

    def add(self, markup: str) -> None:
        self.parts.append(markup)

    # 도형 ---------------------------------------------------------------
    def rect(self, x, y, w, h, *, stroke=LINE, fill="none", r=10, sw=1.4, dash=None,
             opacity=1.0):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" ry="{r}" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d} '
                 f'opacity="{opacity}"/>')

    def text(self, x, y, s, *, size=14, fill=INK, weight=400, anchor="start",
             family=FONT, spacing=0.0, opacity=1.0):
        ls = f' letter-spacing="{spacing}"' if spacing else ""
        self.add(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
                 f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"'
                 f'{ls} opacity="{opacity}">{esc(s)}</text>')

    def line(self, x1, y1, x2, y2, *, stroke=LINE, sw=1.3, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
                 f'stroke-width="{sw}"{d}/>')

    # 합성 ---------------------------------------------------------------
    def panel(self, x, y, w, h, label=None, *, stroke=FAINT, fill="#fafbfd", dash="6 5"):
        self.rect(x, y, w, h, stroke=stroke, fill=fill, r=14, sw=1.4, dash=dash)
        if label:
            self.text(x + 16, y + 24, label, size=13, fill=MUTED, weight=600)

    def box(self, x, y, w, h, title, lines=(), color=NEUTRAL, *, badge=None,
            title_size=15, dash=None, align="left"):
        stroke, fill = color
        self.rect(x, y, w, h, stroke=stroke, fill=fill, r=10, sw=1.5, dash=dash)
        if align == "center":
            tx, anchor = x + w / 2, "middle"
        else:
            tx, anchor = x + 14, "start"
        ty = y + (26 if lines else h / 2 + 5)
        self.text(tx, ty, title, size=title_size, fill=stroke, weight=700, anchor=anchor)
        for i, ln in enumerate(lines):
            self.text(tx, ty + 19 + i * 16, ln, size=12, fill=MUTED, anchor=anchor)
        if badge:
            bw = text_w(badge, 11) + 14
            self.rect(x + w - bw - 10, y + 9, bw, 18, stroke=stroke, fill="#ffffff", r=9,
                      sw=1.0)
            self.text(x + w - bw / 2 - 10, y + 22, badge, size=11, fill=stroke,
                      anchor="middle", weight=600, family=MONO)

    def label(self, x, y, s, *, size=11.5, fill=MUTED, pad=5, anchor="middle",
              weight=500):
        """선 위에 얹는 라벨. 선을 가리도록 흰 판을 먼저 깐다."""
        w = text_w(s, size) + pad * 2
        left = {"middle": x - w / 2, "start": x - pad, "end": x - w + pad}[anchor]
        self.rect(left, y - size, w, size + 8, stroke="none", fill=BG, r=4, sw=0)
        self.text(x, y, s, size=size, fill=fill, anchor=anchor, weight=weight)

    def arrow(self, pts, *, stroke=LINE, sw=1.5, dash=None, head="end", label=None,
              label_at=0.5, label_dy=-8, label_fill=MUTED, label_size=11.5):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        marker = ""
        if head in ("end", "both"):
            marker += ' marker-end="url(#ah)"'
        if head in ("start", "both"):
            marker += ' marker-start="url(#ah-rev)"'
        path = " ".join(f"{px},{py}" for px, py in pts)
        self.add(f'<polyline points="{path}" fill="none" stroke="{stroke}" '
                 f'stroke-width="{sw}" stroke-linejoin="round"{d}{marker}/>')
        if label:
            # 가장 긴 구간의 중간에 라벨을 얹는다.
            best, blen = (pts[0], pts[-1]), -1.0
            for a, b in zip(pts, pts[1:]):
                ln = abs(b[0] - a[0]) + abs(b[1] - a[1])
                if ln > blen:
                    best, blen = (a, b), ln
            (ax, ay), (bx, by) = best
            lx = ax + (bx - ax) * label_at
            ly = ay + (by - ay) * label_at + label_dy
            self.label(lx, ly, label, fill=label_fill, size=label_size)

    def render(self) -> str:
        defs = (
            '<defs>'
            f'<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            f'markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M 0 1 L 9 5 L 0 9 z" fill="{LINE}"/></marker>'
            f'<marker id="ah-rev" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            f'markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M 0 1 L 9 5 L 0 9 z" fill="{LINE}"/></marker>'
            '</defs>'
        )
        body = "".join(self.parts)
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" '
                f'height="{self.h}" viewBox="0 0 {self.w} {self.h}">'
                f'{defs}<rect width="{self.w}" height="{self.h}" fill="{BG}"/>'
                f'{body}</svg>')


def title(svg: Svg, main: str, sub: str = "") -> None:
    svg.text(40, 46, main, size=22, fill=INK, weight=700)
    if sub:
        svg.text(40, 70, sub, size=13, fill=MUTED)


def legend(svg: Svg, x, y, items) -> None:
    """(색, 문구) 목록을 한 줄로 늘어놓는다."""
    cx = x
    for color, text in items:
        stroke, fill = color
        svg.rect(cx, y - 9, 12, 12, stroke=stroke, fill=fill, r=3, sw=1.3)
        svg.text(cx + 18, y + 1, text, size=11.5, fill=MUTED)
        cx += 18 + text_w(text, 11.5) + 22


# --- 1. 시스템 아키텍처 ------------------------------------------------------

def diagram_architecture() -> Svg:
    s = Svg(1280, 800)
    title(s, "시스템 아키텍처",
          "명령은 web → planner로, 로봇은 web → control로. 관측은 perception이 내고 grasp가 완성해 되돌린다.")

    s.box(648, 96, 300, 46, "브라우저 — 제어 · 이력 · 학습 데이터", color=EXT,
          align="center", title_size=13)
    s.box(990, 96, 245, 46, "OpenAI API", color=EXT, align="center", title_size=13)

    s.rect(40, 168, 1200, 412, stroke=FAINT, fill="#fafbfd", r=14, sw=1.4, dash="6 5")
    s.text(58, 192, "로봇 PC — Ubuntu 24.04 · ROS 2 Jazzy · RTX 4060 8GB", size=13,
           fill=MUTED, weight=600)

    row_y, bh, bw = 240, 78, 195
    s.box(70, row_y, bw, bh, "perception", ["온디맨드 관측", "YOLO11-seg / SAM+VLM"],
          color=PERCEPTION)
    s.box(385, row_y, bw, bh, "grasp", ["파지 후보 계산", "2전략"], color=GRASP)
    s.box(700, row_y, bw, bh, "web", ["FastAPI", "+ rclpy 브리지"], color=WEB)
    s.box(1015, row_y, bw, bh, "planner", ["LLM 계획 + 검증기", "ROS2 무관 (HTTP)"],
          color=PLANNER)
    s.box(385, 366, bw, 52, "graspnet", color=GRASP, badge=":8200", title_size=13)
    s.box(1015, 366, bw, 52, "db  PostgreSQL 16", color=DB, badge=":5432",
          title_size=13)
    s.box(700, 470, bw, bh, "control", ["pick · place_into", "home"], color=CONTROL)

    ay = 292
    s.arrow([(798, 142), (798, row_y)], label="HTTP :8000 · /ws/live", label_at=0.45)
    s.arrow([(1112, 142), (1112, row_y)], label="LLM · VLM", label_at=0.5)

    s.arrow([(265, ay), (385, ay)])
    s.label(325, 258, "world_state_raw", size=10.5)
    s.label(325, 274, "instance_masks", size=10.5)
    s.arrow([(580, ay), (700, ay)])
    s.label(640, 266, "/world_state", size=10.5)
    s.arrow([(895, ay), (1015, ay)], head="both")
    s.label(955, 266, "/internal/plan", size=10.5)

    s.arrow([(482, 318), (482, 366)], head="both")
    s.label(560, 346, "HTTP", size=10.5, anchor="start")
    s.arrow([(1112, 318), (1112, 366)], head="both")
    s.arrow([(798, 318), (798, 470)], label="pick · place_into · home 액션",
            label_at=0.55)
    s.arrow([(760, row_y), (760, 222), (180, 222), (180, row_y)],
            label="observe 액션 (온디맨드 관측)", label_at=0.5, label_dy=-6)

    s.box(70, 640, 240, 62, "RealSense (eye-in-hand)", color=HW, align="center",
          title_size=13)
    s.box(385, 640, 240, 62, "M0609 + OnRobot RG2", color=HW, align="center",
          title_size=14)
    s.box(700, 640, 195, 62, "로봇 제어박스", color=HW, align="center", title_size=14)

    s.arrow([(798, 548), (798, 640)], label="이더넷 · dsr 액션 · 서비스", label_at=0.5)
    s.arrow([(700, 671), (625, 671)], head="both")
    s.label(662, 663, "로봇 케이블", size=10.5)
    s.arrow([(385, 671), (310, 671)], head="none")
    s.label(347, 663, "손목 장착", size=10.5)
    s.arrow([(190, 640), (190, 318)], label="color · aligned depth", label_at=0.66)

    legend(s, 40, 762, [(PERCEPTION, "인지"), (GRASP, "파지"), (PLANNER, "계획"),
                        (WEB, "웹"), (CONTROL, "제어"), (DB, "저장"), (HW, "하드웨어")])
    return s


# --- 2. 네트워크 구성도 ------------------------------------------------------

def diagram_network() -> Svg:
    s = Svg(1280, 830)
    title(s, "네트워크 구성",
          "DDS를 봐야 하는 것만 host 네트워크에 둔다. host에는 서비스 이름 DNS가 없어 나머지를 게시된 포트로 부른다.")

    s.box(60, 96, 240, 46, "브라우저", color=EXT, align="center", title_size=13)
    s.box(330, 96, 240, 46, "OpenAI API (외부)", color=EXT, align="center",
          title_size=13)

    s.panel(60, 196, 560, 300, "bridge network — 서비스 이름 DNS로 서로 부른다")
    s.box(90, 244, 240, 56, "db", color=DB, badge=":5432", title_size=14)
    s.box(350, 244, 240, 56, "planner", color=PLANNER, badge=":8100", title_size=14)
    s.box(90, 322, 240, 56, "graspnet", color=GRASP, badge=":8200", title_size=14)
    s.box(350, 322, 240, 56, "web (mock)", color=WEB, badge=":8000", title_size=14)
    for i, ln in enumerate([
            "web(mock)은 MOCK_MODE=1 — data/mock 픽스처로 돈다.",
            "web_ros와 8000이 겹치므로 둘을 동시에 띄우지 않는다.",
            "그래서 web_ros는 profile: ros로 기본 기동에서 빠져 있다."]):
        s.text(90, 416 + i * 22, ln, size=11.5, fill=MUTED)

    s.panel(660, 196, 560, 300, "host network — 호스트 DDS를 그대로 본다")
    s.box(690, 244, 240, 56, "perception", color=PERCEPTION, title_size=14)
    s.box(950, 244, 240, 56, "grasp", color=GRASP, title_size=14)
    s.box(690, 322, 240, 56, "control", color=CONTROL, title_size=14)
    s.box(950, 322, 240, 56, "web_ros", color=WEB, badge=":8000", title_size=14)
    for i, ln in enumerate([
            "DDS 디스커버리가 멀티캐스트에 기대므로 브리지에서는",
            "호스트의 드라이버·카메라 노드를 찾지 못한다.",
            "RMW는 전부 CycloneDDS로 맞춘다 — 다르면 토픽 목록은",
            "보이는데 데이터가 흐르지 않는다."]):
        s.text(690, 416 + i * 22, ln, size=11.5, fill=MUTED)

    # 브라우저 → 둘 중 떠 있는 web 하나
    s.arrow([(180, 142), (180, 196)], head="none", dash="5 4")
    s.arrow([(180, 142), (180, 176), (1070, 176), (1070, 196)], head="none", dash="5 4")
    s.label(700, 170, "http://localhost:8000  —  web 또는 web_ros 중 하나", size=11.5)

    # planner → OpenAI
    s.arrow([(450, 244), (450, 142)])
    s.label(462, 196, "LLM · VLM", anchor="start", size=11)

    # host → bridge
    s.arrow([(660, 356), (620, 356)], head="both")
    s.label(640, 318, "게시된 포트로", size=10.5)
    s.label(640, 334, "localhost 호출", size=10.5)

    s.panel(60, 556, 1160, 140, "호스트에서 직접 띄우는 프로세스 (컨테이너 아님)")
    s.box(85, 600, 340, 74, "로봇 드라이버",
          ["m0609_rg2_bringup", "tools/scripts/run_bringup.sh"], color=HW)
    s.box(455, 600, 340, 74, "카메라 드라이버",
          ["realsense2_camera", "align_depth.enable:=true"], color=HW)
    s.box(825, 600, 370, 74, "웨이크워드 브리지",
          ["tools/voice/wakeword_bridge.py",
           "로봇 PC 마이크 → /api/internal/wake-detected"], color=HW)

    # DDS는 host 네트워크 컨테이너와 호스트 프로세스 사이에서만 흐른다 — bridge 쪽에서
    # 내려오는 선을 그리면 없는 경로를 그린 것이 된다.
    s.arrow([(255, 600), (255, 540), (760, 540), (760, 496)], head="both")
    s.label(470, 534, "DDS", size=11)
    s.arrow([(625, 600), (625, 512), (880, 512), (880, 496)], head="both")
    s.label(790, 506, "DDS", size=11)
    s.arrow([(1010, 600), (1010, 496)], label="HTTP :8000", label_at=0.5)

    s.box(85, 726, 340, 56, "로봇 제어박스 · 192.168.1.100", color=HW, align="center",
          title_size=14)
    s.arrow([(255, 674), (255, 726)], head="both", label="이더넷", label_at=0.78)

    legend(s, 500, 758, [(NEUTRAL, "점선 = 브라우저 접속 경로"), (EXT, "외부 · 사용자"),
                         (HW, "하드웨어 · 호스트 프로세스")])
    return s


# --- 3. 명령 처리 시퀀스 ------------------------------------------------------

def diagram_command_flow() -> Svg:
    s = Svg(1280, 760)
    title(s, "명령 처리 시퀀스",
          "관측 → 계획 → 검증 → 사람 승인 → 실행. 검증을 통과해도 승인 없이는 로봇이 움직이지 않는다.")

    y1, y2, h, w = 130, 330, 84, 196
    xs = [40, 286, 532, 778, 1024]

    s.box(xs[0], y1, w, h, "명령 접수", ["텍스트 또는 음성(STT)", "busy면 즉시 거부"],
          color=WEB)
    s.box(xs[1], y1, w, h, "관측 A", ["Observe MODE_FULL", "SAM + VLM 라벨링"],
          color=PERCEPTION)
    s.box(xs[2], y1, w, h, "계획", ["LLM이 시퀀스 생성", "물체·목적지 그라운딩"],
          color=PLANNER)
    s.box(xs[3], y1, w, h, "검증", ["가반하중 · 작업반경", "파지 가능 · 안전 이벤트"],
          color=PLANNER)
    s.box(xs[4], 246, w, 56, "거부 — 종료", color=DANGER, align="center", title_size=14)

    s.box(xs[0], y2, w, h, "실행 승인",
          ["계획 · 판단 근거 표시", "approve / reject / 라벨 수정"], color=EXT)
    s.box(xs[1], y2, w, h, "pick", ["후보 최종 선택", "Grip detected로 판정"],
          color=CONTROL)
    s.box(xs[2], y2, w, h, "place_into", ["바구니 경계 검사", "순응 하강 후 놓기"],
          color=CONTROL)
    s.box(xs[3], y2, w, h, "관측 B", ["Observe MODE_REPROMPT", "VLM 호출 없음"],
          color=PERCEPTION)
    s.box(xs[4], y2, w, h, "완료", ["실행 로그", "데이터셋 적재"], color=GRASP)

    for i in range(3):
        s.arrow([(xs[i] + w, y1 + h / 2), (xs[i + 1], y1 + h / 2)])
    for i in range(4):
        s.arrow([(xs[i] + w, y2 + h / 2), (xs[i + 1], y2 + h / 2)])

    # 검증 결과에 따라 갈린다
    s.arrow([(xs[3] + w, y1 + h / 2), (1122, y1 + h / 2), (1122, 246)],
            label="rejected", label_at=0.4, label_dy=-9, label_fill=DANGER[0])
    s.arrow([(876, y1 + h), (876, 246), (138, 246), (138, y2)], label="approved",
            label_at=0.5)

    # 되돌아가는 갈래 (빨간 점선)
    # 관측 A로는 위쪽으로 돌아 들어간다 — 같은 높이로 들어오면 명령 접수 박스를 관통한다.
    s.arrow([(384, y2 + h), (384, 560), (20, 560), (20, 100), (384, 100), (384, y1)],
            dash="6 4", stroke=DANGER[0])
    s.label(230, 580, "pick 실패 → home 복귀 후 관측·재계획 (최대 2회)",
            fill=DANGER[0], size=11.5)
    s.arrow([(600, y2 + h), (600, 500), (700, 500), (700, y2 + h)], dash="6 4",
            stroke=DANGER[0])
    s.label(650, 524, "place_into 실패 → 같은 스텝 재전송 (최대 2회)", fill=DANGER[0],
            size=11.5)
    s.box(60, 470, 160, 40, "reject — 종료", color=DANGER, align="center",
          title_size=12.5)
    s.arrow([(138, y2 + h), (138, 470)], dash="6 4", stroke=DANGER[0])

    s.panel(40, 620, 1200, 92, "승인이 필요한 이유")
    s.text(62, 670, "SAM+VLM 경로는 등록 어휘 없이 물체 이름과 속성(파지력에 직결되는 grip_level 포함)까지 스스로 판단한다.",
           size=12.5, fill=MUTED)
    s.text(62, 692, "그래서 검증기를 지났다는 것만으로 실행하지 않고, 명령 1건당 한 번은 사람이 본다. 라벨을 고치면 재계획 후 다시 묻는다.",
           size=12.5, fill=MUTED)
    return s


# --- 4. 동작 순서도 ----------------------------------------------------------

def diagram_operation_flow() -> Svg:
    s = Svg(1280, 700)
    title(s, "동작 순서도",
          "액션 4종이 내보내는 phase. 화면의 진행바는 이 값을 그대로 받아 그린다.")

    tracks = [
        ("observe", "perception", PERCEPTION,
         ["capturing", "segmenting", "labeling", "recapturing", "publishing"],
         "no_frame · no_robot_pose · vlm_unavailable · nothing_to_reprompt"),
        ("pick", "control", CONTROL,
         ["approaching", "contact_detected", "lifting", "verifying"],
         "no_feasible_grasp · unreachable · grasp_failed · no_contact · collision_expected"),
        ("place_into", "control", CONTROL,
         ["moving", "inserting", "releasing", "verifying"],
         "place_failed · unreachable · no_contact · collision_expected"),
        ("home", "control", CONTROL,
         ["moving", "opening_gripper"],
         "unreachable · collision_expected"),
    ]

    y = 120
    for name, owner, color, phases, reasons in tracks:
        stroke, fill = color
        s.box(40, y, 190, 58, name, [f"{owner} 액션 서버"], color=color)
        x = 262
        for i, ph in enumerate(phases):
            pw = text_w(ph, 13) + 34
            s.rect(x, y + 8, pw, 42, stroke=stroke, fill="#ffffff", r=21, sw=1.4)
            s.text(x + pw / 2, y + 34, ph, size=13, fill=stroke, anchor="middle",
                   family=MONO)
            if i < len(phases) - 1:
                s.arrow([(x + pw, y + 29), (x + pw + 22, y + 29)], stroke=stroke)
            x += pw + 22
        s.text(262, y + 74, f"실패 사유:  {reasons}", size=11.5, fill=MUTED)
        y += 122

    s.panel(40, 604, 590, 72)
    s.text(62, 634, "성공 판정", size=12.5, fill=INK, weight=700)
    s.text(62, 656, "pick은 RG2 컨트롤러의 Grip detected 비트 하나로 본다 — 개폭 추정은 쓰지 않는다.",
           size=12, fill=MUTED)

    s.panel(650, 604, 590, 72)
    s.text(672, 634, "후보 선택", size=12.5, fill=INK, weight=700)
    s.text(672, 656, "planner가 후보 전체를 넘기고, 개폭·IK·관절 한계를 아는 control이 실행할 하나를 고른다.",
           size=12, fill=MUTED)
    return s


# --- 5. 장비 구성 ------------------------------------------------------------

def diagram_hardware() -> Svg:
    s = Svg(1280, 800)
    title(s, "장비 구성",
          "팔은 제어박스를 거쳐, 그리퍼는 툴체인저를 거쳐, 카메라·마이크는 USB로 — 셋 다 로봇 PC로 모인다.")

    s.box(430, 150, 420, 96, "로봇 PC",
          ["Ubuntu 24.04 · ROS 2 Jazzy · Python 3.12",
           "NVIDIA RTX 4060 8GB (드라이버 595.84)",
           "웹 · planner · ROS 노드 · DB를 전부 구동"], color=HW)
    s.box(430, 290, 420, 76, "로봇 제어박스",
          ["두산 표준 제어박스 · 192.168.1.100",
           "모터 드라이브 · 안전 I/O · 비상정지 하드와이어"], color=HW)
    s.box(430, 430, 420, 96, "Doosan M0609",
          ["6축 협동로봇 · 가반하중 6kg · 작업반경 900mm",
           "관절 토크센서 내장 · 반복정밀도 ±0.03mm",
           "컨트롤러 TCP 이름 GripperDA_v1"], color=HW)
    s.box(430, 570, 420, 62, "OnRobot RG2",
          ["2지 평행 그리퍼 · Grip detected 비트로 파지 판정"], color=HW)

    s.box(60, 430, 340, 96, "RealSense RGB-D",
          ["손목 장착 (eye-in-hand)", "color · aligned depth · CameraInfo",
           "align_depth 필수"], color=HW)
    s.box(60, 570, 340, 62, "USB 마이크", ['"hello rokey" 웨이크워드 감지'], color=HW)

    s.box(920, 290, 300, 76, "비상정지",
          ["제어박스에 하드웨어 직결 (NFR-01)", "소프트웨어 경로를 거치지 않는다"],
          color=DANGER)
    s.box(920, 430, 300, 96, "OnRobot 툴체인저",
          ["Modbus TCP · 192.168.1.1:502", "onrobot_rg_control 드라이버",
           "/onrobot/sendCommand · status"], color=HW)

    s.arrow([(640, 246), (640, 290)], head="both", label="이더넷", label_at=0.5)
    s.arrow([(640, 366), (640, 430)], head="both", label="제어 통신", label_at=0.5)
    s.arrow([(640, 526), (640, 570)], head="none")
    s.label(640, 554, "엔드이펙터 장착", size=11)
    s.arrow([(920, 328), (850, 328)], head="none", dash="5 4", stroke=DANGER[0])

    # 비상정지 박스를 관통하지 않도록 오른쪽 바깥 채널로 돌린다.
    s.arrow([(850, 190), (1252, 190), (1252, 478), (1220, 478)])
    s.label(1040, 182, "이더넷 · Modbus TCP", size=11)
    s.arrow([(1070, 526), (1070, 548), (885, 548), (885, 601), (850, 601)])
    s.label(968, 542, "그리퍼 제어 · 상태", size=11)

    s.arrow([(230, 430), (230, 198), (430, 198)])
    s.label(230, 392, "USB", size=11)
    s.arrow([(400, 601), (415, 601), (415, 214), (430, 214)])
    s.label(415, 392, "USB", size=11)

    s.panel(60, 676, 790, 96, "eye-in-hand — 카메라 자세는 매 프레임 TCP를 따라간다")
    s.text(82, 724, "T_base_camera = posx_to_matrix(get_current_posx()) @ T_gripper2camera",
           size=12.5, fill=MUTED, family=MONO)
    s.text(82, 750, "T_gripper2camera는 컨트롤러에 TCP가 선택돼 있는 상태로 풀려 있다.",
           size=12, fill=MUTED)

    s.rect(920, 676, 300, 96, stroke=DANGER[1], fill="#fefaf9", r=14, sw=1.4)
    s.text(942, 706, "TCP가 풀리면", size=12.5, fill=DANGER[0], weight=700)
    s.text(942, 730, "좌표 전체가 약 208mm 어긋난다.", size=11.5, fill=DANGER[0])
    s.text(942, 752, "상태바의 TCP 표시가 그것을 본다.", size=11.5, fill=DANGER[0])
    return s


# --- 6. 작업 셀 배치 ---------------------------------------------------------

# src/control/config/bins.yaml 실측값 (base 좌표계, mm). 여기서 지어낸 값이 아니다.
BINS = {
    "left_box": {
        "name": "왼쪽 박스 (left_box)",
        "corners": [(385.898, -371.494), (264.086, -368.680),
                    (256.111, -578.083), (391.268, -569.062)],
        "floor_z": -5.512,
    },
    "right_box": {
        "name": "오른쪽 박스 (right_box)",
        "corners": [(67.499, -364.838), (-74.162, -360.260),
                    (-70.257, -581.734), (61.445, -574.749)],
        "floor_z": -7.903,
    },
}

OBJECTS = [
    ("치약", "toothpaste", 4), ("물티슈", "wet_wipes", 4), ("선크림", "sunscreen", 3),
    ("토끼 인형", "rabbit_doll", 4), ("접이 우산", "umbrella", 3),
    ("섬유탈취제", "fabric_spray", 3), ("젤네일", "gel_nail", 3),
]


def diagram_cell_layout() -> Svg:
    s = Svg(1280, 760)
    title(s, "작업 셀 배치 · 작업물",
          "bins.yaml의 실측 좌표를 그대로 그린 평면도 (base 좌표계, mm).")

    # base 좌표 → 화면 좌표. 화면 위쪽이 +X(로봇 정면), 화면 왼쪽이 +Y.
    # 범위는 실측값이 차지하는 구간(X -74~391, Y -582~-360)에 여백을 더해 잡았다.
    ox, oy, sc = 130, 150, 0.72

    def px(yb):
        return ox + (130 - yb) * sc

    def py(xb):
        return oy + (500 - xb) * sc

    s.panel(60, 104, 700, 600)

    # 격자 100mm
    for yb in range(-600, 101, 100):
        s.line(px(yb), py(480), px(yb), py(-180), stroke="#f0f2f6", sw=1)
    for xb in range(-100, 501, 100):
        s.line(px(100), py(xb), px(-620), py(xb), stroke="#f0f2f6", sw=1)

    # 축
    s.arrow([(px(0), py(0)), (px(0), py(430))], stroke=MUTED, sw=1.2)
    s.arrow([(px(0), py(0)), (px(115), py(0))], stroke=MUTED, sw=1.2)
    s.text(px(0) + 10, py(440), "+X (로봇 정면)", size=11.5, fill=MUTED, family=MONO)
    s.text(px(120), py(0) - 12, "+Y", size=11.5, fill=MUTED, family=MONO)

    # 로봇 base
    s.add(f'<circle cx="{px(0)}" cy="{py(0)}" r="16" fill="{HW[1]}" stroke="{HW[0]}" '
          f'stroke-width="1.6"/>')
    s.text(px(0), py(0) + 5, "R", size=13, fill=HW[0], anchor="middle", weight=700)
    s.text(px(0), py(0) + 38, "M0609 base (0, 0)", size=11.5, fill=MUTED,
           anchor="middle")

    # 바구니
    for key, bin_ in BINS.items():
        pts = " ".join(f"{px(y)},{py(x)}" for x, y in bin_["corners"])
        stroke, fill = (GRASP if key == "left_box" else PERCEPTION)
        s.add(f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" '
              f'stroke-width="1.8"/>')
        cx = sum(px(y) for x, y in bin_["corners"]) / 4
        cy = sum(py(x) for x, y in bin_["corners"]) / 4
        s.text(cx, cy - 10, bin_["name"].split(" (")[0], size=14, fill=stroke,
               anchor="middle", weight=700)
        s.text(cx, cy + 12, key, size=12, fill=stroke, anchor="middle", family=MONO)
        # 크기는 실측 모서리에서 직접 잰다 — 따로 적어 두면 좌표만 고칠 때 어긋난다.
        (ax, ay), (bx, by), (c_x, c_y) = bin_["corners"][:3]
        width = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
        depth = ((bx - c_x) ** 2 + (by - c_y) ** 2) ** 0.5
        s.text(cx, cy + 32, f"{width:.0f} x {depth:.0f} mm", size=11, fill=MUTED,
               anchor="middle", family=MONO)

    s.text(78, 672, "격자 100mm · 상단 내부 모서리 4점과 바닥점을 닫힌 손끝으로 접촉해 실측한 값이다.",
           size=11.5, fill=MUTED)
    s.text(78, 692, "바닥 z — left_box -5.5mm · right_box -7.9mm. place는 이 값과 물체 높이로 놓는 높이를 정한다.",
           size=11.5, fill=MUTED)

    # 오른쪽: 작업물과 규칙
    s.panel(790, 104, 450, 320, "작업물 — YOLO11-seg 7클래스")
    yy = 156
    for ko, en, level in OBJECTS:
        s.text(818, yy, ko, size=13, fill=INK)
        s.text(940, yy, en, size=12, fill=MUTED, family=MONO)
        bw = 34
        s.rect(1170, yy - 13, bw, 19, stroke=CONTROL[0], fill=CONTROL[1], r=9, sw=1.2)
        s.text(1170 + bw / 2, yy + 1, f"g{level}", size=11.5, fill=CONTROL[0],
               anchor="middle", weight=600, family=MONO)
        yy += 32
    s.text(818, 400, "SAM+VLM 경로는 이 목록을 보지 않는다 — 처음 보는 물건도 인지한다.",
           size=11.5, fill=MUTED)

    s.panel(790, 448, 450, 256, "파지력 — 5단계 grip_level")
    rows = [("g1", "40N", "가장 강하게"), ("g2", "35N", ""), ("g3", "30N", "보통"),
            ("g4", "10N", "약하게 (무른 물체 예외)"), ("g5", "20N", "가장 약하게 · 신규 클래스 기본값")]
    yy = 500
    for tag, force, note in rows:
        s.text(818, yy, tag, size=12.5, fill=CONTROL[0], family=MONO, weight=600)
        s.text(858, yy, force, size=12.5, fill=INK, family=MONO)
        s.text(910, yy, note, size=11.5, fill=MUTED)
        yy += 30
    s.text(818, 682, "단계→힘 매핑은 objects.yaml과 skill_params.yaml이 소유한다.",
           size=11.5, fill=MUTED)
    return s


# --- 렌더 --------------------------------------------------------------------

DIAGRAMS = {
    "system_architecture": diagram_architecture,
    "network": diagram_network,
    "command_flow": diagram_command_flow,
    "operation_flow": diagram_operation_flow,
    "hardware_stack": diagram_hardware,
    "cell_layout": diagram_cell_layout,
}

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium-browser",
    "/usr/bin/chromium",
]


def find_chrome() -> str:
    for path in CHROME_CANDIDATES:
        if os.path.exists(path):
            return path
    found = shutil.which("google-chrome") or shutil.which("chromium")
    if found:
        return found
    sys.exit("Chrome/Chromium을 찾지 못했다 — PNG 변환에 필요하다.")


def svg_to_png(svg: Svg, out_path: pathlib.Path, chrome: str) -> None:
    html = (f'<!doctype html><meta charset="utf-8">'
            f'<style>html,body{{margin:0;padding:0;background:{BG};'
            f'width:{svg.w}px;height:{svg.h}px;overflow:hidden}}</style>'
            f'{svg.render()}')
    with tempfile.TemporaryDirectory() as tmp:
        page = pathlib.Path(tmp) / "page.html"
        page.write_text(html, encoding="utf-8")
        shot = pathlib.Path(tmp) / "shot.png"
        subprocess.run(
            [chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
             f"--screenshot={shot}", f"--window-size={svg.w},{svg.h}",
             "--force-device-scale-factor=2", "--default-background-color=FFFFFFFF",
             page.as_uri()],
            check=True, capture_output=True,
        )
        shutil.move(str(shot), out_path)


def main() -> None:
    chrome = find_chrome()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, build in DIAGRAMS.items():
        svg = build()
        out = OUT_DIR / f"{name}.png"
        svg_to_png(svg, out, chrome)
        print(f"{out.relative_to(REPO)}  ({svg.w}x{svg.h} @2x, "
              f"{out.stat().st_size // 1024}KB)")


if __name__ == "__main__":
    main()

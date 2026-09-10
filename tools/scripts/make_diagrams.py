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
    s = Svg(1280, 860)
    title(s, "시스템 아키텍처",
          "명령은 web → planner로, 로봇은 web → control로. 관측은 perception이 내고 grasp가 완성해 되돌린다.")

    s.box(648, 100, 300, 48, "브라우저 — 제어 · 이력 · 학습 데이터", color=EXT,
          align="center", title_size=14)
    s.box(990, 100, 245, 48, "OpenAI API", color=EXT, align="center", title_size=14)

    # 패널 테두리를 라벨 흰판이 갉아먹지 않도록, 경계에 닿는 라벨을 두지 않는다.
    s.rect(40, 186, 1200, 420, stroke=FAINT, fill="#fafbfd", r=14, sw=1.4, dash="6 5")
    s.text(58, 212, "로봇 PC — Ubuntu 24.04 · ROS 2 Jazzy · RTX 4060 8GB", size=13.5,
           fill=MUTED, weight=600)

    row_y, bh, bw = 268, 82, 195
    s.box(70, row_y, bw, bh, "perception", ["온디맨드 관측", "YOLO11-seg / SAM+VLM"],
          color=PERCEPTION, title_size=16)
    s.box(385, row_y, bw, bh, "grasp", ["파지 후보 계산", "2전략"], color=GRASP,
          title_size=16)
    s.box(700, row_y, bw, bh, "web", ["FastAPI", "+ rclpy 브리지"], color=WEB,
          title_size=16)
    s.box(1015, row_y, bw, bh, "planner", ["LLM 계획 + 검증기", "ROS2 무관 (HTTP)"],
          color=PLANNER, title_size=16)
    s.box(385, 396, bw, 54, "graspnet", color=GRASP, badge=":8200", title_size=14)
    s.box(1015, 396, bw, 54, "db  PostgreSQL 16", color=DB, badge=":5432",
          title_size=14)
    s.box(700, 496, bw, 82, "control", ["pick · place_into", "home"], color=CONTROL,
          title_size=16)

    ay = row_y + bh / 2
    s.arrow([(798, 148), (798, row_y)])
    s.label(810, 182, "HTTP :8000 · /ws/live", size=12, anchor="start")
    s.arrow([(1112, 148), (1112, row_y)])
    s.label(1124, 182, "LLM · VLM", size=12, anchor="start")

    s.arrow([(265, ay), (385, ay)])
    s.label(325, 332, "world_state_raw", size=11)
    s.label(325, 348, "instance_masks", size=11)
    s.arrow([(580, ay), (700, ay)])
    s.label(640, 332, "/world_state", size=11)
    s.arrow([(895, ay), (1015, ay)], head="both")
    s.label(955, 332, "/internal/plan", size=11)

    s.arrow([(482, 350), (482, 396)], head="both")
    s.arrow([(1112, 350), (1112, 396)], head="both")
    s.arrow([(798, 350), (798, 496)])
    s.label(810, 430, "pick · place_into · home 액션", size=12, anchor="start")
    s.arrow([(760, row_y), (760, 244), (180, 244), (180, row_y)])
    s.label(470, 232, "observe 액션 (온디맨드 관측)", size=12)

    # 하드웨어 — 패널 아래로 충분히 띄워 경계와 라벨이 겹치지 않게 한다.
    hw_y = 692
    s.box(70, hw_y, 240, 66, "RealSense", color=HW, align="center", title_size=15)
    s.box(385, hw_y, 240, 66, "M0609 + RG2", color=HW, align="center", title_size=15)
    s.box(700, hw_y, 195, 66, "로봇 제어박스", color=HW, align="center", title_size=15)

    s.arrow([(798, 578), (798, hw_y)])
    s.label(810, 646, "이더넷 · dsr 액션", size=12, anchor="start")
    s.arrow([(700, hw_y + 33), (625, hw_y + 33)], head="both")
    s.arrow([(385, hw_y + 33), (310, hw_y + 33)], head="none")
    s.label(347, hw_y + 16, "장착", size=11)
    s.arrow([(190, hw_y), (190, 350)])
    s.label(202, 646, "color · depth", size=12, anchor="start")

    legend(s, 40, 820, [(PERCEPTION, "인지"), (GRASP, "파지"), (PLANNER, "계획"),
                        (WEB, "웹"), (CONTROL, "제어"), (DB, "저장"), (HW, "하드웨어")])
    return s


# --- 2. 네트워크 구성도 ------------------------------------------------------

def diagram_network() -> Svg:
    """세 덩어리로만 보여준다 — bridge / host / 호스트에서 직접 띄우는 것.

    포트·주소를 전부 적으면 정작 "왜 갈라놨는지"가 안 보인다. 그건 docker-compose.yml이
    정본이므로 여기서는 경계와 그 이유만 남긴다.
    """
    s = Svg(1280, 660)
    title(s, "네트워크 구성",
          "DDS를 봐야 하는 것만 host에 둔다. host에는 서비스 이름 DNS가 없어 나머지를 게시된 포트로 부른다.")

    s.box(60, 104, 220, 50, "브라우저", color=EXT, align="center", title_size=15)
    s.arrow([(170, 154), (170, 212)], head="none", dash="5 4")
    s.label(182, 190, ":8000", size=12.5, anchor="start")

    # bridge
    s.panel(60, 212, 520, 232, "bridge network")
    s.box(90, 258, 220, 56, "db", color=DB, badge=":5432", title_size=15)
    s.box(330, 258, 220, 56, "planner", color=PLANNER, badge=":8100", title_size=15)
    s.box(90, 330, 220, 56, "graspnet", color=GRASP, badge=":8200", title_size=15)
    s.box(330, 330, 220, 56, "web (mock)", color=WEB, badge=":8000", title_size=15)
    s.text(90, 418, "서비스 이름으로 서로 부른다", size=12.5, fill=MUTED)

    # host
    s.panel(700, 212, 520, 232, "host network")
    s.box(730, 258, 220, 56, "perception", color=PERCEPTION, title_size=15)
    s.box(970, 258, 220, 56, "grasp", color=GRASP, title_size=15)
    s.box(730, 330, 220, 56, "control", color=CONTROL, title_size=15)
    s.box(970, 330, 220, 56, "web_ros", color=WEB, badge=":8000", title_size=15)
    s.text(730, 418, "호스트 DDS를 그대로 본다", size=12.5, fill=MUTED)

    s.arrow([(700, 328), (580, 328)], head="both")
    s.label(640, 284, "게시된 포트로", size=12)
    s.label(640, 300, "localhost 호출", size=12)

    # 호스트에서 직접 띄우는 것
    s.panel(700, 500, 520, 116, "호스트에서 직접 띄운다 (컨테이너 아님)")
    s.box(730, 542, 220, 56, "로봇 드라이버", color=HW, align="center", title_size=15)
    s.box(970, 542, 220, 56, "realsense2_camera", color=HW, align="center",
          title_size=15)
    s.arrow([(960, 500), (960, 444)], head="both")
    s.label(972, 478, "DDS", size=12.5, anchor="start")

    s.box(60, 542, 520, 56, "로봇 제어박스 · 192.168.1.100", color=HW, align="center",
          title_size=15)
    s.arrow([(730, 570), (580, 570)], head="both")
    s.label(655, 556, "이더넷", size=12.5)

    s.text(60, 474, "web과 web_ros는 8000이 겹친다 — 둘을 동시에 띄우지 않는다.",
           size=12.5, fill=MUTED)
    s.text(60, 496, "RMW는 전부 CycloneDDS로 맞춘다. 다르면 토픽은 보이는데 데이터가 안 흐른다.",
           size=12.5, fill=MUTED)
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
          ["계획된 스텝 표시", "approve / reject / 라벨 수정"], color=EXT)
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

    s.rect(40, 616, 1200, 118, stroke=FAINT, fill="#fafbfd", r=14, sw=1.4)
    s.text(64, 652, "승인이 필요한 이유", size=17, fill=INK, weight=700)
    s.text(64, 684, "SAM+VLM 경로는 등록 어휘 없이 물체 이름과 속성을 스스로 판단한다 — 파지력에 직결되는 grip_level까지.",
           size=15, fill=MUTED)
    s.text(64, 712, "그래서 검증기를 지났다는 것만으로 실행하지 않고, 명령 1건당 한 번은 사람이 본다.",
           size=15, fill=MUTED)
    return s


# --- 4. 동작 순서도 ----------------------------------------------------------

def diagram_operation_flow() -> Svg:
    s = Svg(1280, 760)
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

    y = 118
    for name, owner, color, phases, reasons in tracks:
        stroke, fill = color
        s.box(40, y, 210, 66, name, [f"{owner} 액션 서버"], color=color, title_size=19)
        x = 282
        for i, ph in enumerate(phases):
            pw = text_w(ph, 15) + 38
            s.rect(x, y + 6, pw, 52, stroke=stroke, fill="#ffffff", r=26, sw=1.6)
            s.text(x + pw / 2, y + 39, ph, size=15, fill=stroke, anchor="middle",
                   family=MONO)
            if i < len(phases) - 1:
                s.arrow([(x + pw, y + 32), (x + pw + 24, y + 32)], stroke=stroke, sw=1.7)
            x += pw + 24
        s.text(282, y + 86, f"실패 사유:  {reasons}", size=13.5, fill=MUTED)
        y += 140

    s.rect(40, 676, 590, 62, stroke=FAINT, fill="#fafbfd", r=14, sw=1.4)
    s.text(64, 702, "성공 판정", size=15, fill=INK, weight=700)
    s.text(64, 726, "pick은 RG2의 Grip detected 비트 하나로 본다 — 개폭 추정은 쓰지 않는다.",
           size=13.5, fill=MUTED)

    s.rect(650, 676, 590, 62, stroke=FAINT, fill="#fafbfd", r=14, sw=1.4)
    s.text(674, 702, "후보 선택", size=15, fill=INK, weight=700)
    s.text(674, 726, "planner가 후보 전체를 넘기고, control이 실행할 하나를 고른다.",
           size=13.5, fill=MUTED)
    return s


# --- 5. 장비 구성 ------------------------------------------------------------

def diagram_hardware() -> Svg:
    """장비를 6개로 줄이고 연결선도 6개만 남긴다.

    툴체인저를 따로 그리지 않고 RG2 설명에 넣었고, 마이크는 노트북 내장이라 로봇 PC
    안에 적었다 — 상자를 늘리면 "무엇이 무엇에 붙어 있는가"가 오히려 안 보인다.
    """
    s = Svg(1280, 780)
    title(s, "장비 구성",
          "팔은 제어박스를 거치고, 그리퍼는 Modbus로 직접 간다. 카메라는 손목에 붙어 USB로 들어온다.")

    s.box(420, 118, 440, 134, "로봇 PC (노트북)",
          ["Ubuntu 24.04 · ROS 2 Jazzy · Python 3.12",
           "NVIDIA RTX 4060 8GB",
           "웹 · planner · ROS 노드 · DB를 전부 구동",
           "내장 마이크 — \"hello rokey\" 웨이크워드"],
          color=HW, title_size=20)

    s.box(80, 330, 280, 124, "RealSense RGB-D",
          ["손목 장착 (eye-in-hand)", "color · aligned depth", "align_depth 필수"],
          color=HW, title_size=19)

    s.box(420, 330, 440, 104, "로봇 제어박스",
          ["두산 표준 제어박스 · 192.168.1.100", "모터 드라이브 · 안전 I/O"],
          color=HW, title_size=20)

    s.box(420, 490, 440, 104, "Doosan M0609",
          ["6축 · 가반하중 6kg · 작업반경 900mm", "관절 토크센서 내장"],
          color=HW, title_size=20)

    s.box(420, 650, 440, 104, "OnRobot RG2",
          ["2지 평행 그리퍼 · Grip detected로 파지 판정",
           "툴체인저 경유 Modbus TCP · 192.168.1.1:502"],
          color=HW, title_size=20)

    s.box(80, 650, 280, 104, "비상정지",
          ["제어박스에 하드웨어 직결", "소프트웨어를 거치지 않는다"],
          color=DANGER, title_size=19)

    s.arrow([(220, 330), (220, 185), (420, 185)])
    s.label(232, 260, "USB", size=14, anchor="start")

    s.arrow([(640, 252), (640, 330)], head="both")
    s.label(652, 296, "이더넷", size=14, anchor="start")

    s.arrow([(640, 434), (640, 490)], head="both")
    s.label(652, 468, "제어 통신", size=14, anchor="start")

    s.arrow([(640, 594), (640, 650)], head="none")
    s.label(652, 628, "엔드이펙터 장착", size=14, anchor="start")

    s.arrow([(860, 185), (1040, 185), (1040, 702), (860, 702)])
    s.label(1052, 450, "Modbus TCP", size=14, anchor="start")

    s.arrow([(360, 702), (392, 702), (392, 382), (420, 382)], head="none",
            dash="5 4", stroke=DANGER[0])
    return s


# --- 6. 작업 셀 배치 ---------------------------------------------------------

OBJECTS = [
    ("치약", "toothpaste", 4), ("물티슈", "wet_wipes", 4), ("선크림", "sunscreen", 3),
    ("토끼 인형", "rabbit_doll", 4), ("접이 우산", "umbrella", 3),
    ("섬유탈취제", "fabric_spray", 3), ("젤네일", "gel_nail", 3),
]

# 파지력 5단계. 단계 정의(단조 감소)는 objects.yaml, 실제로 control이 거는 힘은
# skill_params.yaml이 소유한다 — g4만 두 값이 다르다(아래 note 참조).
GRIP_LEVELS = [
    ("g1", "40N", "가장 강하게", None),
    ("g2", "35N", "강하게", None),
    ("g3", "30N", "보통", None),
    ("g4", "25N", "약하게", "control은 10N"),
    ("g5", "20N", "가장 약하게 · 신규 클래스 기본값", None),
]


def diagram_cell_layout() -> Svg:
    """작업 셀 배치 — 실제 셀에서 보이는 대로 그린 모식도(축척 아님).

    좌표를 그대로 찍은 평면도였다가 바꿨다. 목적지의 정확한 좌표는 bins.yaml이 정본이고,
    이 그림이 답해야 하는 것은 "무엇이 어디 있고 무엇을 어디로 옮기는가"다.
    """
    s = Svg(1280, 780)
    title(s, "작업 셀 배치 · 작업물",
          "작업대의 물체를 집어 두 박스로 분류한다. 목적지 좌표의 정본은 bins.yaml이다.")

    s.panel(60, 104, 700, 640)

    # 로봇 — 왼쪽 위
    rx, ry = 210, 250
    s.add(f'<circle cx="{rx}" cy="{ry}" r="46" fill="{HW[1]}" stroke="{HW[0]}" '
          f'stroke-width="2"/>')
    s.text(rx, ry + 7, "M0609", size=17, fill=HW[0], anchor="middle", weight=700)
    # 라벨을 원 위에 둔다 — 아래에 두면 작업대로 가는 점선이 글씨를 가로지른다.
    s.text(rx, ry - 68, "6축 협동로봇 + RG2", size=14, fill=MUTED, anchor="middle")

    # 오른쪽 박스 — 오른쪽 위
    s.rect(470, 150, 250, 150, stroke=PERCEPTION[0], fill=PERCEPTION[1], r=12, sw=2)
    s.text(595, 212, "오른쪽 박스", size=19, fill=PERCEPTION[0], anchor="middle",
           weight=700)
    s.text(595, 240, "right_box", size=14, fill=PERCEPTION[0], anchor="middle",
           family=MONO)

    # 작업대 — 오른쪽 아래, 위에 물체 여러 개
    s.rect(400, 360, 330, 350, stroke=HW[0], fill="#f4f2ee", r=12, sw=2)
    s.text(420, 392, "작업대", size=18, fill=HW[0], weight=700)
    cols, cw, ch = 2, 145, 56
    for i, (ko, en, _lvl) in enumerate(OBJECTS):
        ox = 420 + (i % cols) * (cw + 14)
        oy = 414 + (i // cols) * (ch + 12)
        s.rect(ox, oy, cw, ch, stroke=NEUTRAL[0], fill="#ffffff", r=9, sw=1.4)
        s.text(ox + cw / 2, oy + 24, ko, size=15, fill=INK, anchor="middle")
        s.text(ox + cw / 2, oy + 43, en, size=11.5, fill=MUTED, anchor="middle",
               family=MONO)

    # 왼쪽 박스 — 왼쪽 아래
    s.rect(100, 520, 250, 150, stroke=GRASP[0], fill=GRASP[1], r=12, sw=2)
    s.text(225, 582, "왼쪽 박스", size=19, fill=GRASP[0], anchor="middle", weight=700)
    s.text(225, 610, "left_box", size=14, fill=GRASP[0], anchor="middle", family=MONO)

    # 집어서 분류한다
    s.arrow([(400, 470), (360, 470), (360, 560), (350, 560)], sw=1.8)
    s.label(330, 500, "분류", size=14)
    s.arrow([(560, 360), (560, 300)], sw=1.8)
    s.label(572, 336, "분류", size=14, anchor="start")
    s.arrow([(256, 250), (470, 200)], head="none", dash="5 4")
    s.arrow([(250, 285), (420, 400)], head="none", dash="5 4")
    s.arrow([(230, 296), (225, 520)], head="none", dash="5 4")
    s.text(78, 726, "점선 = 로봇 작업반경 안. 실제 좌표·바닥 높이는 bins.yaml 실측값을 쓴다.",
           size=13.5, fill=MUTED)

    # 오른쪽 — 작업물과 파지력
    s.panel(790, 104, 450, 330, "작업물 — YOLO11-seg 7클래스")
    yy = 158
    for ko, en, level in OBJECTS:
        s.text(816, yy, ko, size=15, fill=INK)
        s.text(940, yy, en, size=13, fill=MUTED, family=MONO)
        s.rect(1168, yy - 15, 40, 22, stroke=CONTROL[0], fill=CONTROL[1], r=11, sw=1.3)
        s.text(1188, yy + 1, f"g{level}", size=13, fill=CONTROL[0], anchor="middle",
               weight=600, family=MONO)
        yy += 36
    s.text(816, 414, "SAM+VLM 경로는 이 목록을 보지 않는다 — 처음 보는 물건도 인지한다.",
           size=13, fill=MUTED)

    s.panel(790, 460, 450, 284, "파지력 — 5단계 grip_level")
    yy = 514
    for tag, force, note, exc in GRIP_LEVELS:
        s.text(816, yy, tag, size=15, fill=CONTROL[0], family=MONO, weight=700)
        s.text(860, yy, force, size=15, fill=INK, family=MONO)
        s.text(920, yy, note, size=13.5, fill=MUTED)
        if exc:
            s.text(1232, yy, exc, size=12.5, fill=DANGER[0], anchor="end")
        yy += 34
    s.text(816, 692, "1이 가장 강하고 5가 가장 약하다(objects.yaml).", size=13,
           fill=MUTED)
    s.text(816, 714, "g4만 예외로 control이 10N을 건다 — 무른 물체는 25N에 걸리기 전에",
           size=13, fill=DANGER[0])
    s.text(816, 732, "눌려버려 Grip detected가 안 켜졌다(skill_params.yaml).", size=13,
           fill=DANGER[0])
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

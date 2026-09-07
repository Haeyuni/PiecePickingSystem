# VLM + SAM 인지 파이프라인

YOLO(`models/best.pt`, 7클래스)가 하던 "무엇이 어디에 있는가"를 VLM으로 대체하는 경로다.
학습한 7클래스 밖의 물건도 이름을 붙일 수 있다는 것이 바꾸는 이유다.

## 흐름

```text
UI "치약 왼쪽으로"
  → 현재 프레임 캡처
  → [1] SAM으로 장면을 조각냄 (everything 모드)      ← 좌표는 SAM이 만든다
  → [2] 조각에 번호를 그려 VLM에 질의                 ← 이름은 VLM이 붙인다
        · 몇 번이 무엇인가 → class_name / name_ko / is_new_class
        · 몇 번은 물체가 아닌가 → 작업대·케이블·그림자
        · 한 물체가 여러 조각이면 → part_of로 묶는다
  → 마스크 + depth → 3D 좌표 (기존 perception 계산 그대로)
  → [3] planner: 물체 목록 + 지시 → 스킬 시퀀스, 없는 물체면 거부
        (기존 grounding / llm_client / validator 그대로)
```

**이 순서는 실측으로 정해졌고, `vlm_sam_test.py`의 기본값(`--mode som`)이다.** 원래 설계는
"VLM에게 사진을 주고 바운딩박스를 받아 SAM에 넣는다"였는데, gpt-4o가 좌표를 못 맞춘다
(아래 실측 결과). 그래서 VLM이 잘하는 일(무엇인지 알아보기)과 못하는 일(어디인지 픽셀로
찍기)을 갈라 놓았다. 원래 순서는 `--mode box`로 남겨 뒀다 — grounding이 되는 모델
(gpt-5 계열, Gemini 등)을 쓰게 되면 더 단순하고 빨라서 다시 유리해질 수 있다.

**역할 분리는 기존과 같다.** 모델이 정하는 것은 "어떤 물체를 어디로"까지고, 파지 자세·
파지력은 코드가 정한다(NFR-03a).

### 인지 단계는 지시를 보지 않는다

`label_marks`에 사용자의 지시를 넣지 않는다. 넣으면 인지가 지시에 끌려간다(실측 결과의
"지시를 인지 단계에 같이 주면 안 된다" 참조). 어느 물체가 지시 대상인지는 물체 목록을
텍스트로 받는 `llm_client.plan`이 정한다 — 원래 그 모듈의 일이다(FR-10/FR-11).
덕분에 같은 프레임에 지시를 여러 번 물어도 인지 결과는 그대로다.

### 클래스 어휘

프롬프트에 `src/perception/config/objects.yaml`의 등록 클래스를 함께 넣는다. 어휘를 안 주면
같은 물체를 매번 다른 이름으로 부르고(치약/toothpaste/tube) 전부 신규품목으로 떨어져
fragile 프로파일이 강제된다(FR-05b) — 오류로 보이지 않고 느려지기만 하는 종류의 문제다.
목록에 없는 물체는 `is_new_class=true`로 표시되어 신규품목 경로를 탄다.

## 테스트

```bash
# 준비 (1회) — 이 파이프라인은 ROS가 필요 없다
python3.12 -m venv .venv
.venv/bin/pip install ultralytics openai        # torch는 ultralytics가 함께 받는다
cp .env.example .env && $EDITOR .env            # OPENAI_API_KEY

# 기본 경로(som): SAM 먼저 → VLM이 이름 → planner가 지시 해석 (3단계 전부)
.venv/bin/python tools/scripts/vlm_sam_test.py services/planner/image.png \
    --command "치약 왼쪽으로"

# SAM 전체 분할(CPU 26초)만 재사용하고 지시만 바꿔 가며
.venv/bin/python tools/scripts/vlm_sam_test.py services/planner/image.png \
    --reuse-marks --command "우산 왼쪽으로"

# 원래 물어본 경로 (VLM 박스 → SAM 프롬프트)
.venv/bin/python tools/scripts/vlm_sam_test.py services/planner/image.png \
    --mode box --command "치약 왼쪽으로" --plan

# API를 안 부르고 뒤 단계만 반복
.venv/bin/python tools/scripts/vlm_sam_test.py services/planner/image.png \
    --from-json data/samples/vlm_sam/image_scene.json
```

GPU에서 돌리려면 `--device 0`. MobileSAM 가중치(~40MB)는 첫 실행에 자동으로 받아
`models/mobile_sam.pt`에 둔다(`*.pt`는 커밋하지 않는다).

출력은 `data/samples/vlm_sam/`에 남는다.

| 파일 | 내용 |
| --- | --- |
| `*_marks.png` | SAM 조각에 번호를 그린 이미지 — **VLM에 실제로 보낸 그림** (som) |
| `*_marks.npz` | 번호별 bool 마스크 (`--reuse-marks`가 읽는다) |
| `*_scene.json` | VLM 원본 응답 (`--from-json`으로 재사용) |
| `*_boxes.png` | 물체 박스 오버레이 |
| `*_masks.png` | 물체 마스크 오버레이 |
| `*_masks.npz` | object_id별 bool 마스크 |
| `*_result.json` | 마스크 통계까지 포함한 전체 결과 |

같은 stem의 `*_depth.npy` / `*_info.json`이 옆에 있으면(=`perception_capture.py`로 뜬 프레임)
마스크 기준 카메라 좌표(mm)와 `depth_valid_ratio`까지 함께 뽑는다.

### 마스크를 판단하는 값

- `box_fill` — 물체의 외접 사각형 안을 마스크가 얼마나 채웠나. 지나치게 낮으면 물체가
  아니라 무늬 조각을 잡은 것이다(길쭉하게 누운 물체는 원래 낮게 나온다).
- `outside_ratio` — 사각형 밖으로 샌 비율. 높으면 배경이나 그림자까지 삼켰다는 뜻이다.

## 실측 결과 (2026-09-07)

측정 조건: `services/planner/image.png`(1280x720, 물티슈·섬유탈취제·치약 3종), gpt-4o
(이 계정에서 쓸 수 있는 이미지 입력 모델은 gpt-4o뿐이다), MobileSAM, **CPU 추론**
(측정 노트북 GPU는 MX450 2GB — 워크스테이션 RTX 4060에서는 SAM 쪽이 훨씬 빠르다).

### 테스트 1 — API가 무엇이 있는지 알아보고 바운딩박스를 주는가

| | 결과 |
| --- | --- |
| 무엇이 있는지 | **된다.** `wet_wipes` / `fabric_spray` / `toothpaste`를 objects.yaml 어휘 그대로 맞혔다 |
| 바운딩박스 좌표 | **못 쓴다.** 좌표가 100px 단위로 뭉개지고 물체 위에 얹히지 않는다 |
| 왕복 시간 | 8~12초 (detail=high, 1280x720) |

박스가 어떻게 틀렸는지가 요점이다. `--mode box` 첫 실행에서 gpt-4o는 `toothpaste`의 박스로
(768,216)-(1152,360)을 줬는데, **그 자리에 있는 것은 치약이 아니라 섬유탈취제였다.**
물티슈 박스는 실제 물체의 위쪽 절반만 덮었고, 3종 중 섬유탈취제는 아예 빠졌다.
0~1000 격자를 그려 넣고 다시 물어도(실험) 좌표는 여전히 100 단위로만 나왔다.
`temperature=0`인데도 같은 사진에 대해 실행마다 박스가 달라졌다(y1이 144 → 216 → 288).

그 박스를 SAM에 넣으면 결과는 박스가 가리키는 것을 정확히 따라간다 — `box_fill=0.15`,
즉 마스크가 박스 안을 15%밖에 못 채웠고, 잡힌 것은 물티슈 포장의 무늬 조각이었다.
**틀린 박스는 SAM을 통과해도 틀린 채로 남는다.**

### 테스트 2 — 받은 좌표로 SAM이 세그멘테이션할 수 있는가

**된다. 박스만 맞으면 마스크는 픽셀 단위로 맞는다.** 박스 프롬프트 방식(`--mode box`)에서
SAM 자체는 1.0초에 박스 2개를 처리했고(CPU), 우연히 섬유탈취제 위에 얹힌 박스로부터
섬유탈취제의 윤곽을 정확히 뽑아냈다. 즉 **테스트 2의 병목은 SAM이 아니라 테스트 1의 좌표다.**

### 그래서 순서를 뒤집었고, 그쪽을 기본값으로 뒀다 (`--mode som`)

VLM이 잘하는 일과 못하는 일을 갈라 놓는다.

```text
SAM everything 모드로 장면을 조각냄  → 좌표는 SAM이 만든다 (정확)
  → 조각마다 번호를 그린 이미지를 VLM에 보냄
  → VLM은 "몇 번이 무엇인가"만 답함   → 분류는 VLM이 한다 (정확)
```

같은 사진에서:

| | box 모드 | som 모드 |
| --- | --- | --- |
| 찾은 물체 | 2 / 3 (섬유탈취제 누락) | **3 / 3** |
| 박스 위치 | 물체와 어긋남 | 마스크에서 뽑으므로 정확 |
| `box_fill` | 0.15 / 0.58 | 0.64 / 0.51 / 0.32 (마스크가 물체 윤곽) |
| 실행 간 재현성 | 실행마다 박스가 달라짐 | 4회 연속 동일 |
| 시간 | VLM 8~12s + SAM 1s | SAM 26s(CPU) + VLM 8~10s |

som 모드의 SAM 전체 분할은 CPU에서 26초다 — GPU에서는 1~2초대가 된다. 명령 하나마다
다시 돌릴 필요는 없다(`--reuse-marks`).

### 지시(명령문)를 인지 단계에 같이 주면 안 된다

처음에는 사진과 지시를 한 번에 보내 "무엇이 있는지 + 어느 것이 지시 대상인지"를 같이
물었다. 그러면 **인지가 지시에 끌려간다.**

- `"우산 왼쪽으로"`를 함께 주자, 사진에 우산이 없는데도 배경 조각 하나(x 0~115)를
  `umbrella`라고 답했다.
- `"치약 왼쪽으로"`, `"우산 왼쪽으로"`, `"화장품 왼쪽으로"` 세 지시 모두에 대해 지시 대상으로
  같은 번호(섬유탈취제)를 돌려줬다.

그래서 `label_marks`는 지시를 받지 않는다. 지시 해석은 물체 목록을 **텍스트로** 받는
`llm_client.plan`이 한다 — 원래 그 모듈의 일이고(FR-10/FR-11), 분리해 두면 같은 프레임에
다른 지시를 여러 번 물어도 인지 결과가 흔들리지 않는다. 실제로 분리 후 네 가지 지시에서
검출 결과가 완전히 동일했다.

분리한 뒤 지시 해석 결과(som 모드, 3단계까지):

| 지시 | 결과 |
| --- | --- |
| `치약 왼쪽으로` | pick `toothpaste_1` → place_into `left_box` |
| `전부 오른쪽으로` | 3종 전부에 대해 pick/place_into 6스텝 |
| `우산 왼쪽으로` | 거부 — "'우산'은 검출된 물체 목록에 없습니다" |
| `화장품 왼쪽으로` | 거부 — "어떤 물체가 화장품인지 명확하지 않습니다" |

마지막 줄이 처음 예시로 든 지시다. 이 사진에는 화장품(선크림·젤네일)이 없으므로 거부가
맞는 동작이다. 다만 **VLM에게 직접 물었을 때는 매번 다른 물체를 화장품이라고 골랐다**
(물티슈, 섬유탈취제) — 범주 지시를 gpt-4o가 안정적으로 처리하지 못한다는 뜻이므로,
선크림이 실제로 있는 장면에서 다시 확인해야 한다.

### 테스트 사진의 오염

`services/planner/image.png`에는 이전 YOLO 검출 결과가 이미 그려져 있다(파란 사각형과
`dentimate_new_total_care_toothpaste 0.98` 배너). SAM은 그 배너를 별도 조각으로 잡았고,
gpt-4o는 배너의 **글자를 읽고** 그것을 치약의 일부라고 답했다. 지금은 조각 합치기에
기하 검사를 넣어(대표 조각보다 크거나 외접 사각형을 1.6배 넘게 키우면 버린다) 걸러내지만,
**다음 측정은 오버레이가 없는 원본 프레임으로 해야 한다**
(`tools/scripts/perception_capture.py`로 뜬 `data/samples/*.png`).

## 아직 안 된 것

- **런타임 노드에 붙지 않았다.** 지금은 정지 이미지 스크립트뿐이고, `perception/node.py`는
  여전히 YOLO를 쓴다. 붙이는 설계는 `on-demand-perception.md`에 정리했다 — 명령당 VLM 1회,
  스텝 사이는 재투영 프롬프트로 SAM 1패스.
- **YOLO와 같은 프레임에서 비교하지 않았다.** som 모드 마스크와 `models/best.pt` 마스크의
  IoU를 재야 "YOLO를 대체할 수 있는가"를 수치로 말할 수 있다. 지금 말할 수 있는 것은
  "3종을 이름까지 맞혔고 마스크가 눈으로 정확하다"까지다.
- **오염되지 않은 사진으로 다시 재야 한다** (위 "테스트 사진의 오염").
- **범주 지시("화장품")가 gpt-4o에서 불안정하다.** 선크림·젤네일이 실제로 있는 장면에서
  다시 확인해야 한다.
- 신규 클래스 등록(FR-05a~c)과 이어지지 않았다. `is_new_class=true`가 나와도 지금은
  `object_attributes`에 기록되지 않는다.

# docs

문서는 네 종류다 — **요구사항**, **계약(인터페이스)**, **기록(문제·결과)**, **환경**.
저장소 전체 개요와 실행 순서는 루트 `README.md`에 있다.

| 문서 | 무엇을 정하나 | 언제 보나 |
|---|---|---|
| [`BR(Business Requirement).md`](BR%28Business%20Requirement%29.md) | 요구사항 — 범위(1차/2차/확장), FR/NFR, 0~10단계 로드맵 | "이게 왜 필요한가"를 물을 때 |
| [`시스템명세서.md`](시스템명세서.md) | DB 스키마, 기술 스택, 하드웨어, 배포 구성 | 서비스를 추가·교체할 때 |
| [`인터페이스_정의서.md`](인터페이스_정의서.md) | `sort_msgs` 메시지·액션, `objects.yaml`/`bins.yaml` 스키마, 단위 규칙 | 계층 경계를 건드릴 때 |
| [`웹_인터페이스_정의서.md`](웹_인터페이스_정의서.md) | 브라우저↔web↔planner의 HTTP/WebSocket 계약 | API를 추가·변경할 때 |
| [`화면정의서.md`](화면정의서.md) | 화면 3종의 구성 요소와 데이터 출처 | UI를 고칠 때 |
| [`개발계획.md`](개발계획.md) | 개발 순서, 결정 기록(D-1~D-8), 마일스톤 | "왜 이렇게 정했나"를 물을 때 |
| [`on-demand-perception.md`](on-demand-perception.md) | 주기 발행 → 온디맨드 관측 전환 설계와 그 결정들 | 관측 트리거·타임아웃을 만질 때 |
| [`vlm_sam_pipeline.md`](vlm_sam_pipeline.md) | SAM+VLM 인지 파이프라인과 실측 결과 | `detector=vlm_sam`을 다룰 때 |
| [`environment.md`](environment.md) | 런타임·CUDA/PyTorch·베이스 이미지 버전 고정 | 장비를 옮기거나 의존성을 올릴 때 |
| [`problem/`](problem) | 실물에서 겪은 문제의 증상 → 증거 → 원인 → 조치 | 같은 증상이 다시 보일 때 |
| [`results/`](results) | 단계별 실험 리포트 (BR 4.1절 산출물) | 지표를 인용할 때 |
| [`diagrams/`](diagrams) | 루트 `README.md`가 쓰는 그림 6장 | 구조가 바뀌었을 때 |

**인터페이스 문서와 구현이 어긋나면 문서를 먼저 고치고 구현을 맞춘다** — 각 계층이 서로의
내부를 모른 채 이 계약만 보고 개발한다는 전제가 깨지지 않도록(`개발계획.md` 8절).

**`diagrams/`의 PNG는 손으로 그린 것이 아니다.** `tools/scripts/make_diagrams.py`가 좌표와
문구를 들고 있고, 그것을 고쳐 다시 돌리면 6장이 한 번에 갱신된다.

```bash
python3 tools/scripts/make_diagrams.py
```

원본 없는 그림은 낡았을 때 고치는 대신 지우게 된다 — 실제로 이 폴더에 있던 ERD.png가
제거된 테이블을 그대로 그린 채 남아 있다가 그렇게 됐다.

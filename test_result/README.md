# test_result/

`tools/scripts/vlm_sam_test.py`가 [`../test_image/`](../test_image/)를 돌린 결과가 쌓이는
곳이다. json은 안 만들고, "LLM에 보내기 전"과 "VLM이 걸러낸 후"를 나란히 볼 수 있게
둘 다 남긴다. 이미지 stem별로:

| 파일 | 내용 |
| --- | --- |
| `<stem>_before.png` | SAM 원본 조각에 번호만 붙인 것 — **LLM에 실제로 보낸 그림** (`--no-vlm`이면 안 만든다) |
| `<stem>_after.png` | 전체 이미지에 물체별 윤곽선+라벨을 그린 **최종 결과** — 웹에서 보는 `/grasp/debug_image`와 같은 스타일(채우기 없음) |
| `<stem>_objects/<object_id>.png` | 물체별로 배경을 지우고 마스크만 잘라낸 컷아웃(투명 PNG) |
| `<stem>_before.npz` | 번호별 원시 마스크 (`--reuse-marks`가 SAM 재계산을 건너뛸 때 읽는 캐시 — 이미지 아님) |

내용물은 실행할 때마다 새로 생기므로 git에는 올리지 않는다(`.gitignore`).

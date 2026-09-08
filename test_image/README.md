# test_image/

SAM 세그멘테이션 파라미터를 튜닝할 때 쓸 정지 이미지를 여기 넣는다. png/jpg/jpeg만 인식한다.

```
.venv/bin/python tools/scripts/vlm_sam_test.py                       # 여기 전체를 돈다
.venv/bin/python tools/scripts/vlm_sam_test.py test_image/scene1.png # 파일 하나만
```

결과는 [`../test_result/`](../test_result/)에 쌓인다. 자세한 옵션은
`tools/scripts/vlm_sam_test.py`의 docstring과 `docs/vlm_sam_pipeline.md` 참고.

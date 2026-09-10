# data

원본 이미지, 포인트클라우드, 모델 가중치 등 대용량 파일을 두는 볼륨 마운트 경로. DB에는 경로와 메타데이터만 저장한다(시스템명세서 2.4절).

- `samples/`: 학습·평가용 샘플 데이터
- `unknown_crops/`: 신규 클래스 크롭 이미지 (VLM 속성 제안용, 시스템명세서 3.1a절)
- `scenes/`: 파지 비교용 `scene_XXX.npz` (rgb / depth_m / mask / K). `scripts/make_scene_npz.py`로 생성한다.
- `point_clouds/`: graspnet_baseline이 추론에 실제로 넣은 물체 단일 포인트클라우드
  (`<trace_id>/<observation_id>_<object_id>.npz`, camera frame mm). GraspNet fine-tuning용
  — execution_logs.point_cloud_path가 이 아래 경로를 가리킨다(grasp/node.py:_save_point_cloud).

이 디렉터리의 실제 데이터 파일은 git으로 버전관리하지 않는다 (`.gitignore` 참조).

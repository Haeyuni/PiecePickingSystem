# Grasp Runtime

`heuristic_pca`는 ROS host 또는 compose 컨테이너에서 실행할 수 있다. `graspnet_baseline`은
PyTorch/CUDA를 ROS 환경과 섞지 않기 위해 **host에서 실행한 grasp node**가 별도 Docker runtime을
호출한다. compose의 grasp 컨테이너에는 Docker socket/CLI를 넣지 않는다.

## 개발 PC

개발 PC는 Docker image, PyTorch, GPU, checkpoint가 없어도 된다. 기본 전략은 `heuristic_pca`이며
`graspnet_baseline`을 선택하지 않는 한 Docker를 검사하거나 build/pull하지 않는다.

```bash
colcon build --merge-install --packages-select perception_common sort_msgs perception grasp control web
source install/setup.bash
pytest -q src/grasp/test/test_pointcloud_pca.py src/grasp/test/test_graspnet_baseline.py
```

`PIECE_PICKING_ASSETS_DIR` 설정은 개발 PC에서는 선택 사항이다. GraspNet 실제 GPU 추론은
**HW_UNVERIFIED**이며, 테스트 PC에서만 준비·검증한다. 이 단계에서는 Docker 명령을 실행하지 않는다.

## 테스트 PC 최초 1회

1. Git pull 후 공용 Drive의 `piece_picking_assets`를 `$HOME/piece_picking_assets`에 준비한다.
2. assets 안에 내려받은 archive가 있는지 확인한다.

```text
$HOME/piece_picking_assets/models/graspnet/checkpoint-rs.tar
```

3. Docker와 NVIDIA Container Toolkit이 준비된 host terminal에서 실행한다.

```bash
export PIECE_PICKING_ASSETS_DIR="$HOME/piece_picking_assets"
bash src/grasp/scripts/setup_graspnet_runtime.sh
```

이 스크립트만 Docker image build와 CUDA/checkpoint load를 수행한다. image
`piece-picking-graspnet-baseline:0.1.0`가 이미 있으면 다시 build/pull하지 않는다. archive는 이
최초 setup에서만 풀리고, 최종 checkpoint는 아래 경로에 보관된다.

```text
$HOME/piece_picking_assets/models/graspnet/checkpoint.tar
```

runtime은 이 단일 `checkpoint.tar`만 `/checkpoint.tar:ro`로 read-only mount해 `torch.load()`한다.
runtime에는 archive 해제, checkpoint 다운로드, Docker build/pull, pip install 코드가 없다. archive가
`checkpoint.tar`를 포함하지 않으면 setup은 실패하므로, 공용 Drive의 파일 형식을 먼저 확인해야 한다.
이 단계는 로봇을 움직이지 않는다.

이미지 tag를 바꿀 때는 `config/grasp_params.yaml`의 `graspnet_baseline.image`와 setup 명령의
`GRASPNET_IMAGE`를 같은 값으로 맞춘다.

```bash
GRASPNET_IMAGE="piece-picking-graspnet-baseline:team-v1" \
  bash src/grasp/scripts/setup_graspnet_runtime.sh
```

## 테스트 PC 이후

assets 경로를 유지한 채 Git pull, build, launch만 한다. 실행 단계는 Docker build/pull, pip install,
checkpoint download를 절대 수행하지 않는다.

```bash
export PIECE_PICKING_ASSETS_DIR="$HOME/piece_picking_assets"
colcon build --merge-install --packages-select perception_common sort_msgs perception grasp control web
source install/setup.bash
ros2 launch grasp grasp_launch.py config_path:=/absolute/path/grasp_test_pc.yaml
```

`grasp_test_pc.yaml`은 기본 `config/grasp_params.yaml`의 전체 복사본으로 만들고,
`strategy.name: graspnet_baseline` 및 테스트 셀 값을 설정한다. launch의 `config_path`가 이 파일을
node parameter로 전달한다. 필요한 경우 YAML을 바꾸지 않고 아래처럼 strategy만 override할 수 있다.

```bash
ros2 launch grasp grasp_launch.py \
  config_path:=/absolute/path/grasp_test_pc.yaml \
  strategy:=graspnet_baseline
```

`graspnet_baseline.T_graspnet_tcp_mm`은 **TCP frame을 GraspNet predicted-gripper frame으로 변환하는
mm 4x4 행렬**이다. 최종 후보는
`T_base_tcp_mm = T_base_camera_mm @ T_camera_graspnet_mm @ T_graspnet_tcp_mm`으로 계산된다.
이 값은 테스트 셀의 RG2 TCP 정의와 GraspNet 축 관계를 측정해 넣어야 하며, 기본값은 비어 있다.
보정값 없이 나온 파지 자세는 실제 로봇에 신뢰할 수 없다. 준비하지 않은 image나 transform을 선택하면
node는 오류만 안내하며, 자동 설치나 PCA fallback을 하지 않는다.

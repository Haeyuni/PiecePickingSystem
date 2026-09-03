"""mock 픽스처를 실제 WorldState 메시지로 발행한다 (개발계획.md 4.2절 B4).

검출 모델이 준비되기 전까지(트랙 A) `/world_state`의 발행자 역할을 대신한다. 실제
`perception`+`grasp` 대신 `data/mock/world_state_*.json`을 읽어 그대로 발행할 뿐이다.

이 노드의 가치는 편의가 아니라 **검증**에 있다: JSON 픽스처를 실제 `.msg` 타입에 채워
넣으므로, 픽스처와 메시지 정의가 어긋나 있으면 여기서 예외로 드러난다. mock으로 개발한
상위 계층이 실물 메시지에서도 성립하는지 확인하는 지점이다(계획서 7절 "mock과 실제
world_state의 괴리" 리스크).

실행:
    ros2 run perception fake_world_publisher --ros-args -p fixture:=world_state_normal
"""
import json
import os
import pathlib

import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point, Pose
from rclpy.node import Node

from sort_msgs.msg import DetectedObject, GraspCandidate, WorldState

def default_mock_dir() -> pathlib.Path:
    """`data/mock`의 위치. 소스 트리와 colcon 설치 트리 양쪽에서 찾아야 한다.

    `__file__` 기준 상대경로만 쓰면 **설치본에서 조용히 깨진다** — 설치되면 이 파일은
    `install/perception/lib/python3.12/site-packages/`로 복사되는데, 픽스처는 저장소의
    `data/mock/`에 남아 있기 때문이다(패키지 밖 데이터라 share에 설치되지 않는다).
    `place_server.bins_yaml_path()`가 막아 둔 것과 같은 함정이다.

    픽스처를 이미지·설치본에 굽지 않는 이유는 마이그레이션·시드와 같다: 픽스처 한 줄
    고치려고 colcon 빌드를 다시 돌릴 이유가 없다.
    """
    override = os.environ.get("MOCK_DIR")
    if override:
        return pathlib.Path(override)

    here = pathlib.Path(__file__).resolve()
    # 1) 소스에서 직접 실행: 조상 어딘가에 data/mock이 있다
    for parent in here.parents:
        candidate = parent / "data" / "mock"
        if candidate.is_dir():
            return candidate
    # 2) colcon 설치본: 워크스페이스 루트(install/ 위)에서 src/*/data/mock을 찾는다
    for parent in here.parents:
        if parent.name == "install":
            for found in sorted((parent.parent / "src").glob("*/data/mock")):
                if found.is_dir():
                    return found
            break
    raise RuntimeError(
        "data/mock을 찾지 못했습니다. MOCK_DIR 환경변수나 "
        "--ros-args -p mock_dir:=<경로>로 지정하세요."
    )


def _pose(data: dict) -> Pose:
    pose = Pose()
    pose.position.x = float(data["position"]["x"])
    pose.position.y = float(data["position"]["y"])
    pose.position.z = float(data["position"]["z"])
    pose.orientation.x = float(data["orientation"]["x"])
    pose.orientation.y = float(data["orientation"]["y"])
    pose.orientation.z = float(data["orientation"]["z"])
    pose.orientation.w = float(data["orientation"]["w"])
    return pose


def to_msg(fixture: dict) -> WorldState:
    """JSON 픽스처 → WorldState.msg. 필드가 어긋나면 여기서 KeyError로 드러난다."""
    msg = WorldState()
    msg.schema_version = fixture["schema_version"]
    msg.trace_id = fixture.get("trace_id", "")
    msg.stamp = Time(sec=fixture["stamp"]["sec"], nanosec=fixture["stamp"]["nanosec"])
    msg.frame_id = fixture.get("frame_id", "base")
    msg.needs_reobserve = list(fixture.get("needs_reobserve", []))

    for item in fixture.get("objects", []):
        obj = DetectedObject()
        obj.object_id = item["object_id"]
        obj.class_name = item["class_name"]
        obj.name_ko = item.get("name_ko", "")
        obj.confidence = float(item.get("confidence", 0.0))
        obj.position_base_mm = Point(
            x=float(item["position_base_mm"]["x"]),
            y=float(item["position_base_mm"]["y"]),
            z=float(item["position_base_mm"]["z"]),
        )
        obj.depth_valid_ratio = float(item.get("depth_valid_ratio", 0.0))
        obj.graspable = bool(item.get("graspable", True))
        obj.not_graspable_reason = item.get("not_graspable_reason", "")
        obj.mass_g = float(item.get("mass_g") or 0.0)
        obj.fragile = bool(item.get("fragile", True))
        obj.deformable = bool(item.get("deformable", False))
        obj.transparent = bool(item.get("transparent", False))
        obj.profile = item.get("profile", "fragile")
        obj.attr_source = item.get("attr_source", "yaml_seed")
        obj.needs_confirmation = bool(item.get("needs_confirmation", False))

        for cand in item.get("grasp_candidates", []):
            candidate = GraspCandidate()
            candidate.pose = _pose(cand["pose"])
            candidate.score = float(cand.get("score", 0.0))
            candidate.strategy = cand.get("strategy", GraspCandidate.STRATEGY_HEURISTIC_PCA)
            obj.grasp_candidates.append(candidate)

        msg.objects.append(obj)

    return msg


class FakeWorldPublisher(Node):
    def __init__(self):
        super().__init__("fake_world_publisher")
        self.declare_parameter("fixture", "world_state_normal")
        self.declare_parameter("mock_dir", str(default_mock_dir()))
        self.declare_parameter("period_s", 1.0)

        # 실제 grasp 노드와 같은 토픽에 발행한다 — 구독자(planner/web)는 차이를 모른다
        self._pub = self.create_publisher(WorldState, "/world_state", 10)

        fixture_name = self.get_parameter("fixture").value
        mock_dir = pathlib.Path(self.get_parameter("mock_dir").value)
        path = mock_dir / f"{fixture_name}.json"
        self._msg = to_msg(json.loads(path.read_text(encoding="utf-8")))

        period = float(self.get_parameter("period_s").value)
        self.create_timer(period, self._publish)
        self.get_logger().info(
            f"픽스처 {path} 발행 시작 (물체 {len(self._msg.objects)}개, {period}초 주기)"
        )

    def _publish(self) -> None:
        now = self.get_clock().now().to_msg()
        self._msg.stamp = now
        self._pub.publish(self._msg)


def main(args=None):
    rclpy.init(args=args)
    node = FakeWorldPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

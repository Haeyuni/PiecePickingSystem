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

DEFAULT_MOCK_DIR = pathlib.Path(
    os.environ.get(
        "MOCK_DIR",
        pathlib.Path(__file__).resolve().parents[2] / "data" / "mock",
    )
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
        self.declare_parameter("mock_dir", str(DEFAULT_MOCK_DIR))
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

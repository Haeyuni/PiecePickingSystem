"""control 노드 5종을 **한 프로세스**에서 함께 띄운다.

노드를 각각 다른 프로세스로 띄우면 `robot_state_publisher.store`(로봇 상태의 단일 소유자)를
공유할 수 없다. mode 전이를 액션 서버가 만들고 발행은 상태 노드가 하므로, 둘이 같은 메모리를
봐야 한다. 상태 소유자를 하나로 두기 위한 구성이다.

실행: ros2 run control control_node   (또는 launch/control_launch.py)
"""
import rclpy
from rclpy.executors import MultiThreadedExecutor

from .home_server import HomeServer
from .pick_server import PickServer
from .place_server import PlaceServer
from .robot_state_publisher import RobotStatePublisherNode, is_fake_robot
from .safety_monitor import SafetyMonitorNode


def main(args=None):
    rclpy.init(args=args)
    nodes = [
        PickServer(),
        PlaceServer(),
        HomeServer(),
        RobotStatePublisherNode(),
        SafetyMonitorNode(),
    ]
    # 액션 실행이 상태 발행을 막지 않도록 멀티스레드 실행기를 쓴다 — 단일 스레드면
    # pick이 도는 동안 robot_state가 멈춰 web의 busy 판단이 낡은 값에 걸린다.
    executor = MultiThreadedExecutor()
    for node in nodes:
        executor.add_node(node)

    nodes[0].get_logger().info(
        f"control 기동 완료: pick / place_into / home / robot_state / safety_monitor "
        f"({'fake' if is_fake_robot() else '실물'} 모드)")
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        for node in nodes:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

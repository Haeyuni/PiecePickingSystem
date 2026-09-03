"""단위 환산 (인터페이스_정의서.md 1.1절).

시스템 전 계층은 **mm**를 쓴다. ROS 표준 스택(MoveIt2·tf2·`JointState`)만 REP-103에 따라
m로 동작하므로, 그 경계에서만 환산한다. **이 파일이 그 유일한 지점이다** — 환산이 두 곳
이상에서 일어나는 순간 "전 계층 mm" 규칙은 지킬 수 없는 약속이 된다.
"""

MM_PER_M = 1000.0


def mm_to_m(value_mm: float) -> float:
    return value_mm / MM_PER_M


def m_to_mm(value_m: float) -> float:
    return value_m * MM_PER_M


def pose_mm_to_m(pose):
    """geometry_msgs/Pose(mm) → 같은 타입의 새 Pose(m). MoveIt2에 넘기기 직전에 쓴다."""
    from geometry_msgs.msg import Pose

    converted = Pose()
    converted.position.x = mm_to_m(pose.position.x)
    converted.position.y = mm_to_m(pose.position.y)
    converted.position.z = mm_to_m(pose.position.z)
    converted.orientation = pose.orientation  # 회전은 무차원이라 그대로
    return converted


def pose_m_to_mm(pose):
    """geometry_msgs/Pose(m) → 같은 타입의 새 Pose(mm). MoveIt2/tf2에서 받은 직후에 쓴다."""
    from geometry_msgs.msg import Pose

    converted = Pose()
    converted.position.x = m_to_mm(pose.position.x)
    converted.position.y = m_to_mm(pose.position.y)
    converted.position.z = m_to_mm(pose.position.z)
    converted.orientation = pose.orientation
    return converted

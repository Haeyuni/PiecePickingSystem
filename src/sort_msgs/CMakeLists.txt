cmake_minimum_required(VERSION 3.8)
project(sort_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/GraspCandidate.msg"
  "msg/DetectedObject.msg"
  "msg/InstanceMasks.msg"
  "msg/WorldState.msg"
  "msg/RobotState.msg"
  "msg/SafetyEvent.msg"
  "action/Pick.action"
  "action/PlaceInto.action"
  "action/Home.action"
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs builtin_interfaces
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

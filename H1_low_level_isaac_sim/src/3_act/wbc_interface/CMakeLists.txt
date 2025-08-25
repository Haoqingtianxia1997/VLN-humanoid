cmake_minimum_required(VERSION 3.8)
project(wbc_interface)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(wbc_interface
    "msg/WBCReference.msg"
    DEPENDENCIES
    "std_msgs"
    "geometry_msgs"
)

install(
    DIRECTORY launch/
    DESTINATION share/${PROJECT_NAME}
)


ament_export_dependencies(rosidl_default_runtime)

ament_package()
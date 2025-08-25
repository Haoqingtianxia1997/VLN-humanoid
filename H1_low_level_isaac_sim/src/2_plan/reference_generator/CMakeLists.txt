cmake_minimum_required(VERSION 3.8)
project(reference_generator)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(mpc_interface REQUIRED)
find_package(mpc REQUIRED)

add_executable(reference_generator_node 
    src/reference_generator_node.cpp
)

target_include_directories(reference_generator_node PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${mpc_INCLUDE_DIRS}
)

target_link_libraries(reference_generator_node
    ${mpc_LIBRARIES}
)

target_compile_features(reference_generator_node PUBLIC c_std_99 cxx_std_17)

# Declare dependencies
set(COMMON_DEPENDENCIES 
    rclcpp
    geometry_msgs
    mpc_interface
    mpc
)

ament_target_dependencies(reference_generator_node ${COMMON_DEPENDENCIES})

install(TARGETS reference_generator_node
  DESTINATION lib/${PROJECT_NAME})

# if(BUILD_TESTING)
#   find_package(ament_lint_auto REQUIRED)
#   # the following line skips the linter which checks for copyrights
#   # comment the line when a copyright and license is added to all source files
#   set(ament_cmake_copyright_FOUND TRUE)
#   # the following line skips cpplint (only works in a git repo)
#   # comment the line when this package is in a git repo and when
#   # a copyright and license is added to all source files
#   set(ament_cmake_cpplint_FOUND TRUE)
#   ament_lint_auto_find_test_dependencies()
# endif()

ament_package()

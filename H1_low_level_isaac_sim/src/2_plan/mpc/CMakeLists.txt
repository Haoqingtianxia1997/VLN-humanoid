cmake_minimum_required(VERSION 3.8)
project(mpc)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --- PYTHON ---
find_package(Python3 COMPONENTS Interpreter REQUIRED)
find_package(h1_description REQUIRED)

if(NOT VENV_PYTHON_EXECUTABLE)
    message(FATAL_ERROR "Please provide the path to the Python executable in your venv via --cmake-args -DVENV_PYTHON_EXECUTABLE=/path/to/venv/bin/python")
endif()
# This is needed to get the directory of the activate script.
get_filename_component(VENV_BIN_DIR ${VENV_PYTHON_EXECUTABLE} DIRECTORY)
set(VENV_ACTIVATE_SCRIPT ${VENV_BIN_DIR}/activate)
set(URDF_PATH ${h1_description_DIR}/../urdf/h1.urdf)
message(STATUS "URDF Path is: ${URDF_PATH}")
set(ACADOS_GENERATED_CODE_DIR ${CMAKE_CURRENT_BINARY_DIR}/c_generated_code)
set(ACADOS_GENERATED_LIB ${ACADOS_GENERATED_CODE_DIR}/libacados_ocp_solver_srbm_robot.so)
set(ACADOS_PYTHON_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/scripts/srbm_mpc.py)

add_custom_command(
    OUTPUT ${ACADOS_GENERATED_LIB}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_solver.sh
            ${VENV_ACTIVATE_SCRIPT}
            ${ACADOS_PYTHON_SCRIPT}
            ${URDF_PATH}
            ${ACADOS_GENERATED_CODE_DIR}

    DEPENDS ${ACADOS_PYTHON_SCRIPT} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_solver.sh
    COMMENT "Generating ACADOS solver via wrapper script..."
    USES_TERMINAL 
)

add_custom_target(generate_acados_code
    DEPENDS ${ACADOS_GENERATED_LIB}
)

# --- ROS Abhängigkeiten ---
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(mpc_interface REQUIRED)
find_package(wbc_interface REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

# --- ACADOS ---
set(ACADOS_SOURCE_DIR_VAR "$ENV{ACADOS_SOURCE_DIR}")
set(ACADOS_INCLUDE_PATH ${ACADOS_SOURCE_DIR_VAR}/include)
set(ACADOS_LIB_DIR ${ACADOS_SOURCE_DIR_VAR}/lib)
if(NOT EXISTS ${ACADOS_LIB_DIR})
    message(FATAL_ERROR "Acados lib directory not found: ${ACADOS_LIB_DIR}")
endif()

# --- EXECUTABLE ---
add_executable(mpc_node 
    src/mpc_node.cpp
)

add_dependencies(mpc_node generate_acados_code)

# --- INCLUDE DIRECTORIES ---
include_directories(
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${ACADOS_GENERATED_CODE_DIR}
    ${ACADOS_INCLUDE_PATH}
    ${ACADOS_INCLUDE_PATH}/acados
    ${ACADOS_INCLUDE_PATH}/blasfeo/include
    ${ACADOS_INCLUDE_PATH}/hpipm/include
)

# Include Directories
target_include_directories(mpc_node PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
# Link Libraries
target_link_libraries(mpc_node 
    ${ACADOS_GENERATED_LIB}
    ${ACADOS_LIB_DIR}/libacados.so
    ${ACADOS_LIB_DIR}/libblasfeo.so
    ${ACADOS_LIB_DIR}/libhpipm.so
    m
    Eigen3::Eigen
)

# Declare dependencies
set(COMMON_DEPENDENCIES 
    rclcpp
    geometry_msgs
    visualization_msgs
    nav_msgs
    mpc_interface
    wbc_interface
)

ament_target_dependencies(mpc_node ${COMMON_DEPENDENCIES})

# --- INSTALLATIONS ---
install(
    FILES ${ACADOS_GENERATED_LIB}
    DESTINATION lib
)

install(
    DIRECTORY ${ACADOS_GENERATED_CODE_DIR}/
    DESTINATION include/mpc/generated_acados
)


install(
    DIRECTORY include/
    DESTINATION include
)

install(
    TARGETS mpc_node
    RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# --- Finalization ---
ament_export_include_directories(
    "include"
    ${ACADOS_INCLUDE_PATH}
    ${ACADOS_INCLUDE_PATH}/acados
    ${ACADOS_INCLUDE_PATH}/blasfeo/include
    ${ACADOS_INCLUDE_PATH}/hpipm/include
)

ament_export_libraries(
    acados_ocp_solver_srbm_robot
    ${ACADOS_LIB_DIR}/libacados.so
    ${ACADOS_LIB_DIR}/libblasfeo.so
    ${ACADOS_LIB_DIR}/libhpipm.so
)
ament_package()
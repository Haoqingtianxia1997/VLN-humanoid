cmake_minimum_required(VERSION 3.8)
project(wbc)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --- Dependencies ---
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(pluginlib REQUIRED)
find_package(controller_interface REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(realtime_tools REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(wbc_interface REQUIRED)
find_package(pinocchio REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(proxsuite CONFIG REQUIRED)
find_package(generate_parameter_library REQUIRED)

# --- Parameter Library ---
# Generiert C++ Code aus Ihrer YAML-Datei für typsicheren Parameter-Zugriff
generate_parameter_library(
    ${PROJECT_NAME}_parameters
    "config/wbc_params.yaml"
)

# --- Controller Library (das Plugin) ---
# Erstellt eine EINZIGE SHARED Library, die vom ControllerManager geladen wird.
add_library(${PROJECT_NAME}_controller SHARED
    src/wbc_controller.cpp
    src/whole_body_controller.cpp
)

target_link_libraries(${PROJECT_NAME}_controller PUBLIC
    # 1. ROS 2-Abhängigkeiten über ihre `${PACKAGE_LIBRARIES}`-Variablen
    ${rclcpp_LIBRARIES}
    ${rclcpp_lifecycle_LIBRARIES}
    ${realtime_tools_LIBRARIES}
    ${controller_interface_LIBRARIES}
    ${pluginlib_LIBRARIES}
    ${sensor_msgs_LIBRARIES}
    ${nav_msgs_LIBRARIES}
    ${wbc_interface_LIBRARIES}
    # Nachrichtenpakete sind meist Header-Only und müssen hier nicht gelinkt werden

    # 2. Externe C++-Abhängigkeiten über ihre `::`-Ziele
    pinocchio::pinocchio
    Eigen3::Eigen
    proxsuite::proxsuite

    # 3. Lokal generierte Bibliotheken über ihren Target-Namen
    ${PROJECT_NAME}_parameters
)

# Include-Verzeichnisse für die Bibliothek
target_include_directories(${PROJECT_NAME}_controller PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    # $<BUILD_INTERFACE:${${PROJECT_NAME}_parameters_BUILD_DIR}/include>
    # $<INSTALL_INTERFACE:include>
    
    # Füge die Include-Verzeichnisse der ROS-Abhängigkeiten explizit hinzu
    ${rclcpp_INCLUDE_DIRS}
    ${rclcpp_lifecycle_INCLUDE_DIRS}
    ${realtime_tools_INCLUDE_DIRS}
    ${controller_interface_INCLUDE_DIRS}
    ${pluginlib_INCLUDE_DIRS}
    ${sensor_msgs_INCLUDE_DIRS}
    ${nav_msgs_INCLUDE_DIRS}
    # ${ament_cmake_core_INCLUDE_DIRS}
    ${wbc_interface_INCLUDE_DIRS}
)

# C++ Standard setzen
target_compile_features(${PROJECT_NAME}_controller PUBLIC cxx_std_17)

# Registriert das Plugin, damit ros2_control es finden kann
pluginlib_export_plugin_description_file(controller_interface wbc_controller_plugins.xml)

# --- Installation ---
install(TARGETS
    ${PROJECT_NAME}_controller
    ${PROJECT_NAME}_parameters
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)
install(FILES wbc_controller_plugins.xml DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

# --- Ament Package ---
ament_package()
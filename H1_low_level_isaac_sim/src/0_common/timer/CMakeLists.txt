cmake_minimum_required(VERSION 3.8)
project(timer)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)

# Declare header-only library
add_library(timer INTERFACE)
target_include_directories(timer INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
# Link dependencies to the interface target
target_link_libraries(timer INTERFACE
)

# Export targets and include directories
ament_export_include_directories(include)

# Install header files
install(
  DIRECTORY include/
  DESTINATION include
)

if(BUILD_TESTING)
endif()

ament_package()

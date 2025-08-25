cmake_minimum_required(VERSION 3.8)
project(h1_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

foreach(dir meshes urdf mjcf)
    install(DIRECTORY ${dir}
            DESTINATION share/h1_description)
endforeach(dir)


ament_package()

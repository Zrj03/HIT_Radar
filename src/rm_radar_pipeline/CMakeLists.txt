cmake_minimum_required(VERSION 3.22)
project(rm_radar_pipeline LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-O3 -Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(radar_interface REQUIRED)
find_package(geometry_msgs REQUIRED)
if(EXISTS "/usr/local/lib/cmake/opencv4/OpenCVConfig.cmake")
  set(OpenCV_DIR "/usr/local/lib/cmake/opencv4" CACHE PATH "" FORCE)
endif()
find_package(message_filters REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io kdtree segmentation)
find_package(Eigen3 3.4 REQUIRED)
find_package(TBB REQUIRED)

set(RM_RADAR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/rm_radar")

add_library(rm_radar_core SHARED
  ${RM_RADAR_DIR}/locate/locate.cpp
  ${RM_RADAR_DIR}/robot/robot.cpp
  ${RM_RADAR_DIR}/track/tracker.cpp
)

target_include_directories(rm_radar_core PUBLIC
  ${RM_RADAR_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)
target_link_libraries(rm_radar_core PUBLIC
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
  Eigen3::Eigen
  TBB::tbb
)

add_executable(rm_radar_pipeline_node src/rm_radar_pipeline_node.cpp)
target_include_directories(rm_radar_pipeline_node PRIVATE include ${RM_RADAR_DIR})
target_link_libraries(rm_radar_pipeline_node rm_radar_core ${OpenCV_LIBS} ${PCL_LIBRARIES})
ament_target_dependencies(rm_radar_pipeline_node
  rclcpp
  sensor_msgs
  std_msgs
  radar_interface
  geometry_msgs
  message_filters
  tf2
  tf2_ros
  tf2_eigen
  pcl_conversions
)

install(TARGETS rm_radar_core rm_radar_pipeline_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY models DESTINATION share/${PROJECT_NAME})

ament_package()

cmake_minimum_required(VERSION 3.0.2)
project(capston_package)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES opencv
  CATKIN_DEPENDS
    roscpp
    rospy
    std_msgs
    cv_bridge
)


find_package(OpenCV REQUIRED)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(lane_detection 
	src/lane_detection.cpp
	src/windowbox.cpp)
target_link_libraries(lane_detection ${catkin_LIBRARIES} ${OpenCV_LIBS} ) 

add_executable(imu_detection 
	src/imu_detection.cpp)
target_link_libraries(imu_detection ${catkin_LIBRARIES} ) 

add_executable(lidar_detection 
	src/lidar_detection.cpp)
target_link_libraries(lidar_detection ${catkin_LIBRARIES} ) 

add_executable(yolo_detection 
	src/yolo_detection.cpp)
target_link_libraries(yolo_detection ${catkin_LIBRARIES} ) 

add_executable(control_node 
	src/control_node.cpp)
target_link_libraries(control_node ${catkin_LIBRARIES} ) 

add_executable(planning_node 
	src/planning_node.cpp)
target_link_libraries(planning_node ${catkin_LIBRARIES} ) 
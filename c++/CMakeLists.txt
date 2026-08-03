cmake_minimum_required(VERSION 4.2)
project(YOLO_ONNX CXX)




# Force C++17 compilation for standard compatibility
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # Enable aggressive compiler optimization options for Raspberry Pi CPU
# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -pipe")
# endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# Force standard compiler flags to include debug symbols (-g)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")


# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include OpenCV header paths
include_directories(${OpenCV_INCLUDE_DIRS})

# Define the target executable
add_executable(yolo_infer main.cpp)

# Link OpenCV libraries to your executable
target_link_libraries(yolo_infer ${OpenCV_LIBS})

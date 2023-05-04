cmake_minimum_required(VERSION 2.6)

project(PIDNet)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

#add_definitions(-DCVT_TO_GIE_MODEL)

find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_75;code=sm_75;)


include_directories(${PROJECT_SOURCE_DIR})

include_directories(/path-to-trt/TensorRT-7.2.1.6/include/)
link_directories(/path-to-trt/TensorRT-7.2.1.6/lib/)

find_package(OpenCV 3.4.8 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


cuda_add_library(segresize SHARED ${PROJECT_SOURCE_DIR}/segresize.cu)

set(LIBRARIES nvinfer  cudart cudart cublas curand opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs segresize)

add_executable(main pidnet.cpp)
target_link_libraries(main ${LIBRARIES} ${ENGINE_LIB})

add_definitions(-O2 -pthread)


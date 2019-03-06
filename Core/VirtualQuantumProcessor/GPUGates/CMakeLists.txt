cmake_minimum_required(VERSION 3.1)
project(QPanda-2.0.GPUQGates)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -gencode arch=compute_50,code=sm_50")
if (MSVC)
    list(APPEND CUDA_NVCC_FLAGS -Xcompiler /FS -Xcompiler /w)
else()
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -w -Xcompiler -fPIC")   
endif()
file(GLOB_RECURSE CURRENT_HEADERS "*.hpp")
file(GLOB_RECURSE CURRENT_SOURCES "*.cu")
cuda_add_library(${PROJECT_NAME} STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)



cmake_minimum_required(VERSION 3.1)
project(GPUQGates)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -gencode arch=compute_50,code=sm_50")
if (MSVC)
    list(APPEND CUDA_NVCC_FLAGS -Xcompiler /FS -Xcompiler /w)
else()
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -w -Xcompiler -fPIC")   
endif()
file(GLOB_RECURSE CURRENT_SOURCES "*.cu" "*.cpp")
cuda_add_library(${LIB_GPUGATES} STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})
set_target_properties(${LIB_GPUGATES} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

INSTALL(TARGETS ${LIB_GPUGATES}
        EXPORT  ${TARGETS_EXPORT_NAME}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

target_include_directories(${LIB_GPUGATES} INTERFACE
                           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include/qpanda-2>)
        
export(TARGETS ${LIB_GPUGATES} 
       APPEND FILE ${PROJECT_BINARY_DIR}/qpanda-2.0.gpugates-target.cmake
       NAMESPACE QPANDA::
       )





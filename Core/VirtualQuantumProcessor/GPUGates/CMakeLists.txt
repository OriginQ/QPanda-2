cmake_minimum_required(VERSION 3.1)

include(FindCUDA)
include(FindCUDA/select_compute_arch)
#cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS 6.0)
cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS "Auto")


string(REPLACE ";" " "  CUDA_ARCH_FLAGS_EXPAND "${CUDA_ARCH_FLAGS}")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH_FLAGS_EXPAND} -isystem -use_fast_math --expt-extended-lambda")

if (MSVC)
    list(APPEND CUDA_NVCC_FLAGS -Xcompiler /FS -Xcompiler /w)
else()
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14 -w -Xcompiler -fPIC")   
endif()

file(GLOB_RECURSE CURRENT_HEADERS ${PROJECT_SOURCE_DIR}/include/Core/VirtualQuantumProcessor/GPUGates/*.cuh)
file(GLOB_RECURSE CURRENT_SOURCES *.cu *.cpp)
cuda_add_library(${LIB_GPUGATES} STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})
set_target_properties(${LIB_GPUGATES} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

INSTALL(TARGETS ${LIB_GPUGATES}
        EXPORT  ${TARGETS_EXPORT_NAME}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

target_include_directories(${LIB_GPUGATES} INTERFACE
                           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include/qpanda2>)
        
export(TARGETS ${LIB_GPUGATES} 
       APPEND FILE ${PROJECT_BINARY_DIR}/qpanda2.0.gpugates-target.cmake
       NAMESPACE QPANDA::
       )
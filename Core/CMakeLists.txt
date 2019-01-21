cmake_minimum_required(VERSION 3.1)
project(QPanda2.0)

if (FIND_CUDA)
    find_package(CUDA)
endif(FIND_CUDA)

if(CUDA_FOUND)
    option (USE_CUDA "Use cuda" ON)
    include_directories(${CUDA_INCLUDE_DIRS}
                        ${PROJECT_SOURCE_DIR}/VirtualQuantumProcessor/GPUGates)
    add_subdirectory(VirtualQuantumProcessor/GPUGates)
else(CUDA_FOUND)
    option (USE_CUDA "Use cuda" OFF)
endif(CUDA_FOUND)

file(GLOB_RECURSE QPandaCore_SRCS_H ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB_RECURSE QPandaCore_SRCS_CPP ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp )
add_library(${PROJECT_NAME} STATIC  ${QPandaCore_SRCS_H} ${QPandaCore_SRCS_CPP} )

if(CUDA_FOUND)
    target_link_libraries(${PROJECT_NAME} TinyXML QPanda-2.0.GPUQGates)   
else (CUDA_FOUND)
    target_link_libraries(${PROJECT_NAME} TinyXML)
endif(CUDA_FOUND)







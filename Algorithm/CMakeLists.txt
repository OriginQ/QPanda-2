cmake_minimum_required(VERSION 3.1)
project(QPandaAlgorithm)

file(GLOB_RECURSE QPandaAlgorithm_SRC_H ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB_RECURSE QPandaAlgorithm_SRC_CPP ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_library(${PROJECT_NAME} STATIC ${QPandaAlgorithm_SRC_H} ${QPandaAlgorithm_SRC_CPP} )
target_link_libraries(${PROJECT_NAME} QPanda2.0)

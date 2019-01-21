cmake_minimum_required(VERSION 2.8)
project(QAlg)

file(GLOB_RECURSE ALG_SRC_H ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB_RECURSE ALG_SRC_CPP ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_library(${PROJECT_NAME} STATIC ${ALG_SRC_H} ${ALG_SRC_CPP})
target_link_libraries(${PROJECT_NAME} QPanda2.0)

cmake_minimum_required(VERSION 2.8)
project(QPanda2.0)

file(GLOB_RECURSE QPandaCore_SRCS_H ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB_RECURSE QPandaCore_SRCS_CPP ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_library(${PROJECT_NAME} STATIC  ${QPandaCore_SRCS_H} ${QPandaCore_SRCS_CPP})
target_link_libraries(${PROJECT_NAME} TinyXML)




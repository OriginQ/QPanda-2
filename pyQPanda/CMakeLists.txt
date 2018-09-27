cmake_minimum_required(VERSION 2.8)
project(pyQPanda)

file(GLOB_RECURSE pyQPanda_SRC_CPP *.cpp)
add_library(${PROJECT_NAME} SHARED ${pyQPanda_SRC_CPP})
target_link_libraries(${PROJECT_NAME} QPanda2.0)

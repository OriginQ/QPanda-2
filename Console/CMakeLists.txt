cmake_minimum_required(VERSION 3.1)
project(Console)
add_executable(${PROJECT_NAME} QuantumProgramSystem.cpp)
target_link_libraries(${PROJECT_NAME} QPanda2.0 QPandaAlgorithm)

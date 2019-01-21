cmake_minimum_required(VERSION 3.1)
project(Variational)
file(GLOB_RECURSE VariationalQuantumCircuit_SRCS_H ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB_RECURSE VariationalQuantumCircuit_SRCS_CPP ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp )
add_library(${PROJECT_NAME} STATIC ${VariationalQuantumCircuit_SRCS_H} ${VariationalQuantumCircuit_SRCS_CPP} )


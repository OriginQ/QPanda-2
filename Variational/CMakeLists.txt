cmake_minimum_required(VERSION 3.1)
project(Variational)
file(GLOB_RECURSE VariationalQuantumCircuit_SRCS_H ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB_RECURSE VariationalQuantumCircuit_SRCS_CPP ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp )
add_library(${PROJECT_NAME} STATIC ${VariationalQuantumCircuit_SRCS_H} ${VariationalQuantumCircuit_SRCS_CPP} )
INSTALL(TARGETS ${PROJECT_NAME}
        EXPORT  ${TARGETS_EXPORT_NAME}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)


target_include_directories(${PROJECT_NAME} INTERFACE
                           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include/qpanda-2>)
        
export(TARGETS ${PROJECT_NAME} 
       APPEND FILE ${PROJECT_BINARY_DIR}/variationa-target.cmake
       NAMESPACE QPANDA::
       )





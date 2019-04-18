cmake_minimum_required(VERSION 3.1)
project(Variational)

set(LIB_VAR Variational)
file(GLOB_RECURSE VariationalQuantumCircuit_SRCS_H ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB_RECURSE VariationalQuantumCircuit_SRCS_CPP ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp )
add_library(${LIB_VAR} STATIC ${VariationalQuantumCircuit_SRCS_H} ${VariationalQuantumCircuit_SRCS_CPP} )
target_link_libraries(${LIB_VAR} ${LIB_QALG} ${LIB_QPANDA})

if (UNIX)
    set(LIB_SHARED_VAR DLL_VAR)
    add_library(${LIB_SHARED_VAR} SHARED ${VariationalQuantumCircuit_SRCS_CPP})
    target_link_libraries(${LIB_SHARED_VAR} ${LIB_QALG} ${LIB_QPANDA})
    set_target_properties(${LIB_SHARED_VAR} PROPERTIES
                          VERSION ${QPANDA_VERSION} SOVERSION ${QPANDA_MAJOR_VERSION})
    set_target_properties(${LIB_SHARED_VAR} PROPERTIES OUTPUT_NAME "${LIB_VAR}")
    set_target_properties(${DLL_VAR} PROPERTIES CLEAN_DIRECT_OUTPUT 1)
    set_target_properties(${DLL_VAR} PROPERTIES CLEAN_DIRECT_OUTPUT 1)
endif()

INSTALL(TARGETS ${LIB_VAR}
        EXPORT  ${TARGETS_EXPORT_NAME}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)


target_include_directories(${LIB_VAR} INTERFACE
                           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include/qpanda-2>)
        
export(TARGETS ${LIB_VAR}
       APPEND FILE ${PROJECT_BINARY_DIR}/variationa-target.cmake
       NAMESPACE QPANDA::
       )





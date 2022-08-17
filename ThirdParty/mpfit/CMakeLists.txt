cmake_minimum_required(VERSION 3.1)
project(mpfit)

file(GLOB_RECURSE SRC_H ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB_RECURSE SRC_C ${CMAKE_CURRENT_SOURCE_DIR}/*.c)

add_library(${LIB_MPFIT} STATIC ${SRC_H} ${SRC_C})

INSTALL(TARGETS ${LIB_MPFIT}
        EXPORT  ${TARGETS_EXPORT_NAME}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

target_include_directories(${LIB_MPFIT} INTERFACE
                           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/ThirdParty/mpfit/include>
                           $<INSTALL_INTERFACE:include/qpanda2/ThirdParty/mpfit/include>)

export(TARGETS ${LIB_MPFIT}
       APPEND FILE ${PROJECT_BINARY_DIR}/mpfit-target.cmake
       NAMESPACE QPANDA::
       )
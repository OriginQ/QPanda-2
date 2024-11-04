cmake_minimum_required(VERSION 3.1)
project(BZ2Project)

file(GLOB_RECURSE LIB_BZ2_SRC_H ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB_RECURSE LIB_BZ2_SRC_CPP ${CMAKE_CURRENT_SOURCE_DIR}/*.c)
add_library(${LIB_BZ2} STATIC ${LIB_BZ2_SRC_H} ${LIB_BZ2_SRC_CPP})

INSTALL(TARGETS ${LIB_BZ2}
        EXPORT  ${TARGETS_EXPORT_NAME}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

target_include_directories(${LIB_BZ2} INTERFACE
                           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/ThirdParty/bz2>
                           $<INSTALL_INTERFACE:include/qpanda2/ThirdParty/bz2>)

export(TARGETS ${LIB_BZ2}
       APPEND FILE ${PROJECT_BINARY_DIR}/bz2-target.cmake
       NAMESPACE QPANDA::
       )
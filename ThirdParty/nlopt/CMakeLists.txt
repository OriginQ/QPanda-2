cmake_minimum_required(VERSION 3.1)
project(NLOPT)


file(GLOB_RECURSE OPT_SRC_H ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
file(GLOB_RECURSE OPT_SRC_CPP ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_library(${LIB_NLOPT} STATIC ${OPT_SRC_H} ${OPT_SRC_CPP})



install(FILES ${OPT_SRC_H} DESTINATION 
        ${CMAKE_INSTALL_PREFIX}/include/qpanda2/ThirdParty/nlopt/include)

INSTALL(TARGETS ${LIB_NLOPT}
       EXPORT  ${TARGETS_EXPORT_NAME}
       ARCHIVE DESTINATION lib
       LIBRARY DESTINATION lib
       RUNTIME DESTINATION bin)

target_include_directories(${LIB_NLOPT} INTERFACE
                           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/ThirdParty/nlopt>
                           $<INSTALL_INTERFACE:include/qpanda2/ThirdParty/nlopt>)

export(TARGETS ${LIB_NLOPT}
       APPEND FILE ${PROJECT_BINARY_DIR}/nlopt-target.cmake
       NAMESPACE QPANDA::
       )
cmake_minimum_required(VERSION 3.1)
project(TinyXML)

set(LIB_XML TinyXML)
file(GLOB_RECURSE TinyXML_SRC_H ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB_RECURSE TinyXML_SRC_CPP ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_library(${LIB_XML} STATIC ${TinyXML_SRC_H} ${TinyXML_SRC_CPP})
INSTALL(TARGETS ${LIB_XML}
        EXPORT  ${TARGETS_EXPORT_NAME}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)


target_include_directories(${LIB_XML} INTERFACE
                           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/ThirdParty>
                           $<INSTALL_INTERFACE:include/qpanda-2/ThirdParty>)

export(TARGETS ${LIB_XML}
       APPEND FILE ${PROJECT_BINARY_DIR}/tinyxml-target.cmake
       NAMESPACE QPANDA::
       )








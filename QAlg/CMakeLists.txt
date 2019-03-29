cmake_minimum_required(VERSION 2.8)
project(QAlg)

file(GLOB_RECURSE ALG_SRC_CPP ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_library(${PROJECT_NAME} STATIC ${ALG_SRC_CPP})
target_link_libraries(${PROJECT_NAME} QPanda2.0)

INSTALL(TARGETS ${PROJECT_NAME}
        EXPORT  ${TARGETS_EXPORT_NAME}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)


target_include_directories(${PROJECT_NAME} INTERFACE
                           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include/qpanda-2>)
        
export(TARGETS ${PROJECT_NAME} 
       APPEND FILE ${PROJECT_BINARY_DIR}/qalg-target.cmake
       NAMESPACE QPANDA::
       )





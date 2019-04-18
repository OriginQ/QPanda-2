cmake_minimum_required(VERSION 2.8)
project(QAlg)

file(GLOB_RECURSE ALG_SRC_CPP ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_library(${LIB_QALG} STATIC ${ALG_SRC_CPP})
target_link_libraries(${LIB_QALG} ${LIB_QPANDA})

if (UNIX)
    set(LIB_SHARED_QALG DLL_QALG)
    add_library(${LIB_SHARED_QALG} SHARED ${ALG_SRC_CPP})
    target_link_libraries(${LIB_SHARED_QALG} ${LIB_QPANDA})
    set_target_properties(${LIB_SHARED_QALG} PROPERTIES
                          VERSION ${QPANDA_VERSION} SOVERSION ${QPANDA_MAJOR_VERSION})
    set_target_properties(${LIB_SHARED_QALG} PROPERTIES OUTPUT_NAME "${LIB_QALG}")
    set_target_properties(${DLL_QALG} PROPERTIES CLEAN_DIRECT_OUTPUT 1)
    set_target_properties(${DLL_QALG} PROPERTIES CLEAN_DIRECT_OUTPUT 1)
endif()

INSTALL(TARGETS ${LIB_QALG}
        EXPORT  ${TARGETS_EXPORT_NAME}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)


target_include_directories(${LIB_QALG} INTERFACE
                           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include/qpanda-2>)
        
export(TARGETS ${LIB_QALG}
       APPEND FILE ${PROJECT_BINARY_DIR}/qalg-target.cmake
       NAMESPACE QPANDA::
       )





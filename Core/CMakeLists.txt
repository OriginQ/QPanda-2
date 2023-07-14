cmake_minimum_required(VERSION 3.1)

if(CUDA_FOUND)
    set(LIB_GPUGATES GPUQGates)
    add_subdirectory(VirtualQuantumProcessor/GPUGates)
    list(APPEND LIB_LIST ${LIB_GPUGATES})
endif (CUDA_FOUND)

set(QPandaCore_SRCS_SEARCH_DIR ${CMAKE_CURRENT_SOURCE_DIR})

if (USE_EXTENSION)
    file(GLOB_RECURSE QPandaCore_SRCS_CPP ${QPandaCore_SRCS_SEARCH_DIR}/*.cpp ${PROJECT_SOURCE_DIR}/Extensions/*.cpp)
else()
    file(GLOB_RECURSE QPandaCore_SRCS_CPP ${QPandaCore_SRCS_SEARCH_DIR}/*.cpp )
endif()

add_library(${LIB_QPANDA} STATIC ${QPandaCore_SRCS_CPP} )
list(APPEND LIB_LIST ${LIB_ANTLR4})
list(APPEND LIB_LIST ${LIB_MPFIT})
target_link_libraries(${LIB_QPANDA} ${LIB_LIST})

if (UNIX)
    set(LIB_SHARED_QPANDA DLL_QPANDA)
    add_library(${LIB_SHARED_QPANDA} SHARED ${QPandaCore_SRCS_CPP})
    target_link_libraries(${LIB_SHARED_QPANDA} ${LIB_LIST})
    set_target_properties(${LIB_SHARED_QPANDA} PROPERTIES
                          VERSION ${QPANDA_VERSION} SOVERSION ${QPANDA_MAJOR_VERSION})
    set_target_properties(${LIB_SHARED_QPANDA} PROPERTIES OUTPUT_NAME "${LIB_QPANDA}")
    set_target_properties(${LIB_QPANDA} PROPERTIES CLEAN_DIRECT_OUTPUT 1)
    set_target_properties(${DLL_QPANDA} PROPERTIES CLEAN_DIRECT_OUTPUT 1)
endif()

INSTALL(TARGETS ${LIB_QPANDA}
        EXPORT  ${TARGETS_EXPORT_NAME}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

target_include_directories(${LIB_QPANDA} INTERFACE
                           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include/qpanda2>)
        
export(TARGETS ${LIB_QPANDA}
       APPEND FILE ${PROJECT_BINARY_DIR}/qpanda2.0-target.cmake
       NAMESPACE QPANDA::
       )

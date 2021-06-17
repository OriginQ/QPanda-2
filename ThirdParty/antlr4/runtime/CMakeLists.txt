
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/src/atn
  ${CMAKE_CURRENT_SOURCE_DIR}/src/dfa
  ${CMAKE_CURRENT_SOURCE_DIR}/src/misc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/support
  ${CMAKE_CURRENT_SOURCE_DIR}/src/tree
  ${CMAKE_CURRENT_SOURCE_DIR}/src/tree/pattern
  ${CMAKE_CURRENT_SOURCE_DIR}/src/tree/xpath
)

if (MSVC)
    set(extra_static_compile_flags "-DANTLR4CPP_STATIC")
    set(disabled_compile_warnings "/wd4251")
endif()

file(GLOB_RECURSE libantlrcpp_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_library(${LIB_ANTLR4} STATIC ${libantlrcpp_SRC})

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_libraries(${LIB_ANTLR4} ${UUID_LIBRARIES})
elseif(APPLE)
    target_link_libraries(${LIB_ANTLR4} ${COREFOUNDATION_LIBRARY})
endif()

set_target_properties(${LIB_ANTLR4}
                      PROPERTIES COMPILE_FLAGS "${disabled_compile_warnings} ${extra_static_compile_flags}")

INSTALL(TARGETS ${LIB_ANTLR4}
        EXPORT  ${TARGETS_EXPORT_NAME}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)


target_include_directories(${LIB_ANTLR4} INTERFACE
                           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/ThirdParty>
                           $<INSTALL_INTERFACE:include/qpanda2/ThirdParty>)

export(TARGETS ${LIB_ANTLR4}
       APPEND FILE ${PROJECT_BINARY_DIR}/antlr4-target.cmake
       NAMESPACE QPANDA::
       )




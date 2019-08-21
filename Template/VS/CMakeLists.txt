cmake_minimum_required(VERSION 3.1)
project(testQPanda)
SET(CMAKE_INSTALL_PREFIX  "C:/QPanda2") # QPanda2安装的路径,用户可根据自己设置的安装路径配置CMAKE_INSTALL_PREFIX
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_INSTALL_PREFIX}/lib/cmake")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT USE_MSVC_RUNTIME_LIBRARY_DLL)
    foreach (flag
        CMAKE_C_FLAGS
        CMAKE_C_FLAGS_DEBUG
        CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_MINSIZEREL
        CMAKE_C_FLAGS_RELWITHDEBINFO
        CMAKE_CXX_FLAGS
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL
        CMAKE_CXX_FLAGS_RELWITHDEBINFO)

        if (${flag} MATCHES "/MD")
            string(REGEX REPLACE "/MD" "/MT" ${flag} "${${flag}}")
        endif()
        if (${flag} MATCHES "/MDd")
            string(REGEX REPLACE "/MDd" "/MTd" ${flag} "${${flag}}")
        endif()
        if (${flag} MATCHES "/W3")
            string(REGEX REPLACE "/W3" "/W0" ${flag} "${${flag}}")
        endif()
    endforeach()
endif()

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

find_package(OpenMP)
if(OPENMP_FOUND)
    option(USE_OPENMP "find OpenMP" ON)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else(OPENMP_FOUND)
    option(USE_OPENMP "not find OpenMP" OFF)
endif(OPENMP_FOUND)

find_package(QPANDA REQUIRED)
if (QPANDA_FOUND)
    include_directories(${QPANDA_INCLUDE_DIR})
endif (QPANDA_FOUND)

add_executable(${PROJECT_NAME} test.cpp)
target_link_libraries(${PROJECT_NAME} ${QPANDA_LIBRARIES})
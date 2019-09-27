cmake_minimum_required(VERSION 3.1)
project(testQPanda)
SET(QPANDA_INSTALL_DIR  "/usr/local") # QPanda2安装的路径,用户可根据自己设置的安装路径配置QPANDA_INSTALL_DIR
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${QPANDA_INSTALL_DIR}/lib/cmake")


add_definitions("-w -DGTEST_USE_OWN_TR1_TUPLE=1")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3")
add_compile_options(-fpermissive)

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
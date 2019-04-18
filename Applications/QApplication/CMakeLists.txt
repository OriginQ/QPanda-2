cmake_minimum_required(VERSION 3.1)
project(QAlgTest)

if (MPI_FOUND)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR})
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Test)
    file(GLOB_RECURSE TEST_SRC_H ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
    file(GLOB_RECURSE TEST_SRC_CPP ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
    add_executable(${PROJECT_NAME} ${TEST_SRC_CPP} )
    target_link_libraries(${PROJECT_NAME} ${LIB_QPANDA} ${LIB_QALG})
endif (MPI_FOUND)




cmake_minimum_required(VERSION 2.8)
project(QAlgTest)


find_package(MPI)
if (MPI_FOUND)
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})
    include_directories(${CMAKE_CURRENT_SOURCE_DIR})

    file(GLOB_RECURSE TEST_SRC_H ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
    file(GLOB_RECURSE TEST_SRC_CPP ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
    add_executable(${PROJECT_NAME} ${TEST_SRC_CPP} )
    target_link_libraries(${PROJECT_NAME} ${MPI_LIBRARIES} QAlg QPanda2.0)
endif (MPI_FOUND)




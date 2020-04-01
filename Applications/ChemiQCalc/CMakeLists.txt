cmake_minimum_required(VERSION 3.1)
project(ChemiQCalc)
file(GLOB_RECURSE CHEMIQ_CALC_CPP ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp )
add_executable(${PROJECT_NAME} ${CHEMIQ_CALC_CPP})
target_link_libraries(${PROJECT_NAME} ${LIB_QALG})

set(TEST_SCRIPT "script.py")
add_custom_command(OUTPUT ${TEST_SCRIPT}
   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_SCRIPT} ${EXECUTABLE_OUTPUT_PATH}/${TEST_SCRIPT}
 )
add_custom_target(CopyTestSript ALL DEPENDS ${TEST_SCRIPT})
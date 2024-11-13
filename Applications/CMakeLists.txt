cmake_minimum_required(VERSION 3.1)

add_subdirectory(PennyGame_Algorithm)
add_subdirectory(QPEAlgorithm)
add_subdirectory(QuantumWalk)
add_subdirectory(SimonAlgorithm)
add_subdirectory(OptimizerTest)

if (USE_EXTENSION)
    add_subdirectory(HHL_Algorithm)
    add_subdirectory(B_V_Algorithm)
    add_subdirectory(DJ_Algorithm)
    add_subdirectory(Grover_Algorithm)
endif(USE_EXTENSION)
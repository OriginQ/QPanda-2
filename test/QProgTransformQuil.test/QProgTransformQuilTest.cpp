#include <iostream>
#include "QPanda.h"
#include "gtest/gtest.h"
#include "QPanda.h"
#include <map>
#include <cstdlib>
#include <sstream>
#include <string>
#include <algorithm>  

USING_QPANDA
  
TEST(QProgTransformQuil, QUIL)
{
    auto qvm = initQuantumMachine();
    auto qubits = qAllocMany(4);
    auto cbits = cAllocMany(4);
    QProg prog;
    QCircuit circuit;

    circuit << RX(qubits[0], PI / 6) << H(qubits[1]) << Y(qubits[2])
        << iSWAP(qubits[2], qubits[3]);
    prog << circuit << MeasureAll(qubits, cbits);

    auto result_1 = runWithConfiguration(prog, cbits, 100);

    for (auto aiter : result_1)
    {
        std::cout << aiter.first << " : " << aiter.second << std::endl;
    }

    auto quil = transformQProgToQuil(prog,qvm);
    std::cout << quil << std::endl;

    auto result_2 = runWithConfiguration(prog, cbits, 100);
    for (auto aiter : result_2)
    {
        std::cout << aiter.first << " : " << aiter.second << std::endl;
    }

    destroyQuantumMachine(qvm);
    system("pause");
    return;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
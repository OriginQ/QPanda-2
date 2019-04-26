#include <iostream>
#include "gtest/gtest.h"
#include "QPanda.h"
#include <map>
#include <cstdlib>
#include <sstream>
#include <string>
#include <algorithm>
#include "Core/Utilities/Transform/QGateCounter.h"
#include "Core/Utilities/Transform/QGateCompare.h"
#include <vector>


using std::vector;
using std::string;

USING_QPANDA
  
TEST(QProgQGateCount, COUNT)
{
    auto qvm = initQuantumMachine();
    auto qubits = qAllocMany(4);
    auto cbits = cAllocMany(4);
    QProg prog;
    QCircuit circuit;

    circuit << RX(qubits[0], PI / 6) << H(qubits[1]) << Y(qubits[2])
        << iSWAP(qubits[2], qubits[3]) << CNOT(qubits[0], qubits[1]);
    prog << circuit << H(qubits[0]) << H(qubits[1]);

//    QGateCounter counter;
//    counter.traversal(circuit);
    size_t num = getQGateNumber(circuit);
    std::cout << "QGate count: " << num << std::endl;
    destroyQuantumMachine(qvm);
    system("pause");
    return;
}

TEST(QProgQGateCompare, COMPARE)
{
    auto qvm = initQuantumMachine();
    auto qubits = qAllocMany(4);
    auto cbits = cAllocMany(4);
    QProg prog;
    QCircuit circuit;

    circuit << RX(qubits[0], PI / 6) << H(qubits[1]) << Y(qubits[2])
        << iSWAP(qubits[2], qubits[3]) << CNOT(qubits[0], qubits[1]);
    prog << circuit << H(qubits[0]) << H(qubits[1]);

    vector<string> single_gates = {"H"};
    vector<string> double_gates = {"CNOT"};
    vector<vector<string>> gates = {single_gates, double_gates};
    auto gate = CNOT(qubits[0], qubits[1]);
    size_t num = getUnSupportQGateNumber(prog, gates);
    //size_t num = getUnSupportQGateNumber(prog, gates);
    std::cout << "unsupport QGate count: " << num << std::endl;

    destroyQuantumMachine(qvm);
    system("pause");
    return;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

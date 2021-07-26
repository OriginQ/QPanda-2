#include <iostream>
#include "gtest/gtest.h"
#include "QPanda.h"
#include <map>
#include <cstdlib>
#include <sstream>
#include <string>
#include <algorithm>
#include "Core/Utilities/QProgInfo/QGateCounter.h"
#include "Core/Utilities/QProgInfo/QGateCompare.h"
#include <vector>


using std::vector;
using std::string;

USING_QPANDA
  
TEST(QProgQGateCount, COUNT)
{
    auto qvm = initQuantumMachine();
    auto qubits = qvm->allocateQubits(4);
    auto cbits = qvm->allocateCBits(4);
    QProg prog;
    QCircuit circuit;

    circuit << RX(qubits[0], PI / 6) << H(qubits[1]) << Y(qubits[2])
        << iSWAP(qubits[2], qubits[3]) << CNOT(qubits[0], qubits[1]);
    prog << circuit << H(qubits[0]) << H(qubits[1]);

//    QGateCounter counter;
//    counter.traversal(circuit);
    size_t num = getQGateNumber(circuit);
    //std::cout << "QGate count: " << num << std::endl;
    destroyQuantumMachine(qvm);
    ASSERT_EQ(num, 5);
    //std::cout << "QProgQGateCount.COUNT tests over." << std::endl;
    return;
}

TEST(QProgQGateCompare, COMPARE)
{
    auto qvm = initQuantumMachine();
    auto qubits = qvm->allocateQubits(4);
    auto cbits = qvm->allocateCBits(4);
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
    //std::cout << "unsupport QGate count: " << num << std::endl;
    destroyQuantumMachine(qvm);
    ASSERT_EQ(num, 3);
    //std::cout << "QProgQGateCompare.COMPARE tests over." << std::endl;
    return;
}


#include "QPanda.h"
#include "gtest/gtest.h"

USING_QPANDA

TEST(QGatecounterTest, test) {
    auto qvm = CPUQVM();
    qvm.init();
    auto qubits = qvm.qAllocMany(4);
    auto cbits = qvm.cAllocMany(4);

    QCircuit cir;

    cir << X(qubits[0])
        << X(qubits[1])
        << Y(qubits[1])
        << H(qubits[0])
        << I(qubits[0])
        << Z(qubits[1])
        << RX(qubits[0], 3.14);

    size_t num = count_qgate_num(cir , PAULI_X_GATE);
    size_t xnum = count_qgate_num(cir, PAULI_X_GATE);
    size_t gnum = count_qgate_num(cir);

    //std::cout << "XGate number: " << num << std::endl;
   //std::cout << "Gate number: " << gnum << std::endl;
    EXPECT_EQ(gnum, 7);
    EXPECT_EQ(num, 2);
    EXPECT_EQ(xnum, 2);

    QProg prog;

    prog << cir
         << X(qubits[0])
         << Y(qubits[1])
         << H(qubits[0])
         << RX(qubits[0], 3.14)
         << Measure(qubits[1], cbits[0]);

    size_t num1 = count_qgate_num(prog);
    size_t xnum1 = count_qgate_num(prog, PAULI_X_GATE);

    EXPECT_EQ(num1, 11);
    EXPECT_EQ(xnum1, 3);


    auto circuit = CreateEmptyCircuit();
    circuit << H(qubits[0]) << X(qubits[1]) << S(qubits[2])
        << iSWAP(qubits[1], qubits[2]) << RX(qubits[3], PI / 4);
    auto count = count_qgate_num(circuit);
    EXPECT_EQ(count, 5);
}
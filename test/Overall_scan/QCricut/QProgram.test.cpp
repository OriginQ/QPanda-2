/// QProgram  test

#include "QPanda.h"
#include "gtest/gtest.h"


TEST(QProgramInfaceTest, test) {
    auto qvm = CPUQVM();
    qvm.init();
    auto qubits = qvm.allocateQubits(4);
    auto cbits = qvm.allocateCBits(4);
    QCircuit circuit = createEmptyCircuit();
    QProg prog;

    auto HGate = H(qubits[0]);
    auto IGate = I(qubits[0]);
    auto SGate = S(qubits[0]);
    auto XGate = X(qubits[0]);

    circuit << HGate << IGate << SGate << XGate; 

    prog << HGate
        << XGate
        << iSWAP(qubits[0], qubits[1])
        << CNOT(qubits[1], qubits[2]);

    prog.pushBackNode(dynamic_pointer_cast<QNode>(HGate.getImplementationPtr()));
    prog.insertQNode(prog.getHeadNodeIter(), dynamic_pointer_cast<QNode>(HGate.getImplementationPtr()));
    EXPECT_EQ(2, prog.getNodeType());

    EXPECT_EQ(3 , prog.get_used_qubits(qubits));

    EXPECT_EQ(6, prog.get_qgate_num());

    EXPECT_EQ(2 , prog.get_max_qubit_addr());

    EXPECT_EQ(0, prog.get_used_cbits(cbits));
    
    for (auto iter = prog.getFirstNodeIter(); iter != prog.getEndNodeIter(); ++iter) {
        iter = prog.deleteQNode(iter);
    }
    EXPECT_EQ(0, prog.get_qgate_num());

    EXPECT_TRUE(prog.is_empty());

    prog << HGate<< XGate << iSWAP(qubits[0], qubits[1]) << CNOT(qubits[1], qubits[2]) << HGate;

    EXPECT_TRUE(prog.is_measure_last_pos());

    prog.clear();
    EXPECT_EQ(0, prog.get_qgate_num());
    EXPECT_TRUE(prog.is_empty());

}

TEST(QGateCaseQProCase, test) {
    auto qvm = CPUQVM();
    qvm.init();
    auto q = qvm.qAllocMany(3);
    QVec qubits = { q[0],q[1] };
    auto qvec = qvm.qAllocMany(4);
    auto cvec = qvm.cAllocMany(4);
    QProg pro;
    pro << H(qvec[0])
        << X(qvec[1])
        << iSWAP(qvec[0], qvec[1])
        << CNOT(qvec[1], qvec[2])
        << H(qvec[3])
        << MeasureAll(qvec, cvec);

    auto result = qvm.runWithConfiguration(pro, cvec, 1000);

    for (auto& val : result)
    {
        //std::cout << val.first << ", " << val.second << std::endl;
    }
}





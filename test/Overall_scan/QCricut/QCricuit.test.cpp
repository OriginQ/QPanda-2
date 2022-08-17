
#include "QPanda.h"
#include "gtest/gtest.h"


TEST(QCricuitInfaceTest, test) {
    auto qvm = CPUQVM();
    qvm.init();
    auto qubits = qvm.allocateQubits(4);
    auto cbits = qvm.allocateCBits(4);
    QProg prog;
 
    QCircuit circuit = createEmptyCircuit();
    EXPECT_TRUE(circuit.is_empty());

    circuit << I(qubits[0]);
    auto HGate = H(qubits[0]);
    circuit.pushBackNode(dynamic_pointer_cast<QNode>(HGate.getImplementationPtr()));
    
    QGateCounter t;
    t.traversal(circuit);
    size_t num = t.count();
    EXPECT_EQ(2, num);

    QGateCounter t1;
    EXPECT_EQ(2, count_qgate_num(circuit));

    circuit = QCircuit(circuit);
    EXPECT_EQ(1, circuit.getNodeType());

    circuit << X(qubits[0]) << S(qubits[0]);
    EXPECT_FALSE(circuit.is_empty());


    QStat cir_matrix = getCircuitMatrix(circuit);

    int gate_num = circuit.get_qgate_num();

    for (auto iter = circuit.getFirstNodeIter(); iter != circuit.getEndNodeIter(); ++iter) {
        iter = circuit.deleteQNode(iter);
    }

    gate_num = circuit.get_qgate_num();

    EXPECT_TRUE(circuit.is_empty());


}


TEST(OriginCircuitInfaceTest, test ) {
    auto qvm = CPUQVM();
    qvm.init();
    auto qubits = qvm.allocateQubits(4);

    OriginCircuit ocircuit ;

    auto HGate = H(qubits[0]);
    auto IGate = I(qubits[0]);
    auto SGate = S(qubits[0]);
    auto XGate = X(qubits[0]);
    ocircuit.insertQNode(ocircuit.getHeadNodeIter(), dynamic_pointer_cast<QNode>(XGate.getImplementationPtr()));
    ocircuit.pushBackNode(dynamic_pointer_cast<QNode>(IGate.getImplementationPtr()));
    ocircuit.pushBackNode(dynamic_pointer_cast<QNode>(HGate.getImplementationPtr()));
    ocircuit.insertQNode(ocircuit.getLastNodeIter(), dynamic_pointer_cast<QNode>(XGate.getImplementationPtr()));
    ocircuit.insertQNode(ocircuit.getLastNodeIter(), dynamic_pointer_cast<QNode>(SGate.getImplementationPtr()));

    int gate_num = ocircuit.get_qgate_num();
    EXPECT_EQ(5, gate_num);

    for (auto iter = ocircuit.getFirstNodeIter(); iter != ocircuit.getEndNodeIter(); ++iter) {
        iter = ocircuit.deleteQNode(iter);
    }
    EXPECT_EQ(0, ocircuit.get_qgate_num());
   

    ocircuit.pushBackNode(dynamic_pointer_cast<QNode>(IGate.getImplementationPtr()));
    ocircuit.pushBackNode(dynamic_pointer_cast<QNode>(HGate.getImplementationPtr()));
    ocircuit.insertQNode(ocircuit.getLastNodeIter(), dynamic_pointer_cast<QNode>(XGate.getImplementationPtr()));
    ocircuit.insertQNode(ocircuit.getLastNodeIter(), dynamic_pointer_cast<QNode>(SGate.getImplementationPtr()));

    for (auto iter = ocircuit.getFirstNodeIter(); iter != ocircuit.getEndNodeIter(); ++iter) {
        iter = ocircuit.deleteQNode(iter);
    }
    EXPECT_EQ(0, ocircuit.get_qgate_num());

    ocircuit.pushBackNode(dynamic_pointer_cast<QNode>(IGate.getImplementationPtr()));
    ocircuit.pushBackNode(dynamic_pointer_cast<QNode>(HGate.getImplementationPtr()));
    ocircuit.insertQNode(ocircuit.getLastNodeIter(), dynamic_pointer_cast<QNode>(XGate.getImplementationPtr()));
    ocircuit.insertQNode(ocircuit.getLastNodeIter(), dynamic_pointer_cast<QNode>(SGate.getImplementationPtr()));

    ocircuit.clear();      

    EXPECT_EQ(0, ocircuit.get_qgate_num());

}

TEST(QcricuitCase,test) {
    auto qvm = CPUQVM();
    qvm.init();
    auto qvec = qvm.qAllocMany(4);
    auto cbits = qvm.cAllocMany(4);
    auto circuit = QCircuit();
    auto prog = QProg();

    circuit << H(qvec[0]) << CNOT(qvec[0], qvec[1])
        << CNOT(qvec[1], qvec[2]) << CNOT(qvec[2], qvec[3]);

    circuit.setDagger(true);
    EXPECT_TRUE(circuit.isDagger());

    prog << H(qvec[3]) << circuit << Measure(qvec[0], cbits[0]);

    auto result = qvm.runWithConfiguration(prog, cbits, 1000);

    for (auto& val : result)
    {
        //std::cout << val.first << ", " << val.second << std::endl;
    }
}

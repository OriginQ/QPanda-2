/// QMeasure class inface test

#include "QPanda.h"
#include "gtest/gtest.h"

TEST(QMeasureInfaceTest, test) {
    auto qvm = CPUQVM();
    qvm.init();
    auto qubits = qvm.qAllocMany(4);
    auto cbits = qvm.cAllocMany(4);
    Qubit* q = qubits[0];
    ClassicalCondition cc = cbits[0];
    cc.set_val(0);
    auto target_cbit = cc.getExprPtr()->getCBit();

    QMeasure QCBit_qmeasure = QMeasure(q, target_cbit);
    QMeasure QC_measure = Measure(q , cc);
    QMeasure mes = QMeasure(QC_measure.getImplementationPtr());
    
    EXPECT_STREQ("c0", mes.getCBit()->getName().c_str());
    auto cbit = mes.getCBit();
    cbit->setOccupancy(true);
    EXPECT_TRUE(cbit->getOccupancy());

    auto qbit = mes.getQuBit();
    EXPECT_TRUE(qbit->getOccupancy());
    EXPECT_EQ(0,qbit->get_phy_addr());
    EXPECT_EQ(3, QCBit_qmeasure.getNodeType());
    EXPECT_EQ(3, QC_measure.getNodeType());
    EXPECT_EQ(3, mes.getNodeType());
    
}

TEST(QMeasureCase,test) {
    auto qvm = CPUQVM();
    qvm.init();
    auto qubits = qvm.qAllocMany(4);
    auto cbits = qvm.cAllocMany(4);

    QProg prog;
    prog << H(qubits[0])
        << H(qubits[1])
        << H(qubits[2])
        << H(qubits[3])
        << MeasureAll(qubits, cbits);

    auto result = qvm.runWithConfiguration(prog, cbits, 1000);

    for (auto& val : result)
    {
        //std::cout << val.first << ", " << val.second << std::endl;
    }


}

TEST(PMeasureCase,test) {
    auto qvm = CPUQVM();
    qvm.init();
    auto qubits = qvm.qAllocMany(2);

    QProg prog;
    prog << H(qubits[0])
        << CNOT(qubits[0], qubits[1]);

    //std::cout << "probRunDict: " << std::endl;
    auto result1 = qvm.probRunDict(prog, qubits);
    for (auto& val : result1)
    {
       // std::cout << val.first << ", " << val.second << std::endl;
    }

    //std::cout << "probRunTupleList: " << std::endl;
    auto result2 = probRunTupleList(prog, qubits);
    for (auto& val : result2)
    {
        //std::cout << val.first << ", " << val.second << std::endl;
    }

   // std::cout << "probRunList: " << std::endl;
    auto result3 = qvm.probRunList(prog, qubits);
    for (auto& val : result3)
    {
        //std::cout << val << std::endl;
    }

}
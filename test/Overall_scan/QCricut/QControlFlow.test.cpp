
#include "QPanda.h"
#include "gtest/gtest.h"

TEST(ControlFlowInfaceTest, test) {
    
    CPUQVM* m_qvm = new CPUQVM();
    m_qvm->init();
    auto qubits = m_qvm->allocateQubits(4);
    auto c1 = m_qvm->allocateCBit();
    auto c2 = m_qvm->allocateCBit();
    c1.set_val(0);

    c1 = c1 + 2;
    //ASSERT_EQ(2, c1.get_val());

    auto prog = QProg();
    auto m_prog = CreateEmptyQProg();
    prog << (c1 = c1 + 1);

    auto qwhile = QWhileProg(c1 < 11, prog);
    m_prog << qwhile;
    m_qvm->directlyRun(prog);

    //ASSERT_EQ(c1.get_val(), 11);
    EXPECT_EQ(4, qwhile.getNodeType());

    auto cc = qwhile.getClassicalCondition();
    //cout << "condition : " << cc.get_val() << endl;
    auto truenode = qwhile.getTrueBranch();
    
    auto cqwhile = CreateWhileProg(c1 == c2, prog);
    EXPECT_EQ(4, cqwhile.getNodeType());
}

TEST(ControlFlow, test)
{
    return;
    CPUQVM* m_qvm = new CPUQVM();
    m_qvm->init();
    auto c1 = m_qvm->allocateCBit();
    auto c2 = m_qvm->allocateCBit();

    c1.set_val(10);
    c2.set_val(20);
    EXPECT_EQ(c1.get_val(), 10);
    EXPECT_EQ(c2.get_val(), 20);

    auto prog = QProg();
    auto m_prog = CreateEmptyQProg();
    prog << (c1 = c1 + 1) << (c2 = c2 + c1);
    auto qwhile = CreateWhileProg(c1 < 11, prog);
    m_prog << qwhile;

    m_qvm->directlyRun(m_prog);
    std::cout << " c1-val: " << c1.get_val() << std::endl;

    EXPECT_EQ(c1.get_val(), 11);
    EXPECT_EQ(c2.get_val(), 83);

    m_qvm->finalize();
    delete m_qvm;
   
    
}

TEST(QWhileCase, test) {
    return;
    init();
    QProg prog;
    auto qvec = qAllocMany(3);
    auto cvec = cAllocMany(3);
    cvec[0].set_val(0);

    QProg prog_in;
    prog_in << cvec[0] << H(qvec[cvec[0]]) << (cvec[0] = cvec[0] + 1);
    auto qwhile = CreateWhileProg(cvec[0] < 3, prog_in);
    prog << qwhile;
    auto result = probRunTupleList(prog, qvec);

    for (auto& val : result)
    {
        std::cout << val.first << ", " << val.second << std::endl;
    }

    finalize();
 
}


TEST(QIFCase, test) {
    return;
    init();
    QProg prog;

    auto qvec = qAllocMany(3);
    auto cvec = cAllocMany(3);
    cvec[1].set_val(0);
    cvec[0].set_val(0);

    QProg branch_true, branch_false;

    branch_true << H(qvec[cvec[0]]) << (cvec[0] = cvec[0] + 1);
    branch_false << H(qvec[0]) << CNOT(qvec[0], qvec[1]) << CNOT(qvec[1], qvec[2]);

    auto qif = QIfProg(cvec[1] > 5, branch_true, branch_false);

    prog << qif;

    auto result = probRunTupleList(prog, qvec);

    for (auto& val : result)
    {
        //std::cout << val.first << ", " << val.second << std::endl;
    }

    finalize();


}
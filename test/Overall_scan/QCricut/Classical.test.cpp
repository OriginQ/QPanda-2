
#include "QPanda.h"
#include "gtest/gtest.h"

//TEST(ClassicalConditionCon, test) {
//
//    CPUQVM* m_qvm = new CPUQVM();
//    m_qvm->init();
//    auto c1 = m_qvm->allocateCBit();
//    auto c2 = m_qvm->allocateCBit();
//
//    auto copyc = ClassicalCondition(c1);
//    EXPECT_EQ(0, copyc.get_val());
//
//    ClassicalCondition cc1 = c1;
//
//    cbit_size_t num = 7;
//    ClassicalCondition cc2 = num;
//    auto classical = ClassicalCondition(num);
//    classical.getExprPtr();
//    EXPECT_EQ(7, classical.get_val());
//}

TEST(ClassicalConditionTest, test) {
    CPUQVM* m_qvm = new CPUQVM();
    m_qvm->init();
    auto c1 = m_qvm->allocateCBit();
    auto c2 = m_qvm->allocateCBit();

    ///set_val , get_val test
    c1.set_val(5);
    c2.set_val(10);
    EXPECT_EQ(5, c1.get_val());
    EXPECT_EQ(10, c2.get_val());

    /// PLUS
    auto CplusC = c1 + c2;
    auto CplusInt = CplusC + 15;
    auto IntplusC = 30 + CplusInt;
    EXPECT_EQ(CplusC.get_val(), 15);
    EXPECT_TRUE(CplusC.checkValidity());
    EXPECT_EQ(CplusInt.get_val(), 30);
    EXPECT_EQ(IntplusC.get_val(), 60);

    ///MINUS
    auto CminusC = c2 - c1;
    auto CminusNum = c2 - 1;
    auto NumMinusC = 30 - c2;
    EXPECT_TRUE(CminusC.checkValidity());
    EXPECT_EQ(5, CminusC.get_val());
    EXPECT_EQ(9, CminusNum.get_val());
    EXPECT_EQ(20, NumMinusC.get_val());

    /// MUL
    auto CmulC = c2 * c1;
    auto CmulNum = c2 * 2;
    auto NummulC = 3 * c1;
    EXPECT_TRUE(CmulC.checkValidity());
    EXPECT_EQ(50, CmulC.get_val());
    EXPECT_EQ(20, CmulNum.get_val());
    EXPECT_EQ(15, NummulC.get_val());

    /// DIV
    auto CdivC = c2 / c1;
    auto CdivNum = c2 / 2;
    auto NdivC = 30 /  c2;

    //ASSERT_THROW(auto cc4 = c1 / 0, std::invalid_argument); 

    EXPECT_TRUE(CdivC.checkValidity());
    EXPECT_EQ(2, CdivC.get_val());
    EXPECT_EQ(5, CdivNum.get_val());
    EXPECT_EQ(3, NdivC.get_val());

    auto CequalC = (c2 == c1) ;
    auto CequalNum =(c2 == 2) ;
    auto NequalC = (30 == c2) ;
    auto c3 = ClassicalCondition(NequalC);
    EXPECT_EQ(0, c3.get_val());

    auto CnoequalC = (c2 != c1);
    EXPECT_TRUE(CnoequalC.checkValidity());
    
}

/// classicalPro 
TEST(ClassicalProgTest, test) {
    CPUQVM* m_qvm = new CPUQVM();
    m_qvm->init();
    auto c1 = m_qvm->allocateCBit();
    auto c2 = m_qvm->allocateCBit();

    ClassicalProg cprog = ClassicalProg(c1);
    auto copyprog = ClassicalProg(cprog);
    auto cprog1 = ClassicalProg(dynamic_pointer_cast<AbstractClassicalProg>(copyprog.getImplementationPtr()));
    EXPECT_EQ(6, cprog.getNodeType());
    EXPECT_EQ(6, cprog1.getNodeType());
    EXPECT_EQ(6, copyprog.getNodeType());



}
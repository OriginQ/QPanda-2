/// OriginclassExp class inface test 

#include "QPanda.h"
#include "gtest/gtest.h"
#include <OriginClassicalExpression.h>

TEST(OriginClassExpInfaceText, test) {
    CPUQVM* m_qvm = new CPUQVM();
    m_qvm->init();
    auto c1 = m_qvm->allocateCBit();
    auto c2 = m_qvm->allocateCBit();

    c1.set_val(2);
    c2.set_val(4);

    CBit* cbit = c1.getExprPtr()->getCBit();
    OriginCExpr ocexpr = OriginCExpr(cbit);
    OriginCExpr numexpr = OriginCExpr(7);

    // this constructor can cause memory error
    //OriginCExpr lrexpr = OriginCExpr( c1.getExprPtr().get(), c2.getExprPtr().get(), PLUS);

    /// ocexpr.getPosition()error position don't init 
    //std::cout << "position:  " << ocexpr.getPosition() << std::endl;
    //std::cout << "position:  " << numexpr.getPosition() << std::endl;
    //this constructor call getPosition cause error;
    //std::cout << "position:  " << lrexpr.getPosition() << std::endl;

    /// ocexpr.getNodeType()error Nodetype don't init 
    //std::cout << " expr : " << ocexpr.getNodeType() << std::endl;
   // std::cout << " expr : " << numexpr.getNodeType() << std::endl;
    //this constructor call getNodeType cause error;
    //std::cout << " expr : " << lrexpr.getNodeType() << std::endl;

    ocexpr.setPosition(10);
    EXPECT_EQ(10 , ocexpr.getPosition());
  
    EXPECT_EQ(0 ,  ocexpr.getContentSpecifier());
   

}
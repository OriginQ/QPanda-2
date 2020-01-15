#include <iostream>
#include <limits>
#include "QPanda.h"
#include "gtest/gtest.h"
USING_QPANDA
using namespace std;

TEST(QVec,test)
{
    init();
    auto prog = QProg();

    //vector<Qubit*> qvec;
    auto qvec = qAllocMany(5);
    auto cvec = cAllocMany(2);
    cvec[1].setValue(0);
    cvec[0].setValue(0);
    auto prog_in = QProg();
    prog_in<<(cvec[1]=cvec[1]+1)<<H(qvec[cvec[0]])<<(cvec[0]=cvec[0]+1);
    auto qwhile = CreateWhileProg(cvec[1]<5,prog_in); 
    prog<<qwhile;
    directlyRun(prog);
    auto result =PMeasure_no_index(qvec);
    for(auto & aiter : result)
    {
        std::cout<<aiter<<std::endl;
    }
    finalize();
}
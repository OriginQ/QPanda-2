#include <iostream>
#include "Utilities/OriginCollection.h"
#include "QPanda.h"
#include "gtest/gtest.h"
USING_QPANDA
using namespace std;
TEST(CPUQVMTest, testInit)
{
    CPUQVM qvm;
    ASSERT_THROW(auto qvec = qvm.allocateQubits(2), qvm_attributes_error);
    ASSERT_THROW(auto cvec = qvm.allocateCBits(2), qvm_attributes_error);

    qvm.init();
    ASSERT_NO_THROW(auto qvec = qvm.allocateQubits(2));
    ASSERT_NO_THROW(auto cvec = qvm.allocateCBits(2));

    ASSERT_THROW(auto qvec = qvm.allocateQubits(26), qalloc_fail);
    ASSERT_THROW(auto cvec = qvm.allocateCBits(257), calloc_fail); 


}

TEST(NoiseMachineTest, test)
{
    NoiseQVM qvm;
    rapidjson::Document doc1;
    doc1.Parse("{}");
    qvm.init();
    auto qvec = qvm.allocateQubits(2);
    auto cvec = qvm.allocateCBits(2);
    auto prog = QProg();
    prog << X(qvec[0])
         << X(qvec[1])
         << Measure(qvec[0], cvec[0])
         << Measure(qvec[1], cvec[1]);

    rapidjson::Document doc;
    doc.Parse("{}");
    auto &alloc = doc.GetAllocator();
    doc.AddMember("shots", 1000, alloc);
    auto result = qvm.runWithConfiguration(prog, cvec, doc);
    for (auto &aiter : result)
    {
        std::cout << aiter.first << " : " << aiter.second << endl;
    }
    auto state = qvm.getQState();
    for (auto &aiter : state)
    {
        std::cout << aiter << endl;
    }
    qvm.finalize();
}

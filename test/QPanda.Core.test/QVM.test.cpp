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

    qvm.finalize();
    ASSERT_THROW(auto qvec = qvm.allocateQubits(2), qvm_attributes_error);
    ASSERT_THROW(auto cvec = qvm.allocateCBits(2), qvm_attributes_error);
    ASSERT_THROW(auto qvec = qvm.getAllocateQubit(), qvm_attributes_error);
    ASSERT_THROW(auto qvec = qvm.getAllocateCMem(), qvm_attributes_error);
    ASSERT_THROW(auto qvec = qvm.getResultMap(), qvm_attributes_error);


}

TEST(NoiseMachineTest, test)
{
    rapidjson::Document doc1;
    doc1.Parse("{}");
    Value value(rapidjson::kObjectType);
    Value value_rx(rapidjson::kArrayType);
    value_rx.PushBack(2, doc1.GetAllocator());
    value_rx.PushBack(10.0, doc1.GetAllocator());
    value_rx.PushBack(2.0, doc1.GetAllocator());
    value_rx.PushBack(0.03, doc1.GetAllocator());
    value.AddMember("RX", value_rx, doc1.GetAllocator());

    Value value_ry(rapidjson::kArrayType);
    value_ry.PushBack(2, doc1.GetAllocator());
    value_ry.PushBack(10.0, doc1.GetAllocator());
    value_ry.PushBack(2.0, doc1.GetAllocator());
    value_ry.PushBack(0.03, doc1.GetAllocator());
    value.AddMember("RY", value_ry, doc1.GetAllocator());

    Value value_rz(rapidjson::kArrayType);
    value_rz.PushBack(2, doc1.GetAllocator());
    value_rz.PushBack(10.0, doc1.GetAllocator());
    value_rz.PushBack(2.0, doc1.GetAllocator());
    value_rz.PushBack(0.03, doc1.GetAllocator());
    value.AddMember("RX", value_rz, doc1.GetAllocator());

    Value value_H(rapidjson::kArrayType);
    value_H.PushBack(2, doc1.GetAllocator());
    value_H.PushBack(10.0, doc1.GetAllocator());
    value_H.PushBack(2.0, doc1.GetAllocator());
    value_H.PushBack(0.03, doc1.GetAllocator());
    value.AddMember("H", value_H, doc1.GetAllocator());

    doc1.AddMember("noisemodel", value, doc1.GetAllocator());

    

    NoiseQVM qvm;
    qvm.init(doc1);
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

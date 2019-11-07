#include <iostream>
#include "Core/Utilities/Tools/OriginCollection.h"
#include "QPanda.h"
#include <time.h>
#include "gtest/gtest.h"
USING_QPANDA
using namespace std;
TEST(CPUQVMTest, testInit)
{
    return;
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
    //return;
    rapidjson::Document doc;
    doc.Parse("{}");
    Value value(rapidjson::kObjectType);
    Value value_h(rapidjson::kArrayType);
    value_h.PushBack(DAMPING_KRAUS_OPERATOR, doc.GetAllocator());
    value_h.PushBack(0.5, doc.GetAllocator());
    value.AddMember("H", value_h, doc.GetAllocator());

    Value value_rz(rapidjson::kArrayType);
    value_rz.PushBack(DAMPING_KRAUS_OPERATOR, doc.GetAllocator());
    value_rz.PushBack(0.5, doc.GetAllocator());
    value.AddMember("RZ", value_rz, doc.GetAllocator());

    Value value_cnot(rapidjson::kArrayType);
    value_cnot.PushBack(DAMPING_KRAUS_OPERATOR, doc.GetAllocator());
    value_cnot.PushBack(0.5, doc.GetAllocator());
    value.AddMember("CPHASE", value_cnot, doc.GetAllocator());
    doc.AddMember("noisemodel", value, doc.GetAllocator());

    NoiseQVM qvm;
    qvm.init(doc);
    auto qvec = qvm.allocateQubits(16);
    auto cvec = qvm.allocateCBits(16);
    auto prog = QProg();

    QCircuit  qft = CreateEmptyCircuit();
    for (auto i = 0; i < qvec.size(); i++)
    {
        qft << H(qvec[qvec.size() - 1 - i]);
        for (auto j = i + 1; j < qvec.size(); j++)
        {
            qft << CR(qvec[qvec.size() - 1 - j], qvec[qvec.size() - 1 - i], 2 * PI / (1 << (j - i + 1)));
        }
    }

    prog << qft << qft.dagger()
        << MeasureAll(qvec,cvec);

    rapidjson::Document doc1;
    doc1.Parse("{}");
    auto &alloc = doc1.GetAllocator();
    doc1.AddMember("shots",10 , alloc);

    clock_t start = clock();
    auto result = qvm.runWithConfiguration(prog, cvec, doc1);
    clock_t end = clock();
    std::cout << end - start << endl;

    for (auto &aiter : result)
    {
        std::cout << aiter.first << " : " << aiter.second << endl;
    }
    //auto state = qvm.getQState();
    //for (auto &aiter : state)
    //{
    //    std::cout << aiter << endl;
    //}
    qvm.finalize();

    getchar();
}

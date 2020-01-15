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

double getStateProb(complex<double> val)
{
    return val.real()*val.real() + val.imag()*val.imag();
}


TEST(QVM, TotalAmplitudeQVM)
{
    auto machine = initQuantumMachine(CPU_SINGLE_THREAD);
    auto qlist = machine->allocateQubits(10);
    auto clist = machine->allocateCBits(10);

    auto prog = QProg();
    for_each(qlist.begin(), qlist.end(), [&](Qubit *val) { prog << H(val); });
    prog << CZ(qlist[1], qlist[5])
         << CZ(qlist[3], qlist[7])
         << CZ(qlist[0], qlist[4])
         << RZ(qlist[7], PI / 4)
         << RX(qlist[5], PI / 4)
         << RX(qlist[4], PI / 4)
         << RY(qlist[3], PI / 4)
         << CZ(qlist[2], qlist[6])
         << RZ(qlist[3], PI / 4)
         << RZ(qlist[8], PI / 4)
         << CZ(qlist[9], qlist[5])
         << RY(qlist[2], PI / 4)
         << RZ(qlist[9], PI / 4)
         << CR(qlist[2], qlist[7], PI / 2);


    machine->directlyRun(prog);

    auto res = machine->getQState();
    machine->finalize();

    for (int i = 0; i < 12; ++i)
    {
        cout << i << " : " << res[i] << endl;
    }

    cout << "--------------" << endl;

}

TEST(QVM, PartialAmplitudeQVM)
{
    auto machine = new PartialAmplitudeQVM();
    machine->init();
    auto qlist = machine->allocateQubits(40);
    auto clist = machine->allocateCBits(40);

    auto Toffoli = X(qlist[20]);
    Toffoli.setControl({ qlist[18], qlist[19] });

    auto prog = QProg();
    prog << H(qlist[18])
         << X(qlist[19])
         << Toffoli;

    std::vector<string> subSet = { "0000000000000000000001000000000000000000" ,
                                   "0000000000000000000010000000000000000000" ,
                                   "0000000000000000000011000000000000000000" ,
                                   "0000000000000000000100000000000000000000" ,
                                   "0000000000000000000101000000000000000000" ,
                                   "0000000000000000000110000000000000000000" ,
                                   "0000000000000000000111000000000000000000" ,
                                   "1000000000000000000000000000000000000000" };
    auto result = machine->pMeasureSubset(prog, subSet);

    for (auto val : result)
    {
        std::cout << val.first << " : " << val.second << std::endl;
    }
    getchar();
}

TEST(QVM, SingleAmplitudeQVM)
{
    //throw exception();
    auto machine = new SingleAmplitudeQVM();
    machine->init();
    auto qlist = machine->allocateQubits(10);
    auto clist = machine->allocateCBits(10);

    auto prog = QProg();
    for_each(qlist.begin(), qlist.end(), [&](Qubit *val) { prog << H(val); });
    prog << CZ(qlist[1], qlist[5])
        << CZ(qlist[3], qlist[5])
        << CZ(qlist[2], qlist[4])
        << CZ(qlist[3], qlist[7])
        << CZ(qlist[0], qlist[4])
        << RY(qlist[7], PI / 2)
        << RX(qlist[8], PI / 2)
        << RX(qlist[9], PI / 2)
        << CR(qlist[0], qlist[1], PI)
        << CR(qlist[2], qlist[3], PI)
        << RY(qlist[4], PI / 2)
        << RZ(qlist[5], PI / 4)
        << RX(qlist[6], PI / 2)
        << RZ(qlist[7], PI / 4)
        << CR(qlist[8], qlist[9], PI)
        << CR(qlist[1], qlist[2], PI)
        << RY(qlist[3], PI / 2)
        << RX(qlist[4], PI / 2)
        << RX(qlist[5], PI / 2)
        << CR(qlist[9], qlist[1], PI)
        << RY(qlist[1], PI / 2)
        << RY(qlist[2], PI / 2)
        << RZ(qlist[3], PI / 4)
        << CR(qlist[7], qlist[8], PI);

    //machine->run(prog);

    cout << machine->PMeasure_bin_index(prog, "0000000000") << endl;
    cout << machine->PMeasure_dec_index(prog, "1") << endl;

    machine->finalize();
    getchar();
}

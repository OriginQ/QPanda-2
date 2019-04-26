#include "QPanda.h"
#include "gtest/gtest.h"
#include "include/Core/QuantumMachine/PartialAmplitudeQVM.h"
#include "include/Core/QuantumMachine/SingleAmplitudeQVM.h"
using namespace std;
USING_QPANDA

double getStateProb(complex<double> val)
{
    return val.real()*val.real() + val.imag()*val.imag();
}

TEST(QVM, SingleAmplitudeQVM)
{
    auto machine = new SingleAmplitudeQVM();
    machine->init();
    auto qlist = machine->allocateQubits(10);
    auto clist = machine->allocateCBits(10);

#if 0
    auto machine1 = new SingleAmplitudeQVM();
    machine1->init();
    auto qlist1 = machine1->allocateQubits(20);
    auto clist1 = machine1->allocateCBits(20);

    auto prog1 = QProg();
    for_each(qlist.begin(), qlist.end(), [&](Qubit *val) { prog1 << H(val); });
    prog1 << Y(qlist[1])
         << iSWAP(qlist[3], qlist[5])
         << SqiSWAP(qlist[3], qlist[5])
         << H(qlist[14])
         << CNOT(qlist[14], qlist[18])
         << H(qlist[14])
         << T(qlist[8])
         << S(qlist[3])
         << X1(qlist[9])
         << Y1(qlist[2])
         << Z1(qlist[14])
         << CZ(qlist[18], qlist[7]);

    machine1->finalize();

#endif


    auto prog = QProg();
    prog <<H(qlist[7])
        << H(qlist[8])
        << H(qlist[9])
        << CNOT(qlist[7], qlist[1])
        << CNOT(qlist[8], qlist[2])
        << CNOT(qlist[9], qlist[3]);

    machine->directlyRun(prog);
    auto res = machine->getProbDict(qlist,1024);

    init();
    auto qlist1 = qAllocMany(10);

    auto res2 =probRunDict(prog, qlist1, 1024);

    for (auto val :res)
    {
        std::cout << val.first <<" :(single) "<<val.second<<"  " <<res2[val.first]<< endl;
    }

    machine->finalize();
    delete machine;
    getchar();
}

//TEST(QVM, PartialAmplitudeQVM)
//{
//    auto machine = new PartialAmplitudeQVM();
//    machine->init();
//    auto qlist = machine->allocateQubits(10);
//    auto clist = machine->allocateCBits(10);
//
//
//#if 0
//    auto machine1 = new PartialAmplitudeQVM();
//    machine1->init();
//    auto qlist1 = machine1->allocateQubits(20);
//    auto clist1 = machine1->allocateCBits(20);
//
//    auto prog1 = QProg();
//    for_each(qlist.begin(), qlist.end(), [&](Qubit *val) { prog1 << H(val); });
//    prog1 << Y(qlist[1])
//          << iSWAP(qlist[3], qlist[5])
//          << SqiSWAP(qlist[3], qlist[5])
//          << H(qlist[14])
//          << CNOT(qlist[14], qlist[18])
//          << H(qlist[14])
//          << T(qlist[8])
//          << S(qlist[3])
//          << X1(qlist[9])
//          << Y1(qlist[2])
//          << Z1(qlist[14])
//          << CZ(qlist[18], qlist[7]);
//
//    machine1->finalize();
//#endif
//
//    auto prog = QProg();
//    for_each(qlist.begin(), qlist.end(), [&](Qubit *val) { prog << H(val); });
//    prog    << CZ(qlist[1], qlist[5])
//            << CZ(qlist[3], qlist[7])
//            << CZ(qlist[0], qlist[4])
//            << RZ(qlist[7], PI / 4)
//            << RX(qlist[5], PI / 4)
//            << RX(qlist[4], PI / 4)
//            << RY(qlist[3], PI / 4)
//            << CZ(qlist[2], qlist[6])
//            << RZ(qlist[3], PI / 4)
//            << RZ(qlist[8], PI / 4)
//            << CZ(qlist[9], qlist[5])
//            << RY(qlist[2], PI / 4)
//            << RZ(qlist[9], PI / 4)
//            << CZ(qlist[2], qlist[3]);
//    machine->run(prog);
//
//    /*Test PMeasure*/
//    EXPECT_NO_THROW(machine->PMeasure(-1));
//    EXPECT_THROW(machine->PMeasure(1280), qprog_syntax_error);
//
//    auto res0 = machine->PMeasure(64);
//    EXPECT_NEAR(8.377582e-05, res0[0].second, 1e-6);
//    EXPECT_NEAR(8.377582e-05, res0[1].second, 1e-6);
//
//    EXPECT_NEAR(4.882813e-04, res0[4].second, 1e-6);
//    EXPECT_NEAR(4.882813e-04, res0[5].second, 1e-6);
//
//    EXPECT_NEAR(2.845912e-03, res0[12].second, 1e-6);
//    EXPECT_NEAR(2.845912e-03, res0[13].second, 1e-6);
//
//    /*Test PMeasure  QVec*/
//    QVec qvec;
//    QVec qvec1;
//    QVec qvec2 = {qlist[0],qlist[0]};
//    for_each(qlist.begin() + 3, qlist.end(), [&](Qubit *val) { qvec.emplace_back(val); });
//
//    EXPECT_NO_THROW(machine->PMeasure(qvec, -1));
//    EXPECT_NO_THROW(machine->PMeasure(qvec, 128));
//    EXPECT_THROW(machine->PMeasure(qvec, 1280), qprog_syntax_error);
//
//    EXPECT_THROW(machine->PMeasure(qvec1, 128), qprog_syntax_error);
//    EXPECT_THROW(machine->PMeasure(qvec2, 128), qprog_syntax_error);
//
//    auto res1 = machine->getProbDict(qvec, 1);
//
//    EXPECT_STREQ("0000000", res1.begin()->first.c_str());
//    EXPECT_NEAR(2.28823e-03, res1.begin()->second, 1e-6);
//
//    /*Test getQStat*/
//    auto res2 = machine->getQStat();
//    EXPECT_NEAR(8.377582e-05, getStateProb(res2[0]), 1e-6);
//    EXPECT_NEAR(4.882813e-04, getStateProb(res2[4]), 1e-6);
//
//    machine->finalize();
//    getchar();
//}

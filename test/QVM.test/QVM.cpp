#include "QPanda.h"
#include "gtest/gtest.h"
#include "include/Core/QuantumMachine/PartialAmplitudeQVM.h"
#include "include/Core/QuantumMachine/SingleAmplitudeQVM.h"
using namespace std;
USING_QPANDA

const string sQRunesPath("D:\\QRunes20_22.txt");

double getStateProb(complex<double> val)
{
    return val.real()*val.real() + val.imag()*val.imag();
}

TEST(QVM, SingleAmplitudeQVM)
{
    init();
    auto prog = QProg();

    auto machine = new SingleAmplitudeQVM();
    qRunesToQProg(sQRunesPath, prog);

    machine->run(prog);

    /*Test PMeasure*/
    auto res0 = machine->PMeasure(3);
  
    EXPECT_NEAR(2.94933e-07, res0[0].second,1e-8);
    EXPECT_NEAR(1.33458e-06, res0[1].second,1e-8);
    EXPECT_NEAR(1.62163e-06, res0[2].second,1e-8);

    /*Test getQStat*/
    auto res1 = machine->getQStat();

    EXPECT_NEAR(2.94933e-07, getStateProb(res1[0]), 1e-8);
    EXPECT_NEAR(1.33458e-06, getStateProb(res1[1]), 1e-8);
    EXPECT_NEAR(1.62163e-06, getStateProb(res1[2]), 1e-8);

    /*Test PMeasure_index*/
    auto res2 = machine->PMeasure_index(0);
    EXPECT_NEAR(2.94933e-07, res2, 1e-8);

    finalize();
    getchar();
}

TEST(QVM, PartialAmplitudeQVM)
{
    init();
    auto prog = QProg();
    auto qlist = qAllocMany(10);
    auto machine = new PartialAmplitudeQVM();

    for_each(qlist.begin(), qlist.end(), [&](Qubit *val) { prog << H(val); });
    prog    << CZ(qlist[1], qlist[5])
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
            << CZ(qlist[2], qlist[3]);
    machine->run(prog);

    /*Test PMeasure*/
    auto res0 = machine->PMeasure(64);
    EXPECT_NEAR(8.377582e-05, res0[0].second, 1e-6);
    EXPECT_NEAR(8.377582e-05, res0[1].second, 1e-6);

    EXPECT_NEAR(4.882813e-04, res0[4].second, 1e-6);
    EXPECT_NEAR(4.882813e-04, res0[5].second, 1e-6);

    EXPECT_NEAR(2.845912e-03, res0[12].second, 1e-6);
    EXPECT_NEAR(2.845912e-03, res0[13].second, 1e-6);

    /*Test getProbDict*/
    QVec qvec;
    for_each(qlist.begin() + 3, qlist.end(), [&](Qubit *val) { qvec.emplace_back(val); });
    auto res1 = machine->getProbDict(qvec, 1);

    EXPECT_STREQ("0000000", res1.begin()->first.c_str());
    EXPECT_NEAR(2.28823e-03, res1.begin()->second, 1e-6);

    /*Test getQStat*/
    auto res2 = machine->getQStat();
    EXPECT_NEAR(8.377582e-05, getStateProb(res2[0]), 1e-6);
    EXPECT_NEAR(4.882813e-04, getStateProb(res2[4]), 1e-6);

    finalize();
    getchar();
}


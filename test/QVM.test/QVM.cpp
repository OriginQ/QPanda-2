#include "QPanda.h"
#include "gtest/gtest.h"
#include "time.h"
#include "Core/QuantumMachine/PartialAmplitudeQVM.h"
#include "Core/QuantumMachine/SingleAmplitudeQVM.h"
using namespace std;
USING_QPANDA

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

int main(int argc, char **argv) 
{
    testing::GTEST_FLAG(catch_exceptions) = 1;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

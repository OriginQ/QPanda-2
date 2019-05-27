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

TEST(QVM, PartialAmplitudeQVM)
{
    auto machine = new PartialAmplitudeQVM();
    machine->init();
    auto qlist = machine->allocateQubits(10);
    auto clist = machine->allocateCBits(10);


#if 0
    auto machine1 = new PartialAmplitudeQVM();
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
        << CZ(qlist[2], qlist[3]);
    machine->run(prog);

    /*Test getQStat*/
    auto res2 = machine->getQStat();

    for (int i = 0; i < 3; ++i)
    {
        cout << res2[i] << endl;
    }

    machine->finalize();
    getchar();
}



TEST(QVM, TotalAmplitudeQVM)
{
    auto machine = initQuantumMachine(CPU_SINGLE_THREAD);
    auto qlist = machine->allocateQubits(10);
    auto clist = machine->allocateCBits(10);

    auto prog = QProg();
    for_each(qlist.begin(), qlist.end(), [&](Qubit *val) { prog << H(val); });
    prog 
        << RY(qlist[7], PI / 3)
        << RX(qlist[8], PI / 3)
        << RX(qlist[9], PI / 3);


    machine->directlyRun(prog);
    machine->finalize();
    getchar();
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
        << CR(qlist[7], qlist[8], PI)
        << RZ(qlist[9], PI / 4)
        << RZ(qlist[1], PI / 4)
        << RX(qlist[2], PI / 2)
        << CR(qlist[5], qlist[6], PI)
        << RZ(qlist[9], PI / 4)
        << RX(qlist[0], PI / 2)
        << CR(qlist[3], qlist[4], PI)
        << RY(qlist[7], PI / 2)
        << RX(qlist[8], PI / 2)
        << CR(qlist[1], qlist[5], PI)
        << CR(qlist[3], qlist[7], PI)
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
    QVec qvec;
    for_each(qlist.begin(), qlist.end(), [&](Qubit *val) { qvec.emplace_back(val); });

    /*Test PMeasure  selectmax*/
    EXPECT_NO_THROW(machine->PMeasure(-5));
    EXPECT_NO_THROW(machine->PMeasure(128));
    auto res = machine->PMeasure(256);

    for (auto val :res)
    {
        std::cout << val.first << " : " << val.second << endl;
    }
    EXPECT_THROW(machine->PMeasure(1280), qprog_syntax_error);

    /*Test PMeasure Qvec selectmax*/
    EXPECT_NO_THROW(machine->PMeasure(qvec, 6));
    EXPECT_NO_THROW(machine->PMeasure(qvec, 128));

    EXPECT_THROW(machine->PMeasure(qvec, 1280), qprog_syntax_error);

    QVec temp;
    EXPECT_THROW(machine->PMeasure(temp, 128), qprog_syntax_error);

    QVec temp1 = { qlist[0],qlist[0] };
    EXPECT_THROW(machine->PMeasure(temp1, 128), qprog_syntax_error);

    /*Test getQStat*/
    EXPECT_NO_THROW(machine->getQStat());

    /*Test PMeasure_index*/
    EXPECT_THROW(machine->PMeasure_index(-8), qprog_syntax_error);
    EXPECT_THROW(machine->PMeasure_index(1280), qprog_syntax_error);
    EXPECT_NO_THROW(machine->PMeasure_index(124));

    machine->finalize();
    getchar();
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
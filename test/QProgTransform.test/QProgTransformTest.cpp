#include <map>
#include "QPanda.h"
#include <algorithm>  
#include "gtest/gtest.h"
#include "Core/Utilities/Transform/TransformDecomposition.h"
using namespace std;
USING_QPANDA

TEST(QProgTransform, QASM)
{
    auto qvm =initQuantumMachine();

    auto prog = CreateEmptyQProg();
    auto cir = CreateEmptyCircuit();

    auto q0 = qvm->allocateQubit();
    auto q1 = qvm->allocateQubit();
    auto q2 = qvm->allocateQubit();
    auto c0 = qvm->allocateCBit();

    auto qlist = qvm->allocateQubits(2);

    cir << Y(q2) << H(q2);
    cir.setControl(qlist);

    auto h1 = H(q1);
    h1.setDagger(true);

    prog << H(q1) << X(q2) << h1 << RX(q1, 2 / PI) << cir << CR(q1, q2, PI / 2);

    cout << transformQProgToQASM(prog,qvm);

    destroyQuantumMachine(qvm);
    getchar();
}

TEST(QProgTransform, QRunesToQProg)
{
    auto qvm = initQuantumMachine();
    auto prog = CreateEmptyQProg();

    qRunesToQProg("D:\\QRunes", prog);

    cout << transformQProgToQASM(prog,qvm) << endl;

    destroyQuantumMachine(qvm);

    getchar();
}

TEST(QProgTransform, qprogDecomposition)
{
    auto qvm = initQuantumMachine(QPanda::CPU_SINGLE_THREAD);
    auto qlist = qvm->allocateQubits(4);
    auto clist = qvm->allocateCBits(4);
    auto prog = CreateEmptyQProg();
    auto gate = X(qlist[3]);
    auto gate1 = gate.control({ qlist[0],qlist[1],qlist[2] });
    prog << X(qlist[0]) << X(qlist[1]) << X(qlist[2]) << gate1;
    prog << Measure(qlist[0], clist[0]);
    std::vector<std::vector<std::string>> ValidQGateMatrix = { { "RX","RY","RZ" },{ "CNOT","CZ" } };
    std::vector<std::vector<std::string>> QGateMatrix = { { "RX","RY","RZ" },{ "CNOT","CZ" } };
    std::vector<std::vector<int> > vAd;

    TransformDecomposition td(ValidQGateMatrix, QGateMatrix, vAd, qvm);
    td.TraversalOptimizationMerge(prog);

    auto progRunes = transformQProgToQRunes(prog, qvm);

    cout << progRunes << endl;

    destroyQuantumMachine(qvm);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

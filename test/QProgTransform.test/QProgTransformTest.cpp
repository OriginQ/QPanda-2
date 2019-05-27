#include <map>
#include "QPanda.h"
#include <algorithm>  
#include "gtest/gtest.h"
#include "Core/Utilities/Transform/TransformDecomposition.h"
#include "include/Core/Utilities/QNodeDeepCopy.h"
using namespace std;
USING_QPANDA

TEST(QNodeDeepCopy, deepCopy)
{
    auto machine = initQuantumMachine(QMachineType::CPU);
    auto q = machine->allocateQubits(20);
    auto c = machine->allocateCBits(20);

    auto prog = QProg();
    auto cir = QCircuit();
    auto cir1 = QCircuit();


    cir << Y(q[2]) << H(q[2])<<CNOT(q[0],q[1])<<cir1;


    auto while_prog= CreateWhileProg(c[1], &cir);

    auto cprpg = ClassicalProg(c[0]);

    auto me = Measure(q[1], c[1]); 
       
    prog << cprpg << me << while_prog;

    std::cout << transformQProgToQRunes(prog, machine) << endl;

    auto temp = deepCopy(prog);
    prog.clear();

    std::cout << transformQProgToQRunes(temp, machine) << endl;
    std::cout << transformQProgToQRunes(prog, machine) << endl;

    machine->finalize();
    getchar();
}


TEST(QProgTransform, QASM)
{
    exit(0);

    auto qvm = initQuantumMachine();

    auto prog = CreateEmptyQProg();
    auto cir = CreateEmptyCircuit();

    auto q = qvm->allocateQubits(6);
    auto c = qvm->allocateCBits(6);


    cir << Y(q[2]) << H(q[2]);
    cir.setDagger(true);

    auto h1 = H(q[1]);
    h1.setDagger(true);

    prog << H(q[1])
        << X(q[2])
        << h1
        << RX(q[1], 2 / PI)
        << cir
        << CR(q[1], q[2], PI / 2)
        << MeasureAll(q, c);

    cout << transformQProgToQASM(prog, qvm);

    qvm->finalize();
    getchar();
}

TEST(QProgTransform, QRunesToQProg)
{
    auto qvm = initQuantumMachine();
    auto prog = CreateEmptyQProg();

    transformQRunesToQProg("D:\\QRunes", prog, qvm);

    cout << transformQProgToQASM(prog, qvm) << endl;

    qvm->finalize();

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

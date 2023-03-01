#include <map>
#include <cstdlib>
#include <sstream>
#include <string>
#include <complex>
#include <algorithm>
#include <regex>
#include <ctime>
#include "Core/Core.h"
#include "gtest/gtest.h"
#include "Core/QuantumCircuit/QNodeDeepCopy.h"
#include "Core/Utilities/Tools/MatrixDecomposition.h"
#include "Core/VirtualQuantumProcessor/MPSQVM/MPSQVM.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/GraphMatch.h"
#include "Core/Utilities/Tools/JsonConfigParam.h"
#include "Core/Utilities/Tools/QCircuitGenerator.h"


#include "Extensions/Extensions.h"
#ifdef USE_EXTENSION

using namespace std;
USING_QPANDA

static bool test_vf1_0()
{
    auto qvm = CPUQVM();
    qvm.init();
    cout << "111111111" << endl;
    qvm.set_parallel_threads(5);
    auto q = qvm.qAllocMany(20);
    auto c = qvm.cAllocMany(20);
    auto circuit = QCircuit();
    auto prog = QProg();
    for (auto &qbit : q) {
        circuit << RX(qbit, rand());
    }
    for (auto &qbit : q) {
        circuit << RY(qbit, rand());
    }
    for (size_t j = 0; j < 18; ++j) {
        circuit << CNOT(q[j], q[(j + 1)]);
    }
    for (size_t k = 0; k < 8; ++k) {
        for (auto &qbit : q) {
            circuit << RZ(qbit, rand());
        }
        for (auto &qbit : q) {
            circuit << RX(qbit, rand());
        }
        for (auto &qbit : q) {
            circuit << RZ(qbit, rand());
        }
        for (size_t j = 0; j < 8; ++j) {
            circuit << CNOT(q[j], q[(j + 1)]);
        }
    }
    for (auto &qbit : q) {
        circuit << RZ(qbit, rand());
    }
    for (auto &qbit : q) {
        circuit << RX(qbit, rand());
    }

    prog << circuit;
    Fusion fuser;
    fuser.aggregate_operations(prog);
    qvm.directlyRun(prog);
    return true;
}


static bool test_vf1_1()
{
    auto qvm = NoiseQVM();
    qvm.init();
    qvm.set_parallel_threads(5);
    auto q = qvm.qAllocMany(20);
    auto c = qvm.cAllocMany(20);
    auto circuit = QCircuit();
    auto prog = QProg();
    for (auto &qbit : q) {
        circuit << RX(qbit, rand());
    }
    for (auto &qbit : q) {
        circuit << RY(qbit, rand());
    }
    for (size_t j = 0; j < 18; ++j) {
        circuit << CNOT(q[j], q[(j + 1)]);
    }
    for (size_t k = 0; k < 8; ++k) {
        for (auto &qbit : q) {
            circuit << RZ(qbit, rand());
        }
        for (auto &qbit : q) {
            circuit << RX(qbit, rand());
        }
        for (auto &qbit : q) {
            circuit << RZ(qbit, rand());
        }
        for (size_t j = 0; j < 8; ++j) {
            circuit << CNOT(q[j], q[(j + 1)]);
        }
    }
    for (auto &qbit : q) {
        circuit << RZ(qbit, rand());
    }
    for (auto &qbit : q) {
        circuit << RX(qbit, rand());
    }

    prog << circuit;
    qvm.directlyRun(prog);

    return true;
}


TEST(SetParallelThreads, test1)
{
    bool test_val = false;
    try
    {
        test_val = test_vf1_0();
        test_val = test_vf1_1();

    }
    catch (const std::exception& e)
    {
        cout << "Got a exception: " << e.what() << endl;
    }
    catch (...)
    {
        cout << "Got an unknow exception: " << endl;
    }

    //ASSERT_TRUE(test_val);

    //cout << "VF2 test over, press Enter to continue." << endl;
    //getchar();
}

#endif
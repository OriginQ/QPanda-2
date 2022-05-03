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
#include "Core/Utilities/QProgTransform/QProgToDAG/GraphMatch.h"
#include "Core/Utilities/Tools/QCircuitGenerator.h"


#include "Extensions/Extensions.h"
#ifdef USE_EXTENSION

using namespace std;
USING_QPANDA

static bool test_vf1_0()
{
    int deepth = 9;
    for (int i = 10; i < 11; i++)
    {
        auto qvm = CPUQVM();
        qvm.init();
        auto qubits = qvm.qAllocMany(i);
        auto c = qvm.cAllocMany(i);

        auto circuit = CreateEmptyCircuit();
        auto prog = CreateEmptyQProg();
        for (auto &qbit : qubits) {
            circuit << RX(qbit, rand());
        }
        for (auto &qbit : qubits) {
            circuit << RY(qbit, rand());
        }
        for (size_t j = 0; j < i; ++j) {
            circuit << CNOT(qubits[j], qubits[(j + 1) % i]);
        }
        for (size_t k = 0; k < deepth; ++k) {
            for (auto &qbit : qubits) {
                circuit << RZ(qbit, rand()).dagger();
            }
            for (auto &qbit : qubits) {
                circuit << RX(qbit, rand()).dagger();
            }
            for (auto &qbit : qubits) {
                circuit << RZ(qbit, rand()).dagger();
            }
            for (size_t j = 0; j < i; ++j) {
                circuit << CNOT(qubits[j], qubits[(j + 1) % i]);
            }
        }
        for (auto &qbit : qubits) {
            circuit << RZ(qbit, rand());
        }
        for (auto &qbit : qubits) {
            circuit << RX(qbit, rand());
        }
        auto cir = QCircuit();
        cir << H(qubits[0]) << H(qubits[1]);

        auto mat = getCircuitMatrix(cir);
        prog << QOracle({ qubits[0],qubits[1] }, mat);
        prog << circuit /*<< MeasureAll(qubits, c)*/;
        cout << prog << endl;
        std::cout << "===============================" << std::endl;
        auto start = std::chrono::system_clock::now();
        //auto result = qvm.probRunDict(prog, qubits);
        //auto result = qvm.runWithConfiguration(prog, c,1);
        auto q = { qubits[0],qubits[1],qubits[2] };
        qvm.probRunDict(prog, q);
        auto end = std::chrono::system_clock::now();
        auto stat = qvm.getQState();
        for (auto res : stat)
        {
            cout << res << std::endl;
        }
        std::chrono::duration<double>elapsed_seconds = end - start;
        std::cout << "qbit: " << i << ", Time used:  " << elapsed_seconds.count() << std::endl;
    }
    return true;
}


static bool test_vf1_1()
{
    auto qpool = OriginQubitPool::get_instance();
    auto cpool = OriginCMem::get_instance();

    auto qv = qpool->qAllocMany(2);
    auto cv = cpool->cAllocMany(2);

    auto qprog = QProg();
    qprog << X(qv[0]) << I(qv[0]) << MeasureAll(qv, cv);

    auto noise = NoiseModel();
    noise.add_noise_model(NOISE_MODEL::DAMPING_KRAUS_OPERATOR, GateType::I_GATE, 0.7);
    noise.add_noise_model(NOISE_MODEL::BITFLIP_KRAUS_OPERATOR, GateType::I_GATE, 0.7);

    auto qm = CPUQVM();
    qm.init();

    auto r1 = qm.runWithConfiguration(qprog, cv, 1000, noise);
    qm.finalize();

    auto nqm = NoiseQVM();
    nqm.init();
    nqm.set_noise_model(NOISE_MODEL::DAMPING_KRAUS_OPERATOR, GateType::I_GATE, 0.7);
    nqm.set_noise_model(NOISE_MODEL::BITFLIP_KRAUS_OPERATOR, GateType::I_GATE, 0.7);
    auto r2 = nqm.runWithConfiguration(qprog, cv, 1000);
    nqm.finalize();

    for (auto i : r1)
    {
        std::cout << i.first << " : " << i.second << std::endl;
    }

    std::cout << "-----------------------------------\n";

    for (auto i : r2)
    {
        std::cout << i.first << " : " << i.second << std::endl;
    }

    return true;

}




TEST(QProgOptimize, test1)
{
    bool test_val = false;
    try
    {
        test_val = test_vf1_0();

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
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
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include "Core/Variational/var.h"
#include "Variational/expression.h"
#include "Variational/utils.h"
#include "Variational/Optimizer.h"
#include "Variational/VarFermionOperator.h"
#include "Variational/VarPauliOperator.h"

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
        for (size_t k = 0; k < deepth - 1; ++k) {
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

        prog << circuit;
        std::cout << "===============================" << std::endl;
        
        Fusion fuser;
        fuser.aggregate_operations(prog, &qvm);
        prog << RXX(qubits[0], qubits[2], 20)
            << RYY(qubits[1], qubits[4], 20) 
            << RZZ(qubits[3], qubits[5], 20)
            << RZX(qubits[2], qubits[0], 20); 
        cout << prog << endl;

        auto prog_text = convert_qprog_to_originir(prog, &qvm);
        std::cout << prog << std::endl;
        std::cout << "===============================" << std::endl;
        std::cout << prog_text << std::endl;
    }
    return true;
}


static bool test_vf1_1()
{
    
    auto qvm = CPUQVM();
    qvm.init();
    auto q = qvm.qAllocMany(10);
    auto c = qvm.cAllocMany(10);
    auto prog = QProg();
    prog << H(q[0]);
    for (int i = 0; i < 9; i++)
    {
        prog << CNOT(q[i], q[i+1]);
    }

    for (int i = 1; i < 10; i += 2)
    {
        prog << Measure(q[i], c[i]);
    }


    auto res = qvm.runWithConfiguration(prog, c, 1000);
    for (auto &re : res)
    {
        std::cout << re.first << ", " << re.second << std::endl;
    }
    //auto res = qvm.probRunDict(prog, q);
    
    return true;
}
static bool test_vf1_2()
{

    auto qvm = CPUQVM();
    qvm.init();
    auto q = qvm.qAllocMany(10);
    auto c = qvm.cAllocMany(10);
    auto prog = QProg();
    prog << H(q[0]).dagger();

    cout << prog << endl;

    return true;
}



TEST(DrawQOracleQDouble, test1)
{
    bool test_val = false;
    try
    {
        test_val = test_vf1_2();

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
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
    
    auto cpu_qvm = CPUQVM();
    cpu_qvm.init();
    
    var x(MatrixXd::Zero(1, 3));
    var y(MatrixXd::Zero(1, 3));
    MatrixXd tmp1(1, 3);
    tmp1 << 1, 1, 2;
    MatrixXd tmp2(1, 3);
    tmp2 << 0.8, 0.8, 0.8; // p is probability
    x.setValue(tmp1);
    y.setValue(tmp2);
    complex_var a(1,2);
    complex_var b(3,4);
    auto c = a.real();
    auto d = a.imag();
    std::cout << "a,real: " << c.getValue() << ", a.imag: " << d.getValue() << endl;
    a=a*2;
    auto e = a.real();
    auto f = a.imag();
    std::cout << "a,real: " << e.getValue() << ", a.imag: " << f.getValue() << endl;

    
    
    std::cout << "x=" << "\n" << x.getValue() << std::endl;
    x = -x;
    std::cout << "x=" << "\n" << x.getValue() << std::endl;
    return true;
}



TEST(DrawQOracleQDouble, test1)
{
    bool test_val = false;
    try
    {
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
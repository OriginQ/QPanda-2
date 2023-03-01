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

#ifdef USE_EXTENSION
#include "Extensions/Extensions.h"
using namespace std;
USING_QPANDA
static bool test_vf1_0()
{
    for (int i = 4; i < 14; i++)
    {
        for (int x = 0; x < 2; x++)
        {
            bool flag = true;
            if (x)
            {
                flag = !flag;
            }
            auto qvm = CPUQVM();
            qvm.init();
            auto qubits = qvm.qAllocMany(i);
            auto prog = QProg();
            for (auto &qbit : qubits) {
                prog << RX(qbit, rand());
            }
            for (auto &qbit : qubits) {
                prog << RY(qbit, rand());
            }
            for (size_t j = 0; j < i; ++j) {
                prog << CNOT(qubits[j], qubits[(j + 1) % i]);
            }
            for (size_t k = 0; k < 9; ++k) {
                for (auto &qbit : qubits) {
                    prog << RZ(qbit, rand());
                }
                for (auto &qbit : qubits) {
                    prog << RX(qbit, rand());
                }
                for (auto &qbit : qubits) {
                    prog << RZ(qbit, rand());
                }
                for (size_t j = 0; j < i; ++j) {
                    prog << CNOT(qubits[j], qubits[(j + 1) % i]);
                }
            }
            for (auto &qbit : qubits) {
                prog << RZ(qbit, rand());
            }
            for (auto &qbit : qubits) {
                prog << RX(qbit, rand());
            }

            std::cout << "========================" << std::endl;
            auto start1 = std::chrono::system_clock::now();
            auto mat = getCircuitMatrix(prog, flag);
            auto end1 = std::chrono::system_clock::now();
            std::chrono::duration<double>elapsed_seconds1 = end1 - start1;
            std::cout << i << ": bit," << "getCircuitMatrix, Time used:  " << elapsed_seconds1.count() << std::endl;
            std::cout << getQGateNum(prog) << std::endl;
            auto start = std::chrono::system_clock::now();
            /*Fusion fuser;
            fuser.aggregate_operations(prog);*/
            auto res = get_unitary(prog, !flag);
            auto end = std::chrono::system_clock::now();
            std::chrono::duration<double>elapsed_seconds = end - start;
            std::cout << i << ": bit," << "get_unitary, Time used:  " << elapsed_seconds.count() << std::endl;
            std::cout << "========================" << std::endl;


            if (mat != res)
            {
                std::cout << "get_unitary failed" << std::endl;
                return false;
            }
        }
        

    }

    return true;
}

static bool test_vf1_1()
{
    auto qvm = CPUQVM();
    qvm.init();
    auto q = qvm.qAllocMany(3);
    auto prog = QProg();
    
    auto cir = QCircuit();
    auto prog1 = QProg();
    prog1 << U4(q[0], 0.5, 0.6, 0.7, 0.8).dagger() << U4(q[1], 0.2, 0.3, 0.4, 0.5).dagger() << U4(q[2], 0.1, 0.3, 0.4, 0.5).dagger();
    auto res = getCircuitMatrix(prog1, true);
    prog <<CNOT(q[0],q[1])<< QOracle({q[0],q[1] ,q[2] },res) <<H(q[0])<< CNOT(q[1], q[2]);
    auto prog2 = QProg();
    prog2 << CNOT(q[0], q[1]) << U4(q[0], 0.5, 0.6, 0.7, 0.8).dagger() << U4(q[1], 0.2, 0.3, 0.4, 0.5 ).dagger() << U4(q[2], 0.1, 0.3, 0.4, 0.5).dagger()<< H(q[0]) << CNOT(q[1], q[2]);
    
    
    auto res1 = getCircuitMatrix(prog2, false);
    Fusion fuser;
    fuser.aggregate_operations(prog);
    std::cout << prog << std::endl;
    auto mat = get_unitary(prog,true);
    if (res1 != mat)
    {
        std::cout << "get matrix error" << std::endl;
    }
    /*Fusion fuser;
    fuser.aggregate_operations(prog);*/
    
    //std::cout << mat << std::endl;
    /*std::cout << "======================" << std::endl;
    std::cout << res << std::endl;*/
    return true;
}

const std::string test_IR_5 = R"(QINIT 4
CREG 0
DAGGER
CONTROL q[1]
U4 q[3],(0,0,0,0)
ENDCONTROL
ENDDAGGER
DAGGER
CONTROL q[1]
U4 q[3],(1.5707963,2.2855193,1.2958022,2.2855193)
ENDCONTROL
ENDDAGGER
DAGGER
CONTROL q[3]
U4 q[1],(1.5707963,-3.1415927,0,0)
ENDCONTROL
ENDDAGGER
DAGGER
CONTROL q[1]
U4 q[3],(-1.5707963,0.85607337,1.2958022,-2.2855193)
ENDCONTROL
ENDDAGGER
DAGGER
U4 q[1],(-1.5707963,-0,2.3561945,-3.1415927)
ENDDAGGER
DAGGER
CONTROL q[1]
U4 q[3],(-1.5707963,0,0,0)
ENDCONTROL
ENDDAGGER
DAGGER
U4 q[3],(1.5707963,1.5707963,0.78539816,-3.1415927)
ENDDAGGER
)";

static bool test_vf1_2()
{
    auto qvm = new CPUQVM();
    qvm->init();
    auto q = qvm->qAllocMany(2);

   /* auto random_prog = random_qprog(2, 1, 20, qvm, q);

    auto random_prog_matrix = getCircuitMatrix(random_prog,true);

    

    QCircuit out_cir = matrix_decompose_qr(q, random_prog_matrix);
    auto str = convert_qprog_to_originir(out_cir, qvm);
    std::cout << str << std::endl;*/
    //decompose_multiple_control_qgate(out_cir, qvm);
    auto prog = convert_originir_string_to_qprog(test_IR_5, qvm);
    //decompose_multiple_control_qgate(prog, qvm);
    
    /*prog << CNOT(q[0], q[1]);*/
   /*auto prog = QProg(); 
   prog << U3(q[0], 0.984, -1.57906, 0.5531) << U3(q[1], 2.218697, 0, 0.850673) << CZ(q[0],q[1])
        << U3(q[1], 2.218697, 0.3354, 0.850673) << CZ(q[0], q[1])
        << U3(q[0], 0.547, -1.57906, 0) << U3(q[1], 2.218697, 0.3325, 0.850673)
        << CZ(q[1], q[0]) << U3(q[0], 0, -1.57906, 0) << CZ(q[1], q[0])
        << U3(q[0], 0, -1.57906, 0) << U3(q[1], 2.218697, 0, 0.850673)
        << CZ(q[0], q[1]) << U3(q[1], 2.218697, 0, 0.850673)
        << CZ(q[0], q[1]) << U3(q[0], 0, -1.57906, 0) << U3(q[1], 2.218697, 0.154, 0.850673)
        << CZ(q[0], q[1]) << CZ(q[0], q[1]) << U3(q[1], 2.218697, 0, 0.850673);*/
    std::cout << prog << std::endl;
    auto mat = get_unitary(prog);
    auto circuit_matrix = getCircuitMatrix(prog, true);
    cout << "source matrix:" << endl;
    for (int i = 0; i < sqrt(mat.size()); i++)
    {
        for (int j = 0; j < sqrt(mat.size()); j++)
        {
            std::cout << mat[i* sqrt(mat.size()) + j] << ",  ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    //cout << "source matrix:" << endl << mat << endl;

    cout << "the decomposed matrix:" << endl << circuit_matrix << endl;

    if (mat==circuit_matrix)
    {
        cout << "matrix decompose ok !" << endl;
        return true;
    }
    return false;
}

TEST(GetUnitaryMatrix, test1)
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
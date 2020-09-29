#include <map>
#include <cstdlib>
#include <sstream>
#include <string>
#include <complex>
#include <algorithm>
#include "Core/Core.h"
#include "gtest/gtest.h"
#include "Core/QuantumCircuit/QNodeDeepCopy.h"
#include "Core/Utilities/Tools/MatrixDecomposition.h"
#include "Core/VirtualQuantumProcessor/MPSQVM/MPSQVM.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/GraphMatch.h"

using namespace std;
USING_QPANDA

//TEST(MatrixXi, Eigen)
//{
//    MPSQVM qvm;
//
//    Configuration config = { 64,64 };
//    qvm.setConfig(config);
//
//    qvm.init();
//
//#if 1
//    auto q = qvm.qAllocMany(10);
//    auto c = qvm.cAllocMany(10);
//
//
//    auto prog = QProg();
//    for_each(q.begin(), q.end(), [&](Qubit *val) { prog << H(val); });
//    prog << CZ(q[1], q[5])
//        << CZ(q[3], q[7])
//        << CZ(q[0], q[4])
//        << RZ(q[7], PI / 4)
//        << RX(q[5], PI / 4)
//        << RX(q[4], PI / 4)
//        << RY(q[3], PI / 4)
//        << CZ(q[2], q[6])
//        << RZ(q[3], PI / 4)
//        << RZ(q[8], PI / 4)
//        << CZ(q[9], q[5])
//        << RY(q[2], PI / 4)
//        << RZ(q[9], PI / 4) 
//        << CR(q[2], q[7], PI / 2)
//
//        << MeasureAll(q, c);
//#else
//    auto q = qvm.qAllocMany(10);
//    auto c = qvm.cAllocMany(10);
//    QProg prog;
//    prog << H(q[0])
//         << CNOT(q[0], q[1])
//         //<< CNOT(q[0], q[2])
//         << MeasureAll(q, c);
//#endif
//     
//    qvm.set_noise_model(NOISE_MODEL::DAMPING_KRAUS_OPERATOR, { 0.1 });
//
//    auto result = qvm.runWithConfiguration(prog, c, 100);
//    for (auto val : result)
//    {
//        cout << val.first << " : " << val.second << endl;
//    }
//
//    getchar(); 
//}

TEST(MatrixXi, Eigen)
{
    return;
}


TEST(GraphMatch, Query)
{
    return;
    auto qvm = initQuantumMachine();
    auto q = qvm->allocateQubits(4);

#if 0
/*
              ┌─┐┌────────────┐┌──┐
    q_0:  |0>─┤H├┤RX(3.141593)├┤CZ├
              ├─┤├────────────┤└─┬┘
    q_1:  |0>─┤H├┤RX(3.141593)├──■─
              ├─┤├────────────┤┌──┐
    q_2:  |0>─┤H├┤RX(3.141593)├┤CZ├
              ├─┤├────────────┤└─┬┘
    q_3:  |0>─┤H├┤RX(3.141593)├──■─
              └─┘└────────────┘

              ┌──┐
    q_0:  |0>─┤CZ├
              └─┬┘
    q_1:  |0>───■─
              ┌──┐
    q_2:  |0>─┤CZ├
              └─┬┘
    q_3:  |0>───■─

*/
    auto prog = QProg();
    prog << H(q[0]) << H(q[1]) << H(q[2]) << H(q[3]) 
         << RX(q[0],PI)<< RX(q[1], PI)<< RX(q[2], PI)<< RX(q[3], PI)
         << CZ(q[1], q[0]) << CZ(q[3], q[2]);

    auto query_cir = QCircuit();
    query_cir << H(q[0]) << H(q[1])
              << RX(q[0], PI) << RX(q[1], PI)
              << CZ(q[1], q[0]);

    auto replace_cir = QCircuit();
    replace_cir << CZ(q[0],q[1]);
#else
/*
              ┌─┐┌────┐┌─┐
    q_0:  |0>─┤H├┤CNOT├┤H├───────────────
              └─┘└──┬─┘└─┘
    q_1:  |0>───────■─────■──────────────
              ┌─┐      ┌──┴─┐┌─┐
    q_2:  |0>─┤H├──────┤CNOT├┤H├───■─────
              ├─┤      └────┘└─┘┌──┴─┐┌─┐
    q_3:  |0>─┤H├───────────────┤CNOT├┤H├
              └─┘               └────┘└─┘

              ┌──┐
    q_0:  |0>─┤CZ├────────
              └─┬┘
    q_1:  |0>───■───■─────
                  ┌─┴┐
    q_2:  |0>─────┤CZ├──■─
                  └──┘┌─┴┐
    q_3:  |0>─────────┤CZ├
                      └──┘
*/

    auto prog = QProg();
    prog << H(q[0]) << H(q[2]) << H(q[3])
        << CNOT(q[1], q[0]) << H(q[0]) << CNOT(q[1], q[2])
        << H(q[2]) << CNOT(q[2], q[3]) << H(q[3]);

    auto query_cir = QCircuit();
    query_cir << H(q[0]) << CNOT(q[1], q[0]) << H(q[0]);

    auto replace_cir = QCircuit();
    replace_cir << CZ(q[0], q[1]);
#endif

    ResultVector result;
    graph_query(prog, query_cir, result);

    for (auto val : result)
    {
        for (auto i = 0; i < val.size(); ++i)
        {
            cout << val[i].m_node_type << "(" << val[i].m_vertex_num << ")";
        }
        cout << endl;
    }

    QProg update_prog;
    graph_query_replace(prog, query_cir, replace_cir, update_prog, qvm);

    cout << transformQProgToOriginIR(update_prog, qvm);
} 
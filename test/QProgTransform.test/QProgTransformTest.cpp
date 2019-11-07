#include <map>
#include <cstdlib>
#include <sstream>
#include <string>
#include <algorithm>  
#include "gtest/gtest.h"
#include "Core/Core.h"
#include "Core/QuantumCircuit/QNodeDeepCopy.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/GraphMatch.h"
#include "ThirdParty/Eigen/Dense"


using namespace std;
USING_QPANDA

QCircuit _CZ(Qubit * q1, Qubit * q2)
{
    QCircuit cir;
    cir << H(q1)
        << CNOT(q2, q1)
        << H(q1);
    return  cir;
}

TEST(MatrixXi, EigenTest)
{
    throw exception();
    Eigen::ArrayXXf a(2, 2);

    AdjacencyMatrix temp= AdjacencyMatrix::Zero(2,4);
    temp(0, 0) = 1;
    cout << temp << endl;
    cout << temp.row(0) << endl;
    cout << temp.row(1) << endl;
    cout << temp.rows() << endl;
    cout << temp.row(0).minCoeff() << endl;
    cout << temp.row(1).minCoeff() << endl;
    cout << temp.cols() << endl;
}

TEST(MatrixXi, DAG)
{
    throw exception();
    auto qvm = initQuantumMachine();
    auto q = qvm->allocateQubits(6);

    auto prog = QProg();
    prog << H(q[0]) << H(q[2]) << H(q[3])
        << CNOT(q[1], q[0]) << H(q[0]) << CNOT(q[1], q[2])
        << H(q[2]) << CNOT(q[2], q[3]) << H(q[3]);


    GraphMatch match;
    TopologicalSequence seq;
    match.get_topological_sequence(prog, seq);

    getchar();
}


TEST(MatrixXi, Eigen)
{
    //throw exception();
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

    getchar();
}

int main(int argc, char **argv)
{
    testing::GTEST_FLAG(catch_exceptions) = 1;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
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
#include "Core/Utilities/QProgTransform/QProgToDAG/GraphMatch.h"

using namespace std;
USING_QPANDA

TEST(MatrixXi, Eigen)
{
	auto qvm =new CPUQVM();
	qvm->init();
	auto q = qvm->qAllocMany(6);
	auto c = qvm->cAllocMany(6 );

	QCircuit in;

	QStat test_matrix = {
		qcomplex_t(0.15774658, -0.52244755), qcomplex_t(-0.65774658, 0.02244755), qcomplex_t(-0.30419319, 0.16889416), qcomplex_t(0.19580681, -0.33110584),
		qcomplex_t(-0.65774658, 0.02244755), qcomplex_t(0.15774658, -0.52244755), qcomplex_t(-0.19580681, 0.33110584), qcomplex_t(0.30419319, -0.16889416),
		qcomplex_t(-0.30419319, 0.16889416), qcomplex_t(-0.19580681, 0.33110584), qcomplex_t(0.15774658, -0.52244755), qcomplex_t(0.65774658, -0.02244755),
		qcomplex_t(0.19580681, -0.33110584), qcomplex_t(0.30419319, -0.16889416), qcomplex_t(0.65774658, -0.02244755), qcomplex_t(0.15774658, -0.52244755)
	};

	QVec used_qubits = { q[0], q[1] };
	QCircuit out = matrix_decompose(used_qubits, test_matrix);

	auto b = getCircuitMatrix(out);
	cout << b << endl;

	if (test_matrix == b)
	{
		cout << "test ok" << endl;
	}

    getchar();
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
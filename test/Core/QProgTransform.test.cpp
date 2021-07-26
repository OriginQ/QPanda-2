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

static bool test_prog_to_dag_1()
{
	auto qvm = initQuantumMachine();
	auto q = qvm->allocateQubits(4);

	QProg prog;
	prog << H(q[0]) << H(q[2]) << H(q[3])
		<< CNOT(q[1], q[0]) << CNOT(q[0], q[1]) << H(q[0]) << CNOT(q[1], q[2])
		<< H(q[2]) << CNOT(q[2], q[3]) << H(q[3]);
	//cout << "src prog:" << prog << endl;
	auto dag = qprog_to_DAG(prog);
	const std::set<QProgDAGEdge>& edges = dag->get_edges();
	const std::vector<QProgDAGVertex>& vertex_vec = dag->get_vertex();
	/*cout << "vertex_vec:" << endl;
	for (const auto& _v : vertex_vec)
	{
		cout << _v.m_id << " layer:" << _v.m_layer << " "
			<< TransformQGateType::getInstance()[(GateType)(_v.m_type)] << " qubit(";
		for (const auto& _q : _v.m_node->m_control_vec)
		{
			cout << _q->get_phy_addr() << ", ";
		}
		for (const auto& _q : _v.m_node->m_qubits_vec)
		{
			cout << _q->get_phy_addr() << ", ";
		}

		cout << ")" << endl;
	}*/

	auto seq = dag->build_topo_sequence();
	//cout << "seq size = " << seq.size() << endl;

	destroyQuantumMachine(qvm);
	//cout << "---------" << endl;
	if (seq.size() != 7)
		return false;
	else
		return true;
}

TEST(QProgToDAG, test1)
{
	bool test_val = false;
	try
	{
		test_val = test_prog_to_dag_1();
	}
	catch (const std::exception& e)
	{
		cout << "Got a exception: " << e.what() << endl;
	}
	catch (...)
	{
		cout << "Got an unknow exception: " << endl;
	}

	//cout << "QProgToDAG test over, press Enter to continue." << endl;
	//getchar();

	ASSERT_TRUE(test_val);
}
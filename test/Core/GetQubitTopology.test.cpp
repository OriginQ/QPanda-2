#include "gtest/gtest.h"
#include "QPanda.h"

USING_QPANDA

#define MAX_CONNECTIVITY 3

static bool get_qubit_topology_test_1()
{
	auto machine = initQuantumMachine(CPU);
	auto x = machine->allocateCBit();
	std::vector<SearchDataByUInt> search_sapce = { 3, 6, 6, 9, 10, 15, 11, 6/*, 9, 10, 15, 11, 9, 10, 15*//*, 11, 9, 10, 15, 11, 9, 10, 15, 11, 9, 10, 15, 11, 9, 10, 15, 11 */};
	/*search_sapce.push_back(8);
	search_sapce.push_back(7);
	search_sapce.push_back(6);
	search_sapce.push_back(0);
	search_sapce.push_back(6);
	search_sapce.push_back(3);
	search_sapce.push_back(6);
	search_sapce.push_back(4);*/
	/*search_sapce.push_back(6);
	search_sapce.push_back(6);
	search_sapce.push_back(6);
	search_sapce.push_back(6);
	search_sapce.push_back(6);
	search_sapce.push_back(6);
	search_sapce.push_back(7);
	search_sapce.push_back(14);
	search_sapce.push_back(9);
	search_sapce.push_back(12);
	search_sapce.push_back(4);
	search_sapce.push_back(9);
	search_sapce.push_back(9);
	search_sapce.push_back(7);
	search_sapce.push_back(21);
	search_sapce.push_back(15);
	search_sapce.push_back(3);
	search_sapce.push_back(11);
	search_sapce.push_back(3);
	search_sapce.push_back(9);
	search_sapce.push_back(7);
	search_sapce.push_back(21);
	search_sapce.push_back(15);
	search_sapce.push_back(11);
	search_sapce.push_back(11);
	search_sapce.push_back(3);
	search_sapce.push_back(9);
	search_sapce.push_back(7);*/

	for (size_t i = 0; i < 0; i++)
	{
		search_sapce.insert(search_sapce.end(), search_sapce.begin() + 20, search_sapce.end());
	}
	cout << "Grover will search through " << search_sapce.size() << " data." << endl;
	cout << "Start grover search algorithm:" << endl;
	QVec measure_qubits;
	QProg grover_Qprog = build_grover_alg_prog(search_sapce, x == 6, machine, measure_qubits, 2);

	//QProg grover_Qprog;
	//auto q = machine->allocateQubits(6);
	//grover_Qprog << applyQGate(q, H) << CNOT(q[0], q[1]) << U1(q[0], 2) << CNOT(q[0], q[1])
	//	<< CNOT(q[0], q[5]) << CNOT(q[0], q[4]) << CNOT(q[0], q[4]) << CZ(q[0], q[4]) << iSWAP(q[0], q[4], 4)/*.control(q[5])*/
	//	<< CNOT(q[0], q[3]) << CNOT(q[0], q[4]) << CR(q[0], q[2], 3) << CNOT(q[3], q[5]) << CNOT(q[3], q[5])
	//	<< CNOT(q[4], q[5]) << Y(q[4]) << CNOT(q[2], q[5])/*.control(q[0])*/ << X(q[5]) << CNOT(q[5], q[2]) << CNOT(q[3], q[5]);
	//cout << "grover_Qprog:" << grover_Qprog << endl;
	//transform_to_base_qgate(grover_Qprog, machine);
	//cout << "after transform_to_base_qgate:" << grover_Qprog << endl;

	//获取双门块拓扑结构
	decompose_multiple_control_qgate(grover_Qprog, machine);
	TopologyData topolog_matrix = get_double_gate_block_topology(grover_Qprog);

	//{
	//	//for tmp test
	//	cout << "after decompose multip-gate, grover_Qprog:" << grover_Qprog << endl;

	//	cout << "the topolog_matrix:" << endl;
	//	for (const auto row : topolog_matrix)
	//	{
	//		cout << "[";
	//		for (const auto item : row)
	//		{
	//			cout << item << " ";
	//		}
	//		cout << "]" << endl;
	//	}
	//	
	//}

	//聚团查找
	std::vector<int> sub_graph =  get_sub_graph(topolog_matrix);

	//修剪权重较低的连边
	//del_weak_edge(topolog_matrix);
	std::vector<weight_edge> candidate_edges;
	std::vector<int> intermediary_points = del_weak_edge(topolog_matrix, MAX_CONNECTIVITY, sub_graph, candidate_edges);
	//std::vector<int> intermediary_points = del_weak_edge(topolog_matrix, sub_graph, 0.5, 0.5, 0.5);

	std::vector<int> complex_points = get_complex_points(topolog_matrix, MAX_CONNECTIVITY);

	std::vector<std::pair<int, TopologyData>> complex_point_sub_graph = split_complex_points(complex_points, MAX_CONNECTIVITY, topolog_matrix);

	replace_complex_points(topolog_matrix, MAX_CONNECTIVITY, complex_point_sub_graph);

	recover_edges(topolog_matrix, MAX_CONNECTIVITY, candidate_edges);

	double evaluate = estimate_topology(topolog_matrix);

	destroyQuantumMachine(machine);
	return true;
}

static bool get_qubit_topology_test_2()
{
	TopologyData test_topo_data = {
	{0, 2, 12, 2, 2, 0},
	{2, 0, 3, 2, 0, 2},
	{12, 2, 0, 0, 56, 2},
	{2, 6, 0, 0, 7, 2},
	{2, 0, 23, 2, 0, 2},
	{0, 2, 2, 2, 22, 0}
	};

	/*TopologyData test_topo_data = {
		{0, 0, 0, 0, 0, 0, 0, 25, 0, 0},
		{0, 0, 206, 138, 0, 0, 0, 0, 0, 0},
		{0, 206, 0, 276, 0, 40, 0, 0, 1, 0},
		{0, 138, 276, 0, 0, 40, 0, 0, 0, 1},
		{0, 0, 0, 0, 0, 0, 0, 0, 32, 32},
		{0, 0, 40, 40, 0, 0, 0, 20, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 40, 40},
		{25, 0, 0, 0, 0, 20, 0, 0, 24, 24},
		{0, 0, 1, 0, 32, 0, 40, 24, 0, 0},
		{0, 0, 0, 1, 32, 0, 40, 24, 0, 0}
	};*/

	/*TopologyData test_topo_data = {
		{0, 0, 0, 0, 0, 0, 0, 2, 0, 0},
		{0, 0, 2, 2, 0, 0, 0, 0, 0, 0},
		{0, 2, 0, 2, 0, 2, 0, 0, 2, 0},
		{0, 2, 2, 0, 0, 2, 0, 0, 0, 2},
		{0, 0, 0, 0, 0, 0, 0, 0, 2, 2},
		{0, 0, 2, 2, 0, 0, 0, 2, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 2, 2},
		{2, 0, 0, 0, 0, 2, 0, 0, 2, 2},
		{0, 0, 2, 0, 2, 0, 2, 2, 0, 0},
		{0, 0, 0, 2, 2, 0, 2, 2, 0, 0}
	};*/

	/*TopologyData test_topo_data = {
		{0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
		{0, 0, 1, 1, 0, 0, 0, 0, 0, 0},
		{0, 1, 0, 1, 0, 1, 0, 0, 1, 0},
		{0, 1, 1, 0, 0, 1, 0, 0, 0, 1},
		{0, 0, 0, 0, 0, 0, 0, 0, 1, 1},
		{0, 0, 1, 1, 0, 0, 0, 1, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 1, 1},
		{1, 0, 0, 0, 0, 1, 0, 0, 1, 1},
		{0, 0, 1, 0, 1, 0, 1, 1, 0, 0},
		{0, 0, 0, 1, 1, 0, 1, 1, 0, 0}
	};*/

	if (planarity_testing(test_topo_data))
	{
		cout << "planarity_testing PASS." << endl;
	}
	else
	{
		cout << "planarity_testing FAIL ......." << endl;
	}
	return true;
}

TEST(GetQubitTopology, test1)
{
	bool test_val = false;
	try
	{
		//test_val = get_qubit_topology_test_1();
		test_val = get_qubit_topology_test_2();
	}
	catch (const std::exception& e)
	{
		cout << "Got a exception: " << e.what() << endl;
	}
	catch (...)
	{
		cout << "Got an unknow exception: " << endl;
	}

	cout << "GetQubitTopology test over, press Enter to continue." << endl;
	getchar();

	ASSERT_TRUE(test_val);
}
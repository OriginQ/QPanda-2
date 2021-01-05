#include "QPanda.h"
#include "gtest/gtest.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include "Core/Utilities/QProgInfo/Visualization/QVisualization.h"
#include "Core/Utilities/Tools/QCircuitOptimize.h"

using namespace std;
USING_QPANDA

bool test_cir_optimize_fun1() 
{
	auto qvm = initQuantumMachine(QMachineType::CPU);
	auto q = qvm->allocateQubits(4);
	auto c = qvm->allocateCBits(4);

	QCircuit cir;
	QCircuit cir2;

	QProg prog;
	cir << CU(1, 2, 3, 4, q[1], q[0]) << (H(q[1])) << X(q[2]) << RZ(q[1], PI / 2) << Y(q[2])
		<< (CR(q[0], q[3], PI / 2)) << (S(q[2])) << S(q[1]) << RZ(q[1], PI / 2) << RZ(q[1], PI / 2) << RZ(q[1], PI / 2) << RZ(q[1], PI / 2) << Y(q[0]) << SWAP(q[3], q[1])
		<< CU(1, 2, 3, 4, q[1], q[0]) << (H(q[1])) << X(q[2]) << RX(q[1], PI / 2) << RX(q[1], PI / 2) << Y(q[2])
		<< CR(q[2], q[3], PI / 2) << CU(1, 2, 3, 4, q[1], q[0]) << (H(q[1])) << X(q[2]) << RZ(q[1], PI / 2) << Y(q[2]);
	cir2 << H(q[1]) << X(q[2]) << X(q[2]) << H(q[1]) << X(q[3]) << X(q[3]);


	//{
	//	cir2.setControl({ q[4] ,q[5]});
	//	auto layer_info = prog_layer(cir2);
	//	std::vector<std::vector<NodeInfo>> tmp_layer(layer_info.size());
	//	size_t layer_index = 0;
	//	for (auto& cur_layer : layer_info)
	//	{
	//		for (auto& node_item : cur_layer)
	//		{
	//			const pOptimizerNodeInfo& n = node_item.first;
	//			//single gate first
	//			if ((node_item.first->m_control_qubits.size() == 0) && (node_item.first->m_target_qubits.size() == 1))
	//			{
	//				tmp_layer[layer_index].insert(tmp_layer[layer_index].begin(),
	//					NodeInfo(n->m_iter, n->m_target_qubits,
	//						n->m_control_qubits, n->m_type,
	//						n->m_is_dagger));
	//			}
	//			else
	//			{
	//				tmp_layer[layer_index].push_back(NodeInfo(n->m_iter, n->m_target_qubits,
	//					n->m_control_qubits, n->m_type,
	//					n->m_is_dagger));
	//			}
	//		}

	//		++layer_index;
	//	}
	//	cout << endl;
	//}
	


	QCircuit cir3;
	QCircuit cir4;
	cir3 << H(q[1]) << H(q[2]) << CNOT(q[2], q[1]) << H(q[1]) << H(q[2]);
	cir4 << CNOT(q[1], q[2]);

	QCircuit cir5;
	QCircuit cir6;
	double theta_1 = PI / 3.0;
	cir5 << RZ(q[3], PI / 2.0) << CZ(q[3], q[0]) << RX(q[3], PI / 2.0) << RZ(q[3], theta_1) << RX(q[3], -PI / 2.0) << CZ(q[3], q[0]) << RZ(q[3], -PI / 2.0);
	cir6 << CZ(q[3], q[0]) << H(q[3]) << RZ(q[3], theta_1) << H(q[3]) << CZ(q[3], q[0]);

	/*QCircuit cir7;
	double theta_2 = PI / 5.0;
	cir7 << RZ(q[0], PI / 2.0) << CZ(q[0], q[3]) << RX(q[0], PI / 2.0) << RZ(q[0], theta_1) << RX(q[0], -PI / 2.0) << CZ(q[0], q[3]) << RZ(q[0], -PI / 2.0);
*/
	std::vector<std::pair<QCircuit, QCircuit>> optimitzer_cir;
	/*optimitzer_cir.push_back(std::pair<QCircuit, QCircuit>(cir3, cir4));
	optimitzer_cir.push_back(std::pair<QCircuit, QCircuit>(cir5, cir6));*/
	QCircuitOptimizerConfig config_reader;
	config_reader.get_replace_cir(optimitzer_cir);
	for (auto cir_item : optimitzer_cir)
	{
		cout << "target cir:" << endl << cir_item.first << endl;
		cout << "replaceed cir:" << endl << cir_item.second << endl;
	}

	prog << cir << cir2 << Reset(q[1]) << cir3 << cir5 << MeasureAll(q, c);
	cout << "befort optimizered QProg:" << endl;
	cout << prog << endl;

	sub_cir_optimizer(prog, optimitzer_cir);

	//prog << cir << cir2 << Reset(q[1]) << cir4 << cir6 << MeasureAll(q, c);

	cout << "The optimizered QProg:" << endl;
	cout << prog << endl;

	destroyQuantumMachine(qvm);
	return true;
}

bool test_cir_optimize_fun2()
{
	auto qvm = initQuantumMachine(QMachineType::CPU);
	auto q = qvm->allocateQubits(4);
	auto c = qvm->allocateCBits(4);

	QCircuit cir;
	QCircuit cir2;

	QProg prog;
	cir << CU(1, 2, 3, 4, q[1], q[0]) << (H(q[1])) << X(q[2]) << RZ(q[1], PI / 2) << Y(q[2])
		<< (CR(q[0], q[3], PI / 2)) << (S(q[2])) << S(q[1]) << RZ(q[1], PI / 2) << RZ(q[1], PI / 2) << RZ(q[1], PI / 2) << RZ(q[1], PI / 2) << Y(q[0]) << SWAP(q[3], q[1])
		<< CU(1, 2, 3, 4, q[1], q[0]) << (H(q[1])) << X(q[2]) << RX(q[1], PI / 2) << RX(q[1], PI / 2) << Y(q[2])
		<< CR(q[2], q[3], PI / 2) << CU(1, 2, 3, 4, q[1], q[0]) << (H(q[1])) << X(q[2]) << RZ(q[1], PI / 2) << Y(q[2]);
	cir2 << H(q[1]) << X(q[2]) << X(q[2]) << H(q[1]) << X(q[3]) << X(q[3]);

	QCircuit cir3;
	cir3 << H(q[1]) << H(q[2]) << CNOT(q[2], q[1]) << H(q[1]) << H(q[2]);

	QCircuit cir5;
	double theta_1 = PI / 3.0;
	cir5 << RZ(q[3], PI / 2.0).dagger() << CZ(q[3], q[0]).dagger() << RX(q[3], PI / 2.0).dagger() 
		<< RZ(q[3], theta_1) << RX(q[3], -PI / 2.0) << CZ(q[3], q[0]) << RZ(q[3], -PI / 2.0);
	cir5.setDagger(true);
	prog << cir << cir2 /*<< Reset(q[1])*/ << cir3 << cir5 /*<< MeasureAll(q, c)*/;

	const auto src_mat = getCircuitMatrix(prog);
	cout << "src_mat:" << endl << src_mat << endl;

	std::vector<std::pair<QCircuit, QCircuit>> optimitzer_cir;
	sub_cir_optimizer(prog, optimitzer_cir, QCircuitOPtimizerMode::Merge_U3);

	const auto result_mat = getCircuitMatrix(prog);
	cout << "result_mat:" << endl << result_mat << endl;

	cout << "The optimizered QProg:" << endl;
	cout << prog << endl;

	if (src_mat == result_mat)
	{
		cout << "//////////////////////// right,\\\\\\\\\\\\\\\\\\\\\\'" << endl;
	}
	else
	{
		cout << "----------------- wrong-------------" << endl;
	}

	destroyQuantumMachine(qvm);
	return true;
}

TEST(QCircuitOptimizer, test1)
{
	bool test_val = false;
	try
	{
		test_val = test_cir_optimize_fun1();
		//test_val = test_cir_optimize_fun2();
	}
	catch (const std::exception& e)
	{
		cout << "Got a exception: " << e.what() << endl;
	}
	catch (...)
	{
		cout << "Got an unknow exception: " << endl;
	}

	cout << "QCircuitOptimizer test over, press Enter to continue." << endl;
	getchar();

	ASSERT_TRUE(test_val);
}
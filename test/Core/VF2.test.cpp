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
#include "Core/VirtualQuantumProcessor/MPSQVM/MPSQVM.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/GraphMatch.h"
#include "Core/Utilities/Tools/JsonConfigParam.h"
#include "Core/Utilities/Tools/QCircuitGenerator.h"
#include "Core/Utilities/QProgTransform/QCircuitRewrite.h"

using namespace std;
USING_QPANDA

#if 1

void vertices_output(std::shared_ptr<QProgDAG> dag) {
	std::cout << "=========================================" << std::endl;
	std::cout << "Vertice:" << std::endl;
	for (auto i = 0; i < dag->get_vertex().size(); ++i) {
		auto vertice = dag->get_vertex(i);
		std::cout << "\tGate " << i << " GateType: " << vertice.m_type << ", the qubits are ";
		for (int i = 0; i < vertice.m_node->m_qubits_vec.size(); ++i) {
			std::cout << vertice.m_node->m_qubits_vec[i]->get_phy_addr() << " ";
		}
		std::cout << std::endl;
		std::cout << "The Parameter for angle(s) are ";
		for (auto& _angle : vertice.m_node->m_angles) {
			std::cout << _angle << " ";
		}
		std::cout << std::endl;
	}
	std::cout << "=========================================" << std::endl;
}

void edges_output(std::shared_ptr<QProgDAG> dag) {
	std::cout << "========================================" << std::endl;
	std::cout << "Edges:" << std::endl;
	for (auto i = 0; i < dag->get_vertex().size(); ++i) {
		auto vertice = dag->get_vertex(i);
		for (auto& _edge : vertice.m_pre_edges) {
			std::cout << _edge.m_from << "->" << _edge.m_to << ", qubit:" << _edge.m_qubit << std::endl;
		}
		for (auto& _edge : vertice.m_succ_edges) {
			std::cout << _edge.m_from << "->" << _edge.m_to << ", qubit:" << _edge.m_qubit << std::endl;
		}
	}
}

#endif
static bool test_vf2_1()
{
	auto qvm = initQuantumMachine();
	auto q = qvm->qAllocMany(3);
	auto c = qvm->cAllocMany(3);

	auto circuit = CreateEmptyCircuit();
	auto prog = CreateEmptyQProg();

	circuit << CNOT(q[0], q[1]) << CNOT(q[0],q[1]) << H(q[0]);
	prog << circuit << MeasureAll(q, c);

	std::cout << "src_prog: " << prog;
	sub_cir_replace(prog, "F:\\tmpShared\\pattern.json", 1);
	std::cout << "result_prog: " << prog;

	return 1;
}

static bool test_vf2_3() {
	auto qvm = initQuantumMachine();
	auto q = qvm->qAllocMany(3);
	auto c = qvm->cAllocMany(3);

	auto circuit = CreateEmptyCircuit();
	auto prog = CreateEmptyQProg();

	circuit << RZ(q[2], PI / 4) << CNOT(q[0], q[2]) << RZ(q[1], PI / 4) << RZ(q[1], PI / 6) << 
		RZ(q[2], PI / 6) << CNOT(q[0], q[2]) << RZ(q[0], PI / 4) << RZ(q[2], PI / 4) << CNOT(q[2], q[0]);
	prog << circuit << MeasureAll(q, c);

	std::cout << "src_prog: " << prog;
	sub_cir_replace(prog, "F:\\tmpShared\\pattern.json", 1);
	std::cout << "result_prog: " << prog;

	return 1;
}

static bool test_vf2_4() {
	auto qvm = initQuantumMachine(QMachineType::CPU);
	QVec out_qv;
	std::vector<ClassicalCondition> out_cv;
	std::string filename = "F:\\tmpShared\\BIGD\\20QBT_45CYC_.6D1_.2D2_5.qasm";
	std::string outfile = "F:\\tmpShared\\test_out\\" + filename.substr(13);
	outfile = outfile.substr(0, outfile.length() - 5) + ".out";

	ofstream ofs;
	ofs.open(outfile, ios::trunc | ios::out);
	QProg prog = convert_qasm_to_qprog(filename, qvm, out_qv, out_cv);
	
	std::cout << "src_prog: " << prog;
	sub_cir_replace(prog, "F:\\tmpShared\\pattern.json", 1);
	std::cout << "result_prog: " << prog;

	destroyQuantumMachine(qvm);
	ofs.close();
	return 1;
}

static bool test_vf2_1_1() {

	return 1;
} 

static bool test_vf2_4_1() {
	auto qvm = initQuantumMachine(QMachineType::CPU);
	QVec out_qv;
	std::vector<ClassicalCondition> out_cv;
	std::string filename = "F:\\tmpShared\\BIGD\\20QBT_45CYC_.0D1_.1D2_1.qasm";
	std::string outfile = "F:\\tmpShared\\test_out\\" + filename.substr(13);
	outfile = outfile.substr(0, outfile.length() - 5) + ".out";

	ofstream ofs;
	ofs.open(outfile, ios::trunc | ios::out);
	QProg prog = convert_qasm_to_qprog(filename, qvm, out_qv, out_cv);

	std::cout << "src_prog: " << prog;
	sub_cir_replace(prog, "F:\\tmpShared\\pattern.json", 1);
	std::cout << "result_prog: " << prog;

	ofs << convert_qprog_to_qasm(prog, qvm) << std::endl;
	/*auto dag = qprog_to_DAG(prog);
	auto par_list = rewriter.DAGPartition(dag, 2, 0);
	auto subgraph_list = dag->partition(par_list);
	edges_output(subgraph_list[0]);*/
	destroyQuantumMachine(qvm);
	ofs.close();
	return 1;
}

static bool test_vf2_4_2() {
	auto qvm = initQuantumMachine(QMachineType::CPU);
	QVec out_qv;
	std::vector<ClassicalCondition> out_cv;
	std::string filename = "F:\\tmpShared\\BIGD\\20QBT_45CYC_.7D1_.1D2_1.qasm";
	std::string outfile = "F:\\tmpShared\\test_out\\" + filename.substr(13);
	outfile = outfile.substr(0, outfile.length() - 5) + ".out";

	ofstream ofs;
	ofs.open(outfile, ios::trunc | ios::out);
	QProg prog = convert_qasm_to_qprog(filename, qvm, out_qv, out_cv);
	
	std::cout << "src_prog: " << prog;
	sub_cir_replace(prog, "F:\\tmpShared\\pattern.json", 1);
	std::cout << "result_prog: " << prog;
	ofs << convert_qprog_to_qasm(prog, qvm) << std::endl;
	destroyQuantumMachine(qvm);
	ofs.close();
	return 1;
}

static bool test_vf2_1_2() {
	auto qvm = initQuantumMachine(QMachineType::CPU);
	auto q = qvm->qAllocMany(3);
	auto c = qvm->cAllocMany(3);

	auto circuit = CreateEmptyCircuit();
	auto prog = CreateEmptyQProg();

	circuit << X(q[0]) << CNOT(q[2], q[1]) << CNOT(q[2], q[1]) << H(q[0]) << CNOT(q[0], q[2]) << CNOT(q[0], q[2]) << CNOT(q[1], q[2]);
	prog << circuit;

	std::cout << "src_prog: "<< prog;

	/*QCircuitRewrite rewriter;
	auto new_prog = rewriter.circuitRewrite(prog);*/
	sub_cir_replace(prog, "F:\\tmpShared\\pattern.json", 1);
	std::cout << "result_prog: " << prog;

	return true;
}

TEST(VF2, test1)
{
	bool test_val = false;
	try
	{
		test_val = test_vf2_1_2();
		//test_val = test_vf2_4_2();
	}
	catch (const std::exception& e)
	{
		cout << "Got a exception: " << e.what() << endl;
	}
	catch (...)
	{
		cout << "Got an unknow exception: " << endl;
	}

	ASSERT_TRUE(test_val);

	cout << "VF2 test over, press Enter to continue." << endl;
	getchar();
}


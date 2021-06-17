#include "gtest/gtest.h"
#include "QPanda.h"
#include <iostream>
#include <vector>
#include <stdio.h>
#include <memory>
#include <chrono>

#ifndef PI
#define PI 3.1415926
#endif // !PI

USING_QPANDA
using namespace std;
template <class T = CPUQVM>
class QVMInit
{
public:
	QVMInit() : m_qvm(nullptr){
		m_qvm = new(std::nothrow) T;
	}

	~QVMInit() {
		m_qvm->finalize();
		delete m_qvm;
	}

	QVec allocate_qubits(size_t size) { return m_qvm->allocateQubits(size); }
	vector<ClassicalCondition> allocate_class_bits(size_t size) { return m_qvm->allocateCBits(size); }

public:
	QuantumMachine *m_qvm;
};

class CalcFidelity : public TraversalInterface<>
{
public:
	CalcFidelity()
	{
		int qnum = 0;
		JsonConfigParam config;
		const std::string config_data;
		config.load_config(/*config_data*/);
		config.getMetadataConfig(qnum, mCnotReliability);

		//mMeaReliability.resize(qnum);
		//for (int i = 0; i < qnum; i++)
		//{
		//	mMeaReliability[i] = 1.0;
		//}

		auto graph = mCnotReliability;
		for (int i = 0; i < qnum; i++)
		{
			for (int j = 0; j < qnum; j++)
			{
				if (i == j)
					graph[i][j] == 0.0;
				else if (graph[i][j] > 1e-6)
					graph[i][j] = 1.0 - graph[i][j];
				else
					graph[i][j] = DBL_MAX;
			}
		}
		std::vector<std::vector<int>> path(qnum, std::vector<int>(qnum));
		std::vector<std::vector<double>> dist(qnum, std::vector<double>(qnum));

		for (int i = 0; i < qnum; i++)
		{
			for (int j = 0; j < qnum; j++)
			{
				dist[i][j] = graph[i][j];
				path[i][j] = j;
			}
		}

		for (int k = 0; k < qnum; k++)
		{
			for (int i = 0; i < qnum; i++)
			{
				for (int j = 0; j < qnum; j++)
				{
					if ((dist[i][k] + dist[k][j] < dist[i][j])
						&& (dist[i][k] != DBL_MAX)
						&& (dist[k][j] != DBL_MAX)
						&& (i != j))
					{
						dist[i][j] = dist[i][k] + dist[k][j];
						path[i][j] = path[i][k];
					}
				}
			}
		}

		mSwapDist.resize(qnum);
		for (int i = 0; i < qnum; i++)
		{
			mSwapDist[i].resize(qnum);
			for (int j = 0; j < qnum; j++)
			{
				int prev = i;
				double reliability = 1.0;
				int cur = path[i][j];
				while (cur != j)
				{
					reliability *= std::pow(mCnotReliability[prev][cur], 3);
					prev = cur;
					cur = path[cur][j];
				}
				reliability *= std::pow(mCnotReliability[prev][j], 3);

				mSwapDist[i][j] = reliability;
			}
		}
	}
	~CalcFidelity() {};

	template <typename _Ty>
	std::pair<double , int > calc_fidelity(_Ty& node)
	{
		m_fidelity = 1.0;
		m_swap_cnt = 0;
		execute(node.getImplementationPtr(), nullptr);
		return { m_fidelity, m_swap_cnt };
	}

	virtual void execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node)
	{
		auto type = cur_node->getQGate()->getGateType();
		QVec qv;
		cur_node->getQuBitVector(qv);
		switch (type)
		{
			
		case  GateType::CPHASE_GATE:
		case  GateType::CZ_GATE:
		case  GateType::CNOT_GATE:
		{
			auto idx_0 = qv[0]->get_phy_addr();
			auto idx_1 = qv[1]->get_phy_addr();
			m_fidelity *= mCnotReliability[idx_0][idx_1];
		}
		break;
		case  GateType::SWAP_GATE:
		{
			auto idx_0 = qv[0]->get_phy_addr();
			auto idx_1 = qv[1]->get_phy_addr();
			m_fidelity *= mSwapDist[idx_0][idx_1];
			m_swap_cnt++;
		}
		break;
		default:
			break;
		}
	}

	virtual void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node) {}

	virtual void execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node) {}

	virtual void execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node)
	{
		Traversal::traversal(cur_node, *this);
	}

	virtual void execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node)
	{
		Traversal::traversal(cur_node, false, *this);
	}

	virtual void execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node)
	{
		Traversal::traversal(cur_node, *this);
	}

	virtual void execute(std::shared_ptr<AbstractClassicalProg>  cur_node,std::shared_ptr<QNode> parent_node){}

	double m_fidelity;
	std::vector<std::vector<double>> mCnotReliability;
	std::vector<std::vector<double>> mSwapDist;
	int m_swap_cnt;
};

static bool test_opt_BMT_qubit_allocator_1()
{
	auto qvm = new CPUQVM();
	qvm->init();

	std::vector<ClassicalCondition> c;

	//QVec q;
	auto q = qvm->qAllocMany(5);
	//auto c = qvm->cAllocMany(5);
	QProg prog;
	//prog = convert_originir_to_qprog("D:\\test.txt", qvm, q, c);
	prog << QFT(q);
	//prog = random_qprog(1, 8, 20, qvm, q);
	//write_to_originir_file(prog, qvm, "D:\\test.txt");


	qvm->directlyRun(prog);
	auto r_1 = qvm->PMeasure_no_index( q);

	//cout << "srd prog:" << prog << endl;

	// test qubit allocator
	auto old_qv_1 = q;
	auto old_qv_2 = q;

	// 1. bmt
	auto bmt_mapped_prog = OBMT_mapping(prog, qvm, q);

	qvm->directlyRun(bmt_mapped_prog);
	auto r_2 = qvm->PMeasure_no_index(q);
	
	cout << "bmt_mapped_prog:" << bmt_mapped_prog << endl;
	CalcFidelity cf;
	std::cout << "bmt fidelity :  "<< cf.calc_fidelity(bmt_mapped_prog).first << std::endl;
	std::cout << "bmt swap :  " << cf.calc_fidelity(bmt_mapped_prog).second << std::endl;
	
	
	
	// 2. qcodar
	//auto qcodar_mapped_prog = qcodar_match_by_simple_type(prog, old_qv_1, qvm, 2, 4, 10);
	auto qcodar_mapped_prog = qcodar_match_by_config(prog, old_qv_1, qvm, "QPandaConfig.json", 5);

	cout << "qcodar_mapped_prog:" << qcodar_mapped_prog << endl;

	std::cout << "qcodar fidelity :  " << cf.calc_fidelity(qcodar_mapped_prog).first << std::endl;
	std::cout << "qcodar swap :  " << cf.calc_fidelity(qcodar_mapped_prog).second << std::endl;

	// 3. astar
	auto astar_mapped_prog = topology_match(prog, old_qv_2, qvm, SWAP_GATE_METHOD, ORIGIN_VIRTUAL_ARCH);
	cout << "astar_mapped_prog:" << astar_mapped_prog << endl;

	std::cout << "astar fidelity :  " << cf.calc_fidelity(astar_mapped_prog).first << std::endl;
	std::cout << "astar swap :  " << cf.calc_fidelity(astar_mapped_prog).second << std::endl;


	int size = std::min(r_1.size(), r_2.size());
	for (int i = 0; i < size; i++)
	{
		if ((fabs(r_1[i] - r_2[i]) > 1e-6))
		{
			std::cout << r_1[i] << " != " << r_2[i] << "i : " << i << std::endl;
		}
	}

	qvm->finalize();
	delete qvm;
	return true;
}

static bool test_opt_BMT_qubit_allocator_2()
{
	QVMInit<> tmp_qvm;

	//build HHL circuit
	auto machine = tmp_qvm.m_qvm;
	QStat A = {
	qcomplex_t(15.0 / 4.0, 0), qcomplex_t(9.0 / 4.0, 0), qcomplex_t(5.0 / 4.0, 0), qcomplex_t(-3.0 / 4.0, 0),
	qcomplex_t(9.0 / 4.0, 0), qcomplex_t(15.0 / 4.0, 0), qcomplex_t(3.0 / 4.0, 0), qcomplex_t(-5.0 / 4.0, 0),
	qcomplex_t(5.0 / 4.0, 0), qcomplex_t(3.0 / 4.0, 0), qcomplex_t(15.0 / 4.0, 0), qcomplex_t(-9.0 / 4.0, 0),
	qcomplex_t(-3.0 / 4.0, 0), qcomplex_t(-5.0 / 4.0, 0), qcomplex_t(-9.0 / 4.0, 0), qcomplex_t(15.0 / 4.0, 0)
	};
	std::vector<double> b = { 0.5, 0.5, 0.5, 0.5 };
	auto hhl_cir = build_HHL_circuit(A, b, machine);
	QProg hhl_prog(hhl_cir);
	cout << "start decompose_multiple_control_qgate." << endl;
	decompose_multiple_control_qgate(hhl_prog, machine);
	cout << "after decompose_multiple_control_qgate" << endl;

	cout << "start flatten" << endl;
	flatten(hhl_prog);

#define HHL_ORIGINIR_FILE "E:\\hhl_10_qubit.ir"
	cout << "start write_to_originir_file" << endl;
	write_to_originir_file(hhl_prog, machine, HHL_ORIGINIR_FILE);
	cout << "write hhl to file end, press enter to continue." << endl;
	getchar();


	/*auto q = tmp_qvm.allocate_qubits(8);
	auto c = tmp_qvm.allocate_class_bits(8);*/
	QVec q;
	vector<ClassicalCondition> c;
	//QProg hhl_prog = convert_originir_to_qprog(HHL_ORIGINIR_FILE, machine, q, c);

	printf("^^^^^^^^^^^^^^^^^after decompose_multiple_control_qgate the hhl_cir_gate_cnt: %llu ^^^^^^^^^^^^^^^^^^^^\n", getQGateNum(hhl_prog));

	
	cout << "0000000000 press enter to continue." << endl;
	getchar();
	// test qubit allocator
	/*QVec q;
	get_all_used_qubits(hhl_prog, q);*/
	auto bmt_mapped_prog = OBMT_mapping(hhl_prog, machine, q);
	//cout << "bmt_mapped_prog:" << bmt_mapped_prog << endl;
	CalcFidelity cf;
	std::cout << "bmt fidelity :  " << cf.calc_fidelity(bmt_mapped_prog).first << std::endl;
	std::cout << "bmt swap :  " << cf.calc_fidelity(bmt_mapped_prog).second << std::endl;



	// 2. qcodar
	auto start = chrono::system_clock::now();
	//auto qcodar_mapped_prog = qcodar_match_by_simple_type(prog, old_qv_1, qvm, 2, 4, 10);
	auto qcodar_mapped_prog = qcodar_match_by_config(hhl_prog, q, machine, "QPandaConfig.json", 10);
	auto end = chrono::system_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
	cout << "The QCodar takes "
		<< double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den
		<< "seconds" << endl;
	//cout << "qcodar_mapped_prog:" << qcodar_mapped_prog << endl;

	std::cout << "qcodar fidelity :  " << cf.calc_fidelity(qcodar_mapped_prog).first << std::endl;
	std::cout << "qcodar swap :  " << cf.calc_fidelity(qcodar_mapped_prog).second << std::endl;

	return true;
}

static bool test_pressed_layer()
{
	QVMInit<> tmp_qvm;
	auto q = tmp_qvm.allocate_qubits(4);
	auto c = tmp_qvm.allocate_class_bits(4);
	QProg prog;
	prog << S(q[0]) << S(q[1]) << H(q[3]) 
		<< X(q[0]) << CNOT(q[1], q[2])  << S(q[3]) 
		<< H(q[0]) << CZ(q[1], q[2]) << T(q[3])
		<< T(q[0]) << CNOT(q[1], q[3]) 
		<< CNOT(q[1], q[0]) << CZ(q[3], q[2])
		<< S(q[1]) << H(q[2])
		<< CNOT(q[0], q[1]) << CZ(q[2], q[3]) 
		<< CZ(q[0], q[1]) << T(q[2]) << X(q[3]) 
		<< H(q[3]) << CZ(q[0], q[2])
		<< T(q[2]) << H(q[3]) << CZ(q[0], q[1]) 
		<< CZ(q[3], q[1]) << H(q[0]) << X(q[2]);


	PressedTopoSeq pressed_seq_layer = get_pressed_layer(prog);

	return true;
}

static bool test_opt_BMT_qubit_allocator_3()
{
	QVMInit<> tmp_qvm;
	auto machine = tmp_qvm.m_qvm;
	machine->setConfigure({ 128,128 });

#define HHL_ORIGINIR_FILE "E:\\11\\random_100_qubit-1.txt"
	/*auto q = tmp_qvm.allocate_qubits(8);
	auto c = tmp_qvm.allocate_class_bits(8);*/
	QVec q;
	vector<ClassicalCondition> c;
	QProg prog_100qubits = convert_originir_to_qprog(HHL_ORIGINIR_FILE, machine, q, c);

	printf("^^^^^^^^^^^^^^^^^after decompose_multiple_control_qgate the hhl_cir_gate_cnt: %llu ^^^^^^^^^^^^^^^^^^^^\n",
		getQGateNum(prog_100qubits));

	/*write_to_originir_file(hhl_prog, machine, HHL_ORIGINIR_FILE);
	getchar();*/

	cout << "src_prog:" << prog_100qubits << endl;

	// test qubit allocator
	/*QVec q;
	get_all_used_qubits(hhl_prog, q);*/
	auto bmt_mapped_prog = OBMT_mapping(prog_100qubits, machine, q);
	//cout << "bmt_mapped_prog:" << bmt_mapped_prog << endl;
	CalcFidelity cf;
	std::cout << "bmt fidelity :  " << cf.calc_fidelity(bmt_mapped_prog).first << std::endl;
	std::cout << "bmt swap :  " << cf.calc_fidelity(bmt_mapped_prog).second << std::endl;



	// 2. qcodar
	std::cout << "start qcodar >>> " << endl;
	auto start = chrono::system_clock::now();
	//auto qcodar_mapped_prog = qcodar_match_by_simple_type(prog, old_qv_1, qvm, 2, 4, 10);
	auto qcodar_mapped_prog = qcodar_match_by_config(prog_100qubits, q, machine, "QPandaConfig.json", 10);
	auto end = chrono::system_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
	cout << "The QCodar takes "
		<< double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den
		<< "seconds" << endl;
	//cout << "qcodar_mapped_prog:" << qcodar_mapped_prog << endl;

	std::cout << "qcodar fidelity :  " << cf.calc_fidelity(qcodar_mapped_prog).first << std::endl;
	std::cout << "qcodar swap :  " << cf.calc_fidelity(qcodar_mapped_prog).second << std::endl;

	return true;
}

static bool test_opt_BMT_qubit_allocator_4()
{
	QVMInit<> tmp_qvm;
	auto machine = tmp_qvm.m_qvm;
	machine->setConfigure({ 128,128 });

	auto q = tmp_qvm.allocate_qubits(10);
	auto qft_cir = QFT(q);
	cout << "src_prog:" << qft_cir << endl;
	printf("circuit-gate-num = %llu\n", getQGateNum(qft_cir));

	write_to_originir_file(qft_cir, machine, "E:\\11\\qft-10.txt");
	getchar();

	auto bmt_mapped_prog = OBMT_mapping(qft_cir, machine, q);
	//cout << "bmt_mapped_prog:" << bmt_mapped_prog << endl;
	CalcFidelity cf;
	std::cout << "bmt fidelity :  " << cf.calc_fidelity(bmt_mapped_prog).first << std::endl;
	std::cout << "bmt swap :  " << cf.calc_fidelity(bmt_mapped_prog).second << std::endl;


	// 2. qcodar
	std::cout << "start qcodar >>> " << endl;
	auto start = chrono::system_clock::now();
	auto qcodar_mapped_prog = qcodar_match_by_config(qft_cir, q, machine, "QPandaConfig.json", 10);
	auto end = chrono::system_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
	cout << "The QCodar takes "
		<< double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den
		<< "seconds" << endl;
	//cout << "qcodar_mapped_prog:" << qcodar_mapped_prog << endl;

	std::cout << "qcodar fidelity :  " << cf.calc_fidelity(qcodar_mapped_prog).first << std::endl;
	std::cout << "qcodar swap :  " << cf.calc_fidelity(qcodar_mapped_prog).second << std::endl;

	return true;
}

static bool test_SABRE_qubit_mapping_1()
{
	QVMInit<> tmp_qvm;
	auto machine = tmp_qvm.m_qvm;
	machine->setConfigure({ 128,128 });
	auto q = tmp_qvm.allocate_qubits(5);
	QCircuit cir;
	cir << H(q[0]) << CNOT(q[0], q[1]) << H(q[1]) << H(q[1]) << CNOT(q[0], q[2]) << CNOT(q[1], q[2])
		<< CNOT(q[3], q[4]) << CNOT(q[2], q[3]) << CNOT(q[2], q[3]) << H(q[2]) << CNOT(q[2], q[3])
		<< CNOT(q[4], q[1]) << H(q[4]) << CNOT(q[4], q[0]) << CNOT(q[1], q[0]) << H(q[4]) << H(q[0]);

	cout << "srd prog:" << cir << endl;

	/*write_to_originir_file(cir, machine, "E:\\11\\qft-10.txt");
	getchar();*/

	// 1. SABRE
	auto sabre_mapped_prog = SABRE_mapping(cir, machine, q);
	cout << "SABRE_mapped_prog:" << sabre_mapped_prog << endl;
	CalcFidelity cf;
	std::cout << "bmt fidelity :  " << cf.calc_fidelity(sabre_mapped_prog).first << std::endl;
	std::cout << "bmt swap :  " << cf.calc_fidelity(sabre_mapped_prog).second << std::endl;
	return true;
}

static bool test_mapping_overall_1()
{
	QVMInit<> tmp_qvm;
	auto machine = tmp_qvm.m_qvm;
	machine->setConfigure({ 128,128 });

	QVec q;
	vector<ClassicalCondition> c;
	//QProg test_prog = convert_originir_to_qprog("E:\\11\\cir_5_qubit.ir", machine, q, c);
	//QProg test_prog = convert_originir_to_qprog("E:\\11\\cir_8_qubit.ir", machine, q, c);
	//QProg test_prog = convert_originir_to_qprog("E:\\11\\hhl_8_qubit.ir", machine, q, c);
	//QProg test_prog = convert_originir_to_qprog("E:\\11\\qft-10.ir", machine, q, c);
	//QProg test_prog = convert_originir_to_qprog("E:\\11\\hhl_10_qubit.ir", machine, q, c);

	//QProg test_prog = convert_originir_to_qprog("E:\\doc\\Quantum-mapping\\mapping_test\\11\\random_cir_20_qubits.ir", machine, q, c);
	QProg test_prog = convert_originir_to_qprog("E:\\doc\\Quantum-mapping\\mapping_test\\11\\random_cir_20_qubits_cx.ir", machine, q, c);
	//QProg test_prog = convert_originir_to_qprog("E:\\11\\random_cir_20_qubits_2.ir", machine, q, c);
	//QProg test_prog = convert_originir_to_qprog("E:\\11\\random_cir_20_qubits_3.ir", machine, q, c);

	//QProg test_prog = convert_originir_to_qprog("E:\\doc\\Quantum-mapping\\mapping_test\\11\\random_cir_50_qubits_1.ir", machine, q, c);
	//QProg test_prog = convert_originir_to_qprog("E:\\doc\\Quantum-mapping\\mapping_test\\11\\random_cir_50_qubits_2.ir", machine, q, c);
	
	//QProg test_prog = convert_originir_to_qprog("E:\\doc\\Quantum-mapping\\mapping_test\\11\\random_100_qubit-0.ir", machine, q, c);
	//QProg test_prog = convert_originir_to_qprog("E:\\11\\random_100_qubit-1.ir", machine, q, c);
	//QProg test_prog = convert_originir_to_qprog("E:\\11\\random_cir_100_qubits_1.ir", machine, q, c);
	//QProg test_prog = convert_originir_to_qprog("E:\\doc\\Quantum-mapping\\mapping_test\\11\\random_cir_100_qubits_2.ir", machine, q, c);
	/*cout << test_prog << endl;
	write_to_qasm_file(test_prog, machine, "E:\\tmp\\random_20_qubit_1_cx.qasm");*/

	// 1. SABRE
	std::cout << "-------------------- start SABRE >>> " << endl;
	auto start = chrono::system_clock::now();
	auto sabre_mapped_prog = SABRE_mapping(test_prog, machine, q, 20, 10);
	auto end = chrono::system_clock::now();
	//cout << "SABRE_mapped_prog:" << sabre_mapped_prog << endl;
	CalcFidelity cf;
	auto layer_info = prog_layer(sabre_mapped_prog);
	cout << "SABRE_mapped_prog deeps = " << layer_info.size() << endl;
	std::cout << "SABRE fidelity :  " << cf.calc_fidelity(sabre_mapped_prog).first << std::endl;
	std::cout << "SABRE swap :  " << cf.calc_fidelity(sabre_mapped_prog).second << std::endl;
	auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
	cout << "The SABRE takes "
		<< double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den
		<< "seconds" << endl;
	std::cout << " <<<< ------------- SABRE END -------------------------- \n" << endl;

	// 2. opt-bmt
	std::cout << "--------------------  start opt-bmt >>> " << endl;
	start = chrono::system_clock::now();
	auto bmt_mapped_prog = OBMT_mapping(test_prog, machine, q, 200);
	end = chrono::system_clock::now();
	//cout << "opt-bmt mapped_prog:" << bmt_mapped_prog << endl;
	layer_info = prog_layer(bmt_mapped_prog);
	cout << "opt-bmt mapped_prog deeps = " << layer_info.size() << endl;
	std::cout << "opt-bmt fidelity :  " << cf.calc_fidelity(bmt_mapped_prog).first << std::endl;
	std::cout << "opt-bmt swap :  " << cf.calc_fidelity(bmt_mapped_prog).second << std::endl;
	duration = chrono::duration_cast<chrono::microseconds>(end - start);
	cout << "The opt-bmt takes "
		<< double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den
		<< "seconds" << endl;
	std::cout << " <<<< ------------- opt-bmt END -------------------------- \n" << endl;

	// 3. QCodar
	std::cout << "-------------------- start QCodar >>> " << endl;
	start = chrono::system_clock::now();
	//auto qcodar_mapped_prog = qcodar_match_by_config(test_prog, q, machine, CONFIG_PATH, 10);
	auto qcodar_mapped_prog = qcodar_match_by_simple_type(test_prog, q, machine, 2, 10, 10);
	end = chrono::system_clock::now();
	duration = chrono::duration_cast<chrono::microseconds>(end - start);
	//cout << "qcodar_mapped_prog:" << qcodar_mapped_prog << endl;
	layer_info = prog_layer(qcodar_mapped_prog);
	cout << "qcodar_mapped_prog deeps = " << layer_info.size() << endl;
	std::cout << "QCodar fidelity :  " << cf.calc_fidelity(qcodar_mapped_prog).first << std::endl;
	std::cout << "QCodar swap :  " << cf.calc_fidelity(qcodar_mapped_prog).second << std::endl;
	cout << "The QCodar takes "
		<< double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den
		<< "seconds" << endl;
	std::cout << " <<<< ------------- QCodar END -------------------------- \n" << endl;

	// 4. A-star
	std::cout << "--------------------  start A-star >>> " << endl;
	start = chrono::system_clock::now();
	auto astar_mapped_prog = topology_match(test_prog, q, machine, SWAP_GATE_METHOD, ORIGIN_VIRTUAL_ARCH, CONFIG_PATH);
	end = chrono::system_clock::now();
	//cout << "A-star mapped_prog:" << astar_mapped_prog << endl;
	layer_info = prog_layer(astar_mapped_prog);
	cout << "astar_mapped_prog deeps = " << layer_info.size() << endl;
	std::cout << "A-star fidelity :  " << cf.calc_fidelity(astar_mapped_prog).first << std::endl;
	std::cout << "A-star swap :  " << cf.calc_fidelity(astar_mapped_prog).second << std::endl;
	duration = chrono::duration_cast<chrono::microseconds>(end - start);
	cout << "The A-star takes "
		<< double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den
		<< "seconds" << endl;
	std::cout << " <<<< ------------- A-star END -------------------------- \n" << endl;

	return true;
}

static void generate_random_cir()
{
	QVMInit<> tmp_qvm;
	auto machine = tmp_qvm.m_qvm;
	machine->setConfigure({ 128,128 });
	QVec q = machine->allocateQubits(8);
	auto random_cir = random_qcircuit(q, 20, {"RX", "RY", "RZ", "CZ", "H", "X"});
	cout << "random_prog:" << random_cir << endl;
	write_to_originir_file(random_cir, machine, "E:\\tmp\\random_cir_4_20_0.ir");
	cout << "write originir to file end, press enter to continue." << endl;
	getchar();
}

TEST(QubitMapping, test1)
{
	bool test_val = false;
	try
	{
		//generate_random_cir();
		//test_val = test_BMT_qubit_allocator_2();

		/* test for OPT-BMT */
		//test_val = test_opt_BMT_qubit_allocator_1();
		//test_val = test_opt_BMT_qubit_allocator_2();
		//test_val = test_opt_BMT_qubit_allocator_3();

		//test_val = test_opt_BMT_qubit_allocator_4();

		//test_val = test_pressed_layer();

		//test_val = test_SABRE_qubit_mapping_1();

		test_val = test_mapping_overall_1();
	}
	catch (const std::exception& e)
	{
		cout << "Got a exception: " << e.what() << endl;
	}
	catch (...)
	{
		cout << "Got an unknow exception: " << endl;
	}

	cout << "BMTQubitAllocator test over, press Enter to continue." << endl;
	getchar();

	ASSERT_TRUE(test_val);
}
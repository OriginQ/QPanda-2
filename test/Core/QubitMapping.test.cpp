#include "gtest/gtest.h"
#include "QPanda.h"
#include <iostream>
#include <vector>
#include <stdio.h>
#include <memory>
#include <chrono>
#include <string>
#include "Extensions/Extensions.h"

#ifdef USE_EXTENSION

#ifndef PI
#define PI 3.1415926
#endif // !PI

const size_t kShots = 2048;
const size_t kEpsion = kShots * 0.05;
#define CHECK_TIME 1
#define CHECK_SWAP 1

const std::string test_IR_1 = R"(QINIT 6 
CREG 4
H q[0]
X q[1]
X q[5]
CNOT q[0],q[1]
CNOT q[4],q[5]
CNOT q[0],q[2]
RZ q[1],(1.16)
RZ q[4],(0.785)
RZ q[5],(2.78539816)
RZ q[0],(1.9816)
RY q[2],(1.112)
CNOT q[3],q[5]
CR q[2],q[4],(0.567)
MEASURE q[1],c[0]
MEASURE q[2],c[1]
MEASURE q[3],c[2]
MEASURE q[4],c[3]
)";

const std::string test_IR_2 = R"(QINIT 4 
CREG 1
H q[0]
X q[1]
X q[2]
X q[3]
CNOT q[3],q[1]
MEASURE q[1],c[0]
)";

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
	
	//cout << "bmt_mapped_prog:" << bmt_mapped_prog << endl;
	CalcFidelity cf;
	//std::cout << "bmt fidelity :  "<< cf.calc_fidelity(bmt_mapped_prog).first << std::endl;
	//std::cout << "bmt swap :  " << cf.calc_fidelity(bmt_mapped_prog).second << std::endl;
	
	if (cf.calc_fidelity(bmt_mapped_prog).first != 0.0203972 && cf.calc_fidelity(bmt_mapped_prog).second != 6)
		return false;

#ifdef qcodar
	// 2. qcodar
	//auto qcodar_mapped_prog = qcodar_match_by_simple_type(prog, old_qv_1, qvm, 2, 4, 10);
	auto qcodar_mapped_prog = qcodar_match_by_config(prog, old_qv_1, qvm, "QPandaConfig.json", 5);

	cout << "qcodar_mapped_prog:" << qcodar_mapped_prog << endl;

	std::cout << "qcodar fidelity :  " << cf.calc_fidelity(qcodar_mapped_prog).first << std::endl;
	std::cout << "qcodar swap :  " << cf.calc_fidelity(qcodar_mapped_prog).second << std::endl;

	// 3. astar
	auto astar_mapped_prog = topology_match(prog, old_qv_2, qvm);
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

#endif // qcodar

	qvm->finalize();
	delete qvm;
	return true;
}

static bool test_opt_BMT_qubit_allocator_3()
{
	QVMInit<> tmp_qvm;
	auto machine = tmp_qvm.m_qvm;
	machine->setConfigure({ 128,128 });

//#define HHL_ORIGINIR_FILE "E:\\11\\random_100_qubit-1.txt"
	/*auto q = tmp_qvm.allocate_qubits(8);
	auto c = tmp_qvm.allocate_class_bits(8);*/
	QVec q;
	vector<ClassicalCondition> c;
	QProg prog_100qubits = convert_originir_string_to_qprog(test_IR_1, machine, q, c);

	/*printf("^^^^^^^^^^^^^^^^^after decompose_multiple_control_qgate the hhl_cir_gate_cnt: %llu ^^^^^^^^^^^^^^^^^^^^\n",
		getQGateNum(prog_100qubits));*/

	/*write_to_originir_file(hhl_prog, machine, HHL_ORIGINIR_FILE);
	getchar();*/

	//cout << "src_prog:" << prog_100qubits << endl;

	// test qubit allocator
	/*QVec q;
	get_all_used_qubits(hhl_prog, q);*/
	auto bmt_mapped_prog = OBMT_mapping(prog_100qubits, machine, q);
	//cout << "bmt_mapped_prog:" << bmt_mapped_prog << endl;
	CalcFidelity cf;
	//std::cout << "bmt fidelity :  " << cf.calc_fidelity(bmt_mapped_prog).first << std::endl;
	//std::cout << "bmt swap :  " << cf.calc_fidelity(bmt_mapped_prog).second << std::endl;
	if (cf.calc_fidelity(bmt_mapped_prog).first != 0.5832 && cf.calc_fidelity(bmt_mapped_prog).second != 0)
		return false;

#ifdef qcodar

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

#endif // qcodar

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

	//cout << "srd prog:" << cir << endl;

	// 1. SABRE
	auto sabre_mapped_prog = SABRE_mapping(cir, machine, q);
	//cout << "SABRE_mapped_prog:" << sabre_mapped_prog << endl;
	CalcFidelity cf;
	//std::cout << "bmt fidelity :  " << cf.calc_fidelity(sabre_mapped_prog).first << std::endl;
	//std::cout << "bmt swap :  " << cf.calc_fidelity(sabre_mapped_prog).second << std::endl;
	if (cf.calc_fidelity(sabre_mapped_prog).first != 0.098411 && cf.calc_fidelity(sabre_mapped_prog).second != 2)
		return false;
	return true;
}

static bool test_mapping_overall_1(const std::string& ir_str)
{
	QVMInit<> tmp_qvm;
	auto machine = tmp_qvm.m_qvm;
	machine->setConfigure({ 128,128 });

	QVec q;
	vector<ClassicalCondition> c;
	QProg test_prog = convert_originir_string_to_qprog(ir_str, machine, q, c);

	// Get correct result
	const std::map<string, size_t> correct_result = machine->runWithConfiguration(test_prog, c, kShots);

	// 1. SABRE
	auto start = chrono::system_clock::now();
	auto _prog = deepCopy(test_prog);
	auto sabre_mapped_prog = SABRE_mapping(_prog, machine, q, 20, 10);
	auto end = chrono::system_clock::now();

	// check Fidelity
	if (CHECK_TIME)
	{
		CalcFidelity cf;
		std::cout << "SABRE fidelity :  " << cf.calc_fidelity(sabre_mapped_prog).first << std::endl;
		std::cout << "SABRE swap :  " << cf.calc_fidelity(sabre_mapped_prog).second << std::endl;
	}
	
	// check circuit-deep
	if (CHECK_TIME)
	{
		//cout << "SABRE_mapped_prog:" << sabre_mapped_prog << endl;
		auto layer_info = prog_layer(sabre_mapped_prog);
		cout << "SABRE_mapped_prog deeps = " << layer_info.size() << endl;
		auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
		cout << "The SABRE takes "
			<< double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den
			<< "seconds" << endl;
		std::cout << " <<<< ------------- SABRE END -------------------------- \n" << endl;
	}
	
	// check result
	{
		std::map<string, size_t> _result = machine->runWithConfiguration(sabre_mapped_prog, c, kShots);
		for (const auto& i : _result)
		{
            if (abs((long)i.second < kEpsion))
                continue;
			const long _a = i.second - correct_result.at(i.first);
			if (std::abs(_a) > kEpsion){
				return false;
			}
		}
	}
	
	// 2. opt-bmt
	//std::cout << "--------------------  start opt-bmt >>> " << endl;
	start = chrono::system_clock::now();
	_prog = deepCopy(test_prog);
	auto bmt_mapped_prog = OBMT_mapping(_prog, machine, q, 200);
	end = chrono::system_clock::now();
	// check Fidelity
	if (CHECK_TIME)
	{
		CalcFidelity cf;
		std::cout << "opt-bmt fidelity :  " << cf.calc_fidelity(bmt_mapped_prog).first << std::endl;
		std::cout << "opt-bmt swap :  " << cf.calc_fidelity(bmt_mapped_prog).second << std::endl;
	}

	// check circuit-deep
	if (CHECK_TIME)
	{
		//cout << "opt-bmt mapped_prog:" << bmt_mapped_prog << endl;
		auto layer_info = prog_layer(bmt_mapped_prog);
		//cout << "opt-bmt mapped_prog deeps = " << layer_info.size() << endl;
		auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
		cout << "The opt-bmt takes "
			<< double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den
			<< "seconds" << endl;
		std::cout << " <<<< ------------- opt-bmt END -------------------------- \n" << endl;
	}

	// check result
	{
		auto _result = machine->runWithConfiguration(bmt_mapped_prog, c, kShots);
		for (const auto& i : _result)
		{
            if (abs((long)i.second < kEpsion))
                continue;
			if (abs((long)i.second - (long)correct_result.at(i.first)) > kEpsion) {
				return false;
			}
		}
	}

#ifdef QCoadr
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

#endif // QCoadr

	// 4. A-star
	start = chrono::system_clock::now();
	_prog = deepCopy(test_prog);
	auto astar_mapped_prog = topology_match(_prog, q, machine, CONFIG_PATH);
	end = chrono::system_clock::now();

	// check Fidelity
	if (CHECK_TIME)
	{
		CalcFidelity cf;
		std::cout << "A-star fidelity :  " << cf.calc_fidelity(astar_mapped_prog).first << std::endl;
		std::cout << "A-star swap :  " << cf.calc_fidelity(astar_mapped_prog).second << std::endl;
	}

	// check circuit-deep
	if (CHECK_TIME)
	{
		//cout << "A-star mapped_prog:" << astar_mapped_prog << endl;
		auto layer_info = prog_layer(astar_mapped_prog);
		cout << "astar_mapped_prog deeps = " << layer_info.size() << endl;
		auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
		cout << "The A-star takes "
			<< double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den
			<< "seconds" << endl;
		std::cout << " <<<< ------------- A-star END -------------------------- \n" << endl;
	}

	// check result
	{
		auto _result = machine->runWithConfiguration(bmt_mapped_prog, c, kShots);
		for (const auto& i : _result)
		{
            if (abs((long)i.second < kEpsion))
                continue;
			if (abs((long)i.second - (long)correct_result.at(i.first)) > kEpsion) {
				return false;
			}
		}
	}

	return true;
}


TEST(QubitMapping, test1)
{
	bool test_val = false;
	try
	{
		test_val = test_mapping_overall_1(test_IR_1);
		test_val = test_val && test_mapping_overall_1(test_IR_2);
		/*test_val = test_val && test_opt_BMT_qubit_allocator_3();
		test_val = test_val && test_SABRE_qubit_mapping_1();
		test_val = test_val && test_opt_BMT_qubit_allocator_1();*/
	}
	catch (const std::exception& e)
	{
		std::cout << "Got a exception: " << e.what() << endl;
		test_val = false;
	}
	catch (...)
	{
		std::cout << "Got an unknow exception: " << endl;
		test_val = false;
	}

	ASSERT_TRUE(test_val);
}

#endif

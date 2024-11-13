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

const size_t kShots = 10000;
const size_t kEpsion = kShots * 0.07;
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
CREG 4
H q[1]
H q[2]
CONTROL q[1]
RX q[3],(-6.283186)
ENDCONTROL
DAGGER
H q[1]
ENDDAGGER
CONTROL q[2]
RX q[3],(-3.141593)
ENDCONTROL
DAGGER
CR q[1],q[2],(1.570796)
ENDDAGGER
DAGGER
H q[2]
ENDDAGGER
BARRIER q[2],q[3],q[1],q[0]
X q[2]
CONTROL q[1],q[2]
RY q[0],(3.141593)
ENDCONTROL
X q[2]
CONTROL q[1],q[2]
RY q[0],(-3.141593)
ENDCONTROL
BARRIER q[1],q[3],q[2],q[0]
H q[2]
CR q[1],q[2],(1.570796)
H q[1]
DAGGER
CONTROL q[2]
RX q[3],(-3.141593)
ENDCONTROL
ENDDAGGER
DAGGER
CONTROL q[1]
RX q[3],(-6.283186)
ENDCONTROL
ENDDAGGER
DAGGER
H q[1]
ENDDAGGER
DAGGER
H q[2]
ENDDAGGER
MEASURE q[0],c[0]
MEASURE q[1],c[1]
MEASURE q[2],c[2]
)";

const std::string test_IR_3 = R"(QINIT 1
CREG 1
H q[0]
X q[0]
MEASURE q[0],c[0]
)";

const std::string test_IR_4 = R"(QINIT 6
CREG 6
CONTROL q[1]
X q[0]
ENDCONTROL
MEASURE q[1],c[0]
MEASURE q[2],c[1]
MEASURE q[3],c[2]
MEASURE q[4],c[3]
)";


const std::string test_IR_5 = R"(QINIT 5
CREG 5
RY q[0],(-2.352396)
Z1 q[1]
RY q[2],(-2.743066)
RZ q[3],(-6.283185)
RZ q[0],(6.283185)
RY q[1],(-1.570796)
RZ q[2],(-3.141593)
RY q[3],(-2.739645)
RZ q[1],(3.181396)
CNOT q[0],q[1]
RZ q[1],(1.475277)
X1 q[1]
CNOT q[1],q[0]
RX q[0],(0.000000)
RY q[1],(1.475277)
CNOT q[0],q[1]
RX q[0],(-1.570796)
RY q[1],(-1.610599)
RZ q[0],(3.923193)
RZ q[1],(-3.141593)
RY q[0],(-1.570796)
RZ q[1],(-1.570796)
RZ q[0],(-1.570796)
RY q[1],(-1.570796)
RZ q[1],(4.516512)
CNOT q[1],q[2]
RZ q[2],(1.389924)
X1 q[2]
CNOT q[2],q[1]
RX q[1],(0.000000)
RY q[2],(0.000538)
CNOT q[1],q[2]
RX q[1],(-1.570796)
RY q[2],(-1.570796)
RY q[1],(-1.495197)
RZ q[2],(-4.712389)
RZ q[1],(-6.283185)
RZ q[2],(-3.141593)
RY q[2],(-0.791455)
RZ q[2],(-1.570796)
X1 q[2]
CNOT q[2],q[3]
RX q[2],(-1.201981)
RY q[3],(-1.145132)
CNOT q[2],q[3]
RX q[2],(-1.570796)
RY q[3],(-0.867419)
Z1 q[2]
RZ q[3],(-6.283185)
RY q[2],(-1.570796)
RY q[3],(-1.570796)
RZ q[3],(-3.141593)
CONTROL q[0]
RY q[4],(0.418879)
ENDCONTROL
CONTROL q[1]
RY q[4],(0.837758)
ENDCONTROL
CONTROL q[2]
RY q[4],(1.675516)
ENDCONTROL
CONTROL q[3]
RY q[4],(3.351032)
ENDCONTROL
MEASURE q[4],c[4]
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

    double m_fidelity{0};
	std::vector<std::vector<double>> mCnotReliability;
	std::vector<std::vector<double>> mSwapDist;
    int m_swap_cnt{0};
};

static uint64_t get_current_time()
{
	const std::chrono::system_clock::duration duration_since_epoch
		= std::chrono::system_clock::now().time_since_epoch(); // 从1970-01-01 00:00:00到当前时间点的时长
	return std::chrono::duration_cast<std::chrono::milliseconds>(duration_since_epoch).count(); // 将时长转换为微秒数
}

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
	cir << H(q[0]) << CNOT(q[0], q[1])
		<< H(q[1]) << H(q[1]) << CNOT(q[0], q[2]) << CNOT(q[1], q[2])
		<< CNOT(q[3], q[4]) << CNOT(q[2], q[3]) << CNOT(q[2], q[3]) << H(q[2]) << CNOT(q[2], q[3])
		<< CNOT(q[4], q[1]) << H(q[4]) << CNOT(q[4], q[0]) << CNOT(q[1], q[0]) << H(q[4]) << H(q[0]);

	cout << "srd prog:" << cir << endl;

	// 1. SABRE
	auto sabre_mapped_prog = SABRE_mapping(cir, machine, q);
	cout << "SABRE_mapped_prog:" << sabre_mapped_prog << endl;
	CalcFidelity cf;
	//std::cout << "bmt fidelity :  " << cf.calc_fidelity(sabre_mapped_prog).first << std::endl;
	//std::cout << "bmt swap :  " << cf.calc_fidelity(sabre_mapped_prog).second << std::endl;
	if (cf.calc_fidelity(sabre_mapped_prog).first != 0.098411 && cf.calc_fidelity(sabre_mapped_prog).second != 2)
		return false;
    return true;
}

static bool test_SABRE_qubit_mapping_2()
{
	QVMInit<> tmp_qvm;
	auto machine = tmp_qvm.m_qvm;
	machine->setConfigure({ 128,128 });
	auto q = tmp_qvm.allocate_qubits(73);

	// 1. SABRE
	ifstream infile("/home/bylz/workspace/pilotos/build/Release/bin/Config/50_qubits_10_depth.txt");
	stringstream ss;
	ss << infile.rdbuf();
	std::string originir = ss.str();
	//std::string originir = "QINIT 6\nCREG 6\nCNOT q[0],q[1]\nCNOT q[1],q[2]\nCNOT q[2],q[3]\nCNOT q[3],q[0]\nMEASURE q[0],c[0]\nMEASURE q[1],c[1]\nMEASURE q[2],c[2]\nMEASURE q[3],c[3]";
	auto prog = convert_originir_string_to_qprog(originir, machine);

    const std::string config_data = "/home/bylz/workspace/pilotos/build/Release/bin/Config/ChipArchConfig_D72.json";

    QMappingConfig mapping_config = QMappingConfig(config_data);
	auto mapping_result = select_best_qubits_blocks<QPanda::SabreQAllocator, QPanda::QMappingConfig>(prog, machine, mapping_config, 5, 10, 10000, 9);

	auto sabre_mapped_prog = SABRE_mapping(prog, machine, q, mapping_result, 5, 10, mapping_config, 10000, 9);
	auto dest_originir = convert_qprog_to_originir(sabre_mapped_prog, machine);

    return true;
}

static bool test_find_mapped_blocks()
{
    const std::string config_data = CONFIG_PATH;
    QMappingConfig mapping_config = QMappingConfig(config_data);

	QVMInit<> tmp_qvm;
	auto machine = tmp_qvm.m_qvm;
	machine->setConfigure({ 128,128 });
	//auto q = tmp_qvm.allocate_qubits(5);
	std::string originir = "QINIT 6\nCREG 6\nH q[0]\nCNOT q[0],q[1]\nMEASURE q[0],c[0]\nMEASURE q[1],c[1]";
	QVec q;
	vector<ClassicalCondition> c;
	QProg prog = convert_originir_string_to_qprog(originir, machine, q, c);

	auto mapping_result = select_best_qubits_blocks<QPanda::SabreQAllocator, QPanda::QMappingConfig>(prog, machine, mapping_config, 20, 0, 100, 2);

	return true;
}

static bool test_sabre_init_mapping()
{

    const std::string config_data = CONFIG_PATH;
    QMappingConfig mapping_config = QMappingConfig(config_data);
    QVec qv;
    std::vector<uint32_t> init_map;
    uint32_t max_look_ahead = 20;
    uint32_t max_iterations = 10;

    //return std::make_pair(ret_prog, init_map);

    QVMInit<> tmp_qvm;
    auto machine = tmp_qvm.m_qvm;
    machine->setConfigure({128, 128});
    QProg prog = convert_originir_to_qprog("C://work//QPanda//qpanda_dev_240428//testtemp//ir.txt", machine);
    //cout << "srd prog:" << prog << endl;

	auto mapping_result = select_best_qubits_blocks<QPanda::SabreQAllocator, QPanda::QMappingConfig>(prog, machine, mapping_config, 20, 0, 100);

    // 1. SABRE
    auto sabre_mapped_prog = SABRE_mapping(prog, machine, qv, init_map, mapping_result, max_look_ahead, max_iterations, mapping_config);
    //cout << "SABRE_mapped_prog:" << sabre_mapped_prog << endl;
    cout << "init mapping: [ ";
    for (auto & q : init_map)
    {
        std::cout << q << "\t";
    }
    std::cout << " ]" << endl;
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
		cout << "SABRE_mapped_prog:" << sabre_mapped_prog << endl;
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
            if (std::abs((double)i.second) < kEpsion)
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
            if (std::abs((long)i.second) < kEpsion)
                continue;
            if (std::abs((long)i.second - (long)correct_result.at(i.first)) > kEpsion) {
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
            if (std::abs((long)i.second) < kEpsion)
                continue;
            if (std::abs((long)i.second - (long)correct_result.at(i.first)) > kEpsion) {
				return false;
			}
		}
	}

	return true;
}

enum class MappingMode
{
	A_STAR,
	BMT,
	SABRE,
};

bool check_result(const std::map<string, size_t>& result1,
	const std::map<string, size_t>& result2, int shots)
{
	for (const auto& val : result1)
	{
		std::string bit_str = val.first;
		if (result2.find(bit_str) == result2.end()) {
			return false;
		}
		auto res_1 = val.second / (double)shots;
		auto res_2 = result2.at(bit_str) / (double)shots;
		if (abs(res_1 - res_2) > 1e-4) {
			return false;
		}
	}

	return true;
}

bool test_mapping(CPUQVM *qvm, QProg src_prog, QVec qv, int shot, MappingMode mode)
{
	decompose_multiple_control_qgate(src_prog, qvm, CONFIG_PATH, false);
	transform_to_base_qgate_withinarg(src_prog, qvm, { {"U3"}, {"CNOT"} });

	auto src_result = qvm->runWithConfiguration(src_prog, shot);
	auto src_cnot_num = count_qgate_num(src_prog);

	QProg out_prog;
	std::string str_tmp = "";
	auto start = chrono::system_clock::now();
	switch (mode)
	{
	case MappingMode::A_STAR:
	{
		out_prog = topology_match(src_prog, qv, qvm, CONFIG_PATH);
		str_tmp = "a star ";
	}
		break;
	case MappingMode::BMT:
	{
		out_prog = OBMT_mapping(src_prog, qvm, qv);

		str_tmp = "bmt ";
	}
		break;
	case MappingMode::SABRE:
	{
		out_prog = SABRE_mapping(src_prog, qvm, qv);
		str_tmp = "sabre ";
	}
		break;
	default:
		break;
	}
	auto end = chrono::system_clock::now();

	auto out_result = qvm->runWithConfiguration(out_prog, shot);
	transform_to_base_qgate_withinarg(out_prog, qvm, { {"U3"}, {"CNOT"} });
	auto out_cnot_num = count_qgate_num(out_prog);

	bool ret = check_result(src_result, out_result, shot);
	if (!ret) {
		std::cout << "topology_match fail\n";
		return false;
	}

	auto used_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << str_tmp << " used milliseconds : " << used_time << std::endl;

	auto increase_cout_num = out_cnot_num - src_cnot_num;
	std::cout << str_tmp<< " increase  " << increase_cout_num << " cnots" << std::endl;
	return true;
}


static bool test_mapping_overall_fix(const std::string& ir_str)
{
    int shot = 10000;
    QVMInit<> tmp_qvm;
    auto machine = tmp_qvm.m_qvm;
    machine->setConfigure({ 128,128 });
    int n = 5;
    QVec q = machine->qAllocMany(n);
    QVec q1;
    vector<ClassicalCondition> c = machine->cAllocMany(n);

    QProg test_prog;

    // 利用originIR
    //test_prog << random_qcircuit(q, 50) <<Measure(q[0],c[0]) << Measure(q[1],c[1]) << Measure(q[2], c[2]);
    //auto prog_str = convert_qprog_to_originir(test_prog, machine);
    //std::cout << prog_str << std::endl;
    test_prog = convert_originir_string_to_qprog(ir_str, machine, q, c);

    //std::cout << "test_prog:" << test_prog << endl;
    decompose_multiple_control_qgate(test_prog, machine, CONFIG_PATH, false);
    transform_to_base_qgate_withinarg(test_prog, machine, { {"U3"}, {"CNOT"} });

    std::cout << "=====================================" << endl;

    auto in_cnot_num = count_qgate_num(test_prog);

    // Get correct result
    auto _prog = deepCopy(test_prog);
    const std::map<string, size_t> correct_result = machine->runWithConfiguration(_prog, shot);
    std::cout << "===========correct_result============" << std::endl;
    for (auto &val : correct_result)
    {
        std::cout << val.first << ": " << val.second << std::endl;
    }
    std::cout << "===========correct_end============" << std::endl;
    //1. SABRE
    std::cout << "===============start SABRE ===========>>> " << endl;
    auto _prog1 = deepCopy(test_prog);
    auto start2 = chrono::system_clock::now();
    auto sabre_mapped_prog = SABRE_mapping(_prog1, machine, q, 20, 10);
    auto end2 = chrono::system_clock::now();
    auto duration2 = chrono::duration_cast<chrono::microseconds>(end2 - start2);
    std::cout << "The SABRE takes "
        << double(duration2.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den
        << "seconds" << endl;
    auto SABRE_out_result = machine->runWithConfiguration(sabre_mapped_prog, shot);
    std::cout << "===========SABRE_result===============" << std::endl;
    for (auto &val : SABRE_out_result)
    {
        std::cout << val.first << ": " << val.second << std::endl;
    }
    for (const auto& i : SABRE_out_result)
    {
        if (std::abs((long)i.second) < kEpsion)
            continue;
        const long _a = i.second - correct_result.at(i.first);
        if (std::abs(_a) > kEpsion) {
            return false;
        }
    }
    std::cout << "===============SABRE end===========>>> " << endl;

    //// 2. opt-bmt
    std::cout << "--------------------  start opt-bmt >>> " << endl;

    auto _prog2 = deepCopy(test_prog);
    QVec used_qv;
    _prog2.get_used_qubits(used_qv);
    auto start1 = chrono::system_clock::now();
    auto bmt_mapped_prog = OBMT_mapping(_prog2, machine, used_qv, 200, 20, 10, QPanda::QMappingConfig(CONFIG_PATH));
    auto end1 = chrono::system_clock::now();
    auto duration1 = chrono::duration_cast<chrono::microseconds>(end1 - start1);
    std::cout << "The opt-bmt takes "
        << double(duration1.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den
        << "seconds" << endl;
    auto BMT_out_result = machine->runWithConfiguration(bmt_mapped_prog, shot);
    std::cout << "===========BMT_result===============" << std::endl;
    for (auto &val : BMT_out_result)
    {
        std::cout << val.first << ": " << val.second << std::endl;
    }
    for (const auto& i : BMT_out_result)
    {
        if (std::abs((long)i.second) < kEpsion)
            continue;
        const long _a = i.second - correct_result.at(i.first);
        if (std::abs(_a) > kEpsion) {
            return false;
        }
    }
    std::cout << "===============BMT end===========>>> " << endl;
    //3. A-star
    std::cout << "--------------------  start A-star >>> " << endl;
    auto _prog3 = deepCopy(test_prog);
    QVec used_qv1;
    _prog3.get_used_qubits(used_qv1);
    auto start = chrono::system_clock::now();
    auto astar_mapped_prog = topology_match(_prog3, used_qv1, machine, CONFIG_PATH);
    auto end = chrono::system_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    std::cout << "The A-star takes "
        << double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den
        << "seconds" << endl;
    auto astar_out_result = machine->runWithConfiguration(astar_mapped_prog, shot);
    std::cout << "===========astar_result===============" << std::endl;
    for (auto &val : astar_out_result)
    {
        std::cout << val.first << ": " << val.second << std::endl;
    }
    for (const auto& i : astar_out_result)
    {
        if (std::abs((long)i.second) < kEpsion)
            continue;
        const long _a = i.second - correct_result.at(i.first);
        if (std::abs(_a) > kEpsion) {
            return false;
        }
    }
    std::cout << "===========astar_end===============" << std::endl;

    return true;
}
static bool test_mapping_sabre_fix(const std::string& ir_str)
{
    int shot = 10000;
    QVMInit<> tmp_qvm;
    auto machine = tmp_qvm.m_qvm;
    machine->setConfigure({ 128,128 });
    int n = 5;
    QVec q = machine->qAllocMany(n);
    QVec q1;
    vector<ClassicalCondition> c = machine->cAllocMany(n);

    QProg test_prog;

    // 利用originIR
    //test_prog << random_qcircuit(q, 10) << Measure(q[0], c[0]) << Measure(q[1], c[1]) << Measure(q[2], c[2]);
    test_prog = convert_originir_string_to_qprog(ir_str, machine, q, c);
    decompose_multiple_control_qgate(test_prog, machine, CONFIG_PATH, false);
    transform_to_base_qgate_withinarg(test_prog, machine, { {"U3"}, {"CNOT"} });
    auto _prog = deepCopy(test_prog);
    const std::map<string, size_t> correct_result = machine->runWithConfiguration(_prog, shot);
    std::cout << "===========correct_result============" << std::endl;
    for (auto &val : correct_result)
    {
        std::cout << val.first << ": " << val.second << std::endl;
    }
    std::cout << "===========correct_end============" << std::endl;
    //1. SABRE
    std::cout << "===============start SABRE ===========>>> " << endl;
    auto _prog1 = deepCopy(test_prog);
    auto start2 = chrono::system_clock::now();
    auto sabre_mapped_prog = SABRE_mapping(_prog1, machine, q, 20, 10);
    std::cout << sabre_mapped_prog << std::endl;
    auto end2 = chrono::system_clock::now();
    auto duration2 = chrono::duration_cast<chrono::microseconds>(end2 - start2);
    std::cout << "The SABRE takes "
        << double(duration2.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den
        << "seconds" << endl;
    auto SABRE_out_result = machine->runWithConfiguration(sabre_mapped_prog, shot);
    std::cout << "===========SABRE_result===============" << std::endl;
    for (auto &val : SABRE_out_result)
    {
        std::cout << val.first << ": " << val.second << std::endl;
    }
    for (const auto& i : SABRE_out_result)
    {
        if (std::abs((long)i.second) < kEpsion)
            continue;
        const long _a = i.second - correct_result.at(i.first);
        if (std::abs(_a) > kEpsion) {
            return false;
        }
    }
    std::cout << "===============SABRE end===========>>> " << endl;
    return true;
}

static bool test_mapping_obmt_fix(const std::string& ir_str)
{
    int shot = 10000;
    QVMInit<> tmp_qvm;
    auto machine = tmp_qvm.m_qvm;
    machine->setConfigure({ 128,128 });
    int n = 5;
    QVec q = machine->qAllocMany(n);
    QVec q1;
    vector<ClassicalCondition> c = machine->cAllocMany(n);

    QProg test_prog;

    // 利用originIR
    //test_prog << random_qcircuit(q, 10) << Measure(q[0], c[0]) << Measure(q[1], c[1]) << Measure(q[2], c[2]);
    test_prog = convert_originir_string_to_qprog(ir_str, machine, q, c);
    decompose_multiple_control_qgate(test_prog, machine, CONFIG_PATH, false);
    transform_to_base_qgate_withinarg(test_prog, machine, { {"U3"}, {"CNOT"} });
    auto _prog = deepCopy(test_prog);
    const std::map<string, size_t> correct_result = machine->runWithConfiguration(_prog, shot);
    std::cout << "===========correct_result============" << std::endl;
    for (auto &val : correct_result)
    {
        std::cout << val.first << ": " << val.second << std::endl;
    }
    std::cout << "===========correct_end============" << std::endl;
    //1. SABRE
    std::cout << "--------------------  start opt-bmt >>> " << endl;

    auto _prog2 = deepCopy(test_prog);
    auto start1 = chrono::system_clock::now();
    auto bmt_mapped_prog = OBMT_mapping(_prog2, machine, q);
    auto prog_str = convert_qprog_to_originir(bmt_mapped_prog, machine);
    std::cout << prog_str << std::endl;
    auto end1 = chrono::system_clock::now();
    auto duration1 = chrono::duration_cast<chrono::microseconds>(end1 - start1);
    std::cout << "The opt-bmt takes "
        << double(duration1.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den
        << "seconds" << endl;
    auto BMT_out_result = machine->runWithConfiguration(bmt_mapped_prog, shot);
    std::cout << "===========BMT_result===============" << std::endl;
    for (auto &val : BMT_out_result)
    {
        std::cout << val.first << ": " << val.second << std::endl;
    }
    for (const auto& i : BMT_out_result)
    {
        if (std::abs((long)i.second) < kEpsion)
            continue;
        const long _a = i.second - correct_result.at(i.first);
        if (std::abs(_a) > kEpsion) {
            return false;
        }
    }
    std::cout << "===============BMT end===========>>> " << endl;
    return true;
}

static bool test_mapping_a_star_fix(const std::string& ir_str)
{
    int shot = 10000;
    QVMInit<> tmp_qvm;
    auto machine = tmp_qvm.m_qvm;
    machine->setConfigure({ 128,128 });
    int n = 5;
    QVec q = machine->qAllocMany(n);
    QVec q1;
    vector<ClassicalCondition> c = machine->cAllocMany(n);

    QProg test_prog;

    // 利用originIR
    //test_prog << random_qcircuit(q, 10) << Measure(q[0], c[0]) << Measure(q[1], c[1]) << Measure(q[2], c[2]);
    test_prog = convert_originir_string_to_qprog(ir_str, machine, q, c);
    decompose_multiple_control_qgate(test_prog, machine, CONFIG_PATH, false);
    transform_to_base_qgate_withinarg(test_prog, machine, { {"U3"}, {"CNOT"} });
    auto _prog = deepCopy(test_prog);
    const std::map<string, size_t> correct_result = machine->runWithConfiguration(_prog, shot);
    std::cout << "===========correct_result============" << std::endl;
    for (auto &val : correct_result)
    {
        std::cout << val.first << ": " << val.second << std::endl;
    }
    std::cout << "===========correct_end============" << std::endl;
    //1. SABRE
    std::cout << "--------------------  start A-star >>> " << endl;
    auto _prog3 = deepCopy(test_prog);
    auto start = chrono::system_clock::now();
    auto astar_mapped_prog = topology_match(_prog3, q, machine, CONFIG_PATH);
    auto end = chrono::system_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    std::cout << "The A-star takes "
        << double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den
        << "seconds" << endl;
    auto astar_out_result = machine->runWithConfiguration(astar_mapped_prog, shot);
    std::cout << "===========astar_result===============" << std::endl;
    for (auto &val : astar_out_result)
    {
        std::cout << val.first << ": " << val.second << std::endl;
    }
    for (const auto& i : astar_out_result)
    {
        if (std::abs((long)i.second) < kEpsion)
            continue;
        const long _a = i.second - correct_result.at(i.first);
        if (std::abs(_a) > kEpsion) {
            return false;
        }
    }
    std::cout << "===========astar_end===============" << std::endl;
    return true;
}

TEST(QubitMapping, test2)
{
	int shot = 5000;

	int qnum = 4;
	auto qvm = new CPUQVM();
	qvm->init();
	auto qv = qvm->qAllocMany(qnum);
	auto cv = qvm->cAllocMany(qnum);

	QProg src_prog;
	src_prog << random_qcircuit(qv, 50);
	
	MappingMode mode = MappingMode::SABRE;

	auto ret = test_mapping(qvm, src_prog, qv, shot, mode);

	qvm->finalize();
	delete qvm;
	ASSERT_TRUE(ret);
}


bool test_mapping_arg(CPUQVM *qvm, QProg src_prog, QVec qv, int shot, MappingMode mode, QMappingConfig config_data)
{
    decompose_multiple_control_qgate(src_prog, qvm, CONFIG_PATH, false);
    transform_to_base_qgate_withinarg(src_prog, qvm, { {"U3"}, {"CNOT"} });

    auto src_result = qvm->runWithConfiguration(src_prog, shot);
    auto src_cnot_num = count_qgate_num(src_prog);

    QProg out_prog;
    std::string str_tmp = "";
    auto start = chrono::system_clock::now();
    switch (mode)
    {
    case MappingMode::A_STAR:
    {
        out_prog = topology_match(src_prog, qv, qvm, CONFIG_PATH);
        str_tmp = "a star ";
    }
    break;
    case MappingMode::BMT:
    {
        out_prog = OBMT_mapping(src_prog, qvm, qv);

        str_tmp = "bmt ";
    }
    break;
    case MappingMode::SABRE:
    {
        out_prog = SABRE_mapping(src_prog, qvm, qv, 20, 10, config_data);
        str_tmp = "sabre ";
    }
    break;
    default:
        break;
    }
    auto end = chrono::system_clock::now();

    auto out_result = qvm->runWithConfiguration(out_prog, shot);
    transform_to_base_qgate_withinarg(out_prog, qvm, { {"U3"}, {"CNOT"} });
    auto out_cnot_num = count_qgate_num(out_prog);

    bool ret = check_result(src_result, out_result, shot);
    if (!ret) {
        std::cout << "topology_match fail\n";
        return false;
    }

    auto used_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << str_tmp << " used milliseconds : " << used_time << std::endl;

    auto increase_cout_num = out_cnot_num - src_cnot_num;
    std::cout << str_tmp << " increase  " << increase_cout_num << " cnots" << std::endl;
    return true;
}


TEST(QubitMapping, arg_test)
{
    int shot = 5000;

    int qnum = 4;
    auto qvm = new CPUQVM();
    qvm->init();
    auto qv = qvm->qAllocMany(qnum);
    auto cv = qvm->cAllocMany(qnum);

    QProg src_prog;
    src_prog << random_qcircuit(qv, 20);

    MappingMode mode = MappingMode::SABRE;

    //1.default config file
    auto mapping_prog_1 = SABRE_mapping(src_prog, qvm, qv, 20, 10);

    //2.config file
    auto mapping_prog_2 = SABRE_mapping(src_prog, qvm, qv, 20, 10, QPanda::QMappingConfig("QPandaConfig.json"));

    //3.std::map
    std::map<size_t, Qnum> mapping;
    mapping[0] = { 1 };
    mapping[1] = { 0, 2 };
    mapping[2] = { 1, 3 };
    mapping[3] = { 2 };

    auto mapping_prog_3 = SABRE_mapping(src_prog, qvm, qv, 20, 10, mapping);

    //4.Eigen::MatrixXd
    Eigen::MatrixXd arch_graph = Eigen::MatrixXd::Zero(4, 4);

    for (int i = 0; i < 3; ++i)
    {
        arch_graph(i, i + 1) = 1;
        arch_graph(i + 1, i) = 1;
    }

    auto mapping_prog_4 = SABRE_mapping(src_prog, qvm, qv, 20, 10, arch_graph);

    //5.prob_vec
    prob_vec mapping_vector(arch_graph.data(), arch_graph.data() + arch_graph.size());

    auto mapping_prog_5 = SABRE_mapping(src_prog, qvm, qv, 20, 10, mapping_vector);

    std::cout << mapping_prog_1  << std::endl;
    std::cout << mapping_prog_2  << std::endl;
    std::cout << mapping_prog_3  << std::endl;
    std::cout << mapping_prog_4  << std::endl;
    std::cout << mapping_prog_5  << std::endl;

    //auto ret = test_mapping(qvm, src_prog, qv, shot, mode);

    qvm->finalize();
    delete qvm;
    //ASSERT_TRUE(ret);
}


TEST(QubitMapping, test1)
{
	bool test_val = true;
	try
	{
		for (size_t i = 0; i < 10; ++i)
		{
            //test_val = test_val && test_mapping_obmt_fix(test_IR_5);
            /*test_val = test_val && test_mapping_a_star_fix(test_IR_2);
            test_val = test_val && test_mapping_sabre_fix(test_IR_2);
            test_val = test_val && test_mapping_overall_fix(test_IR_2);*/
            //test_val = test_val && test_mapping_overall_1(test_IR_3)
			//test_val = test_val && test_mapping_overall_fix(test_IR_2);
			//test_val = test_val && test_mapping_overall_1(test_IR_2);
			//test_val = test_val && test_mapping_overall_1(test_IR_3);
			//test_val = test_val && test_opt_BMT_qubit_allocator_3();
			//test_val = test_val && test_SABRE_qubit_mapping_1();
			test_val = test_val && test_SABRE_qubit_mapping_2();
			//test_val = test_val && test_opt_BMT_qubit_allocator_1();
			//test_val = test_val && test_find_mapped_blocks();

			if (!test_val){
				break;
			}
		}
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

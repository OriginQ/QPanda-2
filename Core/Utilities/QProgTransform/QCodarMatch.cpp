#include "Core/Utilities/QProgTransform/QCodarMatch.h"
#include "Core/Utilities/QProgInfo/QuantumMetadata.h"
#include "Core/QuantumCircuit/QuantumGate.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"

USING_QPANDA
using namespace std;
using namespace QGATE_SPACE;

static QGate iSWAPGateNotheta(Qubit * targitBit_fisrt, Qubit * targitBit_second)
{
	return iSWAP(targitBit_fisrt, targitBit_second);
}

QCodarMatch::QCodarMatch(QuantumMachine * machine, QProg prog, QCodarGridDevice arch_type, int m, int n, const std::string config_data/* = CONFIG_PATH*/)
	:m_qvm(machine), m_arch_type(arch_type), m_config_data(config_data)
{
	size_t qubits = machine->getAllocateQubit();

	if (qubits < 1)
	{
		QCERR("ERROR, Too few qubits to be mapped!");
		throw runtime_error("ERROR, Too few qubits to be mapped!");
	}

	m_gatetype.insert(pair<int, string>(GateType::PAULI_X_GATE, "X"));
	m_gatetype.insert(pair<int, string>(GateType::PAULI_Y_GATE, "Y"));
	m_gatetype.insert(pair<int, string>(GateType::PAULI_Z_GATE, "Z"));

	m_gatetype.insert(pair<int, string>(GateType::X_HALF_PI, "X1"));
	m_gatetype.insert(pair<int, string>(GateType::Y_HALF_PI, "Y1"));
	m_gatetype.insert(pair<int, string>(GateType::Z_HALF_PI, "Z1"));
	m_gatetype.insert(pair<int, string>(GateType::I_GATE, "I"));
	m_gatetype.insert(pair<int, string>(GateType::HADAMARD_GATE, "H"));
	m_gatetype.insert(pair<int, string>(GateType::T_GATE, "T"));
	m_gatetype.insert(pair<int, string>(GateType::S_GATE, "S"));
	m_gatetype.insert(pair<int, string>(GateType::BARRIER_GATE, "BARRIER"));

	m_gatetype.insert(pair<int, string>(GateType::RX_GATE, "RX"));
	m_gatetype.insert(pair<int, string>(GateType::RY_GATE, "RY"));
	m_gatetype.insert(pair<int, string>(GateType::RZ_GATE, "RZ"));
	m_gatetype.insert(pair<int, string>(GateType::U1_GATE, "U1"));

	m_gatetype.insert(pair<int, string>(GateType::U2_GATE, "U2"));
	m_gatetype.insert(pair<int, string>(GateType::U3_GATE, "U3"));
	m_gatetype.insert(pair<int, string>(GateType::U4_GATE, "U4"));

	m_gatetype.insert(pair<int, string>(GateType::CU_GATE, "CU"));
	m_gatetype.insert(pair<int, string>(GateType::CNOT_GATE, "CNOT"));
	m_gatetype.insert(pair<int, string>(GateType::CZ_GATE, "CZ"));
	m_gatetype.insert(pair<int, string>(GateType::CPHASE_GATE, "CR"));
	m_gatetype.insert(pair<int, string>(GateType::ISWAP_GATE, "ISWAP"));
	m_gatetype.insert(pair<int, string>(GateType::SWAP_GATE, "SWAP"));
	m_gatetype.insert(pair<int, string>(GateType::SQISWAP_GATE, "SQISWAP"));
	m_gatetype.insert(pair<int, string>(GateType::ISWAP_THETA_GATE, "SQISWAP"));


	m_single_gate_func.insert(make_pair(GateType::PAULI_X_GATE, X));
	m_single_gate_func.insert(make_pair(GateType::PAULI_Y_GATE, Y));
	m_single_gate_func.insert(make_pair(GateType::PAULI_Z_GATE, Z));
	m_single_gate_func.insert(make_pair(GateType::X_HALF_PI, X1));
	m_single_gate_func.insert(make_pair(GateType::Y_HALF_PI, Y1));
	m_single_gate_func.insert(make_pair(GateType::Z_HALF_PI, Z1));
	m_single_gate_func.insert(make_pair(GateType::I_GATE, I));

	m_single_gate_func.insert(make_pair(GateType::HADAMARD_GATE, H));
	m_single_gate_func.insert(make_pair(GateType::T_GATE, T));
	m_single_gate_func.insert(make_pair(GateType::S_GATE, S));

	m_single_angle_gate_func.insert(make_pair(GateType::RX_GATE, RX));
	m_single_angle_gate_func.insert(make_pair(GateType::RY_GATE, RY));
	m_single_angle_gate_func.insert(make_pair(GateType::RZ_GATE, RZ));
	m_single_angle_gate_func.insert(make_pair(GateType::U1_GATE, U1));

	m_double_gate_func.insert(make_pair(GateType::SWAP_GATE, SWAP));
	m_double_gate_func.insert(make_pair(GateType::CNOT_GATE, CNOT));
	m_double_gate_func.insert(make_pair(GateType::CZ_GATE, CZ));
	m_double_gate_func.insert(make_pair(GateType::ISWAP_GATE, iSWAPGateNotheta));
	m_double_gate_func.insert(make_pair(GateType::SQISWAP_GATE, SqiSWAP));
	m_double_angle_gate_func.insert(make_pair(GateType::CPHASE_GATE, CR));


	m_logic_qubits_apply.resize(qubits, 0);

	traversalQProgParsingInfo(&prog);


	initGridDevice(arch_type, m, n);
	if (qubits > m*n)
	{
		QCERR("ERROR before mapping: more logical qubits than physical ones!");
		throw runtime_error("ERROR before mapping: more logical qubits than physical ones!");
	}

	initScheduler(arch_type, qubits);
}

QCodarMatch::~QCodarMatch()
{
	if (m_device != nullptr)
	{
		delete m_device;
		m_device = nullptr;
	}
	if (m_scheduler != nullptr)
	{
		delete m_device;
		m_device = nullptr;
	}
}

void QCodarMatch::initScheduler(QCodarGridDevice arch_type, size_t qubits)
{
	if (m_device == nullptr)
	{
		QCERR("m_device is null!");
		throw runtime_error("m_device is null!");
	}
	m_scheduler = new QScheduler(m_device);
	m_scheduler->loadCommutingTable();
	if (arch_type == GOOGLE_Q54)
	{
		int ijlist[] = {
				5, 5,		4, 5,		4, 4,		5, 4,		6, 4,		6, 5,		6, 6,		5, 6,		4, 6,		3, 6,
				3, 5,		3, 4,		3, 3,		4, 3,		5, 3,		6, 3,		7, 3,		7, 4,		7, 5,		7, 6,
				7, 7,		6, 7,		5, 7,		4, 7,		4, 8,		5, 8,		6, 8,		7, 8,		2, 6,		2, 5,
				2, 4,		2, 3,		2, 2,		3, 2,		4, 2,		5, 2,		6, 2,		8, 4,		8, 5,		8, 6,
				8, 7,		5, 9,		6, 9,		3, 1,		4, 1,		5, 1,		1, 3,		1, 4,		1, 5,		0, 4,
				4, 0,		9, 5,		9, 6
		};
		srand(time(nullptr));
		int *random_map = new int[qubits];
		for (int i = 0; i < qubits; i++)
		{
			random_map[i] = i;
		}
		for (int i = qubits - 1; i > 0; i--)
		{
			std::swap(random_map[i], random_map[rand() % i]);
		}

		for (int q = 0; q < qubits; q++)
		{
			int qq = random_map[q];
			m_scheduler->addLogicalQubit(ijlist[2 * qq], ijlist[2 * qq + 1]);
		}
	}
	else
	{
		bool is_order = true;
		if (arch_type == ORIGIN_VIRTUAL)
		{
			int logic_qubits_use_count = 0;
			for (auto val : m_logic_qubits_apply)
			{
				if (val != 0)
					logic_qubits_use_count++;
			}
			int valid_physics_qubits = 0;
			for (auto val : m_physics_qubit_fidelity)
			{
				if (val > 1e-15)
				{
					valid_physics_qubits++;
				}
			}

			if (valid_physics_qubits != m_physics_qubit_fidelity.size())
				is_order = false;


			if (logic_qubits_use_count > valid_physics_qubits)
			{
				QCERR("ERROR before mapping: more logical qubits than effective physical ones!");
				throw runtime_error("ERROR before mapping: more logical qubits than effective physical ones!");
			}

			m_scheduler->setQubitFidelity(m_double_gate_apply, m_physics_qubit_fidelity, m_qubit_error);
		}
		m_scheduler->addLogicalQubits(qubits, is_order);
	}
}

#define QPAIR(Q1, Q2)   lines.emplace_back(Q1, Q2);
#define QPAIRS(Q1, Q2)  QPAIR(Q1, Q2) QPAIR(Q2, Q1)

void QCodarMatch::initGridDevice(QCodarGridDevice arch_type, int &m, int &n)
{
	std::vector<std::pair<int, int>> lines;

	switch (arch_type)
	{
	case QCodarGridDevice::IBM_Q20_TOKYO:
	{
		m = 4;
		n = 5;
		QPAIR(0, 1);    QPAIR(0, 5);
		QPAIR(1, 0);    QPAIR(1, 2);    QPAIR(1, 6);    QPAIR(1, 7);
		QPAIR(2, 1);    QPAIR(2, 3);    QPAIR(2, 6);    QPAIR(2, 7);
		QPAIR(3, 2);    QPAIR(3, 4);    QPAIR(3, 8);    QPAIR(3, 9);
		QPAIR(4, 3);    QPAIR(4, 8);    QPAIR(4, 9);
		QPAIR(5, 0);    QPAIR(5, 6);    QPAIR(5, 10);   QPAIR(5, 11);
		QPAIR(6, 1);    QPAIR(6, 2);    QPAIR(6, 5);    QPAIR(6, 7);    QPAIR(6, 10);   QPAIR(6, 11);
		QPAIR(7, 1);    QPAIR(7, 2);    QPAIR(7, 6);    QPAIR(7, 8);    QPAIR(7, 12);   QPAIR(7, 13);
		QPAIR(8, 3);    QPAIR(8, 4);    QPAIR(8, 7);    QPAIR(8, 9);    QPAIR(8, 12);   QPAIR(8, 13);
		QPAIR(9, 3);    QPAIR(9, 4);    QPAIR(9, 8);    QPAIR(9, 14);
		QPAIR(10, 5);   QPAIR(10, 6);   QPAIR(10, 11);  QPAIR(10, 15);
		QPAIR(11, 5);   QPAIR(11, 6);   QPAIR(11, 10);  QPAIR(11, 12);  QPAIR(11, 16);  QPAIR(11, 17);
		QPAIR(12, 7);   QPAIR(12, 8);   QPAIR(12, 11);  QPAIR(12, 13);  QPAIR(12, 16);  QPAIR(12, 17);
		QPAIR(13, 7);   QPAIR(13, 8);   QPAIR(13, 12);  QPAIR(13, 14);  QPAIR(13, 18);  QPAIR(13, 19);
		QPAIR(14, 9);   QPAIR(14, 13);  QPAIR(14, 18);  QPAIR(14, 19);
		QPAIR(15, 10);  QPAIR(15, 16);
		QPAIR(16, 11);  QPAIR(16, 12);  QPAIR(16, 15);  QPAIR(16, 17);
		QPAIR(17, 11);  QPAIR(17, 12);  QPAIR(17, 16);  QPAIR(17, 18);
		QPAIR(18, 13);  QPAIR(18, 14);  QPAIR(18, 17);  QPAIR(18, 19);
		QPAIR(19, 13);  QPAIR(19, 14);  QPAIR(19, 18);
		m_device = new ExGridDevice(m, n, lines);
	}
	break;
	case QCodarGridDevice::IBM_Q53:
	{
		m = 1;
		n = 53;
		QPAIRS(0, 1);    QPAIRS(1, 2);    QPAIRS(2, 3);    QPAIRS(3, 4);
		QPAIRS(0, 5);    QPAIRS(5, 9);    QPAIRS(4, 6);    QPAIRS(6, 13);
		for (int ii = 7; ii < 15; ii++)
		{
			QPAIRS(ii, ii + 1);
		}

		QPAIRS(7, 16);   QPAIRS(16, 19);
		QPAIRS(11, 17);  QPAIRS(17, 23);
		QPAIRS(15, 18);  QPAIRS(18, 27);
		for (int ii = 19; ii < 27; ii++)
		{
			QPAIRS(ii, ii + 1);
		}

		QPAIRS(21, 28);  QPAIRS(28, 32);
		QPAIRS(25, 29);  QPAIRS(29, 36);
		for (int ii = 30; ii < 38; ii++)
		{
			QPAIRS(ii, ii + 1);
		}

		QPAIRS(30, 39);  QPAIRS(39, 42);
		QPAIRS(34, 40);  QPAIRS(40, 46);
		QPAIRS(38, 41);  QPAIRS(41, 50);
		for (int ii = 42; ii < 50; ii++)
		{
			QPAIRS(ii, ii + 1);
		}

		QPAIRS(44, 51);  QPAIRS(48, 52);
		m_device = new ExGridDevice(m, n, lines);
	}
	break;
	case QCodarGridDevice::GOOGLE_Q54:
	{
		m = 10;
		n = 10;
		const bool available_qubits[] =
		{
				0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
				0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
				0, 0, 1, 1, 1, 1, 1, 0, 0, 0,
				0, 1, 1, 1, 1, 1, 1, 0, 0, 0,
				1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
				0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
				0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
				0, 0, 0, 1, 1, 1, 1, 1, 1, 0,
				0, 0, 0, 0, 1, 1, 1, 1, 0, 0,
				0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
		};

		m_device = new UncompletedGridDevice(m, n, available_qubits);
	}
	break;
	case QCodarGridDevice::SIMPLE_TYPE:
	{
		if (m <= 0 || n <= 0)
		{
			QCERR("m or n error!");
			throw runtime_error("m or n error");
		}
		m_device = new SimpleGridDevice(m, n);
	}
	break;
	case QCodarGridDevice::ORIGIN_VIRTUAL:
	{
		if (m_config_data.empty())
		{
			QCERR_AND_THROW(runtime_error, "Error: failed to initGridDevice, the config data is empty.");
		}

		std::vector<std::vector<double>> qubit_matrix;
		int qubit_num = 0;
		JsonConfigParam config;
		config.load_config(m_config_data);
		config.getMetadataConfig(qubit_num, qubit_matrix);
		m = 1;
		n = qubit_num;
		m_physics_qubit_fidelity.resize(qubit_num, 0);
		m_qubit_error.resize(qubit_matrix.size());
		for (int i = 0; i < qubit_matrix.size(); i++)
		{
			m_qubit_error[i].resize(qubit_matrix[i].size());
			for (int j = 0; j < qubit_matrix[i].size(); j++)
			{
				if (qubit_matrix[i][j] > 1e-6)
				{
					QPAIR(i, j);
					m_physics_qubit_fidelity[i] += qubit_matrix[i][j];
				}
				m_qubit_error[i][j] = (1 - qubit_matrix[i][j]);
			}
		}
		m_device = new ExGridDevice(m, n, lines);
	}
	break;
	default:
	{
		QCERR("QCodarGridDevice invalid type");
		throw runtime_error("QCodarGridDevice invalid type");
	}
	break;
	}
}

void QCodarMatch::traversalQProgParsingInfo(QProg *prog)
{
	if (nullptr == prog)
	{
		QCERR("p_prog is null");
		throw runtime_error("p_prog is null");
		return;
	}

	bool isDagger = false;
	execute(prog->getImplementationPtr(), nullptr, isDagger);
}

void QCodarMatch::execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node, bool &is_dagger)
{
	QVec qgate_ctrl_qubits;
	cur_node->getControlVector(qgate_ctrl_qubits);
	auto type = cur_node->getQGate()->getGateType();
	if (type != GateType::BARRIER_GATE && !qgate_ctrl_qubits.empty())
	{
		QCERR("control qubits in qgate are not supported!");
		throw invalid_argument("control qubits in qgate are not supported!");
	}

	GateInfo g;
	QVec qv;
	cur_node->getQuBitVector(qv);

	g.type = type;
	g.is_dagger = cur_node->isDagger() ^ is_dagger;
	auto iter = m_gatetype.find(g.type);
	if (iter == m_gatetype.end())
	{
		QCERR("invalid gate type.");
		throw invalid_argument("invalid gate type.");
	}
	g.gate_name = iter->second;
	g.barrier_id = -1;
	switch (type)
	{
	case GateType::PAULI_X_GATE:
	case GateType::PAULI_Y_GATE:
	case GateType::PAULI_Z_GATE:
	case GateType::X_HALF_PI:
	case GateType::Y_HALF_PI:
	case GateType::Z_HALF_PI:
	case GateType::HADAMARD_GATE:
	case GateType::T_GATE:
	case GateType::S_GATE:
	{
		g.control = -1;
		g.target = qv[0]->getPhysicalQubitPtr()->getQubitAddr();
	}
	break;
	case GateType::BARRIER_GATE:
	{
		g.control = -1;
		g.target = qv[0]->getPhysicalQubitPtr()->getQubitAddr();
		g.barrier_id = m_transform_barrier_id;
		for (auto iter : qgate_ctrl_qubits)
		{
			auto temp_g = g;
			temp_g.target = iter->get_phy_addr();
			m_original_gates.push_back(temp_g);
		}
		m_transform_barrier_id++;
	}
	break;
	case GateType::RX_GATE:
	case GateType::RY_GATE:
	case GateType::RZ_GATE:
	case GateType::U1_GATE:
	{
		g.control = -1;
		g.target = qv[0]->getPhysicalQubitPtr()->getQubitAddr();

		auto gate_parameter = dynamic_cast<AbstractSingleAngleParameter*>(cur_node->getQGate());
		double angle = gate_parameter->getParameter();
		g.param.push_back(angle);

	}
	break;
	case GateType::CNOT_GATE:
	case GateType::CZ_GATE:
	case GateType::ISWAP_GATE:
	case GateType::SWAP_GATE:
	case GateType::SQISWAP_GATE:
	{
		g.control = qv[0]->getPhysicalQubitPtr()->getQubitAddr();
		g.target = qv[1]->getPhysicalQubitPtr()->getQubitAddr();
	}
	break;
	case  GateType::ISWAP_THETA_GATE:
	{
		g.control = qv[0]->getPhysicalQubitPtr()->getQubitAddr();
		g.target = qv[1]->getPhysicalQubitPtr()->getQubitAddr();
		auto gate_parameter = dynamic_cast<AbstractSingleAngleParameter*>(cur_node->getQGate());
		double theta = gate_parameter->getParameter();
		g.param.push_back(theta);
	}
	break;
	case GateType::CPHASE_GATE:
	{
		g.control = qv[0]->getPhysicalQubitPtr()->getQubitAddr();
		g.target = qv[1]->getPhysicalQubitPtr()->getQubitAddr();

		auto gate_parameter = dynamic_cast<AbstractSingleAngleParameter*>(cur_node->getQGate());
		double angle = gate_parameter->getParameter();
		g.param.push_back(angle);
	}
	break;
	case GateType::U2_GATE:
	{
		g.control = -1;
		g.target = qv[0]->getPhysicalQubitPtr()->getQubitAddr();
		QGATE_SPACE::U2 *u2_gate = dynamic_cast<QGATE_SPACE::U2*>(cur_node->getQGate());
		double phi = u2_gate->get_phi();
		double lam = u2_gate->get_lambda();

		g.param.push_back(phi);
		g.param.push_back(lam);

	}
	break;
	case GateType::U3_GATE:
	{
		g.control = -1;
		g.target = qv[0]->getPhysicalQubitPtr()->getQubitAddr();
		QGATE_SPACE::U3 *u3_gate = dynamic_cast<QGATE_SPACE::U3*>(cur_node->getQGate());
		double theta = u3_gate->get_theta();
		double phi = u3_gate->get_phi();
		double lam = u3_gate->get_lambda();
		g.param.push_back(theta);
		g.param.push_back(phi);
		g.param.push_back(lam);
	}
	break;
	case GateType::U4_GATE:
	{
		g.control = -1;
		g.target = qv[0]->getPhysicalQubitPtr()->getQubitAddr();

		auto angle = dynamic_cast<AbstractAngleParameter *>(cur_node->getQGate());

		double alpha = angle->getAlpha();
		double beta = angle->getBeta();
		double gamma = angle->getGamma();
		double delta = angle->getDelta();
		g.param.push_back(alpha);
		g.param.push_back(beta);
		g.param.push_back(gamma);
		g.param.push_back(delta);
	}
	break;
	case GateType::CU_GATE:
	{
		g.control = qv[0]->getPhysicalQubitPtr()->getQubitAddr();
		g.target = qv[1]->getPhysicalQubitPtr()->getQubitAddr();

		auto angle = dynamic_cast<AbstractAngleParameter *>(cur_node->getQGate());
		double alpha = angle->getAlpha();
		double beta = angle->getBeta();
		double gamma = angle->getGamma();
		double delta = angle->getDelta();

		g.param.push_back(alpha);
		g.param.push_back(beta);
		g.param.push_back(gamma);
		g.param.push_back(delta);
	}
	break;
	default:
	{
		QCERR("error! unsupported QGate");
		throw invalid_argument("error! unsupported QGate");
	}
	break;
	}
	m_logic_qubits_apply[g.target] += 1;
	m_double_gate_apply[g.target] += 0;

	if (g.control != -1)
	{
		m_logic_qubits_apply[g.control] += 1;
		m_double_gate_apply[g.control] += 1;
		m_double_gate_apply[g.target] += 1;
	}
	m_original_gates.push_back(g);
}

void QCodarMatch::execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node, bool &is_dagger)
{
	Traversal::traversal(cur_node, *this, is_dagger);
}

void QCodarMatch::execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node, bool &is_dagger)
{
	QVec circuit_ctrl_qubits;
	cur_node->getControlVector(circuit_ctrl_qubits);
	if (!circuit_ctrl_qubits.empty())
	{
		QCERR("control qubits in circuit are not supported!");
		throw invalid_argument("control qubits in circuit are not supported!");
	}
	bool bDagger = cur_node->isDagger() ^ is_dagger;

	auto pNode = std::dynamic_pointer_cast<QNode>(cur_node);
	if (nullptr == pNode)
	{
		QCERR("Unknown internal error");
		throw std::runtime_error("Unknown internal error");
	}

	if (bDagger)
	{
		auto aiter = cur_node->getLastNodeIter();
		if (nullptr == *aiter)
			return;

		while (aiter != cur_node->getHeadNodeIter())
		{
			if (aiter == nullptr)
				break;

			Traversal::traversalByType(*aiter, pNode, *this, bDagger);
			--aiter;
		}
	}
	else
	{
		auto aiter = cur_node->getFirstNodeIter();
		while (aiter != cur_node->getEndNodeIter())
		{
			auto next = aiter.getNextIter();
			Traversal::traversalByType(*aiter, pNode, *this, bDagger);
			aiter = next;
		}
	}
}

void QCodarMatch::execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, bool &is_dagger)
{
	QCERR("transform error, there shouldn't be quantum measure node here.");
	throw invalid_argument("transform error, there shouldn't be quantum measure node here.");
}

void QCodarMatch::execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node, bool &is_dagger)
{
	QCERR("transform error, there shouldn't be quantum reset node here.");
	throw invalid_argument("transform error, there shouldn't be quantum reset node here.");
}

void QCodarMatch::execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, bool &is_dagger)
{
	QCERR("transform error, there shouldn't be control flow node here.");
	throw invalid_argument("transform error, there shouldn't be control flow node here.");
}

void QCodarMatch::execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node, bool &is_dagger)
{
	QCERR("transform error, there shouldn't be classicalProg here.");
	throw invalid_argument("transform error, there shouldn't be classicalProg here.");
}

void QCodarMatch::mappingQProg(size_t run_times, QVec &qv, QProg &mapped_prog)
{
	if (m_original_gates.size() == 0)
	{
		QCERR("parsing qprog info error! qprog valid info is null.");
		throw invalid_argument("parsing qprog info error! qprog valid info is null.");
	}

	for (auto g : m_original_gates)
	{
		if (g.control == -1)
		{
			m_scheduler->addSingleQubitGate(g.gate_name, g.type, g.target, g.param, g.barrier_id, g.is_dagger);
		}
		else
		{
			m_scheduler->addDoubleQubitGate(g.gate_name, g.type, g.control, g.target, g.param, g.is_dagger);
		}
	}
	int best_max_time = INT_MAX;
	double best_error_rate = INT_MAX;
	int gate_count = 0;
	int sum_min_route_len = 0, sum_cnot_route_len = 0, sum_rx_route_len = 0, max_route_len = 0;
	auto best_map_list = m_scheduler->map_list;
	auto save_gate_list = m_scheduler->logical_gate_list;
	std::vector<GateInfo> best_output;
	int random_cout = 15;
	for (int ridx = 0; ridx < random_cout; ridx++)
	{
		for (int i = 0; i < run_times; i++)
		{
			m_scheduler->start();
			int max_time = m_device->maxTime();
			double error_rate = m_scheduler->double_gate_error_rate;

			if (m_arch_type != ORIGIN_VIRTUAL)
			{
				if (max_time <= best_max_time &&
					(max_time < best_max_time || m_scheduler->gate_count < gate_count))
				{
					best_max_time = max_time;
					gate_count = m_scheduler->gate_count;
					best_output = m_scheduler->mapped_result_gates;
					best_map_list = m_scheduler->map_list;
				}
			}
			else
			{
				if (error_rate < best_error_rate ||
					((error_rate - best_error_rate) < 1e-6 && max_time < best_max_time))
				{
					best_error_rate = error_rate;
					best_max_time = max_time;
					gate_count = m_scheduler->gate_count;
					best_output = m_scheduler->mapped_result_gates;
					best_map_list = m_scheduler->map_list;
				}
			}

			if (ridx == 0 && m_scheduler->swap_gate_count == 0)
				break;

			m_device->clear();
			m_scheduler->double_gate_error_rate = 0;
			m_scheduler->logical_gate_list = save_gate_list;
		}

		if (ridx == 0 && m_scheduler->swap_gate_count == 0)
			break;
		int map_num = m_scheduler->map_list.size();
		m_scheduler->map_list.clear();
		m_scheduler->addLogicalQubits(map_num);
	}
	buildResultingQProg(best_output, best_map_list, qv, mapped_prog);
}

void QCodarMatch::buildResultingQProg(const std::vector<GateInfo> resulting_gates, const std::vector<int>  map_vec, QVec &out_qv, QProg &prog)
{
	std::map<int, int> mapping_result;  // ph => loc
	QVec q;
	for (int i = 0; i < map_vec.size(); i++)
	{
		auto qubit = m_qvm->allocateQubitThroughPhyAddress(map_vec[i]);
		if (qubit == nullptr)
		{
			QCERR("Please set the maximum number of qubits supported by the machine");
			throw(qvm_attributes_error("Please set the maximum number of qubits supported by the machine"));
		}
		q.push_back(qubit);
		mapping_result.insert(pair<int, int>(map_vec[i], i));
	}
	out_qv = q;

	std::map<int, QCircuit> slice_cirs;
	std::map<int, QVec> barrier_qvec;

	for (auto g : resulting_gates)
	{
		if (g.control != -1)
		{
			auto iter = mapping_result.find(g.control);
			if (iter == mapping_result.end())
			{
				auto qubit = m_qvm->allocateQubitThroughPhyAddress(g.control);
				q.push_back(qubit);
				int index = mapping_result.size();
				mapping_result.insert(pair<int, int>(g.control, index));
				g.control = index;
			}
			else
			{
				g.control = iter->second;
			}
		}

		auto iter_t = mapping_result.find(g.target);
		if (iter_t == mapping_result.end())
		{
			auto qubit = m_qvm->allocateQubitThroughPhyAddress(g.target);
			q.push_back(qubit);
			int index = mapping_result.size();
			mapping_result.insert(pair<int, int>(g.target, index));
			g.target = index;
		}
		else
		{
			g.target = iter_t->second;
		}

		if (g.type == GateType::BARRIER_GATE)
		{
			barrier_qvec[g.barrier_id].push_back(q[g.target]);
			m_handle_barrier_id = g.barrier_id;
			continue;
		}

		QCircuit cir = slice_cirs[m_handle_barrier_id];
		switch (g.type)
		{
		case GateType::PAULI_X_GATE:
		case GateType::PAULI_Y_GATE:
		case GateType::PAULI_Z_GATE:
		case GateType::X_HALF_PI:
		case GateType::Y_HALF_PI:
		case GateType::Z_HALF_PI:
		case GateType::HADAMARD_GATE:
		case GateType::T_GATE:
		case GateType::S_GATE:
		{
			auto iter = m_single_gate_func.find(g.type);
			if (m_single_gate_func.end() == iter)
			{
				QCERR("unsupported QGate");
				throw invalid_argument("unsupported QGate");
			}

			QGate single_gate = iter->second(q[g.target]);
			single_gate.setDagger(g.is_dagger);
			cir << single_gate;
		}
		break;

		case GateType::RX_GATE:
		case GateType::RY_GATE:
		case GateType::RZ_GATE:
		case GateType::U1_GATE:
		{
			auto iter = m_single_angle_gate_func.find(g.type);
			if (m_single_angle_gate_func.end() == iter)
			{
				QCERR("unsupported QGate");
				throw invalid_argument("unsupported QGate");
			}
			double angle = g.param[0];
			QGate single_angle_gate = iter->second(q[g.target], angle);
			single_angle_gate.setDagger(g.is_dagger);
			cir << single_angle_gate;
		}
		break;

		case GateType::CNOT_GATE:
		case GateType::SWAP_GATE:
		case GateType::CZ_GATE:
		case GateType::ISWAP_GATE:
		case GateType::SQISWAP_GATE:
		{
			auto iter = m_double_gate_func.find(g.type);
			if (m_double_gate_func.end() == iter)
			{
				QCERR("unsupported QGate");
				throw invalid_argument("unsupported QGate");
			}
			QGate double_gate = iter->second(q[g.control], q[g.target]);
			double_gate.setDagger(g.is_dagger);
			cir << double_gate;
		}
		break;
		case  GateType::ISWAP_THETA_GATE:
		{
			double theta = g.param[0];
			QGate iswap_theta = iSWAP(q[g.control], q[g.target], theta);
			iswap_theta.setDagger(g.is_dagger);
			cir << iswap_theta;
		}
		break;

		case GateType::CPHASE_GATE:
		{
			auto iter = m_double_angle_gate_func.find(g.type);
			if (m_double_angle_gate_func.end() == iter)
			{
				QCERR("unsupported QGate");
				throw invalid_argument("unsupported QGate");
			}
			double angle = g.param[0];
			QGate cr_gate = iter->second(q[g.control], q[g.target], angle);
			cr_gate.setDagger(g.is_dagger);
			cir << cr_gate;
		}
		break;
		case  GateType::CU_GATE:
		{
			QGate cu_gate = CU(g.param[0], g.param[1], g.param[2], g.param[3], q[g.control], q[g.target]);
			cu_gate.setDagger(g.is_dagger);

			cir << cu_gate;
		}
		break;
		case  GateType::U2_GATE:
		{
			QGate u2_gate = U2(q[g.target], g.param[0], g.param[1]);
			u2_gate.setDagger(g.is_dagger);
			cir << u2_gate;
		}
		break;
		case  GateType::U3_GATE:
		{
			QGate u3_gate = U3(q[g.target], g.param[0], g.param[1], g.param[2]);
			u3_gate.setDagger(g.is_dagger);
			cir << u3_gate;
		}
		break;
		case  GateType::U4_GATE:
		{
			QGate u4_gate = U4(g.param[0], g.param[1], g.param[2], g.param[3], q[g.target]);
			u4_gate.setDagger(g.is_dagger);
			cir << u4_gate;
		}
		break;
		default:
		{
			QCERR("error! unsupported QGate");
			throw invalid_argument("error! unsupported QGate");
		}
		break;
		}
	}

	for (auto val : slice_cirs)
	{
		prog << val.second;
		int barr_idx = val.first + 1;
		if (barrier_qvec.find(barr_idx) != barrier_qvec.end())
		{
			QVec barrier_qv = barrier_qvec[barr_idx];
			if (barrier_qv.size() == 1)
			{
				prog << BARRIER(barrier_qv[0]);
			}
			else
			{
				QVec ctrl_qv = QVec(barrier_qv.begin() + 1, barrier_qv.end());
				prog << BARRIER(barrier_qv[0]).control(ctrl_qv);
			}
		}
	}

}

QProg QPanda::qcodar_match_by_simple_type(QProg prog, QVec &qv, QuantumMachine * machine,
	size_t m /*= 2*/, size_t n /*= 4*/, size_t run_times /*= 5*/)
{
	if (nullptr == machine)
	{
		QCERR("Quantum machine is nullptr");
		throw std::invalid_argument("Quantum machine is nullptr");
	}

	QProg outprog;
	QCodarMatch match = QCodarMatch(machine, prog, SIMPLE_TYPE, m, n);
	match.mappingQProg(run_times, qv, outprog);
	return outprog;
}

QProg QPanda::qcodar_match_by_config(QProg prog, QVec &qv, QuantumMachine * machine,
	const std::string config_data/* = CONFIG_PATH*/, size_t run_times/*= 5*/)
{
	if (nullptr == machine)
	{
		QCERR("Quantum machine is nullptr");
		throw std::invalid_argument("Quantum machine is nullptr");
	}

	QProg outprog;
	QCodarMatch match = QCodarMatch(machine, prog, ORIGIN_VIRTUAL, 0, 0, config_data);
	match.mappingQProg(run_times, qv, outprog);
	return outprog;
}

QProg QPanda::qcodar_match_by_target_meachine(QProg prog, QVec &qv, QuantumMachine * machine,
	QCodarGridDevice arch_type, size_t run_times /*= 5*/)
{
	if (nullptr == machine)
	{
		QCERR("Quantum machine is nullptr");
		throw std::invalid_argument("Quantum machine is nullptr");
	}

	QProg outprog;
	QCodarMatch match = QCodarMatch(machine, prog, arch_type, 0, 0);
	match.mappingQProg(run_times, qv, outprog);
	return outprog;
}
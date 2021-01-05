#include "Core/QuantumMachine/PartialAmplitudeQVM.h"
using angleParameter = QGATE_SPACE::AbstractSingleAngleParameter;
using namespace std;
USING_QPANDA

static void get_dec_index(std::vector<string> &bin_index, std::vector<uint128_t> &dec_index)
{
	for (auto val : bin_index)
	{
		uint128_t dec_value = 0;
		size_t len = val.size();

		for (size_t i = 0; i < len; ++i)
		{
			bool bin = (val[len - i - 1] != '0');
			uint128_t temp = static_cast<uint128_t>(bin) << i;
			dec_value |= temp;
		}

		dec_index.emplace_back(dec_value);
	}

    return;
}


static void get_couple_state_index(uint128_t num, uint64_t& under_index, uint64_t& upper_index, uint32_t qubit_num)
{
	uint32_t half_qubit = qubit_num / 2;
	long long lower_mask = (1ull << half_qubit) - 1;
	under_index = (uint64_t)(num & lower_mask);
	upper_index = (uint64_t)(num - under_index) >> (qubit_num - half_qubit);
    return;
}

void PartialAmplitudeQVM::init()
{
	_Config.maxQubit = 80;
	_Config.maxCMem = 80;
	_start();
}

void PartialAmplitudeQVM::computing_graph(int qubit_num,const cir_type& circuit, QStat& state)
{
	state.resize(1ull << qubit_num);
	QPUImpl *pQGate = new CPUImplQPU();

	try
	{
		pQGate->initState(0, 1, qubit_num);
		m_graph_backend.computing_graph(circuit, pQGate);

		auto graph_state = pQGate->getQState();

		state.assign(graph_state.begin(), graph_state.end());
		delete pQGate;
	}
	catch (const std::exception& e)
	{
		delete pQGate;
	}

    return;
}


qcomplex_t PartialAmplitudeQVM::PMeasure_bin_index(std::string amplitude)
{
	uint128_t index = 0;
	size_t qubit_num = amplitude.size();
	for (size_t i = 0; i < qubit_num; ++i)
	{
		index += (amplitude[qubit_num - i - 1] != '0') << i ;
	}

	return PMeasure_dec_index(integerToString(index));
}

qcomplex_t PartialAmplitudeQVM::PMeasure_dec_index(std::string amplitude)
{
	uint128_t dec_amplitude(amplitude.c_str());

	auto qubit_num = m_graph_backend.m_qubit_num;
	auto graph_num = m_graph_backend.m_sub_graph.size();

	qcomplex_t result;
	for (auto graph_index = 0; graph_index < graph_num; ++graph_index)
	{
		QStat under_graph_state;
		computing_graph(qubit_num / 2, m_graph_backend.m_sub_graph[graph_index][0], under_graph_state);

		QStat upper_graph_state;
		computing_graph(qubit_num - (qubit_num / 2), m_graph_backend.m_sub_graph[graph_index][1], upper_graph_state);

		uint64_t under_index, upper_index;
		get_couple_state_index(dec_amplitude, under_index, upper_index, m_graph_backend.m_qubit_num);

		result += under_graph_state[under_index] * upper_graph_state[upper_index];
	}

	return result;
}


stat_map PartialAmplitudeQVM::PMeasure_subset(const std::vector<std::string>& amplitude)
{
	std::vector<uint128_t> dec_state;
	for (auto state : amplitude)
	{
		uint128_t val(state.c_str());
		dec_state.emplace_back(val);
	}

	auto qubit_num = m_graph_backend.m_qubit_num;
	auto graph_num = m_graph_backend.m_sub_graph.size();

	QStat result(dec_state.size());
	for (auto graph_index = 0; graph_index < graph_num; ++graph_index)
	{
		QStat under_graph_state;
		computing_graph(qubit_num / 2, m_graph_backend.m_sub_graph[graph_index][0], under_graph_state);

		QStat upper_graph_state;
		computing_graph(qubit_num - (qubit_num / 2), m_graph_backend.m_sub_graph[graph_index][1], upper_graph_state);

		for (auto idx = 0; idx < dec_state.size(); ++idx)
		{
			uint64_t under_index, upper_index;
			get_couple_state_index(dec_state[idx], under_index, upper_index, m_graph_backend.m_qubit_num);

			result[idx] += under_graph_state[under_index] * upper_graph_state[upper_index];
		}
	}

	stat_map state_result;
	for (auto idx = 0; idx < amplitude.size(); ++idx)
	{
		auto pair = std::make_pair(amplitude[idx], result[idx]);

		state_result.insert(pair);
	}
	return state_result;
}


void PartialAmplitudeQVM::execute(std::shared_ptr<AbstractQuantumMeasure>  cur_node, std::shared_ptr<QNode> parent_node)
{
	QCERR("ignore measure");
}

void PartialAmplitudeQVM::execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node)
{
	QCERR("ignore controlflow");
}

void PartialAmplitudeQVM::execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node)
{
	Traversal::traversal(cur_node, true, *this);
}

void PartialAmplitudeQVM::execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node)
{
	Traversal::traversal(cur_node, *this);
}

void PartialAmplitudeQVM::execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node)
{
	QCERR("ignore classical prog");
}

void PartialAmplitudeQVM::execute(std::shared_ptr<AbstractQuantumReset>  cur_node, std::shared_ptr<QNode> parent_node)
{
	QCERR("ignore reset");
}

void PartialAmplitudeQVM::execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node)
{
	if (nullptr == cur_node || nullptr == cur_node->getQGate())
	{
		QCERR("pQGate is null");
		throw invalid_argument("pQGate is null");
	}

	QVec qubits_vector;
	cur_node->getQuBitVector(qubits_vector);

	auto gate_type = (unsigned short)cur_node->getQGate()->getGateType();
	QGateNode node = { gate_type,cur_node->isDagger() };
	switch (gate_type)
	{
	case GateType::P0_GATE:
	case GateType::P1_GATE:
	case GateType::PAULI_Y_GATE:
	case GateType::PAULI_Z_GATE:
	case GateType::X_HALF_PI:
	case GateType::Y_HALF_PI:
	case GateType::Z_HALF_PI:
	case GateType::HADAMARD_GATE:
	case GateType::T_GATE:
	case GateType::S_GATE:
	{
		node.tar_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
	}
	break;

	case GateType::PAULI_X_GATE:
	{
		QVec control_qvec;
		cur_node->getControlVector(control_qvec);

		if (control_qvec.empty())
		{
			node.tar_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
		}
		else
		{
			node.gate_type = TOFFOLI_GATE;

			auto tar_qubit = node.tar_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();

			auto ctr_qubit = node.ctr_qubit = control_qvec[0]->getPhysicalQubitPtr()->getQubitAddr();
			auto tof_qubit = node.tof_qubit = control_qvec[1]->getPhysicalQubitPtr()->getQubitAddr();

			m_graph_backend.m_spilt_num += (m_graph_backend.is_corss_node(ctr_qubit, tar_qubit)) ||
				(m_graph_backend.is_corss_node(ctr_qubit, tof_qubit)) ||
				(m_graph_backend.is_corss_node(tar_qubit, tof_qubit));
		}
	}
	break;

	case GateType::U1_GATE:
	case GateType::RX_GATE:
	case GateType::RY_GATE:
	case GateType::RZ_GATE:
	{
		node.tar_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
		node.gate_parm = dynamic_cast<angleParameter *>(cur_node->getQGate())->getParameter();
	}
	break;

	case GateType::ISWAP_GATE:
	case GateType::SWAP_GATE:
	case GateType::SQISWAP_GATE:
	{
		auto ctr_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
		auto tar_qubit = qubits_vector[1]->getPhysicalQubitPtr()->getQubitAddr();
		if (m_graph_backend.is_corss_node(ctr_qubit, tar_qubit))
		{
			QCERR("Error");
			throw qprog_syntax_error();
		}
		else
		{
			node.ctr_qubit = ctr_qubit;
			node.tar_qubit = tar_qubit;
		}
	}
	break;

	case GateType::CNOT_GATE:
	case GateType::CZ_GATE:
	{
		auto ctr_qubit = node.ctr_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
		auto tar_qubit = node.tar_qubit = qubits_vector[1]->getPhysicalQubitPtr()->getQubitAddr();
		m_graph_backend.m_spilt_num += m_graph_backend.is_corss_node(ctr_qubit, tar_qubit);
	}
	break;

	case GateType::CPHASE_GATE:
	{
		auto ctr_qubit = node.ctr_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
		auto tar_qubit = node.tar_qubit = qubits_vector[1]->getPhysicalQubitPtr()->getQubitAddr();
		node.gate_parm = dynamic_cast<QGATE_SPACE::AbstractSingleAngleParameter *>(cur_node->getQGate())->getParameter();
		m_graph_backend.m_spilt_num += m_graph_backend.is_corss_node(ctr_qubit, tar_qubit);
	}
	break;

    case GateType::BARRIER_GATE:break;
	default:
	{
		QCERR("UnSupported QGate Node");
		throw undefine_error("QGate");
	}
	break;
	}

	m_graph_backend.m_circuit.emplace_back(node);
}

void PartialAmplitudeQVM::construct_graph()
{
	auto qubit_num = getAllocateQubit();
	if (!m_graph_backend.m_spilt_num)
	{
		m_graph_backend.split_circuit(m_graph_backend.m_circuit);
	}
	else
	{
		m_graph_backend.traversal(m_graph_backend.m_circuit);
	}
}
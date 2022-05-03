#include "Core/Core.h"
#include "Core/QuantumMachine/SingleAmplitudeQVM.h"
#include "Core/VirtualQuantumProcessor/SingleAmplitude/QuantumGates.h"
#include "Core/VirtualQuantumProcessor/SingleAmplitude/QuickBB.h"

using namespace std;
USING_QPANDA
#define  OUTPUT_MAX_QNUM 30

SingleAmplitudeQVM::SingleAmplitudeQVM()
{
	m_backend = ComputeBackend::CPU;
	m_single_gate_none_angle.insert({ GateType::PAULI_X_GATE,      X_Gate });
	m_single_gate_none_angle.insert({ GateType::PAULI_Y_GATE,      Y_Gate });
	m_single_gate_none_angle.insert({ GateType::PAULI_Z_GATE,      Z_Gate });
	m_single_gate_none_angle.insert({ GateType::X_HALF_PI,        X1_Gate });
	m_single_gate_none_angle.insert({ GateType::Y_HALF_PI,        Y1_Gate });
	m_single_gate_none_angle.insert({ GateType::Z_HALF_PI,        Z1_Gate });
	m_single_gate_none_angle.insert({ GateType::HADAMARD_GATE,     H_Gate });
	m_single_gate_none_angle.insert({ GateType::T_GATE,            T_Gate });
	m_single_gate_none_angle.insert({ GateType::S_GATE,            S_Gate });

	m_single_gate_and_angle.insert({ GateType::RX_GATE,     RX_Gate });
	m_single_gate_and_angle.insert({ GateType::RY_GATE,     RY_Gate });
	m_single_gate_and_angle.insert({ GateType::RZ_GATE,     RZ_Gate });
	m_single_gate_and_angle.insert({ GateType::U1_GATE,     U1_Gate });

	m_double_gate_none_angle.insert({ GateType::CNOT_GATE,      CNOT_Gate });
	m_double_gate_none_angle.insert({ GateType::SWAP_GATE,      SWAP_Gate });
	m_double_gate_none_angle.insert({ GateType::CZ_GATE,          CZ_Gate });
	m_double_gate_none_angle.insert({ GateType::ISWAP_GATE,    ISWAP_Gate });
	m_double_gate_none_angle.insert({ GateType::SQISWAP_GATE, SQISWAP_Gate });

	m_double_gate_and_angle.insert({ GateType::CPHASE_GATE, CR_Gate });
}

void SingleAmplitudeQVM::init()
{
	_start();
}

void SingleAmplitudeQVM::run(QProg& prog, QVec& qv, size_t max_rank, size_t alloted_time)
{
	/** 1. get QuickBB vertice  */
	m_prog = prog;
	m_prog_map.clear();
	m_edge_count = 0;
	m_prog_map.setMaxRank(max_rank);
	auto vertice_matrix = m_prog_map.getVerticeMatrix();
	auto qubit_count = qv.size();
	vertice_matrix->initVerticeMatrix(qubit_count);
	m_prog_map.setQubitNum(qubit_count);
	bool is_dagger = false;
	execute(prog.getImplementationPtr(), nullptr, is_dagger);
	std::vector<std::pair<size_t, size_t>> vertice_vect;
	getQuickMapVertice(vertice_vect);

	/** 2. get execute  sequence*/
	auto bb_result = QuickBB::compute(vertice_vect, alloted_time);
	getSequence(bb_result.second, m_sequences);

	/** 3.  execute prog by sequence*/
	m_prog_map.clear();
	m_edge_count = 0;
	m_prog_map.setMaxRank(max_rank);
	vertice_matrix = m_prog_map.getVerticeMatrix();
	vertice_matrix->initVerticeMatrix(qubit_count);
	m_prog_map.setQubitNum(qubit_count);
	execute(prog.getImplementationPtr(), nullptr, is_dagger);
}

void SingleAmplitudeQVM::run(QProg& prog, QVec& qv, size_t max_rank,
	const std::vector<qprog_sequence_t>& sequences)
{
	m_prog = prog;
	m_sequences = sequences;
	m_prog_map.clear();
	m_prog_map.setMaxRank(max_rank);

	auto vertice_matrix = m_prog_map.getVerticeMatrix();
	auto qubit_count = qv.size();

	vertice_matrix->initVerticeMatrix(qubit_count);
	m_prog_map.setQubitNum(qubit_count);

	bool is_dagger = false;
	execute(prog.getImplementationPtr(), nullptr, is_dagger);
}

map<string, bool> SingleAmplitudeQVM::directlyRun(QProg& qProg, const NoiseModel& noise_model)
{
    QCERR("SingleAmplitudeQVM have no directlyRun");
    throw qprog_syntax_error("SingleAmplitudeQVM have no directlyRun");
}

qstate_type SingleAmplitudeQVM::singleAmpBackEnd(const string& bin_index)
{
	if (m_prog_map.isEmptyQProg())
	{
		QCERR("PMeasure error");
		throw qprog_syntax_error("PMeasure");
	}

	if (bin_index.size() != m_prog_map.getQubitNum())
	{
		QCERR("The number of qubit and amplitude is not matched");
		throw  std::runtime_error("The number of qubit and amplitude is not matched");
	}

	auto vertice = m_prog_map.getVerticeMatrix();
	qubit_vertice_t qubit_vertice_end, qubit_vertice_begin;
	auto size = vertice->getQubitCount();
	for (size_t i = 0; i < size; i++)
	{
		auto iter = vertice->getQubitMapIter(i);
		if (0 == iter->size())
		{
			continue;
		}

		auto vertice_map_iter_b = (*iter).begin();
		qubit_vertice_begin.m_qubit_id = i;
		qubit_vertice_begin.m_num = (*vertice_map_iter_b).first;
		TensorEngine::dimDecrementbyValue(m_prog_map, qubit_vertice_begin, 0);
	}

	auto check = [](char bin)
	{
		if ('1' != bin && '0' != bin)
		{
			QCERR("PMeasure parm error");
			throw qprog_syntax_error("PMeasure parm");
		}
		else
		{
			return bin == '0' ? 0 : 1;
		}
	};
	qsize_t flag = 1;
	for (size_t i = 0; i < size; i++)
	{
		auto iter = m_prog_map.getVerticeMatrix()->getQubitMapIter(i);
		auto vertice_map_iter = (*iter).end();
		size_t value = check(bin_index[size - i - 1]);
		if (vertice_map_iter == iter->begin())
		{
			if (value == 1)
			{
				flag = 0;
			}
			continue;
		}
		vertice_map_iter--;

		qubit_vertice_end.m_qubit_id = i;
		qubit_vertice_end.m_num = (*vertice_map_iter).first;
		TensorEngine::dimDecrementbyValue(m_prog_map, qubit_vertice_end, value);
	}
	auto flag_complex = qcomplex_data_t(flag, 0);

	TensorEngine::MergeByVerticeVector(m_prog_map, m_sequences[0]);    //TensorEngine::MergeByVerticeVector(m_prog_map,vertice_vector);
	qcomplex_data_t result = flag_complex * TensorEngine::Merge(m_prog_map, m_sequences[1]);
	return (result.real() * result.real() + result.imag() * result.imag());
}


qstate_type SingleAmplitudeQVM::pMeasureBinindex(std::string index)
{
	return singleAmpBackEnd(index);
}

qstate_type SingleAmplitudeQVM::pMeasureDecindex(std::string index)
{
	uint256_t dec_index(index.c_str());
	auto qubit_num = m_prog_map.getQubitNum();
	auto bin_index = integerToBinary(dec_index, qubit_num);
	return singleAmpBackEnd(bin_index);
}

prob_dict SingleAmplitudeQVM::getProbDict(QVec qlist)
{
	std::vector<size_t> qubits_addr;
	for_each(qlist.begin(), qlist.end(), [&](Qubit* q) {
		qubits_addr.push_back(q->get_phy_addr());
	});

	stable_sort(qubits_addr.begin(), qubits_addr.end());
	qubits_addr.erase(unique(qubits_addr.begin(),
		qubits_addr.end(), [](size_t a, size_t b) {return a == b; }),
		qubits_addr.end());
	size_t measure_qubit_num = qubits_addr.size();

	prob_dict res;
	int alloc_qubit_num = m_prog.get_max_qubit_addr()+1;
	if (alloc_qubit_num > 30)
	{
		QCERR("the number of qubits is too large to output");
		throw runtime_error("the number of qubits is too large to output");
	}
	for (uint256_t i = 0; i < (uint256_t(1) << alloc_qubit_num); i++)
	{
		QProgMap new_map(m_prog_map);
		auto amp = singleAmpBackEnd(integerToBinary(i, alloc_qubit_num));
		m_prog_map = new_map;

		uint256_t idx = 0;
		for (size_t j = 0; j < measure_qubit_num; j++)
		{
			idx += (((i >> (qubits_addr[j])) % 2) << j);
		}

		auto bin_idx = integerToBinary(idx, measure_qubit_num);
		auto iter = res.find(bin_idx);
		if (res.end() == iter)
		{
			res.insert({ bin_idx, amp });
		}
		else
		{
			iter->second += amp;
		}
	}

	return res;
}

prob_dict SingleAmplitudeQVM::probRunDict(QProg& prog, QVec qlist)
{
	run(prog, qlist);
	return getProbDict(qlist);
}


void SingleAmplitudeQVM::execute(shared_ptr<AbstractQGateNode> cur_node,
	shared_ptr<QNode> parent_node, bool& is_dagger)
{
	if (nullptr == cur_node || nullptr == cur_node->getQGate())
	{
		QCERR("pQGate is null");
		throw invalid_argument("pQGate is null");
	}

	QVec qubits;
	cur_node->getQuBitVector(qubits);
	auto gate_type = static_cast<GateType>(cur_node->getQGate()->getGateType());
	qstate_t gate_tensor;

	auto dagger = cur_node->isDagger() ^ is_dagger;
	cur_node->setDagger(dagger);

	switch (gate_type)
	{
	case PAULI_X_GATE:
	{
		QVec control_qubits;
		auto control_qubits_num = cur_node->getControlVector(control_qubits);
		if (0 == control_qubits_num)
		{
			auto qubit_addr0 = qubits[0]->getPhysicalQubitPtr()->getQubitAddr();
			m_single_gate_none_angle[gate_type](gate_tensor, cur_node->isDagger());
			addVerticeAndEdge(gate_tensor, gate_type, qubit_addr0);
		}
		else if (2 == control_qubits_num)
		{
			auto control_qubit_addr0 = control_qubits[0]->getPhysicalQubitPtr()->getQubitAddr();
			auto control_qubit_addr1 = control_qubits[1]->getPhysicalQubitPtr()->getQubitAddr();
			auto qubit_addr0 = qubits[0]->getPhysicalQubitPtr()->getQubitAddr();
			TOFFOLI_Gate(gate_tensor, cur_node->isDagger());
			addThreeNonDiagonalGateVerticeAndEdge(gate_tensor, control_qubit_addr0,
				control_qubit_addr1, qubit_addr0);
		}
		else
		{
			QCERR("undefined error");
			throw runtime_error("undefined error");
		}
		break;
	}
	case PAULI_Y_GATE:
	case PAULI_Z_GATE:
	case X_HALF_PI:
	case Y_HALF_PI:
	case Z_HALF_PI:
	case HADAMARD_GATE:
	case T_GATE:
	case S_GATE:
	{
		auto qubit_addr0 = qubits[0]->getPhysicalQubitPtr()->getQubitAddr();
		m_single_gate_none_angle[gate_type](gate_tensor, cur_node->isDagger());
		addVerticeAndEdge(gate_tensor, gate_type, qubit_addr0);
		break;
	}
	case U1_GATE:
	case RX_GATE:
	case RY_GATE:
	case RZ_GATE:
	{
		auto qubit_addr0 = qubits[0]->getPhysicalQubitPtr()->getQubitAddr();
		auto angle = dynamic_cast<QGATE_SPACE::AbstractSingleAngleParameter*>(cur_node->getQGate());
		auto angle_value = angle->getParameter();
		m_single_gate_and_angle[gate_type](gate_tensor, angle_value, cur_node->isDagger());
		addVerticeAndEdge(gate_tensor, gate_type, qubit_addr0);
		break;
	}
	case U2_GATE:
	{
		auto qubit_addr0 = qubits[0]->getPhysicalQubitPtr()->getQubitAddr();
		QGATE_SPACE::U2* u2_gate = dynamic_cast<QGATE_SPACE::U2*>(cur_node->getQGate());
		auto phi = u2_gate->get_phi();
		auto lambda = u2_gate->get_lambda();
		U2_Gate(gate_tensor, phi, lambda, cur_node->isDagger());
		addVerticeAndEdge(gate_tensor, gate_type, qubit_addr0);
		break;
	}
	case U3_GATE:
	{
		auto qubit_addr0 = qubits[0]->getPhysicalQubitPtr()->getQubitAddr();
		QGATE_SPACE::U3* u3_gate = dynamic_cast<QGATE_SPACE::U3*>(cur_node->getQGate());
		auto theta = u3_gate->get_theta();
		auto phi = u3_gate->get_phi();
		auto lambda = u3_gate->get_lambda();
		U3_Gate(gate_tensor, theta, phi, lambda, cur_node->isDagger());
		addVerticeAndEdge(gate_tensor, gate_type, qubit_addr0);
		break;
	}
	case U4_GATE:
	{
		auto qubit_addr0 = qubits[0]->getPhysicalQubitPtr()->getQubitAddr();
		auto angle_param = dynamic_cast<QGATE_SPACE::AbstractAngleParameter*>(cur_node->getQGate());
		auto alpha = angle_param->getAlpha();
		auto beta = angle_param->getBeta();
		auto gamma = angle_param->getGamma();
		auto delta = angle_param->getDelta();
		U4_Gate(gate_tensor, alpha, beta, gamma, delta, cur_node->isDagger());
		addVerticeAndEdge(gate_tensor, gate_type, qubit_addr0);
		break;
	}


	case ISWAP_GATE:
	case SQISWAP_GATE:
	case CNOT_GATE:
	case SWAP_GATE:
	case CZ_GATE:
	{
		auto qubit_addr0 = qubits[0]->getPhysicalQubitPtr()->getQubitAddr();
		auto qubit_addr1 = qubits[1]->getPhysicalQubitPtr()->getQubitAddr();
		m_double_gate_none_angle[gate_type](gate_tensor, cur_node->isDagger());
		addVerticeAndEdge(gate_tensor, gate_type, qubit_addr0, qubit_addr1);
		break;
	}
	case CPHASE_GATE:
	{
		auto qubit_addr0 = qubits[0]->getPhysicalQubitPtr()->getQubitAddr();
		auto qubit_addr1 = qubits[1]->getPhysicalQubitPtr()->getQubitAddr();
		auto angle = dynamic_cast<QGATE_SPACE::AbstractSingleAngleParameter*>(cur_node->getQGate());
		auto angle_value = angle->getParameter();
		m_double_gate_and_angle[gate_type](gate_tensor, angle_value, cur_node->isDagger());
		addVerticeAndEdge(gate_tensor, gate_type, qubit_addr0, qubit_addr1);
		break;
	}
	case GateType::I_GATE:
	case GateType::BARRIER_GATE:
		break;

	default:
		QCERR("undefined error");
		throw runtime_error("undefined error");
	}
}

void SingleAmplitudeQVM::execute(shared_ptr<AbstractQuantumProgram> cur_node,
	shared_ptr<QNode> parent_node, bool& is_dagger)
{
	Traversal::traversal(cur_node, *this, is_dagger);
}

void SingleAmplitudeQVM::execute(shared_ptr<AbstractQuantumMeasure> cur_node,
	shared_ptr<QNode> parent_node, bool& is_dagger)
{
	QCERR("execute node error");
	throw std::runtime_error("execute node error");
}

void SingleAmplitudeQVM::execute(shared_ptr<AbstractQuantumReset> cur_node,
	shared_ptr<QNode> parent_node, bool& is_dagger)
{
	QCERR("execute node error");
	throw std::runtime_error("execute node error");
}

void SingleAmplitudeQVM::execute(std::shared_ptr<AbstractQNoiseNode> cur_node,
		std::shared_ptr<QNode> parent_node, bool& is_dagger)
{
	QCERR_AND_THROW(std::invalid_argument, "SingleAmplitudeQVM not support execute Virtual Noise Node");
}

void SingleAmplitudeQVM::execute(std::shared_ptr<AbstractQDebugNode> cur_node,
		std::shared_ptr<QNode> parent_node, bool& is_dagger)
{
	QCERR_AND_THROW(std::invalid_argument, "SingleAmplitudeQVM not support debug");
}

void SingleAmplitudeQVM::execute(shared_ptr<AbstractControlFlowNode> cur_node,
	shared_ptr<QNode> parent_node, bool& is_dagger)
{
	QCERR("execute node error");
	throw std::runtime_error("execute node error");
}

void SingleAmplitudeQVM::execute(shared_ptr<AbstractClassicalProg> cur_node,
	shared_ptr<QNode> parent_node, bool& is_dagger)
{
	QCERR("execute node error");
	throw std::runtime_error("execute node error");
}

void SingleAmplitudeQVM::execute(shared_ptr<AbstractQuantumCircuit> cur_node,
	shared_ptr<QNode>  parent_node, bool& is_dagger)
{
	bool dagger = cur_node->isDagger() ^ is_dagger;
	Traversal::traversal(cur_node, true, *this, dagger);
}

void SingleAmplitudeQVM::addVerticeAndEdge(qstate_t& gate_tensor, GateType gate_type,
	qsize_t qubit1, qsize_t qubit2)
{
	switch (gate_type)
	{
	case GateType::T_GATE:
	case GateType::S_GATE:
	case GateType::PAULI_Z_GATE:
	case GateType::Z_HALF_PI:
	case GateType::RZ_GATE:
	case GateType::U1_GATE:
		addSingleGateDiagonalVerticeAndEdge(gate_tensor, qubit1);
		break;
	case GateType::HADAMARD_GATE:
	case GateType::PAULI_X_GATE:
	case GateType::PAULI_Y_GATE:
	case GateType::X_HALF_PI:
	case GateType::Y_HALF_PI:
	case GateType::RX_GATE:
	case GateType::RY_GATE:
	case GateType::U2_GATE:
	case GateType::U3_GATE:
	case GateType::U4_GATE:
		addSingleGateNonDiagonalVerticeAndEdge(gate_tensor, qubit1);
		break;
	case GateType::CZ_GATE:
	case GateType::CPHASE_GATE:
		addDoubleDiagonalGateVerticeAndEdge(gate_tensor, qubit1, qubit2);
		break;
	case GateType::CNOT_GATE:
	case GateType::ISWAP_GATE:
	case GateType::SQISWAP_GATE:
	case GateType::SWAP_GATE:
		addDoubleNonDiagonalGateVerticeAndEdge(gate_tensor, qubit1, qubit2);
		break;

	default:
		throw std::runtime_error("QGate type error");
	}

	return;
}


void SingleAmplitudeQVM::addSingleGateDiagonalVerticeAndEdge(qstate_t& gate_tensor,
	qsize_t qubit)
{
	edge_map_t* edge_map = m_prog_map.getEdgeMap();
	auto max_rank = m_prog_map.getMaxRank();
	ComplexTensor temp(m_backend, 1, gate_tensor, max_rank);
	VerticeMatrix* vertice_matrix = m_prog_map.getVerticeMatrix();
	auto vertice_id = vertice_matrix->getQubitVerticeLastID(qubit);

	vector<pair<qsize_t, qsize_t>> contect_vertice =
	{ { qubit,vertice_id } };
	m_edge_count++;
	Edge edge(1, temp, contect_vertice);
	edge_map->insert(pair<qsize_t, Edge>(m_edge_count, edge));
	vertice_matrix->addContectEdge(qubit, vertice_id, m_edge_count);
}


void SingleAmplitudeQVM::addSingleGateNonDiagonalVerticeAndEdge(qstate_t& gate_tensor,
	qsize_t qubit)
{
	edge_map_t* edge_map = m_prog_map.getEdgeMap();
	auto max_rank = m_prog_map.getMaxRank();
	ComplexTensor temp(m_backend, 2, gate_tensor, max_rank);

	VerticeMatrix* vertice_matrix = m_prog_map.getVerticeMatrix();
	auto vertice_id = vertice_matrix->getQubitVerticeLastID(qubit);
	auto vertice_id2 = vertice_matrix->addVertice(qubit);

	vector<pair<qsize_t, qsize_t>> contect_vertice =
	{ { qubit,vertice_id2 },
	  { qubit,vertice_id } };
	m_edge_count++;
	Edge edge(1, temp, contect_vertice);
	edge_map->insert(pair<qsize_t, Edge>(m_edge_count, edge));
	vertice_matrix->addContectEdge(qubit, vertice_id, m_edge_count);
	vertice_matrix->addContectEdge(qubit, vertice_id2, m_edge_count);
}


void SingleAmplitudeQVM::addDoubleDiagonalGateVerticeAndEdge(qstate_t& gate_tensor,
	qsize_t qubit1, qsize_t qubit2)
{
	edge_map_t* edge_map = m_prog_map.getEdgeMap();
	auto max_rank = m_prog_map.getMaxRank();
	ComplexTensor temp(m_backend, 2, gate_tensor, max_rank);
	VerticeMatrix* vertice_matrix = m_prog_map.getVerticeMatrix();
	auto vertice_qubit1_id = vertice_matrix->getQubitVerticeLastID(qubit1);
	auto vertice_qubit2_id = vertice_matrix->getQubitVerticeLastID(qubit2);

	vector<pair<qsize_t, qsize_t>> contect_vertice =
	{ { qubit1,vertice_qubit1_id },
	  { qubit2,vertice_qubit2_id } };

	m_edge_count++;
	Edge edge(2, temp, contect_vertice);
	edge_map->insert(pair<qsize_t, Edge>(m_edge_count, edge));
	vertice_matrix->addContectEdge(qubit1, vertice_qubit1_id, m_edge_count);
	vertice_matrix->addContectEdge(qubit2, vertice_qubit2_id, m_edge_count);
}

void SingleAmplitudeQVM::addDoubleNonDiagonalGateVerticeAndEdge(qstate_t& gate_tensor,
	qsize_t qubit1, qsize_t qubit2)
{

	edge_map_t* edge_map = m_prog_map.getEdgeMap();
	auto max_rank = m_prog_map.getMaxRank();
	ComplexTensor temp(m_backend, 4, gate_tensor, max_rank);
	VerticeMatrix* vertice_matrix = m_prog_map.getVerticeMatrix();

	auto vertice_qubit1_id = vertice_matrix->getQubitVerticeLastID(qubit1);
	auto vertice_qubit1_id2 = vertice_matrix->addVertice(qubit1);
	auto vertice_qubit2_id = vertice_matrix->getQubitVerticeLastID(qubit2);
	auto vertice_qubit2_id2 = vertice_matrix->addVertice(qubit2);

	vector<pair<qsize_t, qsize_t>> contect_vertice
		= { { qubit1,vertice_qubit1_id },
			{ qubit2,vertice_qubit2_id },
			{ qubit1,vertice_qubit1_id2 },
			{ qubit2,vertice_qubit2_id2 } };

	m_edge_count++;
	Edge edge(2, temp, contect_vertice);
	edge_map->insert(pair<qsize_t, Edge>(m_edge_count, edge));

	vertice_matrix->addContectEdge(qubit1, vertice_qubit1_id, m_edge_count);
	vertice_matrix->addContectEdge(qubit1, vertice_qubit1_id2, m_edge_count);
	vertice_matrix->addContectEdge(qubit2, vertice_qubit2_id, m_edge_count);
	vertice_matrix->addContectEdge(qubit2, vertice_qubit2_id2, m_edge_count);
}

void SingleAmplitudeQVM::addThreeNonDiagonalGateVerticeAndEdge(qstate_t& gate_tensor,
	qsize_t qubit1,
	qsize_t qubit2,
	qsize_t qubit3)
{
	edge_map_t* edge_map = m_prog_map.getEdgeMap();
	auto max_rank = m_prog_map.getMaxRank();
	ComplexTensor temp(m_backend, 6, gate_tensor, max_rank);
	VerticeMatrix* vertice_matrix = m_prog_map.getVerticeMatrix();
	auto vertice_qubit1_id = vertice_matrix->getQubitVerticeLastID(qubit1);
	auto vertice_qubit1_id2 = vertice_matrix->addVertice(qubit1);

	auto vertice_qubit2_id = vertice_matrix->getQubitVerticeLastID(qubit2);
	auto vertice_qubit2_id2 = vertice_matrix->addVertice(qubit2);

	auto vertice_qubit3_id = vertice_matrix->getQubitVerticeLastID(qubit3);
	auto vertice_qubit3_id2 = vertice_matrix->addVertice(qubit3);

	vector<pair<qsize_t, qsize_t>> contect_vertice
		= { { qubit1,vertice_qubit1_id },
			{ qubit2,vertice_qubit2_id },
			{ qubit3,vertice_qubit3_id },
			{ qubit1,vertice_qubit1_id2 },
			{ qubit2,vertice_qubit2_id2 },
			{ qubit3,vertice_qubit3_id2} };

	m_edge_count++;
	Edge edge(3, temp, contect_vertice);
	edge_map->insert(pair<qsize_t, Edge>(m_edge_count, edge));
	vertice_matrix->addContectEdge(qubit1, vertice_qubit1_id, m_edge_count);
	vertice_matrix->addContectEdge(qubit1, vertice_qubit1_id2, m_edge_count);

	vertice_matrix->addContectEdge(qubit2, vertice_qubit2_id, m_edge_count);
	vertice_matrix->addContectEdge(qubit2, vertice_qubit2_id2, m_edge_count);

	vertice_matrix->addContectEdge(qubit3, vertice_qubit3_id, m_edge_count);
	vertice_matrix->addContectEdge(qubit3, vertice_qubit3_id2, m_edge_count);
}


size_t SingleAmplitudeQVM::getSequence(const vector<size_t>& quickbb_vertice, vector<qprog_sequence_t>& sequence_vec)
{
	QProgMap  prog_map = m_prog_map;
	if (prog_map.isEmptyQProg())
	{
		return false;
	}

	auto vertice = prog_map.getVerticeMatrix();
	QubitVertice qubit_vertice_end, qubit_vertice_begin;
	auto size = vertice->getQubitCount();

	for (size_t i = 0; i < size; i++)
	{
		auto iter = vertice->getQubitMapIter(i);
		if (0 == iter->size())
		{
			continue;
		}

		auto vertice_map_iter_b = iter->begin();
		qubit_vertice_begin.m_qubit_id = i;
		qubit_vertice_begin.m_num = vertice_map_iter_b->first;
		TensorEngine::dimDecrementbyValue(prog_map, qubit_vertice_begin, 0);
	}

	for (size_t i = 0; i < size; i++)
	{
		auto iter = prog_map.getVerticeMatrix()->getQubitMapIter(i);
		if (0 == iter->size())
		{
			continue;
		}

		auto vertice_map_iter = iter->end();
		vertice_map_iter--;
		qubit_vertice_end.m_qubit_id = i;
		qubit_vertice_end.m_num = vertice_map_iter->first;
		TensorEngine::dimDecrementbyValue(prog_map, qubit_vertice_end, 0);
	}

	sequence_vec.resize(2);
	TensorEngine::seq_merge_by_vertices(prog_map, quickbb_vertice, sequence_vec[0]);
	TensorEngine::seq_merge(prog_map, sequence_vec[1]);

	return 1ull << prog_map.m_count;
}


void SingleAmplitudeQVM::getQuickMapVertice(std::vector<std::pair<size_t, size_t>>& map_vector)
{
	auto prog_map = m_prog_map;
	auto vertice = prog_map.getVerticeMatrix();

	QubitVertice qubit_vertice_end, qubit_vertice_begin;

	for (size_t i = 0; i < prog_map.getQubitNum(); i++)
	{
		auto iter = vertice->getQubitMapIter(i);
		if (0 == iter->size())
		{
			continue;
		}

		auto vertice_map_iter_b = (*iter).begin();
		qubit_vertice_begin.m_qubit_id = i;
		qubit_vertice_begin.m_num = (*vertice_map_iter_b).first;
		TensorEngine::dimDecrementbyValue(prog_map, qubit_vertice_begin, 0);
	}

	for (size_t i = 0; i < prog_map.getQubitNum(); i++)
	{
		auto iter = prog_map.getVerticeMatrix()->getQubitMapIter(i);
		if (0 == iter->size())
		{
			continue;
		}

		auto vertice_map_iter = (*iter).end();
		vertice_map_iter--;
		size_t value = 0;
		qubit_vertice_end.m_qubit_id = i;
		qubit_vertice_end.m_num = (*vertice_map_iter).first;
		TensorEngine::dimDecrementbyValue(prog_map, qubit_vertice_end, value);
	}

	TensorEngine::getVerticeMap(prog_map, map_vector);
	auto size = map_vector.size();

	size_t max_element = 0;
	for_each(map_vector.begin(), map_vector.end(),
		[&max_element](vector<pair<size_t, size_t>>::reference self)
	{
		max_element = max_element < self.first ? self.first : max_element;
		max_element = max_element < self.second ? self.second : max_element;
	});

}

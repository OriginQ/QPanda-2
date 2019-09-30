#include "QProgStored.h"
#include "QVec.h"
#include "Utilities/MetadataValidity.h"
#include "TransformDecomposition.h"

using namespace std;
USING_QPANDA

const std::map<int, QProgStoredNodeType> kGateTypeAndQProgTypeMap =
{
    {PAULI_X_GATE,  QPROG_PAULI_X_GATE},
    {PAULI_Y_GATE,  QPROG_PAULI_Y_GATE},
    {PAULI_Z_GATE,  QPROG_PAULI_Z_GATE},
    {X_HALF_PI,     QPROG_X_HALF_PI},
    {Y_HALF_PI,     QPROG_Y_HALF_PI},
    {Z_HALF_PI,     QPROG_Z_HALF_PI},
    {HADAMARD_GATE, QPROG_HADAMARD_GATE},
    {T_GATE,        QPROG_T_GATE},
    {S_GATE,        QPROG_S_GATE},
    {RX_GATE,       QPROG_RX_GATE},
    {RY_GATE,       QPROG_RY_GATE},
    {RZ_GATE,       QPROG_RZ_GATE},
    {U1_GATE,       QPROG_U1_GATE},
    {U2_GATE,       QPROG_U2_GATE},
    {U3_GATE,       QPROG_U3_GATE},
    {U4_GATE,       QPROG_U4_GATE},
    {CU_GATE,       QPROG_CU_GATE},
    {CNOT_GATE,     QPROG_CNOT_GATE},
    {CZ_GATE,       QPROG_CZ_GATE},
    {CPHASE_GATE,   QPROG_CPHASE_GATE},
    {ISWAP_THETA_GATE, QPROG_ISWAP_THETA_GATE},
    {ISWAP_GATE,    QPROG_ISWAP_GATE},
	{SQISWAP_GATE,  QPROG_SQISWAP_GATE},
	{SWAP_GATE,    QPROG_SWAP_GATE}
};


QProgStored::QProgStored(QuantumMachine *qm) :
    m_node_counter(0u)
{
    m_operator_map.insert(pair<string, int>("+", PLUS));
    m_operator_map.insert(pair<string, int>("-", MINUS));
    m_operator_map.insert(pair<string, int>("*", MUL));
    m_operator_map.insert(pair<string, int>("/", DIV));
    m_operator_map.insert(pair<string, int>("==", EQUAL));
    m_operator_map.insert(pair<string, int>("!=", NE));
    m_operator_map.insert(pair<string, int>(">", GT));
    m_operator_map.insert(pair<string, int>(">=", EGT));
    m_operator_map.insert(pair<string, int>("<", LT));
    m_operator_map.insert(pair<string, int>("<=", ELT));
    m_operator_map.insert(pair<string, int>("&&", AND));
    m_operator_map.insert(pair<string, int>("||", OR));
    m_operator_map.insert(pair<string, int>("!", NOT));
    m_operator_map.insert(pair<string, int>("=", ASSIGN));

    m_quantum_machine = qm;
}

QProgStored::~QProgStored()
{ }

void QProgStored::transform(QProg &prog)
{
    m_qubit_number = m_quantum_machine->getAllocateQubit();
    m_cbit_number = m_quantum_machine->getAllocateCMem();

	transformQProgByTraversalAlg(&prog);
    return;
}

void QProgStored::store(const string &filename)
{
    std::ofstream out;
	out.open(filename, std::ios::out | std::ios::binary);

    if (!out)
    {
        QCERR("fwrite file failure");
        throw invalid_argument("file open error");
    }

    uint32_t file_length = (sizeof(uint32_t) + sizeof(DataNode)) * (m_node_counter + 2);
    pair<uint32_t, DataNode> qubits_cbits_data(m_qubit_number, m_cbit_number);
    pair<uint32_t, DataNode> file_msg(file_length, m_node_counter);

    // fileMsg
    out.write((char *)&file_msg, sizeof(file_msg));
    // qubits and cbits msg
    out.write((char *)&qubits_cbits_data, sizeof(qubits_cbits_data));
    // QProg msg
    out.write((char *)m_data_vector.data(), m_node_counter * (sizeof(uint32_t) + sizeof(DataNode)));

    out.close();
    return;
}

std::vector<uint8_t> QProgStored::getInsturctions()
{
    size_t size = (sizeof(uint32_t) + sizeof(DataNode));
    pair<uint32_t, DataNode> qubits_cbits_data(m_qubit_number, m_cbit_number);
    pair<uint32_t, DataNode> node_msg(0, m_node_counter);
    vector<uint8_t> data(size * (m_data_vector.size() + 2), 0);

    memcpy(data.data(), &node_msg, size);
    memcpy(data.data() + size, &qubits_cbits_data, size);
    memcpy(data.data() + 2 * size, m_data_vector.data(), size * m_data_vector.size());

    return data;
}

void QProgStored::transformQControlFlow(AbstractControlFlowNode *p_controlflow)
{
    if (nullptr == p_controlflow)
    {
        QCERR("pQControlFlow is null");
        throw invalid_argument("pQControlFlow is null");
    }

    ClassicalCondition p_classcical_condition = p_controlflow->getCExpr();
    auto expr = p_classcical_condition.getExprPtr().get();
    transformCExpr(expr);

    QNode *p_node = dynamic_cast<QNode *>(p_controlflow);
    int node_type = p_node->getNodeType();

    switch (node_type)
    {
    case NodeType::QIF_START_NODE:
        transformQIfProg(p_controlflow);
        break;
    case NodeType::WHILE_START_NODE:
        transformQWhileProg(p_controlflow);
        break;
    default:
        QCERR("NodeType is error");
        throw invalid_argument("NodeType is error");
        break;
    }

    return;
}


void QProgStored::transformQIfProg(AbstractControlFlowNode *p_controlFlow)
{
    if (nullptr == p_controlFlow)
    {
        QCERR("p_controlFlow is null");
        throw invalid_argument("p_controlFlow is null");
    }

    uint32_t true_and_false_node = 0;
    addDataNode(QPROG_QIF_NODE, true_and_false_node);
	auto if_node_index = m_data_vector.size() - 1;
	
	auto truth_branch_node = p_controlFlow->getTrueBranch();
	if (nullptr != truth_branch_node)
	{
		Traversal::traversalByType(truth_branch_node, nullptr, *this);
	}
	true_and_false_node |= (m_node_counter << kCountMoveBit);


	auto false_branch_node = p_controlFlow->getFalseBranch();
	if (nullptr != false_branch_node)
	{
		Traversal::traversalByType(false_branch_node, nullptr, *this);
		true_and_false_node |= m_node_counter;
	}

	m_data_vector[if_node_index].second.qubit_data = true_and_false_node;


    return;
}


void QProgStored::transformQWhileProg(AbstractControlFlowNode *p_controlflow)
{
    if (nullptr == p_controlflow)
    {
        QCERR("p_controlflow is null");
        throw invalid_argument("p_controlflow is null");
    }

    uint32_t true_and_false_node = 0;
    addDataNode(QPROG_QWHILE_NODE, true_and_false_node);
	auto while_node_index = m_data_vector.size() - 1;

	auto truth_branch_node = p_controlflow->getTrueBranch();
	if (nullptr != truth_branch_node)
	{
		Traversal::traversalByType(truth_branch_node, nullptr, *this);
	}

    true_and_false_node |= (m_node_counter << kCountMoveBit);
	m_data_vector[while_node_index].second.qubit_data = true_and_false_node;

    return;
}


void QProgStored::transformQGate(AbstractQGateNode *p_gate)
{
    if (nullptr == p_gate)
    {
        QCERR("pQGate is null");
        throw invalid_argument("pQGate is null");
    }

	QVec ctrl_qubits_vector;
	p_gate->getControlVector(ctrl_qubits_vector);
	if (!ctrl_qubits_vector.empty())
	{
		for (int i = 0; i < ctrl_qubits_vector.size(); i += 2)
		{
			uint32_t qubit_data = 0;
			size_t quibt_addr_1 = 0;
			size_t quibt_addr_2 = 0;
			quibt_addr_1 = ctrl_qubits_vector[i]->getPhysicalQubitPtr()->getQubitAddr();
			if (i + 1 < ctrl_qubits_vector.size())
			{
				quibt_addr_2 = ctrl_qubits_vector[i + 1]->getPhysicalQubitPtr()->getQubitAddr();
				if (0 == quibt_addr_2)
				{
					size_t addr_tmp = quibt_addr_1;
					quibt_addr_1 = quibt_addr_2;
					quibt_addr_2 = addr_tmp;
				}
			}
	
			qubit_data |= quibt_addr_1;
			qubit_data |= (quibt_addr_2 << kCountMoveBit);
			addDataNode(QPROG_CONTROL, qubit_data);
		}
	}

    auto quantum_gate = p_gate->getQGate();
    int gate_type = quantum_gate->getGateType();
    const int kQubitNumberMax = 2;
    QVec qubits;
    p_gate->getQuBitVector(qubits);

    /* only support singleGate and doubleGate */
    if (qubits.size() > kQubitNumberMax)
    {
        QCERR("pQGate is illegal");
        throw invalid_argument("pQGate is illegal");
    }

    uint16_t qubit_array[kQubitNumberMax] = { 0 };
    int qubit_number = 0;
    for (auto qubit : qubits)
    {
        PhysicalQubit *p_physical_qubit = qubit->getPhysicalQubitPtr();
        size_t qubit_addr = p_physical_qubit->getQubitAddr();
        qubit_array[qubit_number] = (uint16_t)qubit_addr;
        qubit_number++;
    }

    uint32_t qubit_data = 0;
    qubit_data |= qubit_array[0];
    qubit_data |= (qubit_array[1] << kCountMoveBit);

    auto iter_prog_type = kGateTypeAndQProgTypeMap.find(gate_type);
    if (iter_prog_type == kGateTypeAndQProgTypeMap.end())
    {
        QCERR("gate type error");
        throw invalid_argument("gate type error");
    }
	addDataNode(iter_prog_type->second, qubit_data, p_gate->isDagger());

    if (GateType::RX_GATE == gate_type || GateType::RY_GATE == gate_type
        || GateType::RZ_GATE == gate_type || GateType::U1_GATE == gate_type
        || GateType::CPHASE_GATE == gate_type || GateType::ISWAP_THETA_GATE == gate_type)
    {
        handleQGateWithOneAngle(p_gate);
    }
    else if (GateType::U4_GATE == gate_type || GateType::CU_GATE == gate_type)
    {
        handleQGateWithFourAngle(p_gate);
    }

    return;
}

void QProgStored::transformQMeasure(AbstractQuantumMeasure *p_measure)
{
    if (nullptr == p_measure)
    {
        QCERR("p_measure is null");
        throw invalid_argument("p_measure is null");
    }

    Qubit *qubit = p_measure->getQuBit();
    auto p_physical_qubit = qubit->getPhysicalQubitPtr();
    size_t qubit_addr = p_physical_qubit->getQubitAddr();
    auto cbit = p_measure->getCBit();

    string cbit_name = cbit->getName();
    string cbit_number_str = cbit_name.substr(1);
    int cbit_number = stoi(cbit_number_str);
    const int kQubitNumberMax = 2;
    uint16_t qubit_array[kQubitNumberMax] = { 0 };

    if (qubit_addr > kUshortMax)
    {
        QCERR("QBit number is out of range");
        throw invalid_argument("QBit number is out of range");
    }
    qubit_array[0] = (uint16_t)qubit_addr;

    if (cbit_number > kUshortMax)
    {
        QCERR("QCit number is out of range");
        throw invalid_argument("QCit number is out of range");
    }

    qubit_array[1] = (uint16_t)cbit_number;
    uint32_t qubit_data = 0;
    qubit_data |= qubit_array[0];
    qubit_data |= (qubit_array[1] << kCountMoveBit);
    addDataNode(QPROG_MEASURE_GATE, qubit_data);

    return;
}

void QProgStored::transformCExpr(CExpr *p_cexpr)
{
    if (nullptr == p_cexpr)
    {
        return;
    }

    transformCExpr(p_cexpr->getLeftExpr());
    transformCExpr(p_cexpr->getRightExpr());
    string cexpr_name = p_cexpr->getName();

    int cexpr_specifier = p_cexpr->getContentSpecifier();
    switch (cexpr_specifier)
    {
    case CBIT:
    {
        string cexpr_number_str = cexpr_name.substr(1);
        uint32_t name_num = std::stoul(cexpr_number_str);
        addDataNode(QPROG_CEXPR_CBIT, name_num);
        addDataNode(QPROG_CEXPR_EVAL, (uint32_t)p_cexpr->eval());
    }
    break;
    case OPERATOR:
    {
        auto iter = m_operator_map.find(cexpr_name);
        if (m_operator_map.end() == iter)
        {
            QCERR("pCExpr is error");
            throw invalid_argument("pCExpr is error");
        }
        uint32_t name_num = (uint32_t)iter->second;
        addDataNode(QPROG_CEXPR_OPERATOR, name_num);
    }
    break;
    case CONSTVALUE:
    {
        addDataNode(QPROG_CEXPR_CONSTVALUE, (uint32_t)p_cexpr->eval());
    }
    break;
    default:
        QCERR("pCExpr is error");
        throw invalid_argument("pCExpr is error");
        break;
    }

    return;
}

void QProgStored::transformClassicalProg(AbstractClassicalProg *cc_pro)
{
    if (nullptr == cc_pro)
    {
        QCERR("AbstractClassicalProg is error");
        throw invalid_argument("AbstractClassicalProg is error");
        return;
    }

    auto expr = dynamic_cast<OriginClassicalProg *>(cc_pro)->getExpr().get();
    transformCExpr(expr);
	uint32_t classical_expr_node = 0;
	addDataNode(QPROG_CEXPR_NODE, classical_expr_node);
}

void QProgStored::handleQGateWithOneAngle(AbstractQGateNode * gate)
{
    if (nullptr == gate)
    {
        QCERR("QGate error");
        throw std::invalid_argument("QGate error");
    }

    QGATE_SPACE::angleParameter * angle;
    angle = dynamic_cast<QGATE_SPACE::angleParameter *>(gate->getQGate());
    if (nullptr == angle)
    {
        QCERR("get angle error");
        throw invalid_argument("get angle error");
    }

    float angle_value = (float)(angle->getParameter());
    addDataNode(QPROG_GATE_ANGLE, angle_value);

    return;
}

void QProgStored::handleQGateWithFourAngle(AbstractQGateNode * gate)
{
    if (nullptr == gate)
    {
        QCERR("QGate error");
        throw std::invalid_argument("QGate error");
    }

    auto quantum_gate = gate->getQGate();
    if (nullptr == quantum_gate)
    {
        QCERR("get Quantum Gate error");
        throw invalid_argument("get Quantum Gate error");
    }

    float alpha, beta, gamma, delta;
    alpha = quantum_gate->getAlpha();
    beta = quantum_gate->getBeta();
    gamma = quantum_gate->getGamma();
    delta = quantum_gate->getDelta();

    addDataNode(QPROG_GATE_ANGLE, alpha);
    addDataNode(QPROG_GATE_ANGLE, beta);
    addDataNode(QPROG_GATE_ANGLE, gamma);
    addDataNode(QPROG_GATE_ANGLE, delta);

    return;
}

void QProgStored::addDataNode(const QProgStoredNodeType &type, const DataNode & data, const bool &is_dagger)
{
    uint32_t type_and_number = 0;
    type_and_number |= is_dagger;
    type_and_number |= (type << 1);
    m_node_counter++;

    if (m_node_counter > kUshortMax)
    {
        QCERR("QNode count is out of range");
        throw invalid_argument("QNode count is out of range");
    }
    type_and_number |= (m_node_counter << kCountMoveBit);
    m_data_vector.emplace_back(pair<uint32_t, DataNode>(type_and_number, data));

    return;
}

void QProgStored::transformQProgByTraversalAlg(QProg *prog)
{
	if (nullptr == prog)
	{
		QCERR("p_prog is null");
		throw runtime_error("p_prog is null");
		return;
	} 

	Traversal::traversalByType(prog->getImplementationPtr(), nullptr, *this);
}

void QProgStored::execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node)
{
	transformQGate(cur_node.get());
}

void QProgStored::execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node)
{
	transformQMeasure(cur_node.get());
}

void QProgStored::execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node)
{
	transformClassicalProg(cur_node.get());
}

void QProgStored::execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node)
{
	transformQControlFlow(cur_node.get());
}

void QProgStored::execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node)
{
	Traversal::traversal(cur_node, *this);
}

void QProgStored::execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node)
{
	QVec ctrl_qubits_vector;
	cur_node->getControlVector(ctrl_qubits_vector);
	if (!ctrl_qubits_vector.empty())
	{
		for (int i = 0; i < ctrl_qubits_vector.size(); i += 2)
		{
			uint32_t qubit_data = 0;
			size_t quibt_addr_1 = 0;
			size_t quibt_addr_2 = 0;
			quibt_addr_1 = ctrl_qubits_vector[i]->getPhysicalQubitPtr()->getQubitAddr();
			if (i + 1 < ctrl_qubits_vector.size())
			{
				quibt_addr_2 = ctrl_qubits_vector[i + 1]->getPhysicalQubitPtr()->getQubitAddr();
				if (0 == quibt_addr_2 )
				{
					size_t addr_tmp = quibt_addr_1;
					quibt_addr_1 = quibt_addr_2;
					quibt_addr_2 = addr_tmp;
				}
			}

			qubit_data |= quibt_addr_1;
			qubit_data |= (quibt_addr_2 << kCountMoveBit);
			addDataNode(QPROG_CONTROL, qubit_data);
		}
	}

	bool bDagger = cur_node->isDagger();
	uint32_t circuit_node = 0;
	addDataNode(QPROG_CIRCUIT_NODE, circuit_node, bDagger);
	auto circuit_node_index = m_data_vector.size() - 1;

	Traversal::traversal(cur_node, false, *this);

	circuit_node |= m_node_counter;
	m_data_vector[circuit_node_index].second.qubit_data = circuit_node;
}


void QPanda::storeQProgInBinary(QProg &prog, QuantumMachine *qm, const string &filename)
{
    QProgStored storeProg(qm);
    storeProg.transform(prog);
    storeProg.store(filename);
}

void QPanda::transformQProgToBinary(QProg &prog, QuantumMachine *qm, const string &filename)
{
	QProgStored storeProg(qm);
	storeProg.transform(prog);
	storeProg.store(filename);
}

std::vector<uint8_t> QPanda::transformQProgToBinary(QProg & prog, QuantumMachine *qm)
{
    QProgStored storeProg(qm);
    storeProg.transform(prog);
    return storeProg.getInsturctions();
}


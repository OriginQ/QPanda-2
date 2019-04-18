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
    {SQISWAP_GATE,  QPROG_SQISWAP_GATE}
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
    transformQProg(&prog);
    return;
}

void QProgStored::store(const string &filename)
{
    std::ofstream out;
    out.open(filename);
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


void QProgStored::transformQProg(AbstractQuantumProgram *p_prog)
{
    if (nullptr == p_prog)
    {
        QCERR("pQProg is null");
        throw invalid_argument("pQProg is null");
    }

    for (auto iter = p_prog->getFirstNodeIter(); iter != p_prog->getEndNodeIter(); iter++)
    {
        QNode * pNode = (*iter).get();
        transformQNode(pNode);
    }

    return;
}


void QProgStored::transformQCircuit(AbstractQuantumCircuit *p_circuit)
{
    if (nullptr == p_circuit)
    {
        QCERR("p_circuit is null");
        throw invalid_argument("p_circuit is null");
    }

    if (p_circuit->isDagger())
    {
        for (auto iter = p_circuit->getLastNodeIter(); iter != p_circuit->getHeadNodeIter(); iter--)
        {
            QNode *p_node = (*iter).get();
            int type = p_node->getNodeType();

            switch (type)
            {
            case NodeType::GATE_NODE:
            {
                AbstractQGateNode *p_gate = dynamic_cast<AbstractQGateNode *>(p_node);
                p_gate->setDagger(true ^ p_gate->isDagger());
            }
            break;
            case NodeType::CIRCUIT_NODE:
            {
                AbstractQuantumCircuit *p_circuit = dynamic_cast<AbstractQuantumCircuit *>(p_node);
                p_circuit->setDagger(true ^ p_circuit->isDagger());
            }
            break;
            case NodeType::MEASURE_GATE:
                break;
            default:
                QCERR("Circuit is error");
                throw invalid_argument("Circuit is error");
                break;
            }
            transformQNode(p_node);
        }
    }
    else
    {
        for (auto iter = p_circuit->getFirstNodeIter(); iter != p_circuit->getEndNodeIter(); iter++)
        {
            QNode * p_node = (*iter).get();
            transformQNode(p_node);
        }
    }

    return;
}


void QProgStored::transformQControlFlow(AbstractControlFlowNode *p_controlflow)
{
    if (nullptr == p_controlflow)
    {
        QCERR("pQControlFlow is null");
        throw invalid_argument("pQControlFlow is null");
    }

    ClassicalCondition *p_classcical_condition = p_controlflow->getCExpr();
    auto expr = p_classcical_condition->getExprPtr().get();
    transformCExpr(expr);

    QNode *p_node = dynamic_cast<QNode *>(p_controlflow);
    int node_type = p_node->getNodeType();

    switch (node_type)
    {
    case NodeType::QIF_START_NODE:
        transformQIfProg(p_controlflow);
        break;
    case NodeType::WHILE_START_NODE:
        transformQWhilePro(p_controlflow);
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
    auto iter_head_node = --m_data_vector.end();
    transformQNode(p_controlFlow->getTrueBranch());

    true_and_false_node |= (m_node_counter << kCountMoveBit);
    transformQNode(p_controlFlow->getFalseBranch());
    true_and_false_node |= m_node_counter;
    iter_head_node->second.qubit_data = true_and_false_node;

    return;
}


void QProgStored::transformQWhilePro(AbstractControlFlowNode *p_controlflow)
{
    if (nullptr == p_controlflow)
    {
        QCERR("p_controlflow is null");
        throw invalid_argument("p_controlflow is null");
    }

    uint32_t true_and_false_node = 0;
    std::cout << "true_and_false_node: " << true_and_false_node << std::endl;
    addDataNode(QPROG_QWHILE_NODE, true_and_false_node);
    auto iter_head_node = --m_data_vector.end();
    transformQNode(p_controlflow->getTrueBranch());

    true_and_false_node |= (m_node_counter << kCountMoveBit);
    iter_head_node->second.qubit_data = true_and_false_node;

    return;
}


void QProgStored::transformQGate(AbstractQGateNode *p_gate)
{
    if (nullptr == p_gate)
    {
        QCERR("pQGate is null");
        throw invalid_argument("pQGate is null");
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
    else if (GateType::U4_GATE == gate_type)
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


void QProgStored::transformQNode(QNode *p_node)
{
    if (nullptr == p_node)
    {
        QCERR("p_node is null");
        throw invalid_argument("p_node is null");
    }

    int type = p_node->getNodeType();
    switch (type)
    {
    case NodeType::GATE_NODE:
        transformQGate(dynamic_cast<AbstractQGateNode *>(p_node));
        break;
    case NodeType::CIRCUIT_NODE:
        transformQCircuit(dynamic_cast<AbstractQuantumCircuit *>(p_node));
        break;
    case NodeType::PROG_NODE:
        transformQProg(dynamic_cast<AbstractQuantumProgram *>(p_node));
        break;
    case NodeType::QIF_START_NODE:
    case NodeType::WHILE_START_NODE:
        QCERR("don't support QIF and QWHILE");
        throw invalid_argument("don't support QIF and QWHILE");
        //traversalQControlFlow(dynamic_cast<AbstractControlFlowNode *>(p_node));
        break;
    case NodeType::MEASURE_GATE:
        transformQMeasure(dynamic_cast<AbstractQuantumMeasure *>(p_node));
        break;
    case NodeType::CLASS_COND_NODE:
        transformClassicalProg(dynamic_cast<AbstractClassicalProg *>(p_node));
        break;
    case NodeType::NODE_UNDEFINED:
        QCERR("NodeType UNDEFINED");
        throw invalid_argument("NodeType UNDEFINED");
        break;
    default:
        QCERR("node type is error");
        throw invalid_argument("node type is error");
        break;
    }

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

void QPanda::storeQProgInBinary(QProg &prog, QuantumMachine *qm, const string &filename)
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

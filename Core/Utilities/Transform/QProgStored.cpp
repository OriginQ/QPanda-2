#include "QProgStored.h"
#include "Utilities/MetadataValidity.h"
#include "TransformDecomposition.h"
using namespace std;
USING_QPANDA
QProgStored::QProgStored(QProg &prog) :
    m_file_length(0u), m_node_counter(0u), m_QProg(prog)
{
    m_gate_type_map.insert(pair<int, string>(PAULI_X_GATE, "X"));
    m_gate_type_map.insert(pair<int, string>(PAULI_Y_GATE, "Y"));
    m_gate_type_map.insert(pair<int, string>(PAULI_Z_GATE, "Z"));

    m_gate_type_map.insert(pair<int, string>(X_HALF_PI, "X1"));
    m_gate_type_map.insert(pair<int, string>(Y_HALF_PI, "Y1"));
    m_gate_type_map.insert(pair<int, string>(Z_HALF_PI, "Z1"));

    m_gate_type_map.insert(pair<int, string>(HADAMARD_GATE, "H"));
    m_gate_type_map.insert(pair<int, string>(T_GATE, "T"));
    m_gate_type_map.insert(pair<int, string>(S_GATE, "S"));

    m_gate_type_map.insert(pair<int, string>(RX_GATE, "RX"));
    m_gate_type_map.insert(pair<int, string>(RY_GATE, "RY"));
    m_gate_type_map.insert(pair<int, string>(RZ_GATE, "RZ"));

    m_gate_type_map.insert(pair<int, string>(U1_GATE, "U1"));
    m_gate_type_map.insert(pair<int, string>(U2_GATE, "U2"));
    m_gate_type_map.insert(pair<int, string>(U3_GATE, "U3"));
    m_gate_type_map.insert(pair<int, string>(U4_GATE, "U4"));

    m_gate_type_map.insert(pair<int, string>(CU_GATE, "CU"));
    m_gate_type_map.insert(pair<int, string>(CNOT_GATE, "CNOT"));
    m_gate_type_map.insert(pair<int, string>(CZ_GATE, "CZ"));
    m_gate_type_map.insert(pair<int, string>(CPHASE_GATE, "CPHASE"));

    m_gate_type_map.insert(pair<int, string>(ISWAP_GATE, "ISWAP"));
    m_gate_type_map.insert(pair<int, string>(SQISWAP_GATE, "SQISWAP"));

    m_operator_map.insert(pair<string, int>("+", PLUS));
    m_operator_map.insert(pair<string, int>("-", MINUS));
    m_operator_map.insert(pair<string, int>("&&", AND));
    m_operator_map.insert(pair<string, int>("||", OR));
    m_operator_map.insert(pair<string, int>("!", NOT));
}

QProgStored::~QProgStored()
{ }


void QProgStored::traversal()
{
    const int kMetadataGateTypeCount = 2;
    vector<vector<string>> valid_gate_matrix(kMetadataGateTypeCount, vector<string>(0));
    vector<vector<string>> gate_matrix(kMetadataGateTypeCount, vector<string>(0));
    vector<vector<int>> adjacent_matrixes;

    gate_matrix[METADATA_SINGLE_GATE].emplace_back(m_gate_type_map[PAULI_X_GATE]);
    gate_matrix[METADATA_SINGLE_GATE].emplace_back(m_gate_type_map[PAULI_Y_GATE]);
    gate_matrix[METADATA_SINGLE_GATE].emplace_back(m_gate_type_map[PAULI_Z_GATE]);

    gate_matrix[METADATA_SINGLE_GATE].emplace_back(m_gate_type_map[X_HALF_PI]);
    gate_matrix[METADATA_SINGLE_GATE].emplace_back(m_gate_type_map[Y_HALF_PI]);
    gate_matrix[METADATA_SINGLE_GATE].emplace_back(m_gate_type_map[Z_HALF_PI]);

    gate_matrix[METADATA_SINGLE_GATE].emplace_back(m_gate_type_map[HADAMARD_GATE]);
    gate_matrix[METADATA_SINGLE_GATE].emplace_back(m_gate_type_map[T_GATE]);
    gate_matrix[METADATA_SINGLE_GATE].emplace_back(m_gate_type_map[S_GATE]);

    gate_matrix[METADATA_SINGLE_GATE].emplace_back(m_gate_type_map[RX_GATE]);
    gate_matrix[METADATA_SINGLE_GATE].emplace_back(m_gate_type_map[RY_GATE]);
    gate_matrix[METADATA_SINGLE_GATE].emplace_back(m_gate_type_map[RZ_GATE]);
    gate_matrix[METADATA_SINGLE_GATE].emplace_back(m_gate_type_map[U1_GATE]);

    gate_matrix[METADATA_DOUBLE_GATE].emplace_back(m_gate_type_map[CNOT_GATE]);
    gate_matrix[METADATA_DOUBLE_GATE].emplace_back(m_gate_type_map[CZ_GATE]);
    gate_matrix[METADATA_DOUBLE_GATE].emplace_back(m_gate_type_map[CPHASE_GATE]);

    gate_matrix[METADATA_DOUBLE_GATE].emplace_back(m_gate_type_map[ISWAP_GATE]);
    gate_matrix[METADATA_DOUBLE_GATE].emplace_back(m_gate_type_map[SQISWAP_GATE]);

    SingleGateTypeValidator::GateType(gate_matrix[METADATA_SINGLE_GATE],
        valid_gate_matrix[METADATA_SINGLE_GATE]);  /* single gate data MetadataValidity */
    DoubleGateTypeValidator::GateType(gate_matrix[METADATA_DOUBLE_GATE],
        valid_gate_matrix[METADATA_DOUBLE_GATE]);  /* double gate data MetadataValidity */
    TransformDecomposition traversalVec(valid_gate_matrix, gate_matrix, adjacent_matrixes);

    AbstractQuantumProgram *p_prog = dynamic_cast<AbstractQuantumProgram *>(&m_QProg);
    traversalVec.TraversalOptimizationMerge(dynamic_cast<QNode *>(p_prog));

    for (auto iter = p_prog->getFirstNodeIter(); iter != p_prog->getEndNodeIter(); iter++)
    {
        QNode * p_node = *iter;
        traversalQNode(p_node);
    }

    m_file_length = 2 * sizeof(uint_t) * (m_node_counter + 1);
    m_data_list.emplace_front(pair<uint_t, DataNode>(m_file_length, m_node_counter));

    return;
}


void QProgStored::store(const string &filename)
{
    FILE *fp = nullptr;

#if defined(_MSC_VER) && (_MSC_VER >= 1400 )
    errno_t errno_number = fopen_s(&fp, filename.c_str(), "wb");
    if (errno_number || !fp)
    {
        QCERR("fopen file failure");
        throw invalid_argument("fopen file failure");
    }
#else
    fp = fopen(filename.c_str(), "wb");
    if (!fp)
    {
        QCERR("fopen file failure");
        throw invalid_argument("fopen file failure");
    }
#endif

    const int kMemNumbe = 1;
    for (auto &val : m_data_list)
    {
        if (kMemNumbe != fwrite((void *)(&val), sizeof(uint_t) + sizeof(DataNode), kMemNumbe, fp))
        {
            QCERR("fwrite file failure");
            throw invalid_argument("fwrite file failure");
        }
    }

    fclose(fp);
    return;
}


void QProgStored::traversalQProg(AbstractQuantumProgram *p_prog)
{
    if (nullptr == p_prog)
    {
        QCERR("pQProg is null");
        throw invalid_argument("pQProg is null");
    }

    for (auto iter = p_prog->getFirstNodeIter(); iter != p_prog->getEndNodeIter(); iter++)
    {
        QNode * pNode = *iter;
        traversalQNode(pNode);
    }

    return;
}


void QProgStored::traversalQCircuit(AbstractQuantumCircuit *p_circuit)
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
            QNode *p_node = *iter;
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
            traversalQNode(p_node);
        }
    }
    else
    {
        for (auto iter = p_circuit->getFirstNodeIter(); iter != p_circuit->getEndNodeIter(); iter++)
        {
            QNode * p_node = *iter;
            traversalQNode(p_node);
        }
    }

    return;
}


void QProgStored::traversalQControlFlow(AbstractControlFlowNode *p_controlflow)
{
    if (nullptr == p_controlflow)
    {
        QCERR("pQControlFlow is null");
        throw invalid_argument("pQControlFlow is null");
    }

    ClassicalCondition *p_classcical_condition = p_controlflow->getCExpr();
    auto expr = p_classcical_condition->getExprPtr().get();
    traversalCExpr(expr);

    QNode *p_node = dynamic_cast<QNode *>(p_controlflow);
    int node_type = p_node->getNodeType();

    switch (node_type)
    {
    case NodeType::QIF_START_NODE:
        traversalQIfProg(p_controlflow);
        break;
    case NodeType::WHILE_START_NODE:
        traversalQWhilePro(p_controlflow);
        break;
    default:
        QCERR("NodeType is error");
        throw invalid_argument("NodeType is error");
        break;
    }

    return;
}


void QProgStored::traversalQIfProg(AbstractControlFlowNode *p_controlFlow)
{
    if (nullptr == p_controlFlow)
    {
        QCERR("p_controlFlow is null");
        throw invalid_argument("p_controlFlow is null");
    }

    m_node_counter++;
    uint_t type_and_number = 0;
    if (m_node_counter > kUshortMax)
    {
        QCERR("Node count is out of range");
        throw invalid_argument("Node count is out of range");
    }

    type_and_number |= (QPROG_NODE_TYPE_QIF_NODE << 1);
    type_and_number |= (m_node_counter << kCountMoveBit);
    uint_t true_and_false_node = 0;
    m_data_list.emplace_back(pair<uint_t, DataNode>(type_and_number, true_and_false_node));

    auto iter_head_node = --m_data_list.end();
    traversalQNode(p_controlFlow->getTrueBranch());
    true_and_false_node |= (m_node_counter << kCountMoveBit);
    traversalQNode(p_controlFlow->getFalseBranch());

    true_and_false_node |= m_node_counter;
    iter_head_node->second.qubit_data = true_and_false_node;

    return;
}


void QProgStored::traversalQWhilePro(AbstractControlFlowNode *p_controlflow)
{
    if (nullptr == p_controlflow)
    {
        QCERR("p_controlflow is null");
        throw invalid_argument("p_controlflow is null");
    }

    m_node_counter++;
    uint_t type_and_number = 0;
    if (m_node_counter > kUshortMax)
    {
        QCERR("Node count is out of range");
        throw invalid_argument("Node count is out of range");
    }

    type_and_number |= (QPROG_NODE_TYPE_QWHILE_NODE << 1);
    type_and_number |= (m_node_counter << kCountMoveBit);
    uint_t true_and_false_node = 0;
    m_data_list.emplace_back(pair<uint_t, DataNode>(type_and_number, true_and_false_node));

    auto iter_head_node = --m_data_list.end();
    traversalQNode(p_controlflow->getTrueBranch());
    true_and_false_node |= (m_node_counter << kCountMoveBit);
    iter_head_node->second.qubit_data = true_and_false_node;

    return;
}


void QProgStored::traversalQGate(AbstractQGateNode *p_gate)
{
    if (nullptr == p_gate)
    {
        QCERR("pQGate is null");
        throw invalid_argument("pQGate is null");
    }

    auto quantum_gate = p_gate->getQGate();
    int gate_type = quantum_gate->getGateType();
    auto iter = m_gate_type_map.find(gate_type);

    if (iter == m_gate_type_map.end())
    {
        QCERR("do not support this gateType");
        throw invalid_argument("do not support this gateType");
    }

    const int kQubitNumberMax = 2;
    vector<Qubit*> qubits;
    p_gate->getQuBitVector(qubits);

    /* only support singleGate and doubleGate */
    if (qubits.size() > kQubitNumberMax)
    {
        QCERR("pQGate is illegal");
        throw invalid_argument("pQGate is illegal");
    }

    ushort_t qubit_array[kQubitNumberMax] = { 0 };
    int qubit_number = 0;
    for (auto qubit : qubits)
    {
        PhysicalQubit *p_physical_qubit = qubit->getPhysicalQubitPtr();
        size_t qubit_addr = p_physical_qubit->getQubitAddr();
        qubit_array[qubit_number] = (ushort_t)qubit_addr;
        qubit_number++;
    }

    uint_t qubit_data = 0;
    qubit_data |= qubit_array[0];
    qubit_data |= (qubit_array[1] << kCountMoveBit);
    uint_t type_and_number = 0;

    if (p_gate->isDagger())
    {
        type_and_number |= 1u;
    }

    switch (gate_type)
    {
    case GateType::PAULI_X_GATE:
        type_and_number |= (QPROG_NODE_TYPE_PAULI_X_GATE << 1);
        break;
    case GateType::PAULI_Y_GATE:
        type_and_number |= (QPROG_NODE_TYPE_PAULI_Y_GATE << 1);
        break;
    case GateType::PAULI_Z_GATE:
        type_and_number |= (QPROG_NODE_TYPE_PAULI_Z_GATE << 1);
        break;
    case GateType::X_HALF_PI:
        type_and_number |= (QPROG_NODE_TYPE_X_HALF_PI << 1);
        break;
    case GateType::Y_HALF_PI:
        type_and_number |= (QPROG_NODE_TYPE_Y_HALF_PI << 1);
        break;
    case GateType::Z_HALF_PI:
        type_and_number |= (QPROG_NODE_TYPE_Z_HALF_PI << 1);
        break;
    case GateType::HADAMARD_GATE:
        type_and_number |= (QPROG_NODE_TYPE_HADAMARD_GATE << 1);
        break;
    case GateType::T_GATE:
        type_and_number |= (QPROG_NODE_TYPE_T_GATE << 1);
        break;
    case GateType::S_GATE:
        type_and_number |= (QPROG_NODE_TYPE_S_GATE << 1);
        break;
    case GateType::RX_GATE:
        type_and_number |= (QPROG_NODE_TYPE_RX_GATE << 1);
        break;
    case GateType::RY_GATE:
        type_and_number |= (QPROG_NODE_TYPE_RY_GATE << 1);
        break;
    case GateType::RZ_GATE:
        type_and_number |= (QPROG_NODE_TYPE_RZ_GATE << 1);
        break;
    case GateType::U1_GATE:
        type_and_number |= (QPROG_NODE_TYPE_U1_GATE << 1);
        break;
    case GateType::U2_GATE:
        type_and_number |= (QPROG_NODE_TYPE_U2_GATE << 1);
        break;
    case GateType::U3_GATE:
        type_and_number |= (QPROG_NODE_TYPE_U3_GATE << 1);
        break;
    case GateType::U4_GATE:
        type_and_number |= (QPROG_NODE_TYPE_U4_GATE << 1);
        break;
    case GateType::CU_GATE:
        type_and_number |= (QPROG_NODE_TYPE_CU_GATE << 1);
        break;
    case GateType::CNOT_GATE:
        type_and_number |= (QPROG_NODE_TYPE_CNOT_GATE << 1);
        break;
    case GateType::CZ_GATE:
        type_and_number |= (QPROG_NODE_TYPE_CZ_GATE << 1);
        break;
    case GateType::CPHASE_GATE:
        type_and_number |= (QPROG_NODE_TYPE_CPHASE_GATE << 1);
        break;
    case GateType::ISWAP_GATE:
        type_and_number |= (QPROG_NODE_TYPE_ISWAP_GATE << 1);
        break;
    case GateType::SQISWAP_GATE:
        type_and_number |= (QPROG_NODE_TYPE_SQISWAP_GATE << 1);
        break;
    default:
        QCERR("do not support this type gate");
        throw invalid_argument("do not support this type gate");
        break;
    }

    m_node_counter++;
    if (m_node_counter > kUshortMax)
    {
        QCERR("QNode count is out of range");
        throw invalid_argument("QNode count is out of range");
    }
    type_and_number |= (m_node_counter << kCountMoveBit);
    m_data_list.emplace_back(pair<uint_t, DataNode>(type_and_number, qubit_data));

    if (GateType::RX_GATE == gate_type || GateType::RY_GATE == gate_type
        || GateType::RZ_GATE == gate_type || GateType::U1_GATE == gate_type
        || GateType::CPHASE_GATE == gate_type)
    {
        angleParameter * angle;
        angle = dynamic_cast<angleParameter *>(p_gate->getQGate());
        if (nullptr == angle)
        {
            QCERR("get angle error");
            throw invalid_argument("get angle error");
        }

        float Angle_value = (float)(angle->getParameter());
        type_and_number = 0;
        type_and_number |= (QPROG_NODE_TYPE_GATE_ANGLE << 1);
        m_node_counter++;

        if (m_node_counter > kUshortMax)
        {
            QCERR("QNode count is out of range");
            throw invalid_argument("QNode count is out of range");
        }
        type_and_number |= (m_node_counter << kCountMoveBit);
        m_data_list.emplace_back(pair<uint_t, DataNode>(type_and_number, Angle_value));
    }

    return;
}

void QProgStored::traversalQMeasure(AbstractQuantumMeasure *p_measure)
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
    ushort_t qubit_array[kQubitNumberMax] = { 0 };

    if (qubit_addr > kUshortMax)
    {
        QCERR("QBit number is out of range");
        throw invalid_argument("QBit number is out of range");
    }
    qubit_array[0] = (ushort_t)qubit_addr;

    if (cbit_number > kUshortMax)
    {
        QCERR("QCit number is out of range");
        throw invalid_argument("QCit number is out of range");
    }

    qubit_array[1] = (ushort_t)cbit_number;
    uint_t qubit_data = 0;
    qubit_data |= qubit_array[0];
    qubit_data |= (qubit_array[1] << kCountMoveBit);

    m_node_counter++;
    uint_t type_and_number = 0;
    type_and_number |= (QPROG_NODE_TYPE_MEASURE_GATE << 1);

    if (m_node_counter > kUshortMax)
    {
        QCERR("QNode counter is out of range");
        throw invalid_argument("QNode counter is out of range");
    }
    type_and_number |= (m_node_counter << kCountMoveBit);
    m_data_list.emplace_back(pair<uint_t, DataNode>(type_and_number, qubit_data));

    return;
}


void QProgStored::traversalQNode(QNode *p_node)
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
        traversalQGate(dynamic_cast<AbstractQGateNode *>(p_node));
        break;
    case NodeType::CIRCUIT_NODE:
        traversalQCircuit(dynamic_cast<AbstractQuantumCircuit *>(p_node));
        break;
    case NodeType::PROG_NODE:
        traversalQProg(dynamic_cast<AbstractQuantumProgram *>(p_node));
        break;
    case NodeType::QIF_START_NODE:
    case NodeType::WHILE_START_NODE:
        traversalQControlFlow(dynamic_cast<AbstractControlFlowNode *>(p_node));
        break;
    case NodeType::MEASURE_GATE:
        traversalQMeasure(dynamic_cast<AbstractQuantumMeasure *>(p_node));
        break;
    case NodeType::NODE_UNDEFINED:
        QCERR("NodeType UNDEFINED");
        throw invalid_argument("NodeType UNDEFINED");
        break;
    default:
        QCERR("p_node is error");
        throw invalid_argument("p_node is error");
        break;
    }

    return;
}


void QProgStored::traversalCExpr(CExpr *p_cexpr)
{
    if (nullptr == p_cexpr)
    {
        return;
    }

    traversalCExpr(p_cexpr->getLeftExpr());
    traversalCExpr(p_cexpr->getRightExpr());

    uint_t data = 0;
    uint_t type_and_number = 0;
    string cbit_name = p_cexpr->getName();

    if (nullptr == p_cexpr->getCBit())
    {
        auto iter = m_operator_map.find(cbit_name);
        if (m_operator_map.end() == iter)
        {
            QCERR("pCExpr is error");
            throw invalid_argument("pCExpr is error");
        }

        data = (uint_t)iter->second;
        type_and_number |= (QPROG_NODE_TYPE_CEXPR_OPERATOR << 1);
    }
    else
    {
        string cbit_number_str = cbit_name.substr(1);
        int cbit_number = stoi(cbit_number_str);
        data = (uint_t)cbit_number;
        type_and_number |= (QPROG_NODE_TYPE_CEXPR_CBIT << 1);
    }

    m_node_counter++;
    if (m_node_counter > kUshortMax)
    {
        QCERR("Node count is out of range");
        throw invalid_argument("Node count is out of range");
    }

    type_and_number |= (m_node_counter << kCountMoveBit);
    m_data_list.emplace_back(pair<uint_t, DataNode>(type_and_number, data));

    return;
}

void QPanda::qProgBinaryStored(QProg &prog, const string &filename)
{
    QProgStored storeProg(prog);
    storeProg.traversal();
    storeProg.store(filename);
}

#include "QProgToQuil.h"
#include <iostream>
#include "QPanda.h"
#include "TransformDecomposition.h"
#include "Utilities/MetadataValidity.h"


USING_QPANDA
using namespace std;
QProgToQuil::QProgToQuil(QuantumMachine * quantum_machine)
{
    m_gate_type_map.insert(pair<int, string>(PAULI_X_GATE, "X"));
    m_gate_type_map.insert(pair<int, string>(PAULI_Y_GATE, "Y"));
    m_gate_type_map.insert(pair<int, string>(PAULI_Z_GATE, "Z"));
    m_gate_type_map.insert(pair<int, string>(HADAMARD_GATE, "H"));

    m_gate_type_map.insert(pair<int, string>(T_GATE, "T"));
    m_gate_type_map.insert(pair<int, string>(S_GATE, "S"));
    m_gate_type_map.insert(pair<int, string>(RX_GATE, "RX"));
    m_gate_type_map.insert(pair<int, string>(RY_GATE, "RY"));

    m_gate_type_map.insert(pair<int, string>(RZ_GATE, "RZ"));
    m_gate_type_map.insert(pair<int, string>(U1_GATE, "PHASE"));   /* U1 --> PHASE */
    m_gate_type_map.insert(pair<int, string>(CU_GATE, "CU"));
    m_gate_type_map.insert(pair<int, string>(CNOT_GATE, "CNOT"));

    m_gate_type_map.insert(pair<int, string>(CZ_GATE, "CZ"));
    m_gate_type_map.insert(pair<int, string>(CPHASE_GATE, "CPHASE"));
    m_gate_type_map.insert(pair<int, string>(ISWAP_GATE, "ISWAP"));
    m_instructs.clear();

    m_quantum_machine = quantum_machine;
}

QProgToQuil::~QProgToQuil()
{
}

void QProgToQuil::transform(QProg &prog)
{
    if (nullptr == m_quantum_machine)
    {
        QCERR("Quantum machine is nullptr");
        throw std::invalid_argument("Quantum machine is nullptr");
    }

    const int kMetadata_gate_type_count = 2;
    vector<vector<string>> valid_gate_matrix(kMetadata_gate_type_count, vector<string>(0));
    vector<vector<string>> gate_matrix(kMetadata_gate_type_count, vector<string>(0));
    vector<vector<int>> adjacent_matrixes;

    gate_matrix[METADATA_SINGLE_GATE].emplace_back(m_gate_type_map[PAULI_X_GATE]);
    gate_matrix[METADATA_SINGLE_GATE].emplace_back(m_gate_type_map[PAULI_Y_GATE]);
    gate_matrix[METADATA_SINGLE_GATE].emplace_back(m_gate_type_map[PAULI_Z_GATE]);
    gate_matrix[METADATA_SINGLE_GATE].emplace_back(m_gate_type_map[HADAMARD_GATE]);

    gate_matrix[METADATA_SINGLE_GATE].emplace_back(m_gate_type_map[T_GATE]);
    gate_matrix[METADATA_SINGLE_GATE].emplace_back(m_gate_type_map[S_GATE]);
    gate_matrix[METADATA_SINGLE_GATE].emplace_back(m_gate_type_map[RX_GATE]);
    gate_matrix[METADATA_SINGLE_GATE].emplace_back(m_gate_type_map[RY_GATE]);

    gate_matrix[METADATA_SINGLE_GATE].emplace_back(m_gate_type_map[RZ_GATE]);
    gate_matrix[METADATA_SINGLE_GATE].emplace_back("U1"); /* QPanda U1 Gate Name */
    gate_matrix[METADATA_DOUBLE_GATE].emplace_back(m_gate_type_map[CU_GATE]);
    gate_matrix[METADATA_DOUBLE_GATE].emplace_back(m_gate_type_map[CNOT_GATE]);

    gate_matrix[METADATA_DOUBLE_GATE].emplace_back(m_gate_type_map[CZ_GATE]);
    gate_matrix[METADATA_DOUBLE_GATE].emplace_back(m_gate_type_map[CPHASE_GATE]);
    gate_matrix[METADATA_DOUBLE_GATE].emplace_back(m_gate_type_map[ISWAP_GATE]);

    SingleGateTypeValidator::GateType(gate_matrix[METADATA_SINGLE_GATE],
                                      valid_gate_matrix[METADATA_SINGLE_GATE]);  /* single gate data MetadataValidity */
    DoubleGateTypeValidator::GateType(gate_matrix[METADATA_DOUBLE_GATE],
                                      valid_gate_matrix[METADATA_DOUBLE_GATE]);  /* double gate data MetadataValidity */
    TransformDecomposition traversal_vec(valid_gate_matrix,gate_matrix,adjacent_matrixes,m_quantum_machine);
    auto p_prog = &prog;
    traversal_vec.TraversalOptimizationMerge(prog);
    transformQProg(p_prog);
}

void QProgToQuil::transformQProg(AbstractQuantumProgram * prog)
{
    if (nullptr == prog)
    {
        QCERR("p_prog is null");
        throw runtime_error("p_prog is null");
    }

    for (auto iter = prog->getFirstNodeIter(); iter != prog->getEndNodeIter(); iter++)
    {
        QNode * p_node = (*iter).get();
        transformQNode(p_node);
    }

    return;
}

string QProgToQuil::getInsturctions()
{
    string instructions;
    for (auto &sInstruct : m_instructs)
    {
        instructions.append(sInstruct).append("\n");
    }
    instructions.erase(instructions.size() - 1);
    
    return instructions;
}


void QProgToQuil::transformQGate(AbstractQGateNode *gate)
{
    if (nullptr == gate)
    {
        QCERR("p_gate is null");
        throw runtime_error("p_gate is null");
    }

    auto circuit = transformQPandaBaseGateToQuilBaseGate(gate);
    for (auto iter = circuit.getFirstNodeIter(); iter != circuit.getEndNodeIter(); iter++)
    {
        QNode * p_node = (*iter).get();
        dealWithQuilGate(dynamic_cast<AbstractQGateNode *>(p_node));
    }

    return;
}


void QProgToQuil::transformQCircuit(AbstractQuantumCircuit *circuit)
{
    if (nullptr == circuit)
    {
        QCERR("p_circuit is null");
        throw runtime_error("p_circuit is null");
    }

    if (circuit->isDagger())
    {
        for (auto iter = circuit->getLastNodeIter(); iter != circuit->getHeadNodeIter(); iter--)
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
                throw runtime_error("Circuit is error");
                break;
            }

            transformQNode(p_node);
        }
    }
    else
    {
        for (auto iter = circuit->getFirstNodeIter(); iter != circuit->getEndNodeIter(); iter++)
        {
            QNode * p_node = (*iter).get();
            transformQNode(p_node);
        }
    }

    return;
}


void QProgToQuil::transformQMeasure(AbstractQuantumMeasure *measure)
{
    if (nullptr == measure)
    {
        QCERR("p_measure is null");
        throw runtime_error("p_measure is null");
    }

    Qubit *p_qubit = measure->getQuBit();
    auto p_physical_qubit = p_qubit->getPhysicalQubitPtr();
    size_t qubit_addr = p_physical_qubit->getQubitAddr();
    string qubit_addr_str = to_string(qubit_addr);

    auto p_cbit = measure->getCBit();
    string cbit_name = p_cbit->getName();
    string cbit_number_str = cbit_name.substr(1);
    string instruct = "MEASURE " + qubit_addr_str + " [" + cbit_number_str + "]";

    m_instructs.emplace_back(instruct);
    return;
}


void QProgToQuil::transformQNode(QNode *node)
{
    if (nullptr == node)
    {
        QCERR("p_node is null");
        throw runtime_error("p_node is null");
    }

    int type = node->getNodeType();

    switch (type)
    {
    case NodeType::GATE_NODE:
        transformQGate(dynamic_cast<AbstractQGateNode *>(node));
        break;
    case NodeType::CIRCUIT_NODE:
        transformQCircuit(dynamic_cast<AbstractQuantumCircuit *>(node));
        break;
    case NodeType::PROG_NODE:
        transformQProg(dynamic_cast<AbstractQuantumProgram *>(node));
        break;
    case NodeType::QIF_START_NODE:
    case NodeType::WHILE_START_NODE:
        QCERR("Don't support QWhileProg or QIfProg");
        throw invalid_argument("Don't support QWhileProg or QIfProg");
        break;
    case NodeType::MEASURE_GATE:
        transformQMeasure(dynamic_cast<AbstractQuantumMeasure *>(node));
        break;
    case NodeType::NODE_UNDEFINED:
        QCERR("NodeType UNDEFINED");
        throw invalid_argument("NodeType UNDEFINED");
        break;
    default:
        QCERR("pNode is error");
        throw invalid_argument("pNode is error");
        break;
    }

    return;
}

void QProgToQuil::transformQControlFlow(AbstractControlFlowNode *controlflow)
{
    throw std::runtime_error("not support control flow");
}


void QProgToQuil::dealWithQuilGate(AbstractQGateNode *p_gate)
{
    if (nullptr == p_gate)
    {
        QCERR("pGate is null");
        throw invalid_argument("pGate is null");
    }

    auto p_quantum_gate = p_gate->getQGate();
    int gate_type = p_quantum_gate->getGateType();
    QVec qubits;
    p_gate->getQuBitVector(qubits);

    auto iter = m_gate_type_map.find(gate_type);
    if (iter == m_gate_type_map.end())
    {
        QCERR("do not support this gateType");
        throw invalid_argument("do not support this gateType");
    }

    string gate_type_str = iter->second;
    string all_qubit_addr_str;

    for (auto qubit : qubits)
    {
        PhysicalQubit *p_physical_qubit = qubit->getPhysicalQubitPtr();
        size_t qubit_addr = p_physical_qubit->getQubitAddr();
        all_qubit_addr_str += " ";
        all_qubit_addr_str += to_string(qubit_addr);
    }

    string instruct;
    angleParameter * p_angle;
    string angle_str;

    switch (gate_type)
    {
    case GateType::PAULI_X_GATE:
    case GateType::PAULI_Y_GATE:
    case GateType::PAULI_Z_GATE:
    case GateType::HADAMARD_GATE:
    case GateType::T_GATE:
    case GateType::S_GATE:
    case GateType::CNOT_GATE:
    case GateType::CZ_GATE:
    case GateType::ISWAP_GATE:
    case GateType::SQISWAP_GATE:
        instruct = gate_type_str + all_qubit_addr_str;
        m_instructs.emplace_back(instruct);
        break;
    case GateType::RX_GATE:
    case GateType::RY_GATE:
    case GateType::RZ_GATE:
    case GateType::U1_GATE:
    case GateType::CPHASE_GATE:
        p_angle = dynamic_cast<angleParameter *>(p_gate->getQGate());
        if (nullptr == p_angle)
        {
            QCERR("dynamic_cast error");
            throw invalid_argument("dynamic_cast error");
        }

        angle_str = to_string(p_angle->getParameter());
        instruct = gate_type_str + "(" + angle_str + ")" + all_qubit_addr_str;
        m_instructs.emplace_back(instruct);
        break;
    default:
        QCERR("do not support this type gate");
        throw invalid_argument("do not support this type gate");
        break;
    }

    return ;
}


QCircuit QProgToQuil::transformQPandaBaseGateToQuilBaseGate(AbstractQGateNode *p_gate)
{
    QVec target_qubits;
    if (p_gate->getQuBitVector(target_qubits) <= 0)
    {
        QCERR("gate is null");
        throw invalid_argument("gate is null");
    }

    QuantumGate* p_quantum_gate = p_gate->getQGate();
    QStat matrix;
    p_quantum_gate->getMatrix(matrix);
    double theta = 0;

    int label = p_gate->isDagger() ? -1 : 1;   /* iLabel is 1 or -1 */
    double alpha = p_quantum_gate->getAlpha();
    double beta = p_quantum_gate->getBeta();
    double delta = p_quantum_gate->getDelta();

    double gamma = p_quantum_gate->getGamma();
    auto qCircuit = CreateEmptyCircuit();
    int gate_type = p_quantum_gate->getGateType();

    switch (gate_type)
    {
    case PAULI_X_GATE:
        qCircuit << X(target_qubits[0]);
        break;
    case PAULI_Y_GATE:
        qCircuit << Y(target_qubits[0]);
        break;
    case PAULI_Z_GATE:
        qCircuit << Z(target_qubits[0]);
        break;
    case X_HALF_PI:
        qCircuit << RX(target_qubits[0], label*PI/2);
        break;
    case Y_HALF_PI:
        qCircuit << RY(target_qubits[0], label*PI / 2);
        break;
    case Z_HALF_PI:
        qCircuit << RZ(target_qubits[0], label*PI / 2);
        break;
    case HADAMARD_GATE:
        qCircuit << H(target_qubits[0]);
        break;
    case T_GATE:
        {
            auto gate = p_gate->isDagger() ? U1(target_qubits[0], label*PI / 4) : T(target_qubits[0]);
            qCircuit << gate;
        }
        break;
    case S_GATE:
        {
            auto gate = p_gate->isDagger() ? U1(target_qubits[0], label*PI / 2) : S(target_qubits[0]);
            qCircuit << gate;
        }
        break;
    case RX_GATE:
        {
            angleParameter  *p_angle = dynamic_cast<angleParameter *>(p_gate->getQGate());
            theta = p_angle->getParameter();
            qCircuit << RX(target_qubits[0], label*theta);
            break;
        }
    case RY_GATE:
        {
            angleParameter * p_angle = dynamic_cast<angleParameter *>(p_gate->getQGate());
            theta = p_angle->getParameter();
            qCircuit << RY(target_qubits[0], label*theta);
            break;
        }
    case RZ_GATE:
        {
            angleParameter * p_angle = dynamic_cast<angleParameter *>(p_gate->getQGate());
            theta = p_angle->getParameter();
            qCircuit << RZ(target_qubits[0], label*theta);
            break;
        }
        case U1_GATE:
        {
            angleParameter * p_angle = dynamic_cast<angleParameter *>(p_gate->getQGate());
            theta = p_angle->getParameter();
            qCircuit << U1(target_qubits[0], label*theta);
            break;
        }
    case U4_GATE:
        {
            if (p_gate->isDagger())
            {
                qCircuit << RZ(target_qubits[0],-p_quantum_gate->getBeta())
                        <<RY(target_qubits[0],-p_quantum_gate->getGamma())
                        <<RZ(target_qubits[0],-p_quantum_gate->getDelta());
            }
            else
            {
                qCircuit << RZ(target_qubits[0],p_quantum_gate->getDelta())
                        <<RY(target_qubits[0],p_quantum_gate->getGamma())
                        <<RZ(target_qubits[0],p_quantum_gate->getBeta());
            }

        }
        break;
    case CU_GATE:
        if (p_gate->isDagger())
        {
            qCircuit << U1(target_qubits[0], -alpha) << RZ(target_qubits[0], -beta)
                << RY(target_qubits[0], -gamma / 2) << CNOT(target_qubits[0], target_qubits[1])
                << RY(target_qubits[0], gamma / 2) << RZ(target_qubits[0], (delta + beta) / 2)
                << CNOT(target_qubits[0], target_qubits[1]) << RZ(target_qubits[0], -(delta - beta) / 2);
        }
        else
        {
            qCircuit << RZ(target_qubits[0], (delta - beta) / 2) << CNOT(target_qubits[0], target_qubits[1])
                << RZ(target_qubits[0], -(delta + beta) / 2) << RY(target_qubits[0], -gamma / 2)
                << CNOT(target_qubits[0], target_qubits[1]) << RY(target_qubits[0], gamma / 2)
                << RZ(target_qubits[0], beta) << U1(target_qubits[0], alpha);
        }
        break;
    case CNOT_GATE:
        qCircuit << CNOT(target_qubits[0], target_qubits[1]);
        break;
    case CZ_GATE:
        qCircuit << CZ(target_qubits[0], target_qubits[1]);
        break;
    case CPHASE_GATE:
        {
            angleParameter * p_angle = dynamic_cast<angleParameter *>(p_gate->getQGate());
            theta = p_angle->getParameter();
            qCircuit << CR(target_qubits[0], target_qubits[1],label*theta);
        }
        break;
    case ISWAP_GATE:
        if (p_gate->isDagger())
        {
            qCircuit << iSWAP(target_qubits[0], target_qubits[1])
                << Z(target_qubits[0]) << Z(target_qubits[1]);
        }
        else
        {
            qCircuit << iSWAP(target_qubits[0], target_qubits[1]);
        }
        break;
    case SQISWAP_GATE:
        {
            theta = PI/4;
            qCircuit << CNOT(target_qubits[1], target_qubits[0])
                    << CZ(target_qubits[0], target_qubits[1])
                    << RX(target_qubits[1], -label * theta)
                    << CZ(target_qubits[0], target_qubits[1])
                    << RX(target_qubits[1], label * theta)
                    << CNOT(target_qubits[1], target_qubits[0]);
        }
        break;
    case ISWAP_THETA_GATE:
        {
            angleParameter * p_angle = dynamic_cast<angleParameter *>(p_gate->getQGate());
            theta = p_angle->getParameter();
            qCircuit << CNOT(target_qubits[1], target_qubits[0])
                << CZ(target_qubits[0], target_qubits[1])
                << RX(target_qubits[1], -label * theta)
                << CZ(target_qubits[0], target_qubits[1])
                << RX(target_qubits[1], label * theta)
                << CNOT(target_qubits[1], target_qubits[0]);
        }
        break;
    case TWO_QUBIT_GATE:
        break;
    default:
        QCERR("unknow error");
        throw runtime_error("unknow error");
    }

    return qCircuit;
}


string QPanda::transformQProgToQuil(QProg& prog, QuantumMachine * quantum_machine)
{
    if (nullptr == quantum_machine)
    {
        QCERR("Quantum machine is nullptr");
        throw std::invalid_argument("Quantum machine is nullptr");
    }

    QProgToQuil quil_traverse(quantum_machine);
    quil_traverse.transform(prog);
    return quil_traverse.getInsturctions();
}



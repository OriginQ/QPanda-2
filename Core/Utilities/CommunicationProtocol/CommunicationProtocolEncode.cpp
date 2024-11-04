#include "QPandaConfig.h"
#include "Core/Utilities/CommunicationProtocol/CommunicationProtocolEncode.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif

USING_QPANDA

static void encode_node_extract(std::shared_ptr<AbstractQGateNode> gate, Qnum &tar_qubits, Qnum &ctr_qubits, bool& is_dagger)
{
    QVec qubits, control_qubits;
    gate->getQuBitVector(qubits);
    gate->getControlVector(control_qubits); 

    is_dagger = gate->isDagger();

    for (auto &val : control_qubits)
        ctr_qubits.emplace_back(val->get_phy_addr());

    for (auto &val : qubits)
        tar_qubits.emplace_back(val->get_phy_addr());

    auto node_type = (GateType)gate->getQGate()->getGateType();
    switch (node_type)
    {
    case QPanda::PAULI_X_GATE:
    case QPanda::PAULI_Z_GATE:
    case QPanda::BARRIER_GATE:

        tar_qubits.insert(tar_qubits.begin(), ctr_qubits.begin(), ctr_qubits.end());
        ctr_qubits.clear();
        break;

    default:
        break;
    }

    return;
}

std::map<GateType, OperationType> CommProtocolEncode::m_encode_gate_map = 
{
       {GateType::I_GATE, OperationType::OP_I},
       {GateType::PAULI_X_GATE, OperationType::OP_PAULI_X},
       {GateType::PAULI_Y_GATE, OperationType::OP_PAULI_Y},
       {GateType::PAULI_Z_GATE, OperationType::OP_PAULI_Z},
       {GateType::X_HALF_PI, OperationType::OP_X_HALF_PI},
       {GateType::Y_HALF_PI, OperationType::OP_Y_HALF_PI},
       {GateType::Z_HALF_PI, OperationType::OP_Z_HALF_PI},
       {GateType::HADAMARD_GATE, OperationType::OP_HADAMARD},
       {GateType::T_GATE, OperationType::OP_T},
       {GateType::S_GATE, OperationType::OP_S},

       {GateType::P_GATE, OperationType::OP_P},
       {GateType::RX_GATE, OperationType::OP_RX},
       {GateType::RY_GATE, OperationType::OP_RY},
       {GateType::RZ_GATE, OperationType::OP_RZ},
       {GateType::U1_GATE, OperationType::OP_U1},
       {GateType::U2_GATE, OperationType::OP_U2},
       {GateType::U3_GATE, OperationType::OP_U3},
       {GateType::U4_GATE, OperationType::OP_U4},
       {GateType::RPHI_GATE, OperationType::OP_RPHI},

       {GateType::CU_GATE, OperationType::OP_CU},
       {GateType::CP_GATE, OperationType::OP_CP},
       {GateType::CPHASE_GATE, OperationType::OP_CPHASE},
       {GateType::RXX_GATE, OperationType::OP_RXX},
       {GateType::RYY_GATE, OperationType::OP_RYY},
       {GateType::RZZ_GATE, OperationType::OP_RZZ},
       {GateType::RZX_GATE, OperationType::OP_RZX},

       {GateType::CNOT_GATE, OperationType::OP_CNOT},
       {GateType::CZ_GATE, OperationType::OP_CZ},
       {GateType::MS_GATE, OperationType::OP_MS},
       {GateType::ISWAP_GATE, OperationType::OP_ISWAP},
       {GateType::SWAP_GATE, OperationType::OP_SWAP},
       {GateType::SQISWAP_GATE, OperationType::OP_SQISWAP},

       {GateType::TOFFOLI_GATE, OperationType::OP_TOFFOLI},
       {GateType::ECHO_GATE, OperationType::OP_ECHO},
};


void CommProtocolEncode::execute(std::shared_ptr<AbstractQuantumMeasure>  measure_node,
    std::shared_ptr<QNode> parent_node, QCircuitConfig &config)
{
    //get qubit addr & cbit addr
    auto qbit_addr = (size_t)measure_node->getQuBit()->get_phy_addr();
    auto cbit_addr = (size_t)measure_node->getCBit()->get_addr();

    QMeasureEncode measure_data(OperationType::OP_MEASURE, { qbit_addr }, { cbit_addr });
    node_encode(measure_data);

    return;
}

void CommProtocolEncode::execute(std::shared_ptr<AbstractControlFlowNode> cur_node,
    std::shared_ptr<QNode> parent_node, QCircuitConfig &config)
{
    QCERR_AND_THROW(std::runtime_error, "not support control flow")
}

void CommProtocolEncode::execute(std::shared_ptr<AbstractQNoiseNode>, std::shared_ptr<QNode>, QCircuitConfig &config)
{
    QCERR_AND_THROW(std::runtime_error, "not support virtual nosie node")
}

void CommProtocolEncode::execute(std::shared_ptr<AbstractQDebugNode>, std::shared_ptr<QNode>, QCircuitConfig &config)
{
    QCERR_AND_THROW(std::runtime_error, "not support debug node")
}


void CommProtocolEncode::execute(std::shared_ptr<AbstractQuantumCircuit> cur_node,
    std::shared_ptr<QNode> parent_node, QCircuitConfig &config)
{
    if (nullptr == cur_node)
        QCERR_AND_THROW(std::invalid_argument, "QCircuit is nullptr");

    auto aiter = cur_node->getFirstNodeIter();

    if (aiter == cur_node->getEndNodeIter())
        return;

    auto pNode = std::dynamic_pointer_cast<QNode>(cur_node);

    if (nullptr == pNode)
        QCERR_AND_THROW(std::runtime_error, "unknown internal error");

    QCircuitConfig before_config = config;
    config._is_dagger = cur_node->isDagger() ^ config._is_dagger;
    QVec ctrl_qubits;
    cur_node->getControlVector(ctrl_qubits);

    size_t before_size = config._contorls.size();
    config._contorls.insert(config._contorls.end(), ctrl_qubits.begin(), ctrl_qubits.end());

    if (config._is_dagger)
    {
        auto aiter = cur_node->getLastNodeIter();
        if (nullptr == *aiter)
            return;

        while (aiter != cur_node->getHeadNodeIter())
        {
            if (aiter == nullptr)
                break;

            Traversal::traversalByType(*aiter, pNode, *this, config);
            --aiter;
        }
    }
    else
    {
        auto aiter = cur_node->getFirstNodeIter();
        while (aiter != cur_node->getEndNodeIter())
        {
            auto next = aiter.getNextIter();
            Traversal::traversalByType(*aiter, pNode, *this, config);
            aiter = next;
        }
    }

    config = before_config;
}

void CommProtocolEncode::execute(std::shared_ptr<AbstractQuantumProgram>  cur_node,
    std::shared_ptr<QNode> parent_node, QCircuitConfig &config)
{
    Traversal::traversal(cur_node, *this, config);
}

void CommProtocolEncode::execute(std::shared_ptr<AbstractClassicalProg>  cur_node,
    std::shared_ptr<QNode> parent_node, QCircuitConfig &config)
{
    QCERR_AND_THROW(std::runtime_error, "not support ClassicalProg");
}

void CommProtocolEncode::execute(std::shared_ptr<AbstractQuantumReset>  cur_node,
    std::shared_ptr<QNode> parent_node, QCircuitConfig &config)
{
    QCERR_AND_THROW(std::runtime_error, "not support Reset");
}

void CommProtocolEncode::execute(std::shared_ptr<AbstractQGateNode> gate_node, std::shared_ptr<QNode> parent_node, QCircuitConfig &config)
{
    Qnum tar_qubits, ctr_qubits;
    bool is_dagger = false;

    encode_node_extract(gate_node, tar_qubits, ctr_qubits, is_dagger);
    is_dagger = is_dagger ^ config._is_dagger;

    if (!ctr_qubits.empty())
        QCERR_AND_THROW(std::runtime_error, "encode unsupported control qubits.")

    auto encode_qubits = ctr_qubits;
    encode_qubits.insert(encode_qubits.end(), tar_qubits.begin(), tar_qubits.end());

    auto node_type = (GateType)gate_node->getQGate()->getGateType();
    switch (node_type)
    {
    //single gate with no angle
    case GateType::PAULI_X_GATE:
    {
        if (encode_qubits.size() == 3)
            node_type = GateType::TOFFOLI_GATE;
    }

    case GateType::I_GATE:
    case GateType::ECHO_GATE:
    case GateType::PAULI_Y_GATE:
    case GateType::PAULI_Z_GATE:
    case GateType::X_HALF_PI:
    case GateType::Y_HALF_PI:
    case GateType::Z_HALF_PI:
    case GateType::HADAMARD_GATE:
    case GateType::T_GATE:
    case GateType::S_GATE:

    //double gate with no angle
    case GateType::CNOT_GATE:
    case GateType::CZ_GATE:
    case GateType::MS_GATE:
    case GateType::SWAP_GATE:
    case GateType::ISWAP_GATE:
    case GateType::SQISWAP_GATE:
    case GateType::TWO_QUBIT_GATE:

    //triple gate with no angle
    case GateType::TOFFOLI_GATE:
    {
        QGateEncode gate_data(m_encode_gate_map[node_type], is_dagger, encode_qubits, {});
        node_encode(gate_data);
        break;
    }

    case GateType::BARRIER_GATE:
    {
        QBarrierEncode barrier_data(OperationType::OP_BARRIER, encode_qubits);
        node_encode(barrier_data);
        break;
    }

    //single gate with single angle
    case GateType::P_GATE:
    case GateType::RX_GATE:
    case GateType::RY_GATE:
    case GateType::RZ_GATE:
    case GateType::U1_GATE:

    //double gate with single angle
    case GateType::CP_GATE:
    case GateType::CPHASE_GATE:
    case GateType::RXX_GATE:
    case GateType::RYY_GATE:
    case GateType::RZZ_GATE:
    case GateType::RZX_GATE:
    {
        auto gate_param = dynamic_cast<QGATE_SPACE::AbstractSingleAngleParameter*>(gate_node->getQGate());
        auto param_data = convert_params({ gate_param->getParameter() });
        QGateEncode gate_data(m_encode_gate_map[node_type], is_dagger, encode_qubits, param_data);
        node_encode(gate_data);
        break;
    }

    case GateType::RPHI_GATE:
    {
        auto u4 = dynamic_cast<QGATE_SPACE::U4*>(gate_node->getQGate());
        auto rphi = dynamic_cast<QGATE_SPACE::RPhi*>(gate_node->getQGate());

        auto param_data = convert_params({ u4->getBeta(), rphi->get_phi() });

        QGateEncode gate_data(m_encode_gate_map[node_type], is_dagger, encode_qubits, param_data);
        node_encode(gate_data);
        break;
    }

    case GateType::U2_GATE:
    {
        auto u2 = dynamic_cast<QGATE_SPACE::U2*>(gate_node->getQGate());
        auto param_data = convert_params({ u2->get_phi(), u2->get_lambda() });

        QGateEncode gate_data(m_encode_gate_map[node_type], is_dagger, encode_qubits, param_data);
        node_encode(gate_data);
        break;
    }

    case GateType::U3_GATE:
    {
        auto u3 = dynamic_cast<QGATE_SPACE::U3*>(gate_node->getQGate());

        prob_vec params(3);
        params[0] = u3->get_theta();
        params[1] = u3->get_phi();
        params[2] = u3->get_lambda();

        auto param_data = convert_params(params);

        QGateEncode gate_data(m_encode_gate_map[node_type], is_dagger, encode_qubits, param_data);
        node_encode(gate_data);
        break;
    }

    case GateType::CU_GATE:
    {
        auto cu = dynamic_cast<AbstractAngleParameter*>(gate_node->getQGate());

        prob_vec params(4);
        params[0] = cu->getAlpha();
        params[1] = cu->getBeta();
        params[2] = cu->getGamma();
        params[3] = cu->getDelta();

        auto param_data = convert_params(params);

        QGateEncode gate_data(m_encode_gate_map[node_type], is_dagger, encode_qubits, param_data);
        node_encode(gate_data);
        break;
    }

    case GateType::U4_GATE:
    {
        auto u4 = dynamic_cast<QGATE_SPACE::U4 *>(gate_node->getQGate());

        prob_vec params(4);
        params[0] = u4->getAlpha();
        params[1] = u4->getBeta();
        params[2] = u4->getGamma();
        params[3] = u4->getDelta();

        auto param_data = convert_params(params);

        QGateEncode gate_data(m_encode_gate_map[node_type], is_dagger, encode_qubits, param_data);
        node_encode(gate_data);
        break;
    }

    default: QCERR_AND_THROW(std::runtime_error, "invalid GateType for comm protocol encode : " + std::to_string(node_type));

    } /*end switch*/

    return;
}

void CommProtocolEncode::encode(QProg& prog)
{
    if (prog.is_empty())
        return;

    std::vector<int> qbits, cbits;
    auto qbits_num = get_all_used_qubits(prog, qbits);
    auto cbits_num = get_all_used_class_bits(prog, cbits);

    auto init_data = InitializationEncode(OperationType::OP_INITIALIZATION, qbits_num, cbits_num);
    node_encode(init_data);

    QCircuitConfig config;
    config._is_dagger = false;
    config._contorls.clear();
    config._can_optimize_measure = false;

    execute(prog.getImplementationPtr(), nullptr, config);

    auto circuit_end_sign = ((uint64_t)0 | static_cast<uint64_t>(OperationType::CIRCUIT_END_SIGN));
    m_protocol_encode_data.emplace_back(circuit_end_sign);
    return;
}

void CommProtocolEncode::encode(CommProtocolConfig config)
{
    ConfigurationEncode config_data(config);
    node_encode(config_data);
    return;
}

static void varint_encode(uint64_t value, std::vector<char>& result)
{
    do {
        char byte = value & 0x7F;
        value >>= 7;

        if (value != 0)
            byte |= 0x80;

        result.emplace_back(byte);

    } while (value != 0);

    return;
}

std::vector<char> CommProtocolEncode::convert_to_char()
{
    std::vector<char> encoded_data;
    for (const auto& data : m_protocol_encode_data)
        varint_encode(data, encoded_data);

    return encoded_data; 
}

void CommProtocolEncode::encode_crc()
{
    uint64_t crc_code = 0;

    for (const auto& value : m_protocol_encode_data)
        crc_code ^= value;

    m_protocol_encode_data.emplace_back(crc_code);
    return;
}

std::vector<char> QPanda::comm_protocol_encode(QProg program, CommProtocolConfig config)
{
    CommProtocolEncode protocol_data;
    protocol_data.encode(config);
    protocol_data.encode(program);
    protocol_data.encode_crc();
    return protocol_data.convert_to_char();
}

std::vector<char> QPanda::comm_protocol_encode(std::vector<QProg> prog_list, CommProtocolConfig config)
{
    ConfigurationEncode config_data(config);
    auto config_encode_data = config_data.encode();
    if (config_encode_data.empty())
        QCERR_AND_THROW(std::runtime_error, "ConfigurationEncode error.");

    auto circuits_num = prog_list.size();
    std::vector<std::vector<uint64_t>> encode_data_list(circuits_num + 1);

#ifdef USE_OPENMP

    int max_threads = omp_get_max_threads();
    int actual_threads = std::min(max_threads, static_cast<int>(circuits_num));

#pragma omp parallel for num_threads(actual_threads)

#endif

    for (int i = 0; i < circuits_num; i++)
    {
        CommProtocolEncode protocol_data;
        protocol_data.encode(prog_list[i]);
        encode_data_list[i + 1] = protocol_data.data();
    }

    encode_data_list[0] = config_encode_data;

    uint64_t crc_code = 0;
    for (const auto& data_list : encode_data_list)
        for (const auto& data : data_list)
            crc_code ^= data;

    encode_data_list.emplace_back(std::vector<uint64_t>{ crc_code });

    std::vector<char> encoded_data;
    for (const auto& data_list : encode_data_list)
        for (const auto& data : data_list)
            varint_encode(data, encoded_data);

    return encoded_data;
}


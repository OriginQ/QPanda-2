#include "Core/Utilities/CommunicationProtocol/CommunicationProtocolEncode.h"
#include "Core/Utilities/CommunicationProtocol/CommunicationProtocolDecode.h"

#include "QPandaConfig.h"

#ifdef USE_OPENMP
    #include <omp.h>
#endif

USING_QPANDA

static uint64_t extract_bits(uint64_t value, int right_most_index, int num_bits) 
{
    uint64_t mask = ((1ULL << num_bits) - 1) << right_most_index;
    uint64_t extracted_value = (value & mask) >> right_most_index;
    return extracted_value;
}

CommProtocolDecode::CommProtocolDecode(QuantumMachine* machine)
{
    m_node_decode_imp.init(machine);
}

void CommProtocolDecode::decode_operation(const std::vector<uint64_t>& operation_data, QProg& prog)
{
    auto op_type = static_cast<OperationType>(extract_bits(operation_data[0], 0, 6));
    
    bool is_dagger = (bool)m_node_decode_imp.decode_values(operation_data[0], 7, 1);
    
    switch (op_type)
    {
    case QPanda::OperationType::OP_UNDEFINED:
        QCERR_AND_THROW(std::runtime_error, "OperationType::OP_UNDEFINED error.");
        break;

    case QPanda::OperationType::OP_I:
    case QPanda::OperationType::OP_PAULI_X:
    case QPanda::OperationType::OP_PAULI_Y:
    case QPanda::OperationType::OP_PAULI_Z:
    case QPanda::OperationType::OP_X_HALF_PI:
    case QPanda::OperationType::OP_Y_HALF_PI:
    case QPanda::OperationType::OP_Z_HALF_PI:
    case QPanda::OperationType::OP_HADAMARD:
    case QPanda::OperationType::OP_T:
    case QPanda::OperationType::OP_S:
    case QPanda::OperationType::OP_ECHO:
    {
        auto qubit_addr = m_node_decode_imp.decode_values(operation_data[0], 8, 8);
        auto node = m_node_decode_imp.decode_single_gate(op_type, qubit_addr);
        node.setDagger(is_dagger);
        prog << node;
        break;
    }

    case QPanda::OperationType::OP_P:
    case QPanda::OperationType::OP_RX:
    case QPanda::OperationType::OP_RY:
    case QPanda::OperationType::OP_RZ:
    case QPanda::OperationType::OP_RPHI:
    case QPanda::OperationType::OP_U1:
    case QPanda::OperationType::OP_U2:
    case QPanda::OperationType::OP_U3:
    case QPanda::OperationType::OP_U4:
    {
        auto qubit_addr = m_node_decode_imp.decode_values(operation_data[0], 8, 8);
        auto qubit_parm = m_node_decode_imp.decode_params(operation_data, op_type);

        auto node = m_node_decode_imp.decode_single_gate_with_angle(op_type, qubit_addr, qubit_parm);
        node.setDagger(is_dagger);
        prog << node;
        break;
    }

    case QPanda::OperationType::OP_CNOT:
    case QPanda::OperationType::OP_CZ:
    case QPanda::OperationType::OP_MS:
    case QPanda::OperationType::OP_ISWAP:
    case QPanda::OperationType::OP_SQISWAP:
    case QPanda::OperationType::OP_SWAP:
    {
        auto target_qubit = m_node_decode_imp.decode_values(operation_data[0], 16, 8);
        auto control_qubit = m_node_decode_imp.decode_values(operation_data[0], 8, 8);
        
        auto node = m_node_decode_imp.decode_double_gate(op_type, control_qubit, target_qubit);
        node.setDagger(is_dagger);
        prog << node;
        break;
    }

    break;
    case QPanda::OperationType::OP_CP:
    case QPanda::OperationType::OP_CU:
    case QPanda::OperationType::OP_RYY:
    case QPanda::OperationType::OP_RXX:
    case QPanda::OperationType::OP_RZZ:
    case QPanda::OperationType::OP_RZX:
    case QPanda::OperationType::OP_CPHASE:
    {
        auto target_qubit = m_node_decode_imp.decode_values(operation_data[0], 16, 8);
        auto control_qubit = m_node_decode_imp.decode_values(operation_data[0], 8, 8);
        auto qubit_parm = m_node_decode_imp.decode_params(operation_data, op_type);
        
        auto node = m_node_decode_imp.decode_double_gate_with_angle(op_type, control_qubit, target_qubit, qubit_parm);
        node.setDagger(is_dagger);
        prog << node;
        break;
    }

    break;
    case QPanda::OperationType::OP_TOFFOLI:
    {
        auto qubit_addrs = m_node_decode_imp.decode_values(operation_data[0], {8, 16, 24}, 8);
        auto node = m_node_decode_imp.decode_multi_control_gate(OperationType::OP_TOFFOLI, qubit_addrs);
        node.setDagger(is_dagger);
        prog << node;
        break;
    }

    case QPanda::OperationType::OP_BARRIER:
    {
        std::vector<uint64_t> barrier_qubits = {};
        for (auto i = 0; i < operation_data.size(); ++i)
        {
            auto qubits_num = m_node_decode_imp.decode_values(operation_data[i], 8, 8);

            Qnum insert_indices(qubits_num, 16);
            for (size_t counts = 0; counts < qubits_num; counts++)
                insert_indices[counts] += counts * 8;

            auto qubits = m_node_decode_imp.decode_values(operation_data[i], insert_indices, 8);
            barrier_qubits.insert(barrier_qubits.end(), qubits.begin(), qubits.end());
        }

        prog << m_node_decode_imp.decode_barrier(barrier_qubits);

        break;
    }
    case QPanda::OperationType::OP_MEASURE:
    {
        auto qbit = m_node_decode_imp.decode_values(operation_data[0], 16, 8);
        auto cbit = m_node_decode_imp.decode_values(operation_data[0], 24, 8);
        prog << m_node_decode_imp.decode_measure(qbit, cbit);
        break;
    }

    default:
        QCERR_AND_THROW(std::runtime_error, "OperationType Decode error.");
        break;
    }

}


void CommProtocolDecode::decode_program()
{
    if (m_origin_data.size() < 2)
        QCERR_AND_THROW(std::runtime_error, "decode data length error.");

    std::vector<size_t> init_data_indices, prog_end_indices;
    for (size_t i = 0; i < m_origin_data.size() - 1; i++)
    {
        auto op_type = static_cast<OperationType>(extract_bits(m_origin_data[i], 0, 6));

        if (op_type == OperationType::OP_INITIALIZATION)
            init_data_indices.emplace_back(i);

        if (op_type == OperationType::CIRCUIT_END_SIGN)
            prog_end_indices.emplace_back(i);
    }

    if (init_data_indices.size() != prog_end_indices.size())
        QCERR_AND_THROW(std::runtime_error, "decode data init error.");

    auto circuits_num = init_data_indices.size();

    for (size_t i = 0; i < circuits_num; i++)
        m_decode_progs.emplace_back(QProg());

#ifdef USE_OPENMP

    int max_threads = omp_get_max_threads();
    int actual_threads = std::min(max_threads, static_cast<int>(circuits_num));

#pragma omp parallel for num_threads(actual_threads)

#endif

    //Batch QProg [circuits_num]
    for (int cir_index = 0; cir_index < circuits_num; cir_index++)
    {
        //auto init_data = m_origin_data[init_data_indices[cir_index]];

        auto iter_start = m_origin_data.begin() + init_data_indices[cir_index] + 1;
        auto iter_end   = m_origin_data.begin() + prog_end_indices[cir_index];

        auto prog_data = std::vector<uint64_t>(iter_start, iter_end);

        //Full QProg data
        for (size_t i = 0; i < prog_data.size(); i++)
        {
            //Single Operation , Takes  1 or 2 uint64_t
            std::vector<uint64_t> sub_operation_data = { prog_data[i] };

            //Select Resume data
            if (extract_bits(prog_data[i], 6, 1))
            {
                size_t resume_end_iter = i + 1;
                for (; resume_end_iter < prog_data.size(); resume_end_iter++)
                {
                    sub_operation_data.emplace_back(prog_data[resume_end_iter]);
                    if (!extract_bits(prog_data[resume_end_iter], 6, 1))
                    {
                        i = resume_end_iter + 1;
                        break;
                    }
                }

                i = resume_end_iter;
            }

            decode_operation(sub_operation_data, m_decode_progs[cir_index]);
        }
    }

    return;
}

void CommProtocolDecode::get_init_progs()
{
    if (m_origin_data.size() < 2)
        QCERR_AND_THROW(std::runtime_error, "decode data length error.");

    //std::vector<size_t> init_data_indices, prog_end_indices;
    for (size_t i = 0; i < m_origin_data.size() - 1; i++)
    {
        auto op_type = static_cast<OperationType>(extract_bits(m_origin_data[i], 0, 6));

        if (op_type == OperationType::OP_INITIALIZATION)
            init_data_indices.emplace_back(i);

        if (op_type == OperationType::CIRCUIT_END_SIGN)
            prog_end_indices.emplace_back(i);
    }

    if (init_data_indices.size() != prog_end_indices.size())
        QCERR_AND_THROW(std::runtime_error, "decode data init error.");

    auto circuits_num = init_data_indices.size();

    for (size_t i = 0; i < circuits_num; i++)
        m_decode_progs.emplace_back(QProg());
}
void CommProtocolDecode::decode_single_program(uint32_t index)
{
    auto iter_start = m_origin_data.begin() + init_data_indices[index] + 1;
    auto iter_end   = m_origin_data.begin() + prog_end_indices[index];

    auto prog_data = std::vector<uint64_t>(iter_start, iter_end);

    //Full QProg data
    for (size_t i = 0; i < prog_data.size(); i++)
    {
        //Single Operation , Takes  1 or 2 uint64_t
        std::vector<uint64_t> sub_operation_data ={prog_data[i]};

        //Select Resume data
        if (extract_bits(prog_data[i], 6, 1))
        {
            size_t resume_end_iter = i + 1;
            for (; resume_end_iter < prog_data.size(); resume_end_iter++)
            {
                sub_operation_data.emplace_back(prog_data[resume_end_iter]);
                if (!extract_bits(prog_data[resume_end_iter], 6, 1))
                {
                    i = resume_end_iter + 1;
                    break;
                }
            }

            i = resume_end_iter;
        }

        decode_operation(sub_operation_data, m_decode_progs[index]);
    }

    return;
}

void CommProtocolDecode::decode_configuration(CommProtocolConfig& config)
{
    if (m_origin_data.empty())
        QCERR_AND_THROW(std::runtime_error, "decode CommProtocolConfig data error");

    auto op_type = (OperationType)extract_bits(m_origin_data[0], 0, 6);
    if (op_type != OperationType::OP_CONFIGURATION)
        QCERR_AND_THROW(std::runtime_error, "decode CommProtocolConfig operation error");

    config.open_mapping = static_cast<bool>(extract_bits(m_origin_data[0], 6, 1));
    config.open_error_mitigation = static_cast<bool>(extract_bits(m_origin_data[0], 7, 1));
    config.optimization_level = static_cast<uint8_t>(extract_bits(m_origin_data[0], 8, 4));
    config.circuits_num = static_cast<uint8_t>(extract_bits(m_origin_data[0], 16, 16));
    config.shots = static_cast<uint32_t>(extract_bits(m_origin_data[0], 32, 32));

    return;
}

static uint64_t varint_decode(const std::vector<char>& varint_data)
{
    uint64_t result = 0;

    uint32_t shift = 0;
    for (size_t i = 0; i < varint_data.size(); ++i)
    {
        result |= static_cast<uint64_t>(varint_data[i] & 0x7F) << shift;
        shift += 7;
        if ((varint_data[i] & 0x80) == 0)
            break;
    }

    return result;
}

void CommProtocolDecode::load(const std::vector<char>& origin_data)
{
    //m_origin_data = varint_decode(origin_data);

    m_origin_data.clear();

    size_t index = 0;
    while (index < origin_data.size())
    {
        size_t varint_length = 0;

        while (origin_data[index + varint_length] & 0x80)
            ++varint_length;

        ++varint_length;

        std::vector<char> varint_chunk(origin_data.begin() + index, origin_data.begin() + index + varint_length);

        m_origin_data.emplace_back(varint_decode(varint_chunk));
        index += varint_length;
    }

    // Verify CRC code
    if (m_origin_data.size() < 2)
        QCERR_AND_THROW(std::runtime_error, "decode verify data error");

    uint64_t check_val = 0;
    for (size_t i = 0; i < m_origin_data.size() - 1; ++i)
        check_val ^= m_origin_data[i];

    if (check_val != m_origin_data.back())
        QCERR_AND_THROW(std::runtime_error, "decode verify value error");

    return;
}


std::vector<QProg> QPanda::comm_protocol_decode(CommProtocolConfig& config, const std::vector<char>& encode_data, QuantumMachine* machine)
{
    CommProtocolDecode protocol_data(machine);
    protocol_data.load(encode_data);
    protocol_data.decode_configuration(config);
    protocol_data.decode_program();
    return protocol_data.get_decode_progs();
}
QPanda::CommProtocolDecode QPanda::comm_protocol_devide(CommProtocolConfig& config, const std::vector<char>& encode_data, QuantumMachine* machine)
{
    CommProtocolDecode protocol_data(machine);
    protocol_data.load(encode_data);
    protocol_data.decode_configuration(config);
    protocol_data.get_init_progs();
    return protocol_data;
}

QProg QPanda::comm_protocol_single_decode(QPanda::CommProtocolDecode& protocol_data, uint32_t index)
{
    protocol_data.decode_single_program(index);
    auto progs = protocol_data.get_decode_progs();
    return progs[index];
}

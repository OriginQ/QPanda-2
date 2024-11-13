#pragma once

#include "Core/Core.h"
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/Utilities/CommunicationProtocol/ProtocolDefinitions.h"

QPANDA_BEGIN

class ProtocolNodeDecode
{
public:

    void init(QuantumMachine* machine)
    {
        m_machine_ptr = machine;
    }

    QuantumMachine* m_machine_ptr;

    QMeasure decode_measure(size_t qubit_addr, size_t cbit_addr)
    {
        auto qubit = m_machine_ptr->allocateQubitThroughPhyAddress(qubit_addr);
        auto cbit = m_machine_ptr->allocateCBit(cbit_addr);

        return Measure(qubit, cbit);
    }

    QGate decode_single_gate(OperationType type, size_t qubit_addr)
    {
        auto qubit = m_machine_ptr->allocateQubitThroughPhyAddress(qubit_addr);

        switch (type)
        {
        case QPanda::OperationType::OP_I:
            return I(qubit);

        case QPanda::OperationType::OP_PAULI_X:
            return X(qubit);

        case QPanda::OperationType::OP_PAULI_Y:
            return Y(qubit);

        case QPanda::OperationType::OP_PAULI_Z:
            return Z(qubit);

        case QPanda::OperationType::OP_X_HALF_PI:
            return X1(qubit);

        case QPanda::OperationType::OP_Y_HALF_PI:
            return Y1(qubit);

        case QPanda::OperationType::OP_Z_HALF_PI:
            return Z1(qubit);

        case QPanda::OperationType::OP_HADAMARD:
            return H(qubit);

        case QPanda::OperationType::OP_T:
            return T(qubit);

        case QPanda::OperationType::OP_S:
            return S(qubit);

        case QPanda::OperationType::OP_ECHO:
            return ECHO(qubit);

        default:
            QCERR_AND_THROW(std::runtime_error, "decode quantum single gate data error.");
        }

    }
    
    QGate decode_double_gate(OperationType type, size_t ctr_addr, size_t tar_addr)
    {
        auto ctr_qubit = m_machine_ptr->allocateQubitThroughPhyAddress(ctr_addr);
        auto tar_qubit = m_machine_ptr->allocateQubitThroughPhyAddress(tar_addr);

        switch (type)
        {
        case QPanda::OperationType::OP_CNOT:
            return CNOT(ctr_qubit, tar_qubit);

        case QPanda::OperationType::OP_CZ:
            return CZ(ctr_qubit, tar_qubit);

        case QPanda::OperationType::OP_MS:
            return MS(ctr_qubit, tar_qubit);

        case QPanda::OperationType::OP_ISWAP:
            return iSWAP(ctr_qubit, tar_qubit);

        case QPanda::OperationType::OP_SQISWAP:
            return SqiSWAP(ctr_qubit, tar_qubit);

        case QPanda::OperationType::OP_SWAP:
            return SWAP(ctr_qubit, tar_qubit);

        default:
            auto error_message = "decode double gate OperationType error : " + std::to_string((size_t)type);
            QCERR_AND_THROW(std::runtime_error, error_message);
        }

    }

    QGate decode_single_gate_with_angle(OperationType type, size_t qubit_addr, prob_vec params)
    {
        if (params.empty())
            QCERR_AND_THROW(std::runtime_error, "decode quantum single gate with angle data error.");

        auto qubit = m_machine_ptr->allocateQubitThroughPhyAddress(qubit_addr);

        switch (type)
        {

        case QPanda::OperationType::OP_P:
            return P(qubit, params[0]);

        case QPanda::OperationType::OP_RX:
            return RX(qubit, params[0]);

        case QPanda::OperationType::OP_RY:
            return RY(qubit, params[0]);

        case QPanda::OperationType::OP_RZ:
            return RZ(qubit, params[0]);

        case QPanda::OperationType::OP_RPHI:
            return RPhi(qubit, params[0], params[1]);

        case QPanda::OperationType::OP_U1:
            return U1(qubit, params[0]);

        case QPanda::OperationType::OP_U2:
            return U2(qubit, params[0], params[1]);

        case QPanda::OperationType::OP_U3:
            return U3(qubit, params[0], params[1], params[2]);

        case QPanda::OperationType::OP_U4:
            return U4(qubit, params[0], params[1], params[2], params[3]);

        default:
            auto error_message = "decode single angle gate OperationType error : " + std::to_string((size_t)type);
            QCERR_AND_THROW(std::runtime_error, error_message);
        }

    }

    QGate decode_double_gate_with_angle(OperationType type, size_t ctr_addr, size_t tar_addr, prob_vec params)
    {
        auto ctr_qubit = m_machine_ptr->allocateQubitThroughPhyAddress(ctr_addr);
        auto tar_qubit = m_machine_ptr->allocateQubitThroughPhyAddress(tar_addr);

        switch (type)
        {
        case QPanda::OperationType::OP_CP:
            return CP(ctr_qubit, tar_qubit, params[0]);

        case QPanda::OperationType::OP_CU:
            return CU(ctr_qubit, tar_qubit, params[0], params[1], params[2], params[3]);

        case QPanda::OperationType::OP_RYY:
            return RYY(ctr_qubit, tar_qubit, params[0]);

        case QPanda::OperationType::OP_RXX:
            return RXX(ctr_qubit, tar_qubit, params[0]);

        case QPanda::OperationType::OP_RZZ:
            return RZZ(ctr_qubit, tar_qubit, params[0]);

        case QPanda::OperationType::OP_RZX:
            return RZX(ctr_qubit, tar_qubit, params[0]);

        case QPanda::OperationType::OP_CPHASE:
            return CR(ctr_qubit, tar_qubit, params[0]);

        default:
            QCERR_AND_THROW(std::runtime_error, "decode quantum double gate data error.");
        }

    }

    QGate decode_multi_control_gate(OperationType type, std::vector<uint64_t> qubits)
    {
        if (qubits.size() < 3)
            QCERR_AND_THROW(std::runtime_error, "decode_multi_control_gate qubits args error.");

        switch (type)
        {
        case QPanda::OperationType::OP_TOFFOLI:
        {
            auto ctr_qubit_0 = m_machine_ptr->allocateQubitThroughPhyAddress(qubits[0]);
            auto ctr_qubit_1 = m_machine_ptr->allocateQubitThroughPhyAddress(qubits[1]);
            auto tar_qubit = m_machine_ptr->allocateQubitThroughPhyAddress(qubits[2]);
            return Toffoli(ctr_qubit_0, ctr_qubit_1, tar_qubit);
        }

        default:
            QCERR_AND_THROW(std::runtime_error, "decode quantum multi gate data error.");
        }
    }


    QGate decode_barrier(std::vector<uint64_t> qubits)
    {
        QVec qvec;

        for (auto qubit : qubits)
            qvec.emplace_back(m_machine_ptr->allocateQubitThroughPhyAddress(qubit));

        return BARRIER(qvec);
    }

    uint64_t decode_values(uint64_t data, int start, int num_bits)
    {
        return (data >> start) & ((1ULL << num_bits) - 1);
    }

    std::vector<uint64_t> decode_values(uint64_t data, Qnum start_list, int num_bits)
    {
        std::vector<uint64_t> result_datas;
        for (size_t i = 0; i < start_list.size(); i++)
        {
            auto val = (data >> start_list[i]) & ((1ULL << num_bits) - 1);
            result_datas.emplace_back(val);
        }

        return result_datas;
    }

    std::vector<uint64_t> decode_values(const std::vector<uint64_t> data, int start, int num_bits)
    {
        std::vector<uint64_t> values(data.size(), 0);
        for (size_t i = 0; i < data.size(); i++)
        {
            values[i] = (data[i] >> start) & ((1ULL << num_bits) - 1);
        }

        return values;
    }

    prob_vec decode_params(const std::vector<uint64_t> data, OperationType type)
    {
        int params_num = 1;
        switch (type)
        { 
            case QPanda::OperationType::OP_RPHI:
            case QPanda::OperationType::OP_U2: params_num = 2; break;
            case QPanda::OperationType::OP_U3: params_num = 3; break;

            case QPanda::OperationType::OP_CU:
            case QPanda::OperationType::OP_U4: params_num = 4; break;

            default: break;
        }

        prob_vec angles_data(params_num, 0);

        if (data.size() > 1 && params_num < 2)
            QCERR_AND_THROW(std::runtime_error, "decode quantum gate angles data error.");

        for (size_t i = 0; i < params_num; i++)
        {
            switch (i + 1)
            {
            case 1: angles_data[i] = decode_values(data[0], 32, 16); break;
            case 2: angles_data[i] = decode_values(data[0], 48, 16); break;
            case 3: angles_data[i] = decode_values(data[1], 32, 16); break;
            case 4: angles_data[i] = decode_values(data[1], 48, 16); break;
            default: break;
            }
        }

        for (auto& param: angles_data)
            param = (static_cast<double>(param) / static_cast<double>(1ull << 16)) * (2 * PI);

        return angles_data;
    }

private:

    std::vector<uint64_t> m_decode_data;
};

class CommProtocolDecode
{
public:

    CommProtocolDecode(QuantumMachine* machine);
    CommProtocolDecode() = delete;

    void decode_program();
    void decode_configuration(CommProtocolConfig& config);

    void load(const std::vector<char>& origin_data);

    std::vector<QProg> get_decode_progs() 
    { 
        return m_decode_progs; 
    }
    
    /*
     * @brief get each protocol ir data from origin_data
     */
    void get_init_progs();

    /*
     * @brief decode single protocol ir data to prog
     * @param[in] binary irs index
     */
    void decode_single_program(uint32_t index);

private:

    void decode_operation(const std::vector<uint64_t>& operation_data, QProg& prog);

private:

    ProtocolNodeDecode m_node_decode_imp;

    std::vector<uint64_t> m_origin_data;

    std::vector<QProg> m_decode_progs;

    std::vector<size_t> init_data_indices;      /* protocol ir vector */
    std::vector<size_t> prog_end_indices;       /* prog vector */
};

std::vector<QProg> comm_protocol_decode(CommProtocolConfig& config, const std::vector<char>& data, QuantumMachine* machine);

/*
 * @brief devide protocol origin_data to each protocol ir data
 * @param[in] CommProtocolConfig& config
 * @param[in] const std::vector<char>& protocol irs data
 * @param[in] QuantumMachine* machine
 */
CommProtocolDecode comm_protocol_devide(CommProtocolConfig& config, const std::vector<char>& encode_data, QuantumMachine* machine);

/*
 * @brief decode one protocol ir to prog
 * @param[in] CommProtocolDecode& protocol ir data
 * @param[in] uint32_t protocol ir index
 */
QProg comm_protocol_single_decode(CommProtocolDecode& protocol_data, uint32_t cir_index);

QPANDA_END

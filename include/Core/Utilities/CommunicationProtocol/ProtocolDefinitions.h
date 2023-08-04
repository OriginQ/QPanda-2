#pragma once

#include "Core/Core.h"
#include "Core/Utilities/QPandaNamespace.h"

QPANDA_BEGIN

enum class OperationType
{
    OP_UNDEFINED,
    OP_CONFIGURATION = 1,
    OP_INITIALIZATION,

    OP_I,
    OP_PAULI_X,
    OP_PAULI_Y,
    OP_PAULI_Z,
    OP_X_HALF_PI,
    OP_Y_HALF_PI,
    OP_Z_HALF_PI,

    OP_P,
    OP_HADAMARD,
    OP_T,
    OP_S,
    OP_RX,
    OP_RY,
    OP_RZ,
    OP_RPHI,
    OP_U1,
    OP_U2,
    OP_U3,
    OP_U4,
    OP_CU,
    OP_CNOT,
    OP_CZ,
    OP_CP,
    OP_MS,
    OP_RYY,
    OP_RXX,
    OP_RZZ,
    OP_RZX,
    OP_CPHASE,
    OP_ISWAP,
    OP_SQISWAP,
    OP_SWAP,
    OP_TOFFOLI,
    OP_ECHO,
    OP_BARRIER,

    OP_MEASURE,

    CIRCUIT_END_SIGN = 0X3F
};

struct CommProtocolConfig
{
    bool open_mapping = true;
    bool open_error_mitigation = true;
    uint8_t optimization_level = 4;
    uint8_t circuits_num = 100;
    uint32_t shots = 1000;
};

template <typename T>
static std::vector<std::vector<T>> container_slicer(const std::vector<T>& origin_vector, size_t slice_size) 
{
    if (!slice_size)
        QCERR_AND_THROW(std::runtime_error, "container_slicer error.");

    std::vector<std::vector<T>> result;

    for (size_t i = 0; i < origin_vector.size(); i += slice_size) 
    {
        auto last_iter = origin_vector.begin() + std::min(i + slice_size, origin_vector.size());

        result.push_back(std::vector<T>(origin_vector.begin() + i, last_iter));
    }

    return result;
}

class ConfigurationEncode
{
private:
    std::vector<uint64_t> m_config_data = { 0 };

public:
    ConfigurationEncode(OperationType op_type, 
                        bool enable_mapping, 
                        bool enable_error_mitigation,
                        uint8_t optimization_level, 
                        uint8_t circuits_num, 
                        uint32_t shot)
    {
        /*
        Operation type: 6 bits;
        Enable Mapping: 1 bit;
        Enable Readout Error Mitigation: 1 bit;
        Optimization Level: 4 bits;
        Padding: 4 bits (for byte alignment);
        Number of Circuits: 8 bits;
        Shot: 32 bits.
        */

        //Pack the configuration data into a uint64
        m_config_data[0] = (static_cast<uint64_t>(op_type) << 0) |
                           (static_cast<uint64_t>(enable_mapping) << 6) |
                           (static_cast<uint64_t>(enable_error_mitigation) << 7) |
                           (static_cast<uint64_t>(optimization_level) << 8) |
                           (static_cast<uint64_t>(circuits_num) << 16) |
                           (static_cast<uint64_t>(shot) << 32);
    }

    ConfigurationEncode(CommProtocolConfig config)
        : ConfigurationEncode(OperationType::OP_CONFIGURATION,
                              config.open_mapping,
                              config.open_error_mitigation,
                              config.optimization_level,
                              config.circuits_num,
                              config.shots)
    {}

    const std::vector<uint64_t>& encode() const
    { 
        return m_config_data; 
    }
};

class InitializationEncode
{
private:
    std::vector<uint64_t> m_initialization_data = { 0 };

public:
    InitializationEncode(OperationType op_type, 
                         uint8_t qubits_num,
                         uint8_t classical_bits_num)
    {
        /*
        Operation type: 6 bits
        Padding: 2 bits (for byte alignment)
        Number of qubits: 8 bits
        Number of classical bits: 8 bits
        Padding: 40 bits (for byte alignment)
        */

        //Pack the initialization data into a uint64
        m_initialization_data[0] = (static_cast<uint64_t>(op_type) << 0) |
                                   (static_cast<uint64_t>(qubits_num) << 8) |
                                   (static_cast<uint64_t>(classical_bits_num) << 16);
    }

    //Output the initialization data as uint64
    const std::vector<uint64_t>& encode() const
    {
        return m_initialization_data;
    }
};

class QMeasureEncode
{
private:
    std::vector<uint64_t> m_measure_data;

public:
    QMeasureEncode(OperationType op_type,
                  const std::vector<size_t>& qubits_list,
                  const std::vector<size_t>& classical_bits_list)
    {

        /*
        The Measure operation is represented as follows.
        Measure operation can support measuring multiple qubits:

        Operation type: 6 bits
        Resume: 1 bit
        Padding: 1 bit (for byte alignment)
        Number of measured qubits: 8 bits

        Quantum bit count 1: 8 bits
        Classical bit count 1: 8 bits
        Quantum bit count 2: 8 bits
        Classical bit count 2: 8 bits
        ...
        If exceeding the range, the Resume bit can be set to 
        continue the information transmission.
        */

        //Check if the size of qubits_num and classical_bits_num is the same
        if (qubits_list.size() != classical_bits_list.size())
            throw std::invalid_argument("Quantum bits and classical bits must have the same size.");

        //Calculate the number of uint64_t needed for m_measure_data
        auto slice_qbits_list = container_slicer(qubits_list, 3);
        auto slice_cbits_list = container_slicer(classical_bits_list, 3);

        size_t num_uint64 = slice_qbits_list.size();

        //Initialize m_measure_data with the appropriate size
        m_measure_data.resize(num_uint64, 0);

        const std::vector<uint32_t> insert_qbit_indices = { 16, 32, 48 };
        const std::vector<uint32_t> insert_cbit_indices = { 24, 40, 56 };

        for (size_t i = 0; i < num_uint64; i++)
        {
            if (i != num_uint64 - 1)
                m_measure_data[i] |= static_cast<uint64_t>(1) << 6;

            m_measure_data[i] |= static_cast<uint64_t>(op_type) << 0;
            m_measure_data[i] |= static_cast<uint64_t>(qubits_list.size()) << 8;

            auto op_qbits = slice_qbits_list[i];
            auto op_cbits = slice_cbits_list[i];

            for (size_t idx = 0; idx < op_qbits.size(); ++idx) // 0, 1, 2
                m_measure_data[i] |= static_cast<uint64_t>(op_qbits[idx]) << insert_qbit_indices[idx];
            
            for (size_t idx = 0; idx < op_cbits.size(); ++idx) // 0, 1, 2
                m_measure_data[i] |= static_cast<uint64_t>(op_cbits[idx]) << insert_cbit_indices[idx];
        }
    }

    // Output the measure data as std::vector<uint64_t>
    const std::vector<uint64_t>& encode() const
    {
        return m_measure_data;
    }
};

class QBarrierEncode
{
private:
    std::vector<uint64_t> m_barrier_data;

public:
    QBarrierEncode(OperationType op_type,
        const std::vector<size_t>& qubits_list)
    {
        /*
        Barrier is represented as follows and can support multiple bits:

        Operation type: 6 bits
        Continue: 1 bit
        Padding: 1 bit (for byte alignment)
        Number of Barrier Bits: 8 bits
        Quantum Bit Count 1: 8 bits
        Quantum Bit Count 2: 8 bits
        Quantum Bit Count 3: 8 bits
        ...
        If the range exceeds, the continue bit can be set to
        continue the information transmission.
        */

        if (qubits_list.empty())
            QCERR_AND_THROW(run_fail, "barrier encode error : qubits_list is empty");

        //Calculate the number of uint64_t needed for m_barrier_data
        auto slice_qubits_list = container_slicer(qubits_list, 6);

        size_t num_uint64 = slice_qubits_list.size();

        //Initialize m_barrier_data with the appropriate size
        m_barrier_data.resize(num_uint64, 0);

        const std::vector<uint32_t> insert_indices = { 16, 24, 32, 40, 48, 56 };

        for (size_t i = 0; i < num_uint64; i++)
        {
            if (i != num_uint64 - 1)
                m_barrier_data[i] |= static_cast<uint64_t>(1) << 6;

            m_barrier_data[i] |= static_cast<uint64_t>(op_type) << 0;
            m_barrier_data[i] |= static_cast<uint64_t>(slice_qubits_list[i].size()) << 8;

            auto op_qubits = slice_qubits_list[i];
            for (size_t idx = 0; idx < op_qubits.size(); ++idx)
                m_barrier_data[i] |= static_cast<uint64_t>(op_qubits[idx]) << insert_indices[idx];
        }
    }

    //Output the barrier data as std::vector<uint64_t>
    const std::vector<uint64_t>& encode() const
    {
        return m_barrier_data;
    }

};

class QGateEncode
{
private:
    std::vector<uint64_t> m_data;

public:
    QGateEncode(OperationType op_type,
                bool is_dagger,
                const std::vector<size_t>& qubits_list,
                const std::vector<size_t>& params_list)
    {
        /*
            Operation type: 6 bits;
            Resume bit: 1 bit;
            Dagger bit: 1 bit;
            Quantum bit index region: 3 regions of 8 bits each, totaling 24 bits;
            Half-precision floating-point number region: 2 regions of 16 bits each, totaling 32 bits.
        */
        const std::vector<uint8_t> qubits_indices = { 8, 16, 24 };
        const std::vector<uint8_t> params_indices = { 32, 48 };

        auto is_resume = params_list.size() > 2 ? 1 : 0;

        if (is_resume)
        {
            m_data.resize(2, 0);

            m_data[0] |= static_cast<uint64_t>(op_type) << 0;
            m_data[0] |= static_cast<uint64_t>(is_resume) << 6;
            m_data[0] |= static_cast<uint64_t>(is_dagger) << 7;

            m_data[1] |= static_cast<uint64_t>(op_type) << 0;
            m_data[1] |= static_cast<uint64_t>(!is_resume) << 6;
            m_data[1] |= static_cast<uint64_t>(is_dagger) << 7;

            auto slice_params = container_slicer(params_list, 2);

            for (auto i = 0; i < qubits_list.size(); i++)
            {
                m_data[0] |= static_cast<uint64_t>(qubits_list[i]) << qubits_indices[i];
                m_data[1] |= static_cast<uint64_t>(qubits_list[i]) << qubits_indices[i];
            }

            for (auto i = 0; i < slice_params[0].size(); i++)
                m_data[0] |= static_cast<uint64_t>(slice_params[0][i]) << params_indices[i];

            for (auto i = 0; i < slice_params[1].size(); i++)
                m_data[1] |= static_cast<uint64_t>(slice_params[1][i]) << params_indices[i];
        }
        else
        {
            m_data.resize(1, 0);

            m_data[0] |= static_cast<uint64_t>(op_type) << 0;
            m_data[0] |= static_cast<uint64_t>(is_resume) << 6;
            m_data[0] |= static_cast<uint64_t>(is_dagger) << 7;

            for (auto i = 0; i < qubits_list.size(); i++)
                m_data[0] |= static_cast<uint64_t>(qubits_list[i]) << qubits_indices[i];

            for (auto i = 0; i < params_list.size(); i++)
                m_data[0] |= static_cast<uint64_t>(params_list[i]) << params_indices[i];
        }
    }

    // Output the measure data as std::vector<uint64_t>
    const std::vector<uint64_t>& encode() const
    {
        return m_data;
    }
};

QPANDA_END

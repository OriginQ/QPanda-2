#include "Core/VirtualQuantumProcessor/MPSQVM/NoiseDefinition.h"
#include "Core/VirtualQuantumProcessor/DensityMatrix/DensityMatrixNoise.h"

USING_QPANDA
using namespace NoiseUtils;

bool DensityMatrixNoise::enabled()
{
    return !m_single_qubits.empty() || !m_double_qubits.empty();
}

bool DensityMatrixNoise::enabled(GateType gate_type, Qnum qubits)
{
    if (is_single_gate(gate_type))
    {
        auto iter = m_single_qubits.find(gate_type);
        QPANDA_RETURN(m_single_qubits.end() == iter, false);

        QPANDA_RETURN(iter->second.empty(), true);

        auto tar_iter = std::find(iter->second.begin(), iter->second.end(), qubits[0]);
        QPANDA_RETURN(tar_iter != iter->second.end(), true);
    }
    else
    {
        auto find_pair = [&](const std::vector<DoubleQubits> &qubits, DoubleQubits value)
        {
            for (auto val : qubits)
            {
                if (val.first == value.first && val.second == value.second)
                {
                    return true;
                }
            }

            return false;
        };

        auto iter = m_double_qubits.find(gate_type);
        QPANDA_RETURN(m_double_qubits.end() == iter, false);

        QPANDA_RETURN(iter->second.empty(), true);

        auto ctr_qubit = qubits[0];
        auto tar_qubit = qubits[1];

        QPANDA_RETURN(find_pair(iter->second, std::make_pair(ctr_qubit, tar_qubit)), true);
    }

    return false;
}

KarusError DensityMatrixNoise::get_karus_error(GateType gate_type, const Qnum& qubits)
{
    if (is_single_gate(gate_type))
    {
        auto tar_qubit = qubits[0];

        for (const auto& val : m_one_qubit_karus_error_tuple)
        {
            auto type = std::get<0>(val);
            auto addr = std::get<1>(val);

            if (gate_type == type)
            {
                QPANDA_RETURN(-1 == addr, std::get<2>(val));
                QPANDA_RETURN(tar_qubit == addr, std::get<2>(val));
            }
        }
    }
    else
    {
        auto ctr_qubit = qubits[0];
        auto tar_qubit = qubits[1];

        for (const auto& val : m_two_qubit_karus_error_tuple)
        {
            auto type = std::get<0>(val);
            auto ctr_addr = std::get<1>(val);
            auto tar_addr = std::get<2>(val);

            if (gate_type == type)
            {
                QPANDA_RETURN((-1 == ctr_addr) && (-1 == tar_addr), std::get<3>(val));
                QPANDA_RETURN((ctr_qubit == ctr_addr) && (tar_qubit == tar_addr), std::get<3>(val));
            }
        }
    }

    QPANDA_ASSERT(true, "get_karus_error");
}

void DensityMatrixNoise::set_gate_and_qnum(GateType gate_type, const Qnum& qubits_vec)
{
    QPANDA_ASSERT(!is_single_gate(gate_type), "set_gate_and_qnum error");

    Qnum qubits = m_single_qubits[gate_type];
    for (const auto &qubit : qubits_vec)
    {
        qubits.emplace_back(qubit);
    }

    unique_vector(qubits);
    m_single_qubits[gate_type] = qubits;

    return;
}

void DensityMatrixNoise::set_gate_and_qnums(GateType gate_type, const std::vector<Qnum>& qubits_vecs)
{
    if (is_single_gate(gate_type))
    {
        Qnum qubits = m_single_qubits[gate_type];
        for (const auto &qvec : qubits_vecs)
        {
            std::for_each(qvec.begin(), qvec.end(), [&](size_t addr) {qubits.emplace_back(addr); });
        }

        unique_vector(qubits);
        m_single_qubits[gate_type] = qubits;
    }
    else
    {
        std::vector<DoubleQubits> double_qubits = m_double_qubits[gate_type];

        for (const auto &qubits : qubits_vecs)
        {
            QPANDA_ASSERT(2 != qubits.size(), "qubits size or gate type error");
            double_qubits.emplace_back(std::make_pair(qubits[0], qubits[1]));
        }

        m_double_qubits[gate_type] = double_qubits;
    }

    return;
}

void DensityMatrixNoise::update_karus_error_tuple(GateType gate_type, int tar_qubit, const KarusError& karus_error)
{
    QPANDA_ASSERT(!is_single_gate(gate_type), "update karus error tuple error");

    for (auto& val : m_one_qubit_karus_error_tuple)
    {
        auto type = std::get<0>(val);
        auto addr = std::get<1>(val);

        if ((gate_type == type) && (-1 == tar_qubit))
        {
            return;
        }

        if ((gate_type == type) && (tar_qubit == addr))
        {
            std::get<2>(val) = karus_error;
            return;
        }
    }

    auto karus_error_tuple = std::make_tuple(gate_type, tar_qubit, karus_error);
    m_one_qubit_karus_error_tuple.emplace_back(karus_error_tuple);
    return;
}

void DensityMatrixNoise::update_karus_error_tuple(GateType gate_type, int ctr_qubit, int tar_qubit, const KarusError& karus_error)
{
    QPANDA_ASSERT(is_single_gate(gate_type), "update karus error tuple error");

    for (auto& val : m_two_qubit_karus_error_tuple)
    {
        auto type = std::get<0>(val);
        auto ctr_addr = std::get<1>(val);
        auto tar_addr = std::get<2>(val);

        if ((gate_type == type) && (-1 == ctr_qubit) && (-1 == tar_qubit))
        {
            return;
        }

        if ((gate_type == type) && (ctr_addr == ctr_qubit) && (tar_addr == tar_qubit))
        {
            std::get<3>(val) = karus_error;
            return;
        }
    }

    auto karus_error_tuple = std::make_tuple(gate_type, ctr_qubit, tar_qubit, karus_error);
    m_two_qubit_karus_error_tuple.emplace_back(karus_error_tuple);
    return;
}

void DensityMatrixNoise::set_single_karus_error_tuple(GateType gate_type, const KarusError &karus_error, const Qnum& qubits)
{
    QPANDA_ASSERT(!is_single_gate(gate_type), "set qubits error");

    //set karus error for all qubits by current quantum gates 
    if (qubits.empty())
    {
        update_karus_error_tuple(gate_type, -1, karus_error);
        return;
    }

    //set karus error for selected qubits by current quantum gates
    for (auto qubit : qubits)
    {
        update_karus_error_tuple(gate_type, qubit, karus_error);
    }

    return;
}

void DensityMatrixNoise::set_double_karus_error_tuple(GateType gate_type, const KarusError &karus_error, const std::vector<Qnum>& qubits_vec)
{
    QPANDA_ASSERT(is_single_gate(gate_type), "set qubits error");

    //set karus error for all qubits by current quantum gates 
    if (qubits_vec.empty())
    {
        update_karus_error_tuple(gate_type, -1, -1, karus_error);
        return;
    }

    //set karus error for selected qubits by current quantum gates
    for (auto qubits : qubits_vec)
    {
        QPANDA_ASSERT(qubits.empty(), "set_double_karus_error_tuple");
        update_karus_error_tuple(gate_type, qubits[0], qubits[1], karus_error);
    }

    return;
}

void DensityMatrixNoise::set_noise_model(const NOISE_MODEL& model, const GateType& gate_type, double prob)
{
    QPANDA_ASSERT(0. > prob || prob > 1., "prob range error");

    if (model == NOISE_MODEL::DAMPING_KRAUS_OPERATOR)
    {
        auto karus_matrices = get_noise_model_karus_matrices(model, { prob });
        auto karus_error = KarusError(karus_matrices, model);

        set_gate_and_qnums(gate_type, {});

        QPANDA_OP(is_single_gate(gate_type), set_single_karus_error_tuple(gate_type, karus_error, {}));
        QPANDA_OP(!is_single_gate(gate_type), set_double_karus_error_tuple(gate_type, karus_error, {}));
        return;
    }
    else
    {
        auto unitary_probs = get_noise_model_unitary_probs(model, prob);
        auto unitary_matrices = get_noise_model_unitary_matrices(model, prob);

        auto karus_error = KarusError(unitary_matrices, unitary_probs, model);

        set_gate_and_qnums(gate_type, {});

        QPANDA_OP(is_single_gate(gate_type), set_single_karus_error_tuple(gate_type, karus_error, {}));
        QPANDA_OP(!is_single_gate(gate_type), set_double_karus_error_tuple(gate_type, karus_error, {}));
    }

    return;
}

void DensityMatrixNoise::set_noise_model(const NOISE_MODEL& model, const GateType& gate_type, double T1, double T2, double time_param)
{
    QPANDA_ASSERT(NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR != model, "model != DECOHERENCE_KRAUS_OPERATOR");

    set_gate_and_qnums(gate_type, {});

    std::vector<QStat> karus_matrices = get_noise_model_karus_matrices(DECOHERENCE_KRAUS_OPERATOR, { T1,T2,time_param });

    auto karus_error = KarusError(karus_matrices);

    QPANDA_OP(is_single_gate(gate_type), set_single_karus_error_tuple(gate_type, karus_error, {}));
    QPANDA_OP(!is_single_gate(gate_type), set_double_karus_error_tuple(gate_type, karus_error, {}));

    return;
}

void DensityMatrixNoise::set_noise_model(const NOISE_MODEL& model, const GateType& gate_type, double prob, const std::vector<Qnum>& qubits_vecs)
{
    QPANDA_ASSERT(0. > prob || prob > 1., "prob range error");

    set_gate_and_qnums(gate_type, qubits_vecs);

    auto unitary_probs = get_noise_model_unitary_probs(model, prob);
    auto unitary_matrices = get_noise_model_unitary_matrices(model, prob);

    auto karus_error = KarusError(unitary_matrices, unitary_probs);

    QPANDA_OP(is_single_gate(gate_type), set_single_karus_error_tuple(gate_type, karus_error, get_qnum(qubits_vecs)));
    QPANDA_OP(!is_single_gate(gate_type), set_double_karus_error_tuple(gate_type, karus_error, qubits_vecs));

    return;
}

void DensityMatrixNoise::set_noise_model(const NOISE_MODEL& model, const GateType& gate_type, double prob, const Qnum& qubits_vec)
{
    QPANDA_ASSERT(0. > prob || prob > 1., "prob range error");
    QPANDA_ASSERT(!is_single_gate(gate_type), "set_noise_model gate type error");

    auto unitary_probs = get_noise_model_unitary_probs(model, prob);
    auto unitary_matrices = get_noise_model_unitary_matrices(model, prob);

    auto karus_error = KarusError(unitary_matrices, unitary_probs);

    set_gate_and_qnum(gate_type, qubits_vec);

    set_single_karus_error_tuple(gate_type, karus_error, qubits_vec);

    return;
}

void DensityMatrixNoise::set_noise_model(const NOISE_MODEL& model, const GateType& gate_type, double T1, double T2, double time_param, const std::vector<Qnum>& qubits_vecs)
{
    QPANDA_ASSERT(NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR != model, "model != DECOHERENCE_KRAUS_OPERATOR");

    set_gate_and_qnums(gate_type, qubits_vecs);

    std::vector<QStat> karus_matrices = get_noise_model_karus_matrices(DECOHERENCE_KRAUS_OPERATOR, { T1,T2,time_param });

    auto karus_error = KarusError(karus_matrices);

    QPANDA_OP(is_single_gate(gate_type), set_single_karus_error_tuple(gate_type, karus_error, get_qnum(qubits_vecs)));
    QPANDA_OP(!is_single_gate(gate_type), set_double_karus_error_tuple(gate_type, karus_error, qubits_vecs));

    return;
}

void DensityMatrixNoise::set_noise_model(const NOISE_MODEL& model, const GateType& gate_type, double T1, double T2, double time_param, const Qnum& qubits_vec)
{
    QPANDA_ASSERT(!is_single_gate(gate_type), "set_noise_model gate type error");
    QPANDA_ASSERT(NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR != model, "model != DECOHERENCE_KRAUS_OPERATOR");

    set_gate_and_qnum(gate_type, qubits_vec);

    std::vector<QStat> karus_matrices = get_noise_model_karus_matrices(DECOHERENCE_KRAUS_OPERATOR, { T1,T2,time_param });

    auto karus_error = KarusError(karus_matrices);
    set_single_karus_error_tuple(gate_type, karus_error, qubits_vec);

    return;
}
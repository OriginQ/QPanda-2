#include <map>
#include "QPandaNamespace.h"
#include "Core/Utilities/Tools/QStatMatrix.h"
#include "Core/Utilities/Tools/QPandaException.h"
#include "Core/VirtualQuantumProcessor/MPSQVM/NoiseDefinition.h"
USING_QPANDA

//pauli x
static std::vector<QStat> get_bit_filp_karus_matrices(const prob_vec& params)
{
    std::vector<QStat> karus_matrices(2);
    double probability = params[0];

    karus_matrices[0] = { static_cast<qstate_type>(sqrt(1 - probability)), 0, 0, static_cast<qstate_type>(sqrt(1 - probability)) };
    karus_matrices[1] = { 0, static_cast<qstate_type>(sqrt(probability)), static_cast<qstate_type>(sqrt(probability)), 0 };
    return karus_matrices;
}

//pauli y
static std::vector<QStat> get_dephasing_karus_matrices(const prob_vec& params)
{
    std::vector<QStat> karus_matrices(2);
    double probability = params[0];

    karus_matrices[0] = { (qstate_type)sqrt(1 - probability),0,0,(qstate_type)sqrt(1 - probability) };
    karus_matrices[1] = { (qstate_type)sqrt(probability),0,0,-(qstate_type)sqrt(probability) };
    return karus_matrices;
}

//pauli z
static std::vector<QStat> get_phase_flip_karus_matrices(const prob_vec& params)
{
    std::vector<QStat> karus_matrices(2);
    double probability = params[0];

    karus_matrices[0] = { static_cast<qstate_type>(sqrt(1 - probability)), 0, 0, static_cast<qstate_type>(sqrt(1 - probability)) };
    karus_matrices[1] = { 0, qcomplex_t(0, -sqrt(probability)), qcomplex_t(0, sqrt(probability)), 0 };
    return karus_matrices;
}


//pauli x,y,x
static std::vector<QStat> get_depolarizing_karus_matrices(const prob_vec& params)
{
    std::vector<QStat> karus_matrices(4);
    double probability = params[0];

    QStat matrix_i = { 1, 0, 0, 1 };
    QStat matrix_x = { 0, 1, 1, 0 };
    QStat matrix_y = { 0, qcomplex_t(0, -1), qcomplex_t(0, 1), 0 };
    QStat matrix_z = { 1, 0, 0, -1 };

    karus_matrices[0] = static_cast<qstate_type>(sqrt(1 - probability * 0.75)) * matrix_i;
    karus_matrices[1] = static_cast<qstate_type>(sqrt(probability) / 2) * matrix_x;
    karus_matrices[2] = static_cast<qstate_type>(sqrt(probability) / 2) * matrix_y;
    karus_matrices[3] = static_cast<qstate_type>(sqrt(probability) / 2) * matrix_z;
    return karus_matrices;
}

static std::vector<QStat> get_phase_damping_karus_matrices(const prob_vec& params)
{
    std::vector<QStat> karus_matrices(2);
    double probability = params[0];

    karus_matrices[0] = { 1, 0, 0, static_cast<qstate_type>(sqrt(1 - probability)) };
    karus_matrices[1] = { 0, 0, 0, static_cast<qstate_type>(sqrt(probability)) };
    return karus_matrices;
}

static std::vector<QStat> get_amplitude_damping_flip_karus_matrices(const prob_vec& params)
{
    std::vector<QStat> karus_matrices(2);
    double probability = params[0];

    karus_matrices[0] = { 1,0,0,(qstate_type)sqrt(1 - probability) };
    karus_matrices[1] = { 0,(qstate_type)sqrt(probability),0,0 };
    return karus_matrices;
}

static std::vector<QStat> get_decoherence_karus_matrices(const prob_vec& params)
{
    double T1 = params[0];
    double T2 = params[1];
    double t_gate_time = params[2];

    double p_damping = 1. - std::exp(-(t_gate_time / T1));
    double p_dephasing = 0.5 * (1. - std::exp(-(t_gate_time / T2 - t_gate_time / (2 * T1))));

    QStat K1 = { std::sqrt(1 - p_dephasing), 0,0,std::sqrt((1 - p_damping)*(1 - p_dephasing)) };
    QStat K2 = { 0, std::sqrt(p_damping*(1 - p_dephasing)), 0, 0 };
    QStat K3 = { 0, std::sqrt(p_damping*(1 - p_dephasing)), 0, 0 };
    QStat K4 = { 0, -std::sqrt(p_damping*p_dephasing), 0, 0 };

    std::vector<QStat> karus_matrices{ K1,K2,K3,K4 };
    return karus_matrices;
}

static std::map<NOISE_MODEL, std::function<std::vector<QStat>(const prob_vec&)>>
karus_matrices_map =
{
    {BITFLIP_KRAUS_OPERATOR,            get_bit_filp_karus_matrices},
    {DEPHASING_KRAUS_OPERATOR,          get_dephasing_karus_matrices},
    {BIT_PHASE_FLIP_OPRATOR,            get_phase_flip_karus_matrices},
    {DEPOLARIZING_KRAUS_OPERATOR,       get_depolarizing_karus_matrices},
    {PHASE_DAMPING_OPRATOR,             get_phase_damping_karus_matrices},
    {DAMPING_KRAUS_OPERATOR,            get_amplitude_damping_flip_karus_matrices},
    {DECOHERENCE_KRAUS_OPERATOR,        get_decoherence_karus_matrices}
};


static std::map<NOISE_MODEL, QStat> 
flip_model_mapping_map =
{
    {NOISE_MODEL::BITFLIP_KRAUS_OPERATOR, {0., 1., 1., 0.}},
    {NOISE_MODEL::BIT_PHASE_FLIP_OPRATOR, {0., qcomplex_t(0., -1.), qcomplex_t(0., 1.), 0.}},
    {NOISE_MODEL::DEPHASING_KRAUS_OPERATOR, {1., 0., 0., -1.}}
};


std::vector<QStat> QPanda::get_noise_model_karus_matrices(NOISE_MODEL model, const std::vector<double>& params)
{
    try
    {
        auto karus_map_iter = karus_matrices_map.find(model);

        QPANDA_ASSERT(karus_matrices_map.cend() == karus_map_iter, "karus_map_iter error");

        return karus_map_iter->second(params);
    }
    catch (...)
    {
        throw run_fail("get_noise_model_karus_matrices error");
    }

}


std::vector<double> QPanda::get_noise_model_unitary_probs(NOISE_MODEL model, double param)
{
    switch (model)
    {
        case NOISE_MODEL::BITFLIP_KRAUS_OPERATOR:
        case NOISE_MODEL::BIT_PHASE_FLIP_OPRATOR:
        case NOISE_MODEL::DEPHASING_KRAUS_OPERATOR:
        {
            return { param,1 - param };
        }
        case NOISE_MODEL::DEPOLARIZING_KRAUS_OPERATOR:
        {
            param /=  4.0;
            return { param, param, param, 1 - 3 * param };
        }
        case NOISE_MODEL::PHASE_DAMPING_OPRATOR:
        {
            double alpha = (1 + std::sqrt(param)) / 2;
            return { alpha,1 - alpha };
        }
        default:
        {
            QCERR("unsupported noise model");
            throw run_fail("unsupported noise model");
            break;
        }
    }
}


std::vector<QStat> QPanda::get_noise_model_unitary_matrices(NOISE_MODEL model, double param)
{
    switch (model)
    {
        case NOISE_MODEL::BITFLIP_KRAUS_OPERATOR:
        case NOISE_MODEL::BIT_PHASE_FLIP_OPRATOR:
        case NOISE_MODEL::DEPHASING_KRAUS_OPERATOR:
        {
            QStat pauli_unitary = flip_model_mapping_map.at(model);

            return { pauli_unitary,{1.,0.,0.,1.} };
        }
        case NOISE_MODEL::DEPOLARIZING_KRAUS_OPERATOR:
        {
            QStat pauli_X_unitary = flip_model_mapping_map.at(BITFLIP_KRAUS_OPERATOR);
            QStat pauli_Y_unitary = flip_model_mapping_map.at(BIT_PHASE_FLIP_OPRATOR);
            QStat pauli_Z_unitary = flip_model_mapping_map.at(DEPHASING_KRAUS_OPERATOR);
            
            return { pauli_X_unitary, pauli_Y_unitary, pauli_Z_unitary, {1.,0.,0.,1.} };
        }
        case NOISE_MODEL::PHASE_DAMPING_OPRATOR:
        {
            double alpha = (1 + std::sqrt(param)) / 2;
            return QPanda::get_noise_model_unitary_matrices(NOISE_MODEL::DEPHASING_KRAUS_OPERATOR, alpha);
        }
        default:
        {
            QCERR("unsupported noise model");
            throw run_fail("unsupported noise model");
            break;
        }
    }
}
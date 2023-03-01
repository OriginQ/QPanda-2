#include <set>
#include "Core/Utilities/Tools/QStatMatrix.h"
#include "Core/Utilities/Tools/QProgFlattening.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include "Core/VirtualQuantumProcessor/DensityMatrix/DensityMatrixSimulator.h"

USING_QPANDA

static void qnum_assert(const Qnum& tar_qubits, const Qnum& ctr_qubits)
{
    for (auto tar_qubit : tar_qubits)
        for (auto ctr_qubit : ctr_qubits)
            if (tar_qubit == ctr_qubit)
                QCERR_AND_THROW(std::runtime_error, "control_qubit == target_qubit");

    Qnum check_qubits = ctr_qubits;
    std::set<size_t> qubits_set(check_qubits.begin(), check_qubits.end());
    check_qubits.assign(qubits_set.begin(), qubits_set.end());

    if (ctr_qubits.size() != check_qubits.size())
        QCERR_AND_THROW(std::runtime_error, "repetitive qubits addr");

    return;
}

static void parse_gate_node(std::shared_ptr<AbstractQGateNode> gate, Qnum &targets, Qnum &controls, bool& is_conj)
{
    QVec qubits;
    QVec control_qubits;
    gate->getQuBitVector(qubits);
    gate->getControlVector(control_qubits);

    is_conj = gate->isDagger();

    for (auto &val : control_qubits)
    {
        auto control_addr = val->get_phy_addr();
        controls.push_back(control_addr);
    }

    for (auto &val : qubits)
    {
        auto target_addr = val->get_phy_addr();
        targets.push_back(target_addr);
    }

    qnum_assert(targets, controls);

    return;
}


void DensityMatrixSimulator::init(bool is_double_precision)
{
    _start();
    if (is_double_precision == true)
        m_simulator = std::make_shared<DensityMatrix<double>>();
    else
        m_simulator = std::make_shared<DensityMatrix<float>>();
    return;
}

void DensityMatrixSimulator::run(QProg& node, bool reset_state)
{
    QVec qubits;
    auto qubits_num = get_allocate_qubits(qubits);

    /*Initialize density matrix space*/
    if (reset_state)
        m_simulator->init_density_matrix(qubits_num);

    flatten(node);
    auto prog_node = node.getImplementationPtr();
    for (auto iterator  = prog_node->getFirstNodeIter();
              iterator != prog_node->getEndNodeIter(); ++iterator)
    {
        auto qnode = std::dynamic_pointer_cast<QNode>(*iterator);

        switch (qnode->getNodeType())
        {
        case NodeType::MEASURE_GATE:
        {
            auto measure_node = std::dynamic_pointer_cast<AbstractQuantumMeasure>(qnode);
            auto qubit_addr = measure_node->getQuBit()->get_phy_addr();
            m_simulator->apply_Measure({ qubit_addr });
            break;
        }

        case NodeType::GATE_NODE:
        {
            auto gate_node = std::dynamic_pointer_cast<AbstractQGateNode>(qnode);

            apply_gate(gate_node);

            /* check if global noisy enabled or not*/
            if (m_noisy.enabled())
                apply_gate_with_noisy(gate_node);

            break;
        }

        default:QCERR_AND_THROW(std::runtime_error, "invalid node for density matrix simulator");
        }

    } /*end iterator*/

    return;
}

void DensityMatrixSimulator::apply_gate(std::shared_ptr<AbstractQGateNode> gate_node)
{
    Qnum targets;
    Qnum controls;
    bool is_conj = false;

    parse_gate_node(gate_node, targets, controls, is_conj);

    Qnum qubits, conj_qubits;
    for (auto qubit : controls)
        qubits.emplace_back(qubit);

    for (auto qubit : targets)
        qubits.emplace_back(qubit);

    for (auto qubit : qubits)
        conj_qubits.push_back(qubit + getAllocateQubitNum());

    switch (gate_node->getQGate()->getGateType())
    {
        case GateType::I_GATE:
        case GateType::BARRIER_GATE:
            break;

        case GateType::PAULI_X_GATE:
        case GateType::CNOT_GATE:
        case GateType::TOFFOLI_GATE:
            m_simulator->apply_mcx(qubits);
            m_simulator->apply_mcx(conj_qubits);
            break;

        case GateType::PAULI_Y_GATE:
            m_simulator->apply_mcy(qubits);
            m_simulator->apply_mcy(conj_qubits, true);
            break;

        case GateType::PAULI_Z_GATE:
        case GateType::CZ_GATE:
            m_simulator->apply_mcphase(qubits, -1);
            m_simulator->apply_mcphase(conj_qubits, -1);
            break;

        case GateType::S_GATE:
        {
            if (is_conj)
            {
                m_simulator->apply_mcphase(qubits, std::complex<double>(0., -1.));
                m_simulator->apply_mcphase(conj_qubits, std::complex<double>(0., 1.));
                //m_simulator->apply_Phase(targets[0], std::complex<double>(0., -1.));
            }
            else
            {
                m_simulator->apply_mcphase(qubits, std::complex<double>(0., 1.));
                m_simulator->apply_mcphase(conj_qubits, std::complex<double>(0., -1.));
                //m_simulator->apply_Phase(targets[0], std::complex<double>(0., 1.));
            }

            break;
        }
        break;

        case GateType::T_GATE:
        {
            const double sqrt2 = { 1. / std::sqrt(2) };

            if (is_conj)
            {
                m_simulator->apply_mcphase(qubits, std::complex<double>(sqrt2, -sqrt2));
                m_simulator->apply_mcphase(conj_qubits, std::complex<double>(sqrt2, sqrt2));
            }
            else
            {
                m_simulator->apply_mcphase(qubits, std::complex<double>(sqrt2, sqrt2));
                m_simulator->apply_mcphase(conj_qubits, std::complex<double>(sqrt2, -sqrt2));
            }

            break;
        }

        case GateType::SWAP_GATE:
            m_simulator->apply_mcswap(qubits);
            m_simulator->apply_mcswap(conj_qubits);
            break;

        case GateType::CP_GATE:
        case GateType::CU_GATE:
        case GateType::CPHASE_GATE:
        {
            auto param_ptr = dynamic_cast<QGATE_SPACE::AbstractSingleAngleParameter *>(gate_node->getQGate());
            auto param = param_ptr->getParameter();

            if (is_conj)
                param *= -1.0;

            std::complex<double> phase = std::exp(std::complex<double>(0, 1) * param);;

            m_simulator->apply_mcphase(qubits, phase);
            m_simulator->apply_mcphase(conj_qubits, std::conj(phase));
            break;
        }

        case GateType::RXX_GATE:
        case GateType::RYY_GATE:
        case GateType::RZZ_GATE:
        case GateType::RZX_GATE:

        case GateType::P00_GATE:
        case GateType::P11_GATE:

        case GateType::ISWAP_GATE:
        case GateType::SQISWAP_GATE:
        case GateType::TWO_QUBIT_GATE:
        case GateType::ISWAP_THETA_GATE:
        {
            Qnum control_conj_qubits, target_conj_qubits;

            for (auto qubit : controls)
                control_conj_qubits.emplace_back(qubit + getAllocateQubitNum());

            for (auto qubit : targets)
                target_conj_qubits.emplace_back(qubit + getAllocateQubitNum());

            auto gate_matrix = getCircuitMatrix(std::dynamic_pointer_cast<QNode>(gate_node));

            auto matrix = column_stacking(gate_matrix);
            m_simulator->apply_multiplexer(controls, targets, matrix);
            m_simulator->apply_multiplexer(control_conj_qubits, target_conj_qubits, vector_conj(matrix));
            break;
        }

        /*
        case GateType::HADAMARD_GATE:
        case GateType::P_GATE:
        case GateType::U1_GATE:
        case GateType::U2_GATE:
        case GateType::U3_GATE:
        case GateType::U4_GATE:

        case GateType::RX_GATE:
        case GateType::RY_GATE:
        case GateType::RZ_GATE:

        case GateType::X_HALF_PI:
        case GateType::Y_HALF_PI:
        case GateType::Z_HALF_PI:
        */
        default:
        {
            QStat gate_matrix;
            gate_node->getQGate()->getMatrix(gate_matrix);

            auto matrix = column_stacking(gate_matrix);

            if (is_conj)
                dagger(matrix);

            m_simulator->apply_mcu(qubits, matrix);
            m_simulator->apply_mcu(conj_qubits, vector_conj(matrix));
            break;
        }

    } /*end switch*/
}

void DensityMatrixSimulator::apply_gate_with_noisy(std::shared_ptr<AbstractQGateNode> gate_node)
{
    Qnum targets;
    Qnum controls;
    bool is_conj = false;

    parse_gate_node(gate_node, targets, controls, is_conj);

    auto gate_type = (GateType)gate_node->getQGate()->getGateType();

    /* check if noisy enabled for current gate & qubits*/
    if (m_noisy.enabled(gate_type, targets))
    {
        auto karus_error = m_noisy.get_karus_error(gate_type, targets);

        std::vector<QStat> karus_matrices;

        if (is_single_gate(gate_type))
            karus_error.get_one_qubit_karus_matrices(karus_matrices);
        else
            karus_error.get_two_qubit_karus_matrices(karus_matrices);

        m_simulator->apply_karus(targets, karus_matrices);
    }

    return;
}

double DensityMatrixSimulator::get_probability(QProg& node, std::string bin_index)
{
    run(node);

    size_t index = 0;
    size_t qubit_num = bin_index.size();
    for (size_t i = 0; i < bin_index.size(); ++i)
    {
        index += (bin_index[qubit_num - i - 1] != '0') << i;
    }

    return m_simulator->probability(index);
}

prob_vec DensityMatrixSimulator::get_probabilities(QProg& node, std::vector<std::string> bin_indices)
{
    run(node);

    prob_vec result;
    for (auto val : bin_indices)
    {
        size_t index = 0;
        size_t qubit_num = val.size();
        for (size_t i = 0; i < val.size(); ++i)
        {
            index += (val[qubit_num - i - 1] != '0') << i;
        }

        result.emplace_back(m_simulator->probability(index));
    }

    return result;
}


double DensityMatrixSimulator::get_probability(QProg& node, size_t index)
{
    run(node);
    return m_simulator->probability(index);
}

prob_vec DensityMatrixSimulator::get_probabilities(QProg& node)
{
    run(node);
    return m_simulator->probabilities();
}

prob_vec DensityMatrixSimulator::get_probabilities(QProg& node, QVec qubits)
{
    run(node);
    return m_simulator->probabilities(NoiseUtils::get_qubits_addr(qubits));
}

prob_vec DensityMatrixSimulator::get_probabilities(QProg& node, Qnum qubits)
{
    run(node);
    return m_simulator->probabilities(qubits);
}

double DensityMatrixSimulator::get_expectation(QProg& node, const QHamiltonian& hamiltonian, const QVec& qubits)
{
    double expval = 0.0;

    auto init_matrix = get_density_matrix(node);

    m_simulator->initialize(init_matrix);

    auto expval_check = [](size_t number)
    {
        bool label = true;

        size_t i = 0;
        while ((number >> i) != 0)
        {
            if ((number >> i) % 2 == 1)
                label = !label;

            ++i;
        }
        return label;
    };

    for (size_t i = 0; i < hamiltonian.size(); i++)
    {
        auto component = hamiltonian[i];
        if (component.first.empty())
        {
            expval += component.second;
            continue;
        }

        QProg prog;
        QVec reduced_qubits;
        for (auto iter : component.first)
        {
            reduced_qubits.emplace_back(qubits[iter.first]);
            if (iter.second == 'X')
                prog << H(qubits[iter.first]);
            else if (iter.second == 'Y')
                prog << RX(qubits[iter.first], PI / 2);
        }

        m_simulator->initialize(init_matrix);
        run(prog, false);
        prob_vec reduced_probs = m_simulator->probabilities(NoiseUtils::get_qubits_addr(reduced_qubits));

        double expectation = 0;
#pragma omp parallel for reduction(+:expectation)
        for (int64_t i = 0; i < reduced_probs.size(); i++)
        {
            if (expval_check(i))
                expectation += reduced_probs[i];
            else
                expectation -= reduced_probs[i];
        }

        expval += component.second * expectation;
    }

    return expval;
}

double DensityMatrixSimulator::get_expectation(QProg& node, const QHamiltonian& hamiltonian, const Qnum& qubits)
{
    QVec phy_qubits;
    for (const auto& val : qubits)
        phy_qubits.emplace_back(allocateQubitThroughPhyAddress(val));

    return get_expectation(node, hamiltonian, phy_qubits);
}

cmatrix_t  DensityMatrixSimulator::get_density_matrix(QProg& node)
{
    run(node);
    return m_simulator->density_matrix();
}
cmatrix_t  DensityMatrixSimulator::get_reduced_density_matrix(QProg& node, const QVec& qubits)
{
    Qnum qubits_addr;
    for (const auto& qubit : qubits)
        qubits_addr.emplace_back(qubit->get_phy_addr());

    return get_reduced_density_matrix(node, qubits_addr);
}

cmatrix_t DensityMatrixSimulator::get_reduced_density_matrix(QProg& node, const Qnum& qubits)
{
    run(node);

    if (qubits.empty()) 
    {
        cmatrix_t reduced_matrix;
        reduced_matrix = cmatrix_t::Zero(1,1);
        reduced_matrix(0, 0) = m_simulator->trace();
        return reduced_matrix;
    }
    else
    {
        return m_simulator->reduced_density_matrix(qubits);
    }
}


/* bit-flip, phase-flip, bit-phase-flip, phase-damping, amplitude-damping, depolarizing*/
void DensityMatrixSimulator::set_noise_model(const NOISE_MODEL& model, const GateType& type, double prob)
{
    m_noisy.set_noise_model(model, type, prob);
    return;
}

void DensityMatrixSimulator::set_noise_model(const NOISE_MODEL& model, const std::vector<GateType> &types, double prob)
{
    for (const auto& val: types)
        m_noisy.set_noise_model(model, val, prob);

    return;
}

void DensityMatrixSimulator::set_noise_model(const NOISE_MODEL& model, const GateType& type, double prob, const QVec& qubits)
{
    m_noisy.set_noise_model(model, type, prob, NoiseUtils::get_qubits_addr(qubits));
    return;
}

void DensityMatrixSimulator::set_noise_model(const NOISE_MODEL& model, const std::vector<GateType> &types, double prob, const QVec& qubits)
{
    for (const auto& val : types)
        m_noisy.set_noise_model(model, val, prob, NoiseUtils::get_qubits_addr(qubits));

    return;
}


void DensityMatrixSimulator::set_noise_model(const NOISE_MODEL& model, const GateType& type, double prob, const std::vector<QVec>& qubits)
{
    m_noisy.set_noise_model(model, type, prob, NoiseUtils::get_qubits_addr(qubits));
    return;
}

/*decoherence error*/
void DensityMatrixSimulator::set_noise_model(const NOISE_MODEL& model, const GateType& type, double T1, double T2, double t_gate)
{
    m_noisy.set_noise_model(model, type, T1, T2, t_gate);
    return;
}

void DensityMatrixSimulator::set_noise_model(const NOISE_MODEL& model, const std::vector<GateType> &types, double T1, double T2, double t_gate)
{
    for (const auto& val : types)
        m_noisy.set_noise_model(model, val, T1, T2, t_gate);

    return;
}
void DensityMatrixSimulator::set_noise_model(const NOISE_MODEL& model, const GateType& type, double T1, double T2, double t_gate, const QVec& qubits)
{
    m_noisy.set_noise_model(model, type, T1, T2, t_gate, NoiseUtils::get_qubits_addr(qubits));
    return;
}

void DensityMatrixSimulator::set_noise_model(const NOISE_MODEL& model, const std::vector<GateType> &types, double T1, double T2, double t_gate, const QVec& qubits)
{
    for (const auto& val : types)
        m_noisy.set_noise_model(model, val, T1, T2, t_gate, NoiseUtils::get_qubits_addr(qubits));

    return;
}

void DensityMatrixSimulator::set_noise_model(const NOISE_MODEL& model, const GateType& type, double T1, double T2, double t_gate, const std::vector<QVec>& qubits)
{
    m_noisy.set_noise_model(model, type, T1, T2, t_gate, NoiseUtils::get_qubits_addr(qubits));
    return;
}
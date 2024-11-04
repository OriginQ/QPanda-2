#include <set>
#include <string>
#include <Core/Core.h>
#include "Core/Utilities/Tools/QStatMatrix.h"
#include "Core/Utilities/Tools/QProgFlattening.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include "Core/VirtualQuantumProcessor/Stabilizer/Stabilizer.h"

USING_QPANDA
using std::string;

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

void Stabilizer::init()
{
    _Config = { 6000,6000 };
    _start();
    m_simulator = std::make_shared<Clifford>();
    return;
}

std::map<std::string, size_t> Stabilizer::runWithConfiguration(QProg &prog, int shots)
{
    TraversalConfig traver_param;
    QProgCheck prog_check;
    prog_check.execute(prog.getImplementationPtr(), nullptr, traver_param);

    std::sort(traver_param.m_measure_cc.begin(), traver_param.m_measure_cc.end(), [&](CBit* a, CBit* b)
    {
        string current_cbit_a_number_str = a->getName().substr(1);
        string current_cbit_b_number_str = b->getName().substr(1);

        return stoul(current_cbit_a_number_str) < stoul(current_cbit_b_number_str);
    });

    std::vector<ClassicalCondition> cbits_vector;
    for (auto cbit : traver_param.m_measure_cc)
        cbits_vector.push_back(ClassicalCondition(cbit));

    std::map<string, size_t> result_map;

    if (m_noisy.enabled())
    {
        for (int i = 0; i < shots; i++)
        {
            auto noise_prog = m_noisy.generate_noise_prog(prog);
            run(noise_prog);

            std::string result_bin_str = _ResultToBinaryString(cbits_vector);
            std::reverse(result_bin_str.begin(), result_bin_str.end());
            if (result_map.find(result_bin_str) == result_map.end())
                result_map[result_bin_str] = 1;
            else
                result_map[result_bin_str] += 1;
        }
    }
    else
    {
        for (int i = 0; i < shots; i++)
        {
            run(prog);

            std::string result_bin_str = _ResultToBinaryString(cbits_vector);
            std::reverse(result_bin_str.begin(), result_bin_str.end());
            if (result_map.find(result_bin_str) == result_map.end())
                result_map[result_bin_str] = 1;
            else
                result_map[result_bin_str] += 1;
        }
    }

    return result_map;
}

void Stabilizer::run(QProg& node, bool reset_state)
{
    QVec qubits;
    auto qubits_num = get_allocate_qubits(qubits);

    /*initialize state*/
    if (reset_state)
        m_simulator->initialize(qubits_num);

    flatten(node);
    auto prog_node = node.getImplementationPtr();
    for (auto iterator = prog_node->getFirstNodeIter();
        iterator != prog_node->getEndNodeIter(); ++iterator)
    {
        auto qnode = std::dynamic_pointer_cast<QNode>(*iterator);

        switch (qnode->getNodeType())
        {
            case NodeType::MEASURE_GATE:
            {
                auto measure_node = std::dynamic_pointer_cast<AbstractQuantumMeasure>(qnode);
                auto qubit_addr = measure_node->getQuBit()->get_phy_addr();
                auto cbit = measure_node->getCBit();

                auto result = m_simulator->measure_and_update({ qubit_addr });
                cbit->set_val(result[0]);
                _QResult->append({ cbit->getName(), cbit->getValue() });

                break;
            }

            case NodeType::GATE_NODE:
            {
                auto gate_node = std::dynamic_pointer_cast<AbstractQGateNode>(qnode);
                apply_gate(gate_node);

                break;
            }
            case NodeType::RESET_NODE:
            {
                auto reset_node = std::dynamic_pointer_cast<AbstractQuantumReset>(qnode);
                apply_reset(reset_node);

                break;
            }

            default:
            {
                auto node_type = qnode->getNodeType();
                QCERR_AND_THROW(std::runtime_error, "invalid node for stablizer simulator");
            }
            
        }

    }

    return;
}

void Stabilizer::apply_reset(std::shared_ptr<AbstractQuantumReset> reset_node)
{
    auto qubit = reset_node->getQuBit()->get_phy_addr();
    auto result = m_simulator->measure_and_update({ qubit });

    if (result.front() == 1)
        m_simulator->append_x(qubit);

    return;
}

void Stabilizer::apply_gate(std::shared_ptr<AbstractQGateNode> gate_node)
{
    Qnum targets;
    Qnum controls;
    bool is_conj = false;

    parse_gate_node(gate_node, targets, controls, is_conj);

    switch (gate_node->getQGate()->getGateType())
    {
        case GateType::I_GATE:
        case GateType::BARRIER_GATE:
            break;

        case GateType::PAULI_X_GATE:
            if (controls.empty())
                m_simulator->append_x(targets[0]);
            else
                m_simulator->append_cx(controls[0], targets[0]);
            break;

        case GateType::PAULI_Y_GATE:
            if (controls.empty())
                m_simulator->append_y(targets[0]);
            else
                m_simulator->append_cy(controls[0], targets[0]);
            break;

        case GateType::PAULI_Z_GATE:
            if (controls.empty())
                m_simulator->append_z(targets[0]);
            else
                m_simulator->append_cz(controls[0], targets[0]);
            break;
            break;

        case GateType::CNOT_GATE:
            m_simulator->append_cx(targets[0], targets[1]);
            break;

        case GateType::CZ_GATE:
            m_simulator->append_cz(targets[0], targets[1]);
            break;

        case GateType::HADAMARD_GATE:
            m_simulator->append_h(targets[0]);
            break;

        case GateType::S_GATE:
        {
            if (is_conj)
            {
                m_simulator->append_z(targets[0]);
                m_simulator->append_s(targets[0]);
            }
            else
            {
                m_simulator->append_s(targets[0]);
            }

            break;
        }
        break;

        case GateType::SWAP_GATE:
            // SWAP(0, 1) => CNOT(0, 1) + CNOT(1, 0) + CNOT(0, 1)
            m_simulator->append_cx(targets[0], targets[1]);
            m_simulator->append_cx(targets[1], targets[0]);
            m_simulator->append_cx(targets[0], targets[1]);
            break;

            /*
            case GateType::CP_GATE:
            case GateType::CU_GATE:
            case GateType::CPHASE_GATE:

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

            case GateType::T_GATE:

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
            QCERR_AND_THROW(std::runtime_error,
                " Basic Clifford Simulator Only Support: { H, S, X, Y, Z, CNOT, CY, CZ, SWAP }");
        }

    }
}

prob_dict Stabilizer::probRunDict(QProg& prog, QVec qubits, int select_max)
{
    run(prog);

    Qnum qubits_addrs;
    for (const auto& qubit : qubits)
        qubits_addrs.emplace_back(qubit->get_phy_addr());

    prob_vec probs = m_simulator->pmeasure(qubits_addrs);

    size_t length = probs.size();

    prob_dict result_dict;
    for (auto i = 0; i < length; i++)
        result_dict.insert({ dec2bin(i, qubits.size()), probs[i] });

    return result_dict;
}


/* bit-flip, phase-flip, bit-phase-flip, phase-damping, depolarizing*/
void Stabilizer::set_noise_model(const NOISE_MODEL& model, const GateType& type, double prob)
{
    m_noisy.set_noise_model(model, type, prob);
    return;
}

void Stabilizer::set_noise_model(const NOISE_MODEL& model, const std::vector<GateType> &types, double prob)
{
    for (const auto& val : types)
        m_noisy.set_noise_model(model, val, prob);

    return;
}

void Stabilizer::set_noise_model(const NOISE_MODEL& model, const GateType& type, double prob, const QVec& qubits)
{
    m_noisy.set_noise_model(model, type, prob, NoiseUtils::get_qubits_addr(qubits));
    return;
}

void Stabilizer::set_noise_model(const NOISE_MODEL& model, const std::vector<GateType> &types, double prob, const QVec& qubits)
{
    for (const auto& val : types)
        m_noisy.set_noise_model(model, val, prob, NoiseUtils::get_qubits_addr(qubits));

    return;
}


void Stabilizer::set_noise_model(const NOISE_MODEL& model, const GateType& type, double prob, const std::vector<QVec>& qubits)
{
    m_noisy.set_noise_model(model, type, prob, NoiseUtils::get_qubits_addr(qubits));
    return;
}

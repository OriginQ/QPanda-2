#include <set>
#include <numeric>
#include "Core/Utilities/Tools/Uinteger.h"
#include "Core/VirtualQuantumProcessor/MPSQVM/NoiseSimulator.h"
#include "Core/VirtualQuantumProcessor/MPSQVM/NoiseDefinition.h"

USING_QPANDA
using namespace std;
using namespace QGATE_SPACE;
using SINGLE_GATE_FUNC = std::function<QGate(Qubit*)>;

static Qnum get_qnum(const std::vector<Qnum>& qnums)
{
    Qnum qubits;
    for (const auto &qvec : qnums)
    {
        std::for_each(qvec.begin(), qvec.end(), [&](size_t addr) {qubits.emplace_back(addr); });
    }

    return qubits;
}

static void assert_probs_equal_to_one(const std::vector<double> &probs)
{
    double sum_probs = accumulate(probs.begin(), probs.end(), .0);
    QPANDA_ASSERT(FLT_EPSILON < std::fabs(1. - sum_probs), "The sum of probabilities is not equal to 1");
}

static bool matrix_equal(const QStat &lhs, const QStat &rhs) 
{
    QPANDA_RETURN(lhs.size() != rhs.size(), false);
    for (size_t i = 0; i < lhs.size(); i++) 
    { 
        QPANDA_RETURN(fabs(lhs[i].real() - rhs[i].real()) > FLT_EPSILON || fabs(lhs[i].imag() - rhs[i].imag()) > FLT_EPSILON, false); 
    }

    return true;
}

static void unique_vector(Qnum &qubits)
{
    std::set<int> qubits_set(qubits.begin(), qubits.end());
    qubits.assign(qubits_set.begin(), qubits_set.end());
    return;
}

static bool optimize_karus_matrices(std::vector<QStat> &ops) 
{
    auto is_zero_matrix = [](const QStat &matrix)->bool 
    {        
        for (auto &ele : matrix) 
        { 
            QPANDA_RETURN(fabs(ele.real()) > FLT_EPSILON || fabs(ele.imag()) > FLT_EPSILON, false); 
        }        
        return true;    
    };

    for (size_t idx = 0; idx < ops.size(); idx++) 
    { 
        if (is_zero_matrix(ops[idx])) 
        { 
            ops.erase(ops.begin() + idx);            
            idx--; 
        } 
    }
    
    std::vector<double> factors;    
    factors.reserve(ops.size());    
    for (auto &op : ops) 
    {
        auto max_iter = std::max_element(op.begin(), op.end(), [](const qcomplex_t &lhs, const qcomplex_t &rhs) 
        {
            return std::norm(lhs) < std::norm(rhs);        
        });

        auto factor = sqrt(std::norm(*max_iter));        
        factors.push_back(factor);        
        for_each(op.begin(), op.end(), [&](qcomplex_t &value)->void { value /= factor; });
    }

    for (size_t i = 0; i < ops.size() - 1; i++) 
    { 
        for (size_t j = i + 1; j < ops.size(); j++) 
        { 
            if (matrix_equal(ops[i], ops[j])) 
            { 
                factors[i] = sqrt(std::norm(factors[i]) + std::norm(factors[j]));                
                factors.erase(factors.begin() + j);                
                ops.erase(ops.begin() + j);                
                j--;
            } 
        } 
    }
    for (size_t i = 0; i < ops.size(); i++) 
    { 
        for (size_t j = 0; j < ops[i].size(); j++) 
        { 
            ops[i][j] *= factors[i]; 
        } 
    }
    return true;
}

static std::vector<QStat> get_tensor_matrices(const std::vector<QStat>& matrices_a, const std::vector<QStat>& matrices_b)
{
    for (const auto& val : matrices_a)
    {
        QPANDA_ASSERT((1ull << 2) != val.size(), "karus matrices size error");
    }

    for (const auto& val : matrices_b)
    {
        QPANDA_ASSERT((1ull << 2) != val.size(), "karus matrices size error");
    }

    std::vector<QStat> tensor_results;

    for (const auto& matrix_a : matrices_a)
    {
        for (const auto& matrix_b : matrices_b)
        {
            tensor_results.emplace_back(QPanda::tensor(matrix_a, matrix_b));
        }
    }

    return tensor_results;
}

static size_t get_karus_error_qubit_num(const std::vector<QStat>& matrices)
{
    QPANDA_ASSERT(matrices.empty(), "karus matrices is empty");

    size_t qubit_num = 1;
    QPANDA_OP((1ull << 4) == matrices.front().size(), qubit_num = 2);
    
    for (const auto& matrix : matrices)
    {
        QPANDA_ASSERT(matrices.front().size() != matrix.size(), "matrices param error");
    }

    return qubit_num;
}

static std::vector<double> get_tensor_probs(const std::vector<double>& probs_a, const std::vector<double>& probs_b)
{
    assert_probs_equal_to_one(probs_a);
    assert_probs_equal_to_one(probs_b);

    QPANDA_ASSERT(probs_a.size() != probs_b.size(), "probs size error");

    std::vector<double> tensor_results;

    for (const auto& prob_a : probs_a)
    {
        for (const auto& prob_b : probs_b)
        {
            tensor_results.emplace_back(prob_a * prob_b);
        }
    }

    return tensor_results;
}

static std::vector<QStat> get_compose_karus_matrices(const std::vector<QStat>& karus_matrices_a,
    const std::vector<QStat>& karus_matrices_b)
{
    std::vector<QStat> karus_compose_results;

    for (const auto& karus_matrix_a : karus_matrices_a)
    {
        for (const auto& karus_matrix_b : karus_matrices_b)
        {
            karus_compose_results.emplace_back(karus_matrix_a * karus_matrix_b);
        }
    }

    optimize_karus_matrices(karus_compose_results);
    return karus_compose_results;
}

static bool is_single_gate(GateType type)
{
    return static_cast<size_t>(GateType::I_GATE)  == static_cast<size_t>(type) || (
           static_cast<size_t>(GateType::P0_GATE) <= static_cast<size_t>(type) &&
           static_cast<size_t>(GateType::U4_GATE) >= static_cast<size_t>(type));
}

static bool is_rotation_gate(GateType type)
{
    return GateType::RX_GATE == type     || GateType::RY_GATE == type || 
           GateType::RZ_GATE == type     || GateType::CPHASE_GATE == type || 
           GateType::ISWAP_THETA_GATE == type;
}

static int random_discrete(const std::vector<double> &probs)
{ 
    static RandomEngine19937 rng;
    return rng.random_discrete(probs);
}

bool NonKarusError::has_readout_error()
{
    return !m_readout_probabilities.empty() && !m_readout_qubits.empty();
}

bool NonKarusError::has_measure_qubit(size_t qubit)
{
    QPANDA_RETURN(m_measure_qubits.empty(), true);

    auto iter = std::find(m_measure_qubits.begin(), m_measure_qubits.end(), qubit);
    return iter != m_measure_qubits.end();
}

void NonKarusError::set_measure_error(int qubit, std::vector<QStat>& karus_matrices)
{
    auto global_iter = m_measure_error_karus_matrices.find(-1);

    if (m_measure_error_karus_matrices.end() != global_iter)
    {
        return;
    }

    m_measure_error_karus_matrices[qubit] = karus_matrices;
}

void NonKarusError::get_measure_error(int qubit, std::vector<QStat>& karus_matrices)
{
    auto iter = m_measure_error_karus_matrices.find(-1);
    if (m_measure_error_karus_matrices.end() != iter)
    {
        karus_matrices = iter->second;
        return;
    }

    iter = m_measure_error_karus_matrices.find(qubit);

    QPANDA_ASSERT(iter == m_measure_error_karus_matrices.end(), "get_measure_error");

    karus_matrices = iter->second;
    return;
}

void NonKarusError::set_measure_qubit(const Qnum& qubits) 
{ 
    for (auto qubit : qubits)
    {
        m_measure_qubits.emplace_back(qubit);
    }
   
    unique_vector(m_measure_qubits);
}

bool NonKarusError::has_non_karus_error()
{
    bool has_reset_error_params = std::fabs(m_reset_p0) > FLT_EPSILON && std::fabs(m_reset_p1) > FLT_EPSILON;
    bool has_rotation_error_params = std::fabs(m_rotation_param) > FLT_EPSILON;

    bool has_measure_error_params = !m_measure_error_karus_matrices.empty();
    bool has_readout_error_params = !m_readout_qubits.empty() && !m_readout_probabilities.empty();

    return has_reset_error_params || has_rotation_error_params || has_measure_error_params || has_readout_error_params;
}

void NonKarusError::set_readout_error(const std::vector<std::vector<double>> &probabilities, const Qnum& qubits)
{ 
    for (const auto& probs_vec : m_readout_probabilities)
    {
        assert_probs_equal_to_one(probs_vec);
    }

    QPANDA_ASSERT(m_readout_qubits.size() != (m_readout_probabilities.size() / 2), "readour error");

    m_readout_qubits = qubits;
    m_readout_probabilities = probabilities;

    return;
}

bool NonKarusError::get_readout_result(bool result, size_t qubit)
{
     auto tar_iter = std::find(m_readout_qubits.begin(), m_readout_qubits.end(), qubit);
     QPANDA_RETURN(tar_iter == m_readout_qubits.end(), result);

     try
     {
         auto probs_idx = 2 * std::distance(m_readout_qubits.begin(), tar_iter);
         auto probs_vec = m_readout_probabilities[probs_idx + (size_t)result];

         return probs_vec[0] > random_generator19937() ? false : true;
     }
     catch (...)
     {
         throw run_fail("get_readout_result error");
     }
}

bool KarusError::has_karus_error()
{
    bool has_karus_matrices_error = !m_karus_matrices.empty();
    bool has_unitary_matrices_error = !m_unitary_probs.empty() && !m_unitary_matrices.empty();

    return has_karus_matrices_error || has_unitary_matrices_error;
}

KarusError::KarusError(const std::vector<QStat>& karus_matrices)
{
    m_karus_matrices = karus_matrices;
    m_karus_error_type = KarusErrorType::KARUS_MATRIICES;
    m_qubit_num = get_karus_error_qubit_num(karus_matrices);
}

KarusError::KarusError(const std::vector<QStat>& unitary_matrices, const std::vector<double>& probs_vec)
{
    m_unitary_probs = probs_vec;
    m_unitary_matrices = unitary_matrices;
    m_karus_error_type = KarusErrorType::UNITARY_MATRIICES;
    m_qubit_num = get_karus_error_qubit_num(unitary_matrices);
}

void KarusError::set_unitary_probs(std::vector<double>& probs_vec)
{ 
    m_unitary_probs.swap(probs_vec);
    return;
}

void KarusError::get_unitary_probs(std::vector<double>& probs_vec) const
{ 
    probs_vec = m_unitary_probs;
    return;
}

void KarusError::set_unitary_matrices(std::vector<QStat>& unitary_matrices) 
{
    m_unitary_matrices.swap(unitary_matrices); 
    return;
}

void KarusError::get_unitary_matrices(std::vector<QStat>& unitary_matrices) const
{
    unitary_matrices = m_unitary_matrices;
    return;
}

void KarusError::set_karus_matrices(std::vector<QStat>& karus_matrices) 
{ 
    m_karus_matrices.swap(karus_matrices);
    return;
}

void KarusError::get_karus_matrices(std::vector<QStat>& karus_matrices) const
{
    if (KarusErrorType::KARUS_MATRIICES == m_karus_error_type) 
    {
        karus_matrices = m_karus_matrices;
    }
    else
    {
        QPANDA_ASSERT(m_unitary_matrices.size() != m_unitary_probs.size(), "unitary matrices size error");

        std::vector<QStat> result_karus_matrices;
        for (int i = 0; i < m_unitary_matrices.size(); i++)
        {
            result_karus_matrices.emplace_back(m_unitary_matrices[i] * sqrt(m_unitary_probs[i]));
        }

        karus_matrices =  result_karus_matrices;
    }
}


KarusError KarusError::tensor(const KarusError& karus_error)
{
    QPANDA_ASSERT(1 != get_qubit_num(), "tensor qubit num error");
    QPANDA_ASSERT(1 != karus_error.get_qubit_num(), "tensor qubit num error");

    std::vector<QStat> karus_matrices_a;
    get_karus_matrices(karus_matrices_a);

    std::vector<QStat> karus_matrices_b;
    karus_error.get_karus_matrices(karus_matrices_b);

    auto tensor_karus_matrices = get_tensor_matrices(karus_matrices_a, karus_matrices_b);
    return KarusError(tensor_karus_matrices);
}

KarusError KarusError::expand(const KarusError& karus_error)
{
    QPANDA_ASSERT(1 != get_qubit_num(), "tensor qubit num error");
    QPANDA_ASSERT(1 != karus_error.get_qubit_num(), "tensor qubit num error");

    std::vector<QStat> karus_matrices_a;
    get_karus_matrices(karus_matrices_a);

    std::vector<QStat> karus_matrices_b;
    karus_error.get_karus_matrices(karus_matrices_b);

    auto tensor_karus_matrices = get_tensor_matrices(karus_matrices_b, karus_matrices_a);
    return KarusError(tensor_karus_matrices);
}

KarusError KarusError::compose(const KarusError& karus_error)
{
    QPANDA_ASSERT(karus_error.get_qubit_num() != get_qubit_num(), "compose qubit num error");

    std::vector<QStat> karus_matrices_a;
    get_karus_matrices(karus_matrices_a);

    std::vector<QStat> karus_matrices_b;
    karus_error.get_karus_matrices(karus_matrices_b);

    auto tensor_karus_matrices = get_compose_karus_matrices(karus_matrices_a, karus_matrices_b);
    return KarusError(tensor_karus_matrices);
}

bool NoiseSimulator::has_error_for_current_gate(GateType gate_type, QVec qubits)
{
    if (is_single_gate(gate_type))
    {
        auto iter = m_single_qubits.find(gate_type);
        QPANDA_RETURN(m_single_qubits.end() == iter, false);

        QPANDA_RETURN(iter->second.empty(), true);

        auto tar_iter = std::find(iter->second.begin(), iter->second.end(), qubits[0]->get_phy_addr());
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

        auto ctr_qubit = qubits[0]->get_phy_addr();
        auto tar_qubit = qubits[1]->get_phy_addr();

        QPANDA_RETURN(find_pair(iter->second, std::make_pair(ctr_qubit, tar_qubit)), true);
    }

    return false;
}

void NoiseSimulator::set_rotation_error(double param)
{
    m_non_karus_error.set_rotation_error(param);
    return;
}

void NoiseSimulator::set_reset_error(double p0_param, double p1_param)
{
    QPANDA_ASSERT(p0_param < 0. || p0_param < 0. || (p0_param + p1_param) > 1., "reset param error");
    m_non_karus_error.set_reset_error(p0_param, p1_param);
    return;
}

void NoiseSimulator::execute(std::shared_ptr<AbstractQuantumMeasure>  cur_node,
    std::shared_ptr<QNode> parent_node, QCircuitConfig &config)
{
    //get qubit addr
    auto qubits_addr = cur_node->getQuBit()->get_phy_addr();

    //measurement error
    if (m_non_karus_error.has_measure_error() && m_non_karus_error.has_measure_qubit(qubits_addr))
    {
        std::vector<QStat> karus_matrices;
        m_non_karus_error.get_measure_error(qubits_addr, karus_matrices);
        handle_karus_matrices(karus_matrices, { cur_node->getQuBit() });
    }

    // if the Measure operation is not at the end of the quantum program , 
    // Measure operation need to be performed immediately
    auto result = m_mps_qpu->qubitMeasure(qubits_addr);

    //readout error
    if (m_non_karus_error.has_readout_error())
    {
        result = m_non_karus_error.get_readout_result(result, qubits_addr);
    }

    auto cbit = cur_node->getCBit();
    cbit->set_val(result);
    m_result->append({ cbit->getName(), cbit->getValue() });

    return;
}

void NoiseSimulator::execute(std::shared_ptr<AbstractControlFlowNode> cur_node,
    std::shared_ptr<QNode> parent_node, QCircuitConfig &config)
{
    QCERR("not support controlflow");
    throw std::runtime_error("not support controlflow");
}

void NoiseSimulator::execute(std::shared_ptr<AbstractQuantumCircuit> cur_node,
    std::shared_ptr<QNode> parent_node, QCircuitConfig &config)
{
    if (nullptr == cur_node)
    {
        QCERR("pQCircuit is nullptr");
        throw std::invalid_argument("pQCircuit is nullptr");
    }

    auto aiter = cur_node->getFirstNodeIter();

    if (aiter == cur_node->getEndNodeIter())
        return;

    auto pNode = std::dynamic_pointer_cast<QNode>(cur_node);

    if (nullptr == pNode)
    {
        QCERR("Unknown internal error");
        throw std::runtime_error("Unknown internal error");
    }

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

void NoiseSimulator::execute(std::shared_ptr<AbstractQuantumProgram>  cur_node,
    std::shared_ptr<QNode> parent_node, QCircuitConfig &config)
{
    Traversal::traversal(cur_node, *this, config);
}

void NoiseSimulator::execute(std::shared_ptr<AbstractClassicalProg>  cur_node,
    std::shared_ptr<QNode> parent_node, QCircuitConfig &config)
{
    QCERR("not support ClassicalProg");
    throw std::runtime_error("not support ClassicalProg");
}

void NoiseSimulator::execute(std::shared_ptr<AbstractQuantumReset>  cur_node,
    std::shared_ptr<QNode> parent_node, QCircuitConfig &config)
{  
    auto reset_p0 = m_non_karus_error.get_reset_p0_error();
    auto reset_p1 = m_non_karus_error.get_reset_p1_error();
    if (fabs(reset_p0 - .0) > FLT_EPSILON || fabs(reset_p1 - .0) > FLT_EPSILON)
    {
        /*probabilities p0 for reset to |0>*/
        /*probabilities p1 for reset to |1>*/
        /*probabilities 1 - p0 - p1 do nothing*/

        prob_vec reset_params = { reset_p0 , reset_p1, 1 - reset_p0 - reset_p1 };

        auto random_index = random_discrete(reset_params);
        if (!random_index)
        {
            /*reset to |0>*/
        }
        else if (1 == random_index)
        {
            /*reset to |0>*/
            /*apply pauli X gate*/
        }
        else
        {
            //do nothing
        }
    }
    else
    {
        /*reset to |0>*/
    }

    QCERR("not support Reset");  
}

KarusError NoiseSimulator::get_karus_error(GateType gate_type, const QVec& qubits)
{
    if (is_single_gate(gate_type))
    {
        auto tar_qubit = qubits[0]->get_phy_addr();

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
        auto ctr_qubit = qubits[0]->get_phy_addr();
        auto tar_qubit = qubits[1]->get_phy_addr();

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


void NoiseSimulator::handle_noise_gate(GateType gate_type, QVec& qubits)
{
    auto karus_error = get_karus_error(gate_type, qubits);
    auto karus_error_qubit = karus_error.get_qubit_num();

    QPANDA_ASSERT(is_single_gate(gate_type) && (2 == karus_error_qubit), "two qubits error can not apply on single qubit gate");

    if (KarusErrorType::KARUS_MATRIICES == karus_error.get_karus_error_type())
    {
        std::vector<QStat> karus_matrices;
        karus_error.get_karus_matrices(karus_matrices);

        if (1 == qubits.size())
        {
            handle_karus_matrices(karus_matrices, qubits);
        }
        else
        {
            if (1 == karus_error_qubit)
            {
                //karus matrices way 1
                //auto tensor_karus_matrices = get_tensor_matrices(karus_matrices, karus_matrices);
                //optimize_karus_matrices(tensor_results);
                //handle_karus_matrices(tensor_karus_matrices, qubits);

                //karus matrices way 2
                auto karus_matrices_a = karus_matrices;
                auto karus_matrices_b = karus_matrices;
                handle_karus_matrices(karus_matrices_a, { qubits[0] });
                handle_karus_matrices(karus_matrices_b, { qubits[1] });
            }
            else
            {
                handle_karus_matrices(karus_matrices, qubits);
            }
        }
    }
    else
    {
        std::vector<double> probs_vec;
        karus_error.get_unitary_probs(probs_vec);

        std::vector<QStat> unitary_matrices;
        karus_error.get_unitary_matrices(unitary_matrices);

        if (1 == qubits.size())
        {
            handle_unitary_matrices(unitary_matrices, probs_vec, qubits);
        }
        else
        {
            if (1 == karus_error_qubit)
            {
                //unitary matrices way 1
                auto tensor_probs = get_tensor_probs(probs_vec, probs_vec);
                auto tensor_unitary_matrices = get_tensor_matrices(unitary_matrices, unitary_matrices);
                handle_unitary_matrices(tensor_unitary_matrices, tensor_probs, qubits);

                //unitary matrices way 2
                //handle_unitary_matrices(unitary_matrices, probs_vec, { qubits[0] });
                //handle_unitary_matrices(unitary_matrices, probs_vec, { qubits[1] });
            }
            else
            {
                handle_unitary_matrices(unitary_matrices, probs_vec, qubits);
            }
        }
    }

    return;
}

void NoiseSimulator::set_readout_error(const std::vector<std::vector<double>>& readout_params, const Qnum& qubits)
{
    m_non_karus_error.set_readout_error(readout_params, qubits);
    return;
}

std::shared_ptr<AbstractQGateNode> NoiseSimulator::handle_rotation_error(std::shared_ptr<AbstractQGateNode> gate_node)
{
    /*gate type*/
    GateType gate_type = (GateType)gate_node->getQGate()->getGateType();
    auto gate_name = TransformQGateType::getInstance()[gate_type];

    /*get rotation error param*/
    double rotation_error = m_non_karus_error.get_rotation_error();
    QPANDA_RETURN(fabs(rotation_error) < FLT_EPSILON || !is_rotation_gate(gate_type), gate_node);

    /*targets*/
    QVec targets;
    gate_node->getQuBitVector(targets);

    /*construct param offset*/
    auto param_offset = random_generator19937(-rotation_error / 2., rotation_error / 2.);

    /*get parameter*/
    auto gate_parameter = dynamic_cast<AbstractSingleAngleParameter *>(gate_node->getQGate());
    auto gate_param = gate_parameter->getParameter() + param_offset;

    switch (gate_type)
    {
    case GateType::RX_GATE:
    {
        auto gate = QPanda::RX(targets[0], gate_param);
        gate.setDagger(gate_node->isDagger());
        return gate.getImplementationPtr();
    }

    case GateType::RY_GATE:
    {
        auto gate = QPanda::RY(targets[0], gate_param);
        gate.setDagger(gate_node->isDagger());
        return gate.getImplementationPtr();
    }

    case GateType::RZ_GATE:
    {
        auto gate = QPanda::RZ(targets[0], gate_param);
        gate.setDagger(gate_node->isDagger());
        return gate.getImplementationPtr();
    }

    case GateType::U1_GATE:
    {
        auto gate = QPanda::U1(targets[0], gate_param);
        gate.setDagger(gate_node->isDagger());
        return gate.getImplementationPtr();
    }

    case GateType::CPHASE_GATE:
    {
        auto gate = QPanda::CR(targets[0], targets[1], gate_param);
        gate.setDagger(gate_node->isDagger());
        return gate.getImplementationPtr();
    }

    case GateType::ISWAP_THETA_GATE:
    {
        auto gate = QPanda::iSWAP(targets[0], targets[1], gate_param);
        gate.setDagger(gate_node->isDagger());
        return gate.getImplementationPtr();
    }

    case GateType::U2_GATE:
    {
        QGATE_SPACE::U2 *u2_gate = dynamic_cast<QGATE_SPACE::U2*>(gate_node->getQGate());
        auto phi = u2_gate->get_phi();
        auto lambda = u2_gate->get_lambda();

        phi = random_generator19937(phi - rotation_error / 2., phi + rotation_error / 2.);
        lambda = random_generator19937(lambda - rotation_error / 2., lambda + rotation_error / 2.);

        auto gate = QPanda::U2(targets[0], phi, lambda);
        gate.setDagger(gate_node->isDagger());
        return gate.getImplementationPtr();
    }

    case GateType::U3_GATE:
    {
        QGATE_SPACE::U3 *u3_gate = dynamic_cast<QGATE_SPACE::U3*>(gate_node->getQGate());
        auto theta = u3_gate->get_theta();
        auto phi = u3_gate->get_phi();
        auto lambda = u3_gate->get_lambda();

        theta = random_generator19937(theta - rotation_error / 2., theta + rotation_error / 2.);
        phi = random_generator19937(phi - rotation_error / 2., phi + rotation_error / 2.);
        lambda = random_generator19937(lambda - rotation_error / 2., lambda + rotation_error / 2.);

        auto gate = QPanda::U3(targets[0], theta, phi, lambda);
        gate.setDagger(gate_node->isDagger());
        return gate.getImplementationPtr();
    }

    default:return gate_node;
    }
}

void NoiseSimulator::execute(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitConfig &config)
{
    QVec controls;
    cur_node->getControlVector(controls);
    QPANDA_ASSERT(!controls.empty(), "unsupported control qubits");

    QVec targets;
    cur_node->getQuBitVector(targets);

    //handle rotation error
    auto gate_node = handle_rotation_error(cur_node);

    //gate type
    auto gate_type = (GateType)gate_node->getQGate()->getGateType();
    bool is_dagger = gate_node->isDagger() ^ config._is_dagger;

    handle_quantum_gate(gate_node, is_dagger);
    QPANDA_OP(has_error_for_current_gate(gate_type, targets), handle_noise_gate(gate_type, targets));
}

void NoiseSimulator::handle_quantum_gate(std::shared_ptr<AbstractQGateNode> gate, bool is_dagger)
{
    auto gate_type = (GateType)gate->getQGate()->getGateType();

    QStat gate_matrix;
    gate->getQGate()->getMatrix(gate_matrix);

    QVec targets;
    gate->getQuBitVector(targets);

    if (is_single_gate(gate_type))
    {
        auto tar_qubit = targets[0]->get_phy_addr();
        m_mps_qpu->unitarySingleQubitGate(tar_qubit, gate_matrix, is_dagger, static_cast<GateType>(gate_type));
    }
    else
    {
        auto ctr_qubit = targets[0]->get_phy_addr();
        auto tar_qubit = targets[1]->get_phy_addr();
        m_mps_qpu->unitaryDoubleQubitGate(ctr_qubit, tar_qubit, gate_matrix, is_dagger, static_cast<GateType>(gate_type));
    }

    return;
}

void NoiseSimulator::set_gate_and_qnum(GateType gate_type, const Qnum& qubits_vec)
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

void NoiseSimulator::set_gate_and_qnums(GateType gate_type, const std::vector<Qnum>& qubits_vecs)
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

void NoiseSimulator::update_karus_error_tuple(GateType gate_type, int tar_qubit, const KarusError& karus_error)
{
    QPANDA_ASSERT(!is_single_gate(gate_type), "update karus error tuple error");

    for (auto& val : m_one_qubit_karus_error_tuple)
    {
        auto type = std::get<0>(val);
        auto addr = std::get<1>(val);

        if ((gate_type == type) && ( -1 == tar_qubit))
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

void NoiseSimulator::update_karus_error_tuple(GateType gate_type, int ctr_qubit, int tar_qubit, const KarusError& karus_error)
{
    QPANDA_ASSERT(is_single_gate(gate_type), "update karus error tuple error");

    for (auto& val : m_two_qubit_karus_error_tuple)
    {
        auto type = std::get<0>(val);
        auto ctr_addr = std::get<1>(val);
        auto tar_addr = std::get<2>(val);

        if ((gate_type == type) && (-1  == ctr_qubit) && (-1 == tar_qubit))
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

void NoiseSimulator::set_single_karus_error_tuple(GateType gate_type, const KarusError &karus_error, const Qnum& qubits)
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


void NoiseSimulator::set_double_karus_error_tuple(GateType gate_type, const KarusError &karus_error, const std::vector<Qnum>& qubits_vec)
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


void NoiseSimulator::set_noise_model(NOISE_MODEL model, GateType gate_type, double param)
{
    QPANDA_ASSERT(0. > param || param > 1., "param range error");

    if (model == NOISE_MODEL::DAMPING_KRAUS_OPERATOR)
    {
        auto karus_matrices = get_noise_model_karus_matrices(DAMPING_KRAUS_OPERATOR, { param });
        auto karus_error = KarusError(karus_matrices);

        set_gate_and_qnums(gate_type, {});

        QPANDA_OP(is_single_gate(gate_type), set_single_karus_error_tuple(gate_type, karus_error, {}));
        QPANDA_OP(!is_single_gate(gate_type), set_double_karus_error_tuple(gate_type, karus_error, {}));
        return;
    }
    else
    {
        auto unitary_probs = get_noise_model_unitary_probs(model, param);
        auto unitary_matrices = get_noise_model_unitary_matrices(model, param);

        auto karus_error = KarusError(unitary_matrices, unitary_probs);

        set_gate_and_qnums(gate_type, {});

        QPANDA_OP(is_single_gate(gate_type), set_single_karus_error_tuple(gate_type, karus_error, {}));
        QPANDA_OP(!is_single_gate(gate_type), set_double_karus_error_tuple(gate_type, karus_error, {}));
    }

    return;
}

void NoiseSimulator::set_noise_model(NOISE_MODEL model, GateType gate_type, double T1, double T2, double time_param)
{
    QPANDA_ASSERT(NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR != model, "model != DECOHERENCE_KRAUS_OPERATOR");

    set_gate_and_qnums(gate_type, {});

    std::vector<QStat> karus_matrices = get_noise_model_karus_matrices(DECOHERENCE_KRAUS_OPERATOR, { T1,T2,time_param });

    auto karus_error = KarusError(karus_matrices);

    QPANDA_OP(is_single_gate(gate_type), set_single_karus_error_tuple(gate_type, karus_error, {}));
    QPANDA_OP(!is_single_gate(gate_type), set_double_karus_error_tuple(gate_type, karus_error, {}));

    return;
}

void NoiseSimulator::set_noise_model(NOISE_MODEL model, GateType gate_type, double param, const std::vector<Qnum>& qubits_vecs)
{
    QPANDA_ASSERT(0. > param || param > 1., "param range error");

    set_gate_and_qnums(gate_type, qubits_vecs);

    auto unitary_probs = get_noise_model_unitary_probs(model, param);
    auto unitary_matrices = get_noise_model_unitary_matrices(model, param);

    auto karus_error = KarusError(unitary_matrices, unitary_probs);

    QPANDA_OP(is_single_gate(gate_type), set_single_karus_error_tuple(gate_type, karus_error, get_qnum(qubits_vecs)));
    QPANDA_OP(!is_single_gate(gate_type), set_double_karus_error_tuple(gate_type, karus_error, qubits_vecs));

    return;
}

void NoiseSimulator::set_noise_model(NOISE_MODEL model, GateType gate_type, double param, const Qnum& qubits_vec)
{
    QPANDA_ASSERT(0. > param || param > 1., "param range error");
    QPANDA_ASSERT(!is_single_gate(gate_type), "set_noise_model gate type error");

    auto unitary_probs = get_noise_model_unitary_probs(model, param);
    auto unitary_matrices = get_noise_model_unitary_matrices(model, param);

    auto karus_error = KarusError(unitary_matrices, unitary_probs);

    set_gate_and_qnum(gate_type, qubits_vec);

    set_single_karus_error_tuple(gate_type, karus_error, qubits_vec);

    return;
}


void NoiseSimulator::set_noise_model(NOISE_MODEL model, GateType gate_type, double T1, double T2, double time_param, const std::vector<Qnum>& qubits_vecs)
{
    QPANDA_ASSERT(NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR != model, "model != DECOHERENCE_KRAUS_OPERATOR");

    set_gate_and_qnums(gate_type, qubits_vecs);

    std::vector<QStat> karus_matrices = get_noise_model_karus_matrices(DECOHERENCE_KRAUS_OPERATOR, { T1,T2,time_param });

    auto karus_error = KarusError(karus_matrices);

    QPANDA_OP(is_single_gate(gate_type), set_single_karus_error_tuple(gate_type, karus_error, get_qnum(qubits_vecs)));
    QPANDA_OP(!is_single_gate(gate_type), set_double_karus_error_tuple(gate_type, karus_error, qubits_vecs));

    return;
}

void NoiseSimulator::set_noise_model(NOISE_MODEL model, GateType gate_type, double T1, double T2, double time_param, const Qnum& qubits_vec)
{
    QPANDA_ASSERT(!is_single_gate(gate_type), "set_noise_model gate type error");
    QPANDA_ASSERT(NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR != model, "model != DECOHERENCE_KRAUS_OPERATOR");

    set_gate_and_qnum(gate_type, qubits_vec);

    std::vector<QStat> karus_matrices = get_noise_model_karus_matrices(DECOHERENCE_KRAUS_OPERATOR, { T1,T2,time_param });

    auto karus_error = KarusError(karus_matrices);
    set_single_karus_error_tuple(gate_type, karus_error, qubits_vec);

    return;
}

void NoiseSimulator::set_measure_error(NOISE_MODEL model, double param)
{
    QPANDA_ASSERT(0. > param || param > 1., "param range error");

    m_non_karus_error.set_measure_qubit({});

    std::vector<QStat> karus_matrices = get_noise_model_karus_matrices(model, { param });
    m_non_karus_error.set_measure_error(-1, karus_matrices);
}

void NoiseSimulator::set_measure_error(NOISE_MODEL model, double param, const Qnum& qubits)
{
    QPANDA_ASSERT(0. > param || param > 1., "param range error");

    m_non_karus_error.set_measure_qubit({qubits});

    std::vector<QStat> karus_matrices = get_noise_model_karus_matrices(model, { param });
    std::for_each(qubits.begin(), qubits.end(), [&](size_t qubit) { m_non_karus_error.set_measure_error(qubit, karus_matrices); });
}

void NoiseSimulator::set_measure_error(NOISE_MODEL model, double T1, double T2, double time_param)
{
    QPANDA_ASSERT(NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR != model, "model != DECOHERENCE_KRAUS_OPERATOR");

    m_non_karus_error.set_measure_qubit({});

    std::vector<QStat> karus_matrices = get_noise_model_karus_matrices(model, { T1,T2,time_param });
    m_non_karus_error.set_measure_error(-1, karus_matrices);
}

void NoiseSimulator::set_measure_error(NOISE_MODEL model, double T1, double T2, double time_param, const Qnum& qubits)
{
    QPANDA_ASSERT(NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR != model, "model != DECOHERENCE_KRAUS_OPERATOR");

    m_non_karus_error.set_measure_qubit({qubits});

    std::vector<QStat> karus_matrices = get_noise_model_karus_matrices(model, { T1,T2,time_param });
    std::for_each(qubits.begin(), qubits.end(), [&](size_t qubit) { m_non_karus_error.set_measure_error(qubit, karus_matrices); });
}

void NoiseSimulator::set_mixed_unitary_error(GateType gate_type, const std::vector<QStat>& karus_matrices)
{
    set_gate_and_qnums(gate_type, {});

    auto qubit = get_karus_error_qubit_num(karus_matrices);
    QPANDA_ASSERT(1 == qubit && !is_single_gate(gate_type), "set_mixed_unitary_error");
    QPANDA_ASSERT(2 == qubit &&  is_single_gate(gate_type), "set_mixed_unitary_error");

    auto karus_error = KarusError(karus_matrices);
    QPANDA_OP(is_single_gate(gate_type), set_single_karus_error_tuple(gate_type, karus_error, {}));
    QPANDA_OP(!is_single_gate(gate_type), set_double_karus_error_tuple(gate_type, karus_error, {}));

    return;
}

void NoiseSimulator::set_mixed_unitary_error(GateType gate_type, const std::vector<QStat>& karus_matrices, const std::vector<Qnum>& qubits_vecs)
{
    set_gate_and_qnums(gate_type, qubits_vecs);

    auto qubit = get_karus_error_qubit_num(karus_matrices);
    QPANDA_ASSERT(1 == qubit && !is_single_gate(gate_type), "set_mixed_unitary_error");
    QPANDA_ASSERT(2 == qubit &&  is_single_gate(gate_type), "set_mixed_unitary_error");

    auto karus_error = KarusError(karus_matrices);
    QPANDA_OP(is_single_gate(gate_type), set_single_karus_error_tuple(gate_type, karus_error, get_qnum(qubits_vecs)));
    QPANDA_OP(!is_single_gate(gate_type), set_double_karus_error_tuple(gate_type, karus_error, qubits_vecs));

    return;
}

void NoiseSimulator::set_mixed_unitary_error(GateType gate_type, const std::vector<QStat>& unitary_matrices, const std::vector<double>& probs_vec, const std::vector<Qnum>& qubits_vecs)
{
    assert_probs_equal_to_one(probs_vec);
    set_gate_and_qnums(gate_type, qubits_vecs);

    auto qubit = get_karus_error_qubit_num(unitary_matrices);
    QPANDA_ASSERT(1 == qubit && !is_single_gate(gate_type), "set_mixed_unitary_error");
    QPANDA_ASSERT(2 == qubit &&  is_single_gate(gate_type), "set_mixed_unitary_error");

    auto karus_error = KarusError(unitary_matrices, probs_vec);
    QPANDA_OP(is_single_gate(gate_type), set_single_karus_error_tuple(gate_type, karus_error, get_qnum(qubits_vecs)));
    QPANDA_OP(!is_single_gate(gate_type), set_double_karus_error_tuple(gate_type, karus_error, qubits_vecs));

    return;
}

void NoiseSimulator::set_mixed_unitary_error(GateType gate_type, const std::vector<QStat>& unitary_matrices, const std::vector<double>& probs_vec)
{
    assert_probs_equal_to_one(probs_vec);
    set_gate_and_qnums(gate_type, {});

    auto qubit = get_karus_error_qubit_num(unitary_matrices);
    QPANDA_ASSERT(1 == qubit && !is_single_gate(gate_type), "set_mixed_unitary_error");
    QPANDA_ASSERT(2 == qubit &&  is_single_gate(gate_type), "set_mixed_unitary_error");

    auto karus_error = KarusError(unitary_matrices, probs_vec);
    QPANDA_OP(is_single_gate(gate_type), set_single_karus_error_tuple(gate_type, karus_error, {}));
    QPANDA_OP(!is_single_gate(gate_type), set_double_karus_error_tuple(gate_type, karus_error, {}));

    return;
}

void NoiseSimulator::set_mps_qpu_and_result(std::shared_ptr<MPSImplQPU> mps_qpu, QResult* result) 
{ 
    QPANDA_ASSERT(nullptr == mps_qpu, "mps_qpu is nullptr");
    QPANDA_ASSERT(nullptr == result , "m_esult is nullptr");

    m_mps_qpu = mps_qpu; 
    m_result = result; 
}

bool NoiseSimulator::has_quantum_error() 
{ 
    return !m_single_qubits.empty()|| !m_double_qubits.empty() || m_non_karus_error.has_non_karus_error();
}

void NoiseSimulator::handle_unitary_matrices(const std::vector<QStat>& unitary_matrices, const std::vector<double> unitary_probs, const QVec& qubits)
{
    auto index = random_discrete(unitary_probs);

    QStat matrix = unitary_matrices[index];

    if (1 == qubits.size())
    {
        QPANDA_ASSERT((1ull << 2) != matrix.size(), "unitary matrix error");

        auto tar_qubit = qubits[0]->get_phy_addr();
        m_mps_qpu->unitarySingleQubitGate(tar_qubit, matrix, false, GateType::GATE_UNDEFINED);
    }
    else
    {
        QPANDA_ASSERT((1ull << 4) != matrix.size(), "unitary matrix error");

        auto ctr_qubit = qubits[0]->get_phy_addr();
        auto tar_qubit = qubits[1]->get_phy_addr();
        m_mps_qpu->unitaryDoubleQubitGate(ctr_qubit, tar_qubit, matrix, false, GateType::GATE_UNDEFINED);
    }

    return;
}

void NoiseSimulator::handle_karus_matrices(std::vector<QStat>& matrixs, const QVec& qubits)
{
    Qnum qubits_addr;
    for (auto qubit : qubits)
    {
        qubits_addr.emplace_back(qubit->get_phy_addr());
    }
    std::sort(qubits_addr.begin(), qubits_addr.end());

    bool complete = false;
    double sum_probs = .0;
    for (size_t j = 0; j < matrixs.size() - 1; j++)
    {
        cmatrix_t matrix = QStat_to_Eigen(matrixs[j]);
        double p = m_mps_qpu->expectation_value(qubits_addr, matrix);
        sum_probs += p;

        if (sum_probs > random_generator19937())
        {
            for (auto &elt : matrixs[j])
            {
                elt *= (1 / std::sqrt(p));
            }

            m_mps_qpu->unitaryQubitGate(qubits_addr, matrixs[j], false);
            complete = true;
            break;
        }
    }

    if (complete == false)
    {
        qcomplex_t renorm = 1 / std::sqrt(1. - sum_probs);
        m_mps_qpu->unitaryQubitGate(qubits_addr, renorm * matrixs.back(), false);
    }

    return;
}
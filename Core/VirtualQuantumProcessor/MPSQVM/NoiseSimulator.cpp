#include "Core/VirtualQuantumProcessor/MPSQVM/NoiseSimulator.h"

using SINGLE_GATE_FUNC = std::function<QGate(Qubit*)>;

static void vector_unique(Qnum &qubits)
{
    std::set<int> qubits_set(qubits.begin(), qubits.end());
    qubits.assign(qubits_set.begin(), qubits_set.end());
    return;
}

static bool find_pair(const std::vector<DoubleQubits> &qubits, DoubleQubits value)
{
    for (auto val : qubits)
    {
        if (val.first == value.first && val.second == value.second)
        {
            return true;
        }
    }

    return false;
}

static bool is_single_gate(GateType type)
{
    return static_cast<size_t>(GateType::I_GATE) == static_cast<size_t>(type)  ||
          (static_cast<size_t>(GateType::P0_GATE) <= static_cast<size_t>(type) &&
           static_cast<size_t>(GateType::U4_GATE) >= static_cast<size_t>(type)) ;
}

static std::map<NOISE_MODEL, SINGLE_GATE_FUNC> flip_model_mapping_map =
{
    {NOISE_MODEL::BITFLIP_KRAUS_OPERATOR, X},
    {NOISE_MODEL::BIT_PHASE_FLIP_OPRATOR, Y},
    {NOISE_MODEL::DEPHASING_KRAUS_OPERATOR, Z}
};

static int random_discrete(const std::vector<double> &probs)
{ 
    static std::mt19937_64 engine;
    engine.seed(std::random_device()());
    return std::discrete_distribution<size_t>(probs.begin(), probs.end())(engine);
}

std::vector<double> TensorNoiseModelConfig::get_params(GateType type)
{
    try
    {
        return m_params.find(type)->second;
    }
    catch (std::exception& e)
    {
        throw run_fail("get_params error");
    }
}

std::vector<size_t> TensorNoiseModelConfig::get_single_qubits(GateType type)
{
    try
    {
        return m_single_qubits.find(type)->second;
    }
    catch (std::exception& e)
    {
        throw run_fail("get_qubits error");
    }
};

bool TensorNoiseModelConfig::is_config(GateType type, QVec qubits)
{
    auto params_iter = m_params.find(type);

    if (is_single_gate(type))
    {
        auto qubits_iter = m_single_qubits.find(type);

        auto tar_qubit = qubits[0]->get_phy_addr();

        if (m_params.cend() != params_iter && m_single_qubits.cend() != qubits_iter)
        {
            auto single_gate_qubits = qubits_iter->second;
            auto tar_iter = std::find(single_gate_qubits.begin(), single_gate_qubits.end(), tar_qubit);

            return tar_iter != single_gate_qubits.end();
        }
    }
    else
    {
        auto qubits_iter = m_double_qubits.find(type);

        auto ctr_qubit = qubits[0]->get_phy_addr();
        auto tar_qubit = qubits[1]->get_phy_addr();

        if (m_params.cend() != params_iter && m_double_qubits.cend() != qubits_iter)
        {
            auto double_gate_qubits = qubits_iter->second;
            return find_pair(double_gate_qubits, std::make_pair(ctr_qubit, tar_qubit));
        }
    }

    return false;
}

void TensorNoiseGenerator::execute(std::shared_ptr<AbstractQuantumMeasure>  cur_node,
    std::shared_ptr<QNode> parent_node, QCircuitConfig &config)
{
    auto qubits_addr = cur_node->getQuBit()->get_phy_addr();
    auto cbit = cur_node->getCBit();

    // if the Measure operation is not at the end of the quantum program , 
    // Measure operation need to be performed immediately
    auto result = m_mps_qpu->qubitMeasure(qubits_addr);
    cbit->set_val(result);
    m_result->append({ cbit->getName(), cbit->getValue() });
    return;
}

void TensorNoiseGenerator::execute(std::shared_ptr<AbstractControlFlowNode> cur_node,
    std::shared_ptr<QNode> parent_node, QCircuitConfig &config)
{
    QCERR("not support controlflow");
    throw std::runtime_error("not support controlflow");
}

void TensorNoiseGenerator::execute(std::shared_ptr<AbstractQuantumCircuit> cur_node,
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

void TensorNoiseGenerator::execute(std::shared_ptr<AbstractQuantumProgram>  cur_node,
    std::shared_ptr<QNode> parent_node, QCircuitConfig &config)
{
    Traversal::traversal(cur_node, *this, config);
}

void TensorNoiseGenerator::execute(std::shared_ptr<AbstractClassicalProg>  cur_node,
    std::shared_ptr<QNode> parent_node, QCircuitConfig &config)
{
    QCERR("not support ClassicalProg");
    throw std::runtime_error("not support ClassicalProg");
}

void TensorNoiseGenerator::execute(std::shared_ptr<AbstractQuantumReset>  cur_node,
    std::shared_ptr<QNode> parent_node, QCircuitConfig &config)
{  
    QCERR("not support Reset");
    throw std::runtime_error("not support Reset");
}

void TensorNoiseGenerator::handle_noise_gate(const std::vector<double>& params, QVec targets)
{
    auto noise_model = m_noise_model.get_model();
    switch (noise_model)
    {
        case NOISE_MODEL::BITFLIP_KRAUS_OPERATOR:
        case NOISE_MODEL::BIT_PHASE_FLIP_OPRATOR:
        case NOISE_MODEL::DEPHASING_KRAUS_OPERATOR:
        {
            handle_flip_noise_model(noise_model, params, targets);
            break;
        }
        case NOISE_MODEL::DEPOLARIZING_KRAUS_OPERATOR:
        {
            handle_depolarizing_noise_model(params, targets);
            break;
        }
        case NOISE_MODEL::DAMPING_KRAUS_OPERATOR:
        {
            handle_amplitude_damping_noise_model(params, targets);
            break;
        }
        case NOISE_MODEL::PHASE_DAMPING_OPRATOR:
        {
            handle_phase_damping_noise_model(params, targets);
            break;
        }
        case NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR:
        {
            handle_decoherence_noise_model(params, targets);
            break;
        }
        default:
        {
            QCERR("unsupported noise model");
            throw run_fail("unsupported noise model");
            break;
        }
    }

    return;
}

void TensorNoiseGenerator::handle_decoherence_noise_model(const std::vector<double>& params, QVec qubits)
{
    double T1 = params[0];
    double T2 = params[1];
    double single_time = params[2];
    double double_time = params[3];

    Qnum qubits_addr;
    for (auto qubit : qubits)
    {
        qubits_addr.emplace_back(qubit->get_phy_addr());
    }
    std::sort(qubits_addr.begin(), qubits_addr.end());

    double gate_time = 1 == qubits.size() ? single_time : double_time;

    std::vector<QStat> karus_matrixs;

    double p_damping = 1. - std::exp(-(gate_time / T1));
    double p_dephasing = 0.5 * (1. - std::exp(-(gate_time / T2 - gate_time / (2 * T1))));

    QStat K1 = { std::sqrt(1 - p_dephasing), 0,0,std::sqrt((1 - p_damping)*(1 - p_dephasing)) };
    QStat K2 = { 0, std::sqrt(p_damping*(1 - p_dephasing)), 0, 0 };
    QStat K3 = { 0, std::sqrt(p_damping*(1 - p_dephasing)), 0, 0 };
    QStat K4 = { 0, -std::sqrt(p_damping*p_dephasing), 0, 0 };

    karus_matrixs.emplace_back(K1);
    karus_matrixs.emplace_back(K2);
    karus_matrixs.emplace_back(K3);
    karus_matrixs.emplace_back(K4);

    std::vector<QStat> matrixs;
    if (1 == qubits.size())
    {
        matrixs = karus_matrixs;
    }
    else
    {
        for (auto val0 : karus_matrixs)
        {
            for (auto val1 : karus_matrixs)
            {
                matrixs.emplace_back(QPanda::tensor(val0, val1));
            }
        }
    }

    double random = random_generator19937();

    double sum = 0.0;

    bool complete = false;
    for (size_t j = 0; j < matrixs.size() - 1; j++)
    {
        cmatrix_t matrix = QStat_to_Eigen(matrixs[j]);
        double p = m_mps_qpu->expectation_value(qubits_addr, matrix);
        sum += p;

        if (sum > random)
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
        qcomplex_t renorm = 1 / std::sqrt(1. - sum);
        m_mps_qpu->unitaryQubitGate(qubits_addr, renorm * matrixs.back(), false);
    }

    return;
}


void TensorNoiseGenerator::handle_amplitude_damping_noise_model(const std::vector<double>& params, QVec qubits)
{
    Qnum qubits_addr;
    for (auto qubit : qubits)
    {
        qubits_addr.emplace_back(qubit->get_phy_addr());
    }
    std::sort(qubits_addr.begin(), qubits_addr.end());

    auto param = params[0];

    QStat E0 = { 1, 0, 0, std::sqrt(1 - param) };
    QStat E1 = { 0, std::sqrt(param), 0, 0 };

    std::vector<QStat> matrixs;

    if (1 == qubits.size())
    {
        matrixs.emplace_back(E0);
        matrixs.emplace_back(E1);
    }
    else
    {
        QStat matrix0 = QPanda::tensor(E0, E0);
        QStat matrix1 = QPanda::tensor(E0, E1);
        QStat matrix2 = QPanda::tensor(E1, E0);
        QStat matrix3 = QPanda::tensor(E1, E1);
         
        matrixs.emplace_back(matrix0);
        matrixs.emplace_back(matrix1);
        matrixs.emplace_back(matrix2);
        matrixs.emplace_back(matrix3);          
    }

    double random = random_generator19937();

    double sum = 0.0;

    bool complete = false;
    for (size_t j = 0; j < matrixs.size() - 1; j++) 
    {
        cmatrix_t matrix = QStat_to_Eigen(matrixs[j]);
        double p = m_mps_qpu->expectation_value(qubits_addr, matrix);
        sum += p;

        if (sum > random) 
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
        qcomplex_t renorm = 1 / std::sqrt(1. - sum);
        m_mps_qpu->unitaryQubitGate(qubits_addr, renorm * matrixs.back(), false);
    }

    return;
}


void TensorNoiseGenerator::handle_phase_damping_noise_model(const std::vector<double>& params, QVec qubits)
{
    //alpha : ( 1 + sqrt(lambda) ) / 2

    double alpha = (1 + std::sqrt(params[0])) / 2;

    return handle_flip_noise_model(NOISE_MODEL::DEPHASING_KRAUS_OPERATOR, { alpha }, qubits);
}

void TensorNoiseGenerator::handle_depolarizing_noise_model(const std::vector<double>& params, QVec qubits)
{
    static std::map<size_t, SINGLE_GATE_FUNC> depolarizing_noise_map = { {0,X} , {1,Y} ,  {2,Z} };

    double param = params[0] / 4.0;
    prob_vec gate_params = { param, param, param, 1 - 3 * param };
     
    if (1 == qubits.size())
    {
        auto index = random_discrete(gate_params);
        auto aiter = depolarizing_noise_map.find(index);

        if (aiter != depolarizing_noise_map.end())
        {
            auto noise_gate = aiter->second(qubits[0]);
            handle_quantum_gate(noise_gate.getImplementationPtr(), false);
        }
    }
    else
    {
        double p_0 = param * param;
        double p_1 = param * (1 - 3 * param);
        double p_2 = (1 - 3 * param) * (1 - 3 * param);
        prob_vec gate_probs = { p_1, p_1, p_1, p_1,
                                p_0, p_0, p_0, p_1,
                                p_0, p_0, p_0, p_1,
                                p_0, p_0, p_0, p_2 };

        auto index = random_discrete(gate_probs);

        //0000 : XI  //0001 : YI  //0010 : ZI  //0011 : IX 
        //0100 : XX  //0101 : YX  //0110 : ZX  //0111 : IY 
        //1000 : XY  //1001 : YY  //1010 : ZY  //1011 : IZ 
        //1100 : XZ  //1101 : YZ  //1110 : ZZ  //1111 : II

        auto aiter = depolarizing_noise_map.find(index % 4);
        if (depolarizing_noise_map.end() != aiter)
        {
            auto noise_gate = aiter->second(qubits[0]);
            handle_quantum_gate(noise_gate.getImplementationPtr(), false);
        }

        auto iter_index = ((index + 1) / 4) - 1;
        aiter = depolarizing_noise_map.find(iter_index);
        if (depolarizing_noise_map.end() != aiter)
        {
            auto noise_gate = aiter->second(qubits[1]);
            handle_quantum_gate(noise_gate.getImplementationPtr(), false);
        }
    }

    return;
}

void TensorNoiseGenerator::handle_flip_noise_model(NOISE_MODEL model, const std::vector<double>& params, QVec qubits)
{
    auto gate = flip_model_mapping_map.at(model);

    if (1 == qubits.size())
    {
        if (params[0] > random_generator19937())
        {
            auto noise_gate = gate(qubits[0]);
            handle_quantum_gate(noise_gate.getImplementationPtr(), false);
        }
    }
    else
    {
        prob_vec gate_probs;
        prob_vec gate_params = { params[0] ,1 - params[0] };

        for (auto i = 0; i < gate_params.size(); ++i)
        {
            for (auto j = 0; j < gate_params.size(); ++j)
            {
                gate_probs.emplace_back(gate_params[i] * gate_params[j]);
            }
        }

        auto index = random_discrete(gate_probs);
        for (auto i = 0; i < qubits.size(); ++i)
        {
            //00 : XX 
            //01 : XI 
            //10 : IX 
            //11 : II 

           if (!((index >> i) & 1))
           {
               auto noise_gate = gate(qubits[qubits.size() - 1 - i]);

               handle_quantum_gate(noise_gate.getImplementationPtr(), false);
           }
        }
    }

    return;
}

void TensorNoiseGenerator::execute(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitConfig &config)
{
    //targets && controls
    QVec targets;
    QVec controls;
    cur_node->getQuBitVector(targets);
    cur_node->getControlVector(controls);

    if (!controls.empty())
    {
        QCERR("not support controls");
        throw std::runtime_error("not support controls");
    }

    //gate type
    auto gate_type = (GateType)cur_node->getQGate()->getGateType();

    bool is_dagger = cur_node->isDagger() ^ config._is_dagger;

    if (NoiseConfigMethod::GLOBAL_CONFIG == m_method)
    {
        auto params = m_noise_model.get_golbal_params();

        handle_quantum_gate(cur_node, is_dagger);
        handle_noise_gate(params, targets);
    }
    else
    {
        if (m_noise_model.is_config(gate_type, targets))
        {
            auto params = m_noise_model.get_params(gate_type);

            handle_quantum_gate(cur_node, is_dagger);
            handle_noise_gate(params, targets);
        }
        else
        {
            handle_quantum_gate(cur_node, is_dagger);
            return;
        }
    }

}

void TensorNoiseGenerator::handle_quantum_gate(std::shared_ptr<AbstractQGateNode> gate, bool is_dagger)
{
    auto gate_type = (GateType)gate->getQGate()->getGateType();

    QStat gate_matrix;
    gate->getQGate()->getMatrix(gate_matrix);

    QVec targets;
    gate->getQuBitVector(targets);

    if (is_single_gate(gate_type))
    {
        unsigned short target;
        target = targets[0]->get_phy_addr();
        m_mps_qpu->unitarySingleQubitGate(target, gate_matrix, is_dagger, static_cast<GateType>(gate_type));
    }
    else
    {
        unsigned short control,target;
        control = targets[0]->get_phy_addr();
        target = targets[1]->get_phy_addr();
        m_mps_qpu->unitaryDoubleQubitGate(control, target, gate_matrix, is_dagger, static_cast<GateType>(gate_type));
    }

    return;
}

void TensorNoiseModelConfig::set_qubits(GateType gate_type, std::vector<size_t> qubits)
{
    if (is_single_gate(gate_type))
    {
        auto aiter = m_single_qubits.find(gate_type);
        if (m_single_qubits.end() != aiter)
        {
            aiter->second.insert(aiter->second.end(), qubits.begin(), qubits.end());
            vector_unique(aiter->second);
        }
        else
        {
            m_single_qubits[gate_type] = qubits;
        }
    }
    else
    {
        auto pairs = std::make_pair(qubits[0], qubits[1]);
        auto aiter = m_double_qubits.find(gate_type);

        if (m_double_qubits.end() != aiter)
        {
            aiter->second.emplace_back(pairs);
        }
        else
        {
            m_double_qubits[gate_type] = { pairs };
        } 
    }
    return;
}

void TensorNoiseModelConfig::set_params(GateType gate_type, std::vector<double> params)
{
    auto iter = m_params.find(gate_type);
    if (m_params.end() != iter)
    {
        iter->second = params;
    }
    else
    {
        m_params[gate_type] = params;
    }
}

void TensorNoiseGenerator::set_noise_model(NOISE_MODEL model,const std::vector<double> params_vec)
{
    for (auto param : params_vec)
    {
        if (0. > param || param > 1.)
        {
            QCERR("param error");
            throw run_fail("param error");
        }
    }

    if (NoiseConfigMethod::GLOBAL_CONFIG != m_method)
    {
        m_method = NoiseConfigMethod::GLOBAL_CONFIG;
    }

    m_noise_model.set_model(model);
    m_noise_model.set_golbal_params(params_vec);
}

void TensorNoiseGenerator::set_noise_model(NOISE_MODEL model, std::vector<double> T_params_vec, std::vector<double> time_params_vec)
{
    if (NoiseConfigMethod::GLOBAL_CONFIG != m_method)
    {
        m_method = NoiseConfigMethod::GLOBAL_CONFIG;
    }

    if (NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR != model)
    {
        QCERR("model != DECOHERENCE_KRAUS_OPERATOR");
        throw run_fail("model != DECOHERENCE_KRAUS_OPERATOR");
    }

    T_params_vec.insert(T_params_vec.end(), time_params_vec.begin(), time_params_vec.end());

    m_noise_model.set_model(model);
    m_noise_model.set_golbal_params(T_params_vec);
}

void TensorNoiseGenerator::set_noise_model(NOISE_MODEL model, std::string gate, Qnum qubits_vec, std::vector<double> params_vec)
{
    if (NoiseConfigMethod::NORMAL_CONFIG != m_method)
    {
        m_method = NoiseConfigMethod::NORMAL_CONFIG;
    }

    m_noise_model.set_model(model);

    auto gate_type = TransformQGateType::getInstance()[gate];
    m_noise_model.set_qubits(gate_type, qubits_vec);
    m_noise_model.set_params(gate_type, params_vec);
}

void TensorNoiseGenerator::set_noise_model(NOISE_MODEL model, std::string gate, Qnum qubits_vec, std::vector<double> T_params_vec, std::vector<double> time_params_vec)
{
    if (NoiseConfigMethod::NORMAL_CONFIG != m_method) 
    {
        m_method = NoiseConfigMethod::NORMAL_CONFIG;
    }

    if (NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR != model)
    {
        QCERR("model != DECOHERENCE_KRAUS_OPERATOR");
        throw run_fail("model != DECOHERENCE_KRAUS_OPERATOR");
    } 

    m_noise_model.set_model(model);

    T_params_vec.insert(T_params_vec.end(), time_params_vec.begin(), time_params_vec.end());

    auto gate_type = TransformQGateType::getInstance()[gate];
    m_noise_model.set_qubits(gate_type, qubits_vec);
    m_noise_model.set_params(gate_type, T_params_vec);
}

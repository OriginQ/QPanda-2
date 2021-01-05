#include "Core/VirtualQuantumProcessor/MPSQVM/MPSQVM.h"
USING_QPANDA
using namespace std;

static void merge_qvec(QVec &v1, const QVec &v2)
{
    if (0 == v2.size())
    {
        return;
    }

    if (0 == v1.size())
    {
        v1 = v2;
        return;
    }

    v1.insert(v1.end(), v2.begin(), v2.end());
}

static void merge_qvec(QVec &v1, const QVec &v2, const QVec &v3)
{
    merge_qvec(v1, v2);
    merge_qvec(v1, v3);
}

static void unique_qvec(QVec &qv)
{
    if (qv.size() <= 1)
    {
        return;
    }

    sort(qv.begin(), qv.end(), [](Qubit *a, Qubit *b)
    {return a->getPhysicalQubitPtr()->getQubitAddr() <
        b->getPhysicalQubitPtr()->getQubitAddr(); });

    qv.erase(unique(qv.begin(), qv.end(),
        [](Qubit *a, Qubit *b)
    {return a->getPhysicalQubitPtr()->getQubitAddr() ==
        b->getPhysicalQubitPtr()->getQubitAddr(); }),
        qv.end());
}

static std::vector<Qnum> get_qubits_addr(const std::vector<QVec>& qvs)
{
    std::vector<Qnum> qubits_addrs;
    for (const auto& qvec : qvs)
    {
        Qnum qubits_addr;
        for_each(qvec.begin(), qvec.end(), [&](Qubit* qubit)
        {
            qubits_addr.emplace_back(qubit->get_phy_addr()); 
        });

        qubits_addrs.emplace_back(qubits_addr);
    }

    return qubits_addrs;
}

static Qnum get_qubits_addr(const QVec &qvs)
{
    Qnum qubits_addrs;
    for (const auto& qubit : qvs)
    {
        qubits_addrs.emplace_back(qubit->get_phy_addr());
    }

    return qubits_addrs;
}

void MPSQVM::init()
{
    try
    {
        _start();
        m_simulator = make_unique<MPSImplQPU>();
    }
    catch (const std::exception &e)
    {
        QCERR(e.what());
        throw init_fail(e.what());
    }
}

static void get_gate_paramter(std::shared_ptr<AbstractQGateNode> gate, const QCircuitConfig &config, unsigned short &target, std::vector<size_t>& controls)
{
    auto gate_type = static_cast<GateType>(gate->getQGate()->getGateType());

    QVec qubits;
    QVec control_qubits;
    gate->getQuBitVector(qubits);
    gate->getControlVector(control_qubits);

    merge_qvec(control_qubits, config._contorls);

    if (qubits.size() > 1)
    {
        QVec single_gate_controls(qubits.begin(), qubits.end() - 1);
        merge_qvec(control_qubits, single_gate_controls);
    }

    unique_qvec(control_qubits);
    target = qubits[qubits.size() - 1]->getPhysicalQubitPtr()->getQubitAddr();

    for (auto &val : control_qubits)
    {
        auto control_addr = val->getPhysicalQubitPtr()->getQubitAddr();
        controls.push_back(control_addr);
    }
}


static void get_gate_paramter(std::shared_ptr<AbstractQGateNode> gate, const QCircuitConfig &config, std::vector<unsigned short> &targets, std::vector<size_t> &controls)
{
    auto gate_type = static_cast<GateType>(gate->getQGate()->getGateType());

    QVec qubits;
    QVec control_qubits;
    gate->getQuBitVector(qubits);
    gate->getControlVector(control_qubits);

    merge_qvec(control_qubits, config._contorls);
    unique_qvec(control_qubits);

    for (auto &val : control_qubits)
    {
        auto control_addr = val->getPhysicalQubitPtr()->getQubitAddr();
        controls.push_back(control_addr);
    }

    for (auto &val : qubits)
    {
        auto target_addr = val->getPhysicalQubitPtr()->getQubitAddr();
        targets.push_back(target_addr);
    }
}

std::map<std::string, bool> MPSQVM::directlyRun(QProg &prog)
{
    run(prog);
    for (auto &val : m_measure_obj)
    {
        auto measure_result = m_simulator->qubitMeasure(val.first);
        val.second->set_val(measure_result);
        _QResult->append({ val.second->getName(), measure_result });
    }

    return _QResult->getResultMap();
}

std::map<std::string, size_t> MPSQVM::run_configuration_without_noise(QProg &prog, std::vector<ClassicalCondition> &cbits, int shots)
{
    map<string, size_t> result_map;
    TraversalConfig traver_param;
    QProgCheck prog_check;
    prog_check.execute(prog.getImplementationPtr(), nullptr, traver_param);

    // at least one Measure operation is not at the end of the quantum program
    if (traver_param.m_can_optimize_measure != true)
    {
        for (int i = 0; i < shots; i++)
        {
            run_cannot_optimize_measure(prog);
            std::string result_bin_str = _ResultToBinaryString(cbits);
            std::reverse(result_bin_str.begin(), result_bin_str.end());
            if (result_map.find(result_bin_str) == result_map.end())
                result_map[result_bin_str] = 1;
            else
                result_map[result_bin_str] += 1;
        }

        return  result_map;
    }

    // all Measure operations are at the end of the quantum program

    run(prog);

    std::vector<size_t> measure_qubits;
    for (auto &val : m_measure_obj)
        measure_qubits.push_back(val.first);

    auto measure_all_result = m_simulator->measure_all_noncollapsing(measure_qubits, shots);

    for (int shot = 0; shot < measure_all_result.size(); shot++)
    {
        for (int qidx = 0; qidx < measure_qubits.size(); qidx++)
        {
            auto cbit = m_measure_obj[qidx].second;
            cbit->set_val(measure_all_result[shot][qidx]);
            _QResult->append({ cbit->getName(), cbit->getValue() });
        }

        string result_bin_str = _ResultToBinaryString(cbits);
        std::reverse(result_bin_str.begin(), result_bin_str.end());
        if (result_map.find(result_bin_str) == result_map.end())
            result_map[result_bin_str] = 1;
        else
            result_map[result_bin_str] += 1;
    }

    return  result_map;
}


std::map<std::string, size_t> MPSQVM::run_configuration_with_noise(QProg &prog, std::vector<ClassicalCondition> &cbits, int shots)
{
    map<string, size_t> result_map;

    for (int i = 0; i < shots; i++)
    { 
        run_cannot_optimize_measure_with_noise(prog);
        std::string result_bin_str = _ResultToBinaryString(cbits);
        std::reverse(result_bin_str.begin(), result_bin_str.end());

        if (result_map.find(result_bin_str) == result_map.end())
            result_map[result_bin_str] = 1;
        else
            result_map[result_bin_str] += 1;
    }

    return  result_map;
}   

std::map<string, size_t> MPSQVM::runWithConfiguration(QProg &prog, std::vector<ClassicalCondition> &cbits, int shots)
{
    if (m_noise_simulator.has_quantum_error())
    {
        return run_configuration_with_noise(prog, cbits, shots);
    }
    else
    {
        return run_configuration_without_noise(prog, cbits, shots);
    }
}

std::map<string, size_t> MPSQVM::runWithConfiguration(QProg &prog, std::vector<ClassicalCondition> &cbits, rapidjson::Document &doc)
{
    if (!doc.HasMember("shots"))
    {
        QCERR("OriginCollection don't  have shots");
        throw run_fail("runWithConfiguration param don't  have shots");
    }
    size_t shots = 0;
    if (doc["shots"].IsUint64())
    {
        shots = doc["shots"].GetUint64();
    }
    else
    {
        QCERR("shots data type error");
        throw run_fail("shots data type error");
    }

    return runWithConfiguration(prog, cbits, shots);
}

QStat MPSQVM::getQState()
{
    if (m_simulator == nullptr)
    {
        QCERR("m_simulator error, need run the prog");
        throw run_fail("m_simulator error, need run the prog");
    }
    return m_simulator->getQState();
}

std::map<std::string, size_t> MPSQVM::quickMeasure(QVec vQubit, size_t shots)
{
    QCERR("quickMeasure");
    throw run_fail("quickMeasure");
}


prob_vec MPSQVM::PMeasure_no_index(QVec qubits)
{
    return getProbList(qubits, -1);
}

prob_tuple MPSQVM::PMeasure(QVec qubits, int select_max)
{
    return pMeasure(qubits, select_max);
}

prob_tuple MPSQVM::pMeasure(QVec qubits, int select_max)
{
    Qnum vqubit;
    for (auto aiter = qubits.begin(); aiter != qubits.end(); ++aiter)
    {
        vqubit.push_back((*aiter)->getPhysicalQubitPtr()->getQubitAddr());
    }

    prob_tuple result_tuple;

    prob_vec pmeasure_vector;
    m_simulator->pMeasure(vqubit, pmeasure_vector);

    for (auto i = 0; i < pmeasure_vector.size(); ++i)
    {
        result_tuple.emplace_back(make_pair(i, pmeasure_vector[i]));
    }

    sort(result_tuple.begin(), result_tuple.end(),
        [=](std::pair<size_t, double> a, std::pair<size_t, double> b) {return a.second > b.second; });

    if ((select_max == -1) || (pmeasure_vector.size() <= select_max))
    {
        return result_tuple;
    }
    else
    {
        result_tuple.erase(result_tuple.begin() + select_max, result_tuple.end());
        return result_tuple;
    }
}

void MPSQVM::initState(const QStat &state)
{
    m_simulator->initState(getAllocateQubitNum(), state);
    return;
}


prob_tuple MPSQVM::getProbTupleList(QVec vQubit, int select_max)
{
    if (0 == vQubit.size())
    {
        QCERR("the size of qubit_vector is zero");
        throw invalid_argument("the size of qubit_vector is zero");
    }

    try
    {
        return pMeasure(vQubit, select_max);
    }
    catch (const std::exception&e)
    {
        QCERR(e.what());
        throw result_get_fail(e.what());
    }
}

prob_vec MPSQVM::getProbList(QVec vQubit, int selectMax)
{
    if (0 == vQubit.size())
    {
        QCERR("the size of qubit_vector is zero");
        throw invalid_argument("the size of qubit_vector is zero");
    }

    Qnum qubits;
    for (auto aiter = vQubit.begin(); aiter != vQubit.end(); ++aiter)
    {
        qubits.push_back((*aiter)->getPhysicalQubitPtr()->getQubitAddr());
    }

    prob_vec result_vec;
    m_simulator->pMeasure(qubits, result_vec);
    return result_vec;
}

prob_dict MPSQVM::getProbDict(QVec vQubit, int selectMax)
{
    auto result_vec = getProbList(vQubit, selectMax);
    prob_dict result_dict;
    size_t bit_length = vQubit.size();
    for (size_t i = 0; i < result_vec.size(); i++)
    {
        result_dict.insert({ dec2bin(i, bit_length), result_vec[i] });
    }

    return result_dict;
}

prob_tuple MPSQVM::probRunTupleList(QProg & qProg, QVec vQubit, int selectMax)
{
    run(qProg);
    return getProbTupleList(vQubit, selectMax);
}

prob_vec MPSQVM::probRunList(QProg & qProg, QVec vQubit, int selectMax)
{
    run(qProg);
    return getProbList(vQubit, selectMax);
}

prob_dict MPSQVM::probRunDict(QProg & qProg, QVec vQubit, int selectMax)
{
    run(qProg);
    return getProbDict(vQubit, selectMax);
}

void MPSQVM::handle_one_target(std::shared_ptr<AbstractQGateNode> gate, const QCircuitConfig &config)
{
    auto gate_type = gate->getQGate()->getGateType();

    QStat gate_matrix;
    gate->getQGate()->getMatrix(gate_matrix);

    bool is_dagger = gate->isDagger() ^ config._is_dagger;

    unsigned short target;
    std::vector<size_t> controls;
    get_gate_paramter(gate, config, target, controls);

    if (controls.size() > 0)
        m_simulator->controlunitarySingleQubitGate(target, controls, gate_matrix, is_dagger, static_cast<GateType>(gate_type));
    else
        m_simulator->unitarySingleQubitGate(target, gate_matrix, is_dagger, static_cast<GateType>(gate_type));

    return;
}


void MPSQVM::handle_two_targets(std::shared_ptr<AbstractQGateNode> gate, const QCircuitConfig &config)
{ 
    auto gate_type = gate->getQGate()->getGateType();

    QStat gate_matrix;
    gate->getQGate()->getMatrix(gate_matrix);

    bool is_dagger = gate->isDagger() ^ config._is_dagger;

    std::vector<unsigned short> targets;
    std::vector<size_t> controls;
    get_gate_paramter(gate, config, targets, controls);

    if (controls.size() > 0)
        m_simulator->controlunitaryDoubleQubitGate(targets[0], targets[1], controls, gate_matrix, is_dagger, static_cast<GateType>(gate_type));
    else
        m_simulator->unitaryDoubleQubitGate(targets[0], targets[1], gate_matrix, is_dagger, static_cast<GateType>(gate_type));

    return;
}

void MPSQVM::run_cannot_optimize_measure_with_noise(QProg &prog)
{
    m_qubit_num = getAllocateQubitNum();
    m_simulator->initState(0, 1, m_qubit_num);

    QCircuitConfig config;
    config._is_dagger = false;
    config._contorls.clear();
    config._can_optimize_measure = false;

    m_noise_simulator.set_mps_qpu_and_result(m_simulator, getResult());
    m_noise_simulator.execute(prog.getImplementationPtr(), nullptr, config);
}

void MPSQVM::run(QProg &prog)
{
    m_qubit_num = getAllocateQubitNum();
    m_simulator->initState(0, 1, m_qubit_num);

    QCircuitConfig config;
    config._is_dagger = false;
    config._contorls.clear();
    config._can_optimize_measure = true;
    execute(prog.getImplementationPtr(), nullptr, config);
}

void MPSQVM::run_cannot_optimize_measure(QProg &prog)
{
    m_qubit_num = getAllocateQubitNum();
    m_simulator->initState(0, 1, m_qubit_num);

    QCircuitConfig config;
    config._is_dagger = false;
    config._contorls.clear();
    config._can_optimize_measure = false;
    execute(prog.getImplementationPtr(), nullptr, config);
}

void MPSQVM::execute(std::shared_ptr<AbstractQuantumMeasure>  cur_node,
    std::shared_ptr<QNode> parent_node, QCircuitConfig &config)
{
    auto qubits_addr = cur_node->getQuBit()->get_phy_addr();
    auto cbit = cur_node->getCBit();

    // if the Measure operation is not at the end of the quantum program , 
    // Measure operation need to be performed immediately
    if (config._can_optimize_measure != true)
    {
        auto result = m_simulator->qubitMeasure(qubits_addr);
        cbit->set_val(result);
        _QResult->append({ cbit->getName(), cbit->getValue() });
    }
    else
    {
        m_measure_obj.push_back({ qubits_addr, cbit });
    }
    return;
}

void MPSQVM::execute(std::shared_ptr<AbstractControlFlowNode> cur_node,
    std::shared_ptr<QNode> parent_node, QCircuitConfig &config)
{
    QCERR("not support controlflow");
    throw std::runtime_error("not support controlflow");
}


void MPSQVM::execute(std::shared_ptr<AbstractQuantumCircuit> cur_node,
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

void MPSQVM::execute(std::shared_ptr<AbstractQuantumProgram>  cur_node,
    std::shared_ptr<QNode> parent_node, QCircuitConfig &config)
{
    Traversal::traversal(cur_node, *this, config);
}

void MPSQVM::execute(std::shared_ptr<AbstractClassicalProg>  cur_node,
    std::shared_ptr<QNode> parent_node, QCircuitConfig &config)
{
    cur_node->get_val();
}

void MPSQVM::execute(std::shared_ptr<AbstractQuantumReset>  cur_node,
    std::shared_ptr<QNode> parent_node, QCircuitConfig &config)
{
    QCERR("not support Reset");
    throw std::runtime_error("not support Reset");
}

void MPSQVM::execute(std::shared_ptr<AbstractQGateNode>  cur_node,
    std::shared_ptr<QNode> parent_node, QCircuitConfig &config)
{
    auto gate_type = static_cast<GateType>(cur_node->getQGate()->getGateType());
    switch (gate_type)
    {
    case PAULI_X_GATE:
    case PAULI_Y_GATE:
    case PAULI_Z_GATE:
    case X_HALF_PI:
    case Y_HALF_PI:
    case Z_HALF_PI:
    case HADAMARD_GATE:
    case T_GATE:
    case S_GATE:
    case I_GATE:
    case RX_GATE:
    case RY_GATE:
    case RZ_GATE:
    case U1_GATE:
    case U2_GATE:
    case U3_GATE:
    case U4_GATE:
        handle_one_target(cur_node, config);
        break;
    case CNOT_GATE:
    case CZ_GATE:
    case CPHASE_GATE:
    case CU_GATE:
    case ISWAP_GATE:
    case SWAP_GATE:
    case SQISWAP_GATE:
    case ISWAP_THETA_GATE:
    case TWO_QUBIT_GATE:
        handle_two_targets(cur_node, config);
        break;
    default:
        QCERR("QGate type error");
        throw run_fail("QGate type error");
    }
}

qcomplex_t MPSQVM::pmeasure_bin_index(QProg prog, std::string str)
{
    run(prog);
    return m_simulator->pmeasure_bin_index(str);
}

qcomplex_t MPSQVM::pmeasure_dec_index(QProg prog, std::string str)
{
    run(prog);
    return m_simulator->pmeasure_dec_index(str);
}

//The all next functions are only for noise simulation
 
void MPSQVM::set_reset_error(double reset_0_param, double reset_1_param) 
{ 
    return m_noise_simulator.set_reset_error(reset_0_param, reset_1_param); 
} 

void MPSQVM::set_rotation_error(double param) 
{
    return m_noise_simulator.set_rotation_error(param); 
}

void MPSQVM::set_measure_error(NOISE_MODEL model, double param)
{
    return m_noise_simulator.set_measure_error(model, param);
}

void MPSQVM::set_measure_error(NOISE_MODEL model, double T1, double T2, double time_param)
{
    m_noise_simulator.set_measure_error(model, T1, T2, time_param);
    return;
}

void MPSQVM::set_measure_error(NOISE_MODEL model, double param, const QVec& qubits_vec)
{
    return m_noise_simulator.set_measure_error(model, param, get_qubits_addr(qubits_vec));
}

void MPSQVM::set_measure_error(NOISE_MODEL model, double T1, double T2, double time_param, const QVec& qubits_vec)
{
    m_noise_simulator.set_measure_error(model, T1, T2, time_param, get_qubits_addr(qubits_vec));
    return;
}

void MPSQVM::set_noise_model(NOISE_MODEL model, GateType gate_type, double param)
{
    m_noise_simulator.set_noise_model(model, gate_type, param);
    return;
}

void MPSQVM::set_noise_model(NOISE_MODEL model, GateType gate_type, double param, const std::vector<QVec>& qubits_vecs)
{
    m_noise_simulator.set_noise_model(model, gate_type, param, get_qubits_addr(qubits_vecs));
    return;
}

void MPSQVM::set_noise_model(NOISE_MODEL model, GateType gate_type, double param, const QVec& qubits_vec)
{
    m_noise_simulator.set_noise_model(model, gate_type, param, get_qubits_addr(qubits_vec));
    return;
}


void MPSQVM::set_noise_model(NOISE_MODEL model, GateType gate_type, double T1, double T2, double time_param)
{
    m_noise_simulator.set_noise_model(model, gate_type, T1, T2, time_param);
    return;
}

void MPSQVM::set_noise_model(NOISE_MODEL model, GateType gate_type, double T1, double T2, double time_param, const std::vector<QVec>& qubits_vecs)
{
    m_noise_simulator.set_noise_model(model, gate_type, T1, T2, time_param, get_qubits_addr(qubits_vecs));
    return;
}


void MPSQVM::set_noise_model(NOISE_MODEL model, GateType gate_type, double T1, double T2, double time_param, const QVec& qubits_vec)
{
    m_noise_simulator.set_noise_model(model, gate_type, T1, T2, time_param, get_qubits_addr(qubits_vec));
    return;
}
void MPSQVM::set_mixed_unitary_error(GateType gate_type, const std::vector<QStat>& unitary_matrices, const std::vector<double>& probs_vec)
{
    m_noise_simulator.set_mixed_unitary_error(gate_type, unitary_matrices, probs_vec);
    return;
}

void MPSQVM::set_mixed_unitary_error(GateType gate_type, const std::vector<QStat>& karus_matrices)
{
    m_noise_simulator.set_mixed_unitary_error(gate_type, karus_matrices);
    return;
}

void MPSQVM::set_mixed_unitary_error(GateType gate_type, const std::vector<QStat>& unitary_matrices, const std::vector<double>& probs_vec, const std::vector<QVec>& qubits_vecs)
{
    return m_noise_simulator.set_mixed_unitary_error(gate_type, unitary_matrices, probs_vec, get_qubits_addr(qubits_vecs));
    return;
}

void MPSQVM::set_mixed_unitary_error(GateType gate_type, const std::vector<QStat>& karus_matrices, const std::vector<QVec>& qubits_vecs)
{
    m_noise_simulator.set_mixed_unitary_error(gate_type, karus_matrices, get_qubits_addr(qubits_vecs));
    return;
}

void MPSQVM::set_readout_error(const std::vector<std::vector<double>>& readout_params, const QVec& qubits)
{
    m_noise_simulator.set_readout_error(readout_params, get_qubits_addr(qubits));
    return;
}

#if 0
void MPSQVM::set_error(GateType gate_type, const KarusError& karus_error)
{
    m_noise_simulator.set_combining_error(gate_type, karus_error, {});
    return;
}

void MPSQVM::set_error(GateType gate_type, const KarusError& karus_error, const std::vector<QVec>& qubits_vecs)
{
    m_noise_simulator.set_combining_error(gate_type, karus_error, get_qubits_addr(qubits_vecs));
    return;
}
#endif
#include "Core/QuantumMachine/PartialAmplitudeQVM.h"
#include "Core/QuantumCircuit/QuantumGate.h"
using namespace std;
USING_QPANDA

static void getAvgBinary(uint128_t num, 
                         uint64_t& low_pos, 
                         uint64_t& high_pos, 
                         size_t qubit_num)
{
    size_t half_qubit = qubit_num / 2;
    long long lower_mask = (1 << half_qubit) - 1;
    low_pos = (uint64_t)(num & lower_mask);
    high_pos = (uint64_t)(num - low_pos) >> (qubit_num - half_qubit);
}

PartialAmplitudeQVM::PartialAmplitudeQVM()
{
    _Config.maxQubit = 64;
    _Config.maxCMem = 64;
    m_prog_map = new PartialAmplitudeGraph();
}

PartialAmplitudeQVM::~PartialAmplitudeQVM()
{
    delete  m_prog_map;
}

void PartialAmplitudeQVM::init()
{
    _Config.maxQubit = 256;
    _Config.maxCMem = 256;
    _start();
}

stat_map PartialAmplitudeQVM::getQState()
{
    if (nullptr == m_prog_map)
    {
        QCERR("prog is null");
        throw run_fail("prog is null");
    }

    vector<vector<QStat>> graph_stat_map;
    getSubGraphStat(graph_stat_map);

    stat_map result_map;
    uint128_t size = (uint128_t)1 << m_prog_map->m_qubit_num;
    uint64_t low_pos, high_pos;
    for (uint128_t i = 0; i < size; i++)
    {
        qcomplex_t addResult(0, 0);
        getAvgBinary(i, low_pos, high_pos, m_prog_map->m_qubit_num);
        for (size_t j = 0; j < graph_stat_map.size(); ++j)
        {
            addResult = addResult + graph_stat_map[j][0][low_pos] * graph_stat_map[j][1][high_pos];
        }
        result_map.insert(make_pair(integerToBinary(i, m_prog_map->m_qubit_num), addResult));
    }
     
    return result_map;
}

void PartialAmplitudeQVM::getSubGraphStat(vector<vector<QStat>> &graph_stat_map)
{
    for (uint64_t i = 0; i < m_prog_map->getMapVecSize(); ++i)
    {
        vector<QStat> calculateMap;
        for (uint64_t j = 0; j < m_prog_map->m_circuit_vec[i].size(); ++j)
        {
            auto m_qubit_number = (j == 0) ? 
                (m_prog_map->m_qubit_num / 2) :
               ((m_prog_map->m_qubit_num) - (m_prog_map->m_qubit_num / 2));
            QPUImpl *pQGate = new CPUImplQPU();

            pQGate->initState(0,1, m_qubit_number);
            m_prog_map->traversalMap(m_prog_map->m_circuit_vec[i][j], pQGate);

            calculateMap.emplace_back(pQGate->getQState());
            delete pQGate;
        }

        graph_stat_map.emplace_back(calculateMap);
    }
}

qstate_type PartialAmplitudeQVM::PMeasure_dec_index(string index)
{
    if (nullptr == m_prog_map)
    {
        QCERR("prog is null");
        throw run_fail("prog is null");
    }

    vector<vector<QStat>> graph_stat_map;
    getSubGraphStat(graph_stat_map);

    qcomplex_t addResult(0, 0);
    uint128_t u_index(index.c_str());
    uint64_t low_pos, high_pos;
    getAvgBinary(u_index, low_pos, high_pos, m_prog_map->m_qubit_num);
    for (size_t j = 0; j < graph_stat_map.size(); ++j)
    {
        addResult = addResult + graph_stat_map[j][0][low_pos] * graph_stat_map[j][1][high_pos];
    }
    return addResult.real()*addResult.real() + addResult.imag()*addResult.imag();
}

qstate_type PartialAmplitudeQVM::PMeasure_bin_index(string index)
{
    auto check = [](char bin)
    {
        if ('1' != bin && '0' != bin)
        {
            QCERR("PMeasure parm error");
            throw qprog_syntax_error("PMeasure parm");
        }
        else
        {
            return bin == '0' ? 0 : 1;
        }
    };

    uint128_t u_index = 0;
    size_t len = index.size();
    for (size_t i = 0; i < len; ++i)
    {
        u_index += check(index[len - i - 1]) << i;
    }

    return PMeasure_dec_index(integerToString(u_index));
}

prob_map PartialAmplitudeQVM::PMeasure(QVec qvec, string select_max)
{
    Qnum  qubit_vec;
    for_each(qvec.begin(), qvec.end(), [&](Qubit *qubit)
    {qubit_vec.emplace_back(qubit->getPhysicalQubitPtr()->getQubitAddr()); });
    sort(qubit_vec.begin(), qubit_vec.end());
    auto iter = adjacent_find(qubit_vec.begin(), qubit_vec.end());

    uint128_t select_max_size(select_max.c_str());
    uint128_t max_size = (uint128_t)1 << qubit_vec.size();
    if ((qubit_vec.end() != iter) || 
        select_max_size > (max_size))
    {
        QCERR("PMeasure error");
        throw qprog_syntax_error("PMeasure");
    }

    size_t qubit_num = m_prog_map->m_qubit_num;

    uint128_t value_size = (uint128_t)1 << (qubit_num - qubit_vec.size());
    prob_map res;
    auto pmeasure_size = qubit_vec.size();

    vector<vector<QStat>> graph_stat_map;
    getSubGraphStat(graph_stat_map);

    if (pmeasure_size <= qubit_num)
    {
        uint64_t low_pos, high_pos;
        for (uint128_t i = 0; i < select_max_size; ++i)
        {
            double temp_value = 0.0;
            for (uint128_t j = 0; j < value_size; ++j)
            {
                qcomplex_t addResult(0, 0);
                uint128_t index = getDecIndex(i, j, qubit_vec, qubit_num);
                getAvgBinary(index, low_pos, high_pos, qubit_num);
                for (size_t k = 0; k < graph_stat_map.size(); ++k)
                {
                    addResult = addResult + graph_stat_map[k][0][low_pos] * graph_stat_map[k][1][high_pos];
                }
                temp_value += addResult.real()*addResult.real() + addResult.imag()*addResult.imag();
            }

            res.insert(make_pair(integerToString(i), temp_value));
        }

        return res;
    }
    else
    {
        QCERR("PMeasure error");
        throw qprog_syntax_error("PMeasure");
    }
}

prob_map PartialAmplitudeQVM::PMeasure(string select_max)
{
    if (nullptr == m_prog_map)
    {
        QCERR("prog is null");
        throw run_fail("prog is null");
    }

    uint128_t select_max_size(select_max.c_str());
    uint128_t max_size = (uint128_t)1 << m_prog_map->m_qubit_num;
    if (select_max_size > max_size)
    {
        QCERR("PMeasure error");
        throw qprog_syntax_error("PMeasure");
    }

    vector<vector<QStat>> graph_stat_map;
    getSubGraphStat(graph_stat_map);

    prob_map result_map;
    uint64_t low_pos, high_pos;
    for (uint128_t i = 0; i < select_max_size; i++)
    {
        qcomplex_t value(0, 0);
        getAvgBinary(i, low_pos, high_pos, m_prog_map->m_qubit_num);
        for (size_t j = 0; j < graph_stat_map.size(); ++j)
        {
            value = value + graph_stat_map[j][0][low_pos] * graph_stat_map[j][1][high_pos];
        }

        string index = integerToString(i);
        result_map.insert(make_pair(index, 
            (value.real()*value.real()+value.imag()*value.imag())));
    }

    return result_map;
}

prob_map PartialAmplitudeQVM::getProbDict(QVec qvec, string select_max)
{
    Qnum  qubit_vec;
    for_each(qvec.begin(), qvec.end(), [&](Qubit *qubit)
    {qubit_vec.emplace_back(qubit->getPhysicalQubitPtr()->getQubitAddr()); });
    sort(qubit_vec.begin(), qubit_vec.end());
    auto iter = adjacent_find(qubit_vec.begin(), qubit_vec.end());

    uint128_t select_max_size(select_max.c_str());
    uint128_t max_size = (uint128_t)1 << qubit_vec.size();
    if ((qubit_vec.end() != iter) ||
        select_max_size > (max_size))
    {
        QCERR("PMeasure error");
        throw qprog_syntax_error("PMeasure");
    }

    size_t qubit_num = m_prog_map->m_qubit_num;

    uint128_t value_size = (uint128_t)1 << (qubit_num - qubit_vec.size());
    prob_map res;
    auto pmeasure_size = qubit_vec.size();

    vector<vector<QStat>> graph_stat_map;
    getSubGraphStat(graph_stat_map);

    if (pmeasure_size <= qubit_num)
    {
        uint64_t low_pos, high_pos;
        for (uint128_t i = 0; i < select_max_size; ++i)
        {
            double temp_value = 0.0;

            for (uint128_t j = 0; j < value_size; ++j)
            {
                qcomplex_t addResult(0, 0);
                uint128_t index = getDecIndex(i, j, qubit_vec, qubit_num);
                getAvgBinary(index, low_pos, high_pos, qubit_num);
                for (size_t k = 0; k < graph_stat_map.size(); ++k)
                {
                    addResult = addResult + graph_stat_map[k][0][low_pos] * graph_stat_map[k][1][high_pos];
                }
                temp_value += addResult.real()*addResult.real() + addResult.imag()*addResult.imag();
            }

            res.insert(make_pair(integerToBinary(i, pmeasure_size), temp_value));
        }

        return res;
    }
    else
    {
        QCERR("PMeasure error");
        throw qprog_syntax_error("PMeasure");
    }
}

prob_map PartialAmplitudeQVM::probRunDict(QProg &prog, QVec qvec, string select_max)
{
    run(prog);
    return getProbDict(qvec, select_max);
}

prob_map PartialAmplitudeQVM::PMeasureSubSet(QProg &prog, std::vector<std::string> subset_vec)
{
    run(prog);
    if (nullptr == m_prog_map || 0 == subset_vec.size())
    {
        QCERR("prog is null");
        throw run_fail("prog is null");
    }

    for_each(subset_vec.begin(), subset_vec.end(), [&](std::string str)
    {
        if (str.length() != m_prog_map->m_qubit_num)
        {
            QCERR("parm error");
            throw run_fail("parm error");
        };
    });

    vector<vector<QStat>> graph_stat_map;
    getSubGraphStat(graph_stat_map);

    auto check = [](char bin)
    {
        if ('1' != bin && '0' != bin)
        {
            QCERR("PMeasure parm error");
            throw qprog_syntax_error("PMeasure parm");
        }
        else
        {
            return bin != '0';
        }
    };

    prob_map result_map;
    uint64_t low_pos, high_pos;
    for (auto val : subset_vec)
    {
        qcomplex_t value(0, 0);
        uint128_t u_index = 0;
        size_t len = val.size();
        for (size_t i = 0; i < len; ++i)
        {
            u_index += check(val[len - i - 1]) << i;
        }

        getAvgBinary(u_index, low_pos, high_pos, m_prog_map->m_qubit_num);
        for (size_t j = 0; j < graph_stat_map.size(); ++j)
        {
            value = value + graph_stat_map[j][0][low_pos] * graph_stat_map[j][1][high_pos];
        }

        result_map.insert(make_pair(val,
            (value.real()*value.real() + value.imag()*value.imag())));
    }

    return result_map;
}

prob_map PartialAmplitudeQVM::pMeasureSubset(QProg &prog, std::vector<std::string> subset_vec)
{
    return PMeasureSubSet(prog, subset_vec);
}



void PartialAmplitudeQVM::execute(std::shared_ptr<AbstractQuantumMeasure>  cur_node, std::shared_ptr<QNode> parent_node)
{
    QCERR("Does not support QuantumMeasure ");
    throw std::runtime_error("Does not support QuantumMeasure");
}

void PartialAmplitudeQVM::execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node)
{
    QCERR("Does not support ControlFlowNode ");
    throw std::runtime_error("Does not support ControlFlowNode");
}

void PartialAmplitudeQVM::execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node)
{
    Traversal::traversal(cur_node, false, *this);
}

void PartialAmplitudeQVM::execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node)
{
    Traversal::traversal(cur_node, *this);
}

void PartialAmplitudeQVM::execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node)
{
    QCERR("Does not support ClassicalProg ");
    throw std::runtime_error("Does not support ClassicalProg");
}

void PartialAmplitudeQVM::execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node)
{
    if (nullptr == cur_node || nullptr == cur_node->getQGate())
    {
        QCERR("pQGate is null");
        throw invalid_argument("pQGate is null");
    }

    QVec qubits_vector;
    cur_node->getQuBitVector(qubits_vector);

    auto gate_type = (unsigned short)cur_node->getQGate()->getGateType();
    QGateNode node = { gate_type,cur_node->isDagger() };
    switch (gate_type)
    {
    case GateType::P0_GATE:
    case GateType::P1_GATE:
    case GateType::PAULI_Y_GATE:
    case GateType::PAULI_Z_GATE:
    case GateType::Y_HALF_PI:
    case GateType::Z_HALF_PI:
    case GateType::X_HALF_PI:
    case GateType::HADAMARD_GATE:
    case GateType::T_GATE:
    case GateType::S_GATE:
    {
        node.tar_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
    }
    break;

    case GateType::PAULI_X_GATE:
    {
        QVec control_qvec;
        cur_node->getControlVector(control_qvec);

        if (control_qvec.empty())
        {
            node.tar_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
        }
        else
        {
            node.gate_type = TOFFOLI_GATE;

            auto tar_qubit = node.tar_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();

            auto ctr_qubit = node.ctr_qubit = control_qvec[0]->getPhysicalQubitPtr()->getQubitAddr();
            auto tof_qubit = node.tof_qubit = control_qvec[1]->getPhysicalQubitPtr()->getQubitAddr();

            m_prog_map->m_spilt_num += (m_prog_map->isCorssNode(ctr_qubit, tar_qubit)) ||
                (m_prog_map->isCorssNode(ctr_qubit, tof_qubit)) ||
                (m_prog_map->isCorssNode(tar_qubit, tof_qubit)) ;
        }
    }
    break;

    case GateType::U1_GATE:
    case GateType::RX_GATE:
    case GateType::RY_GATE:
    case GateType::RZ_GATE:
    {
        node.tar_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
        node.gate_parm = dynamic_cast<QGATE_SPACE::AbstractSingleAngleParameter *>(cur_node->getQGate())->getParameter();
    }
    break;

    case GateType::ISWAP_GATE:
    case GateType::SQISWAP_GATE:
    {
        auto ctr_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
        auto tar_qubit = qubits_vector[1]->getPhysicalQubitPtr()->getQubitAddr();
        if (ctr_qubit == tar_qubit || m_prog_map->isCorssNode(ctr_qubit, tar_qubit))
        {
            QCERR("Error");
            throw qprog_syntax_error();
        }
        else
        {
            node.ctr_qubit = ctr_qubit;
            node.tar_qubit = tar_qubit;
        }
    }
    break;

    case GateType::CNOT_GATE:
    case GateType::CZ_GATE:
    {
        node.ctr_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
        node.tar_qubit = qubits_vector[1]->getPhysicalQubitPtr()->getQubitAddr();
        m_prog_map->m_spilt_num += (m_prog_map->isCorssNode(node.ctr_qubit, node.tar_qubit));
    }
    break;

    case GateType::CPHASE_GATE:
    {
        node.ctr_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
        node.tar_qubit = qubits_vector[1]->getPhysicalQubitPtr()->getQubitAddr();
        node.gate_parm = dynamic_cast<QGATE_SPACE::AbstractSingleAngleParameter *>(cur_node->getQGate())->getParameter();
        m_prog_map->m_spilt_num += (m_prog_map->isCorssNode(node.ctr_qubit, node.tar_qubit));
    }
    break;

    default:
    {
        QCERR("UnSupported QGate Node");
        throw undefine_error("QGate");
    }
    break;
    }

    m_prog_map->m_circuit.emplace_back(node);
}

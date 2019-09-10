#include "QPanda.h"
#include <algorithm>
#include "Core/QuantumMachine/SingleAmplitudeQVM.h"
using namespace std;
USING_QPANDA

SingleAmplitudeQVM::SingleAmplitudeQVM()
{ 
    m_singleGateFunc.insert(make_pair(GateType::PAULI_X_GATE,      X_Gate));
    m_singleGateFunc.insert(make_pair(GateType::PAULI_Y_GATE,      Y_Gate));
    m_singleGateFunc.insert(make_pair(GateType::PAULI_Z_GATE,      Z_Gate));
    m_singleGateFunc.insert(make_pair(GateType::X_HALF_PI,        X1_Gate));
    m_singleGateFunc.insert(make_pair(GateType::Y_HALF_PI,        Y1_Gate));
    m_singleGateFunc.insert(make_pair(GateType::Z_HALF_PI,        Z1_Gate));
    m_singleGateFunc.insert(make_pair(GateType::HADAMARD_GATE,     H_Gate));
    m_singleGateFunc.insert(make_pair(GateType::T_GATE,            T_Gate));
    m_singleGateFunc.insert(make_pair(GateType::S_GATE,            S_Gate));

    m_singleAngleGateFunc.insert(make_pair(GateType::RX_GATE,     RX_Gate));
    m_singleAngleGateFunc.insert(make_pair(GateType::RY_GATE,     RY_Gate));
    m_singleAngleGateFunc.insert(make_pair(GateType::RZ_GATE,     RZ_Gate));
    m_singleAngleGateFunc.insert(make_pair(GateType::U1_GATE,     U1_Gate));

    m_doubleGateFunc.insert(make_pair(GateType::CNOT_GATE,      CNOT_Gate));
    m_doubleGateFunc.insert(make_pair(GateType::CZ_GATE,          CZ_Gate));
    m_doubleGateFunc.insert(make_pair(GateType::ISWAP_GATE,    ISWAP_Gate));
    m_doubleGateFunc.insert(make_pair(GateType::CPHASE_GATE, SQISWAP_Gate));

    m_doubleAngleGateFunc.insert(make_pair(GateType::CPHASE_GATE, CR_Gate));
}

void SingleAmplitudeQVM::init()
{
    _Config.maxQubit = 256;
    _Config.maxCMem = 256;
    _start();
}

void SingleAmplitudeQVM::traversalAll(AbstractQuantumProgram *pQProg)
{
    if (nullptr==pQProg)
    {
        QCERR("pQProg is null");
        throw invalid_argument("pQProg is null");
    }
    VerticeMatrix  *vertice_matrix = m_prog_map.getVerticeMatrix();
    vertice_matrix->initVerticeMatrix(getAllocateQubit());
    m_prog_map.setQubitNum(getAllocateQubit());
    TraversalQProg::traversal(pQProg);
}

void SingleAmplitudeQVM::traversal(AbstractQGateNode *pQGate)
{
    if (nullptr == pQGate || nullptr == pQGate->getQGate())
    {
        QCERR("pQGate is null");
        throw invalid_argument("pQGate is null");
    }
    
    QVec qubits_vector;
    pQGate->getQuBitVector(qubits_vector);

    size_t gate_type = pQGate->getQGate()->getGateType();
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
            {
                auto tar_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
                m_singleGateFunc.find(gate_type)->second(m_prog_map, tar_qubit, pQGate->isDagger());
            }
            break;

        case U1_GATE:
        case RX_GATE:
        case RY_GATE:
        case RZ_GATE:
            {
                auto tar_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
                auto gate_parm = dynamic_cast<angleParameter *>(pQGate->getQGate())->getParameter();
                m_singleAngleGateFunc.find(gate_type)->second(m_prog_map, tar_qubit,gate_parm, pQGate->isDagger());
            }
            break;

        case ISWAP_GATE:
        case SQISWAP_GATE:
        case CNOT_GATE:
        case CZ_GATE:
            {
                auto ctr_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
                auto tar_qubit = qubits_vector[1]->getPhysicalQubitPtr()->getQubitAddr();
                m_doubleGateFunc.find(gate_type)->second(m_prog_map, ctr_qubit, tar_qubit, pQGate->isDagger());
            }
            break;

        case CPHASE_GATE: 
            {
                auto ctr_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
                auto tar_qubit = qubits_vector[1]->getPhysicalQubitPtr()->getQubitAddr();
                auto gate_parm = dynamic_cast<angleParameter *>(pQGate->getQGate())->getParameter();
                m_doubleAngleGateFunc.find(gate_type)->second(m_prog_map, ctr_qubit, tar_qubit, gate_parm, pQGate->isDagger());
            }
            break;
        default:
            {
                QCERR("undefined error");
                throw runtime_error("undefined error");
            }
            break;
    }
}

void SingleAmplitudeQVM::run(QProg &prog)
{
    m_prog = prog;
    m_prog_map.clear();
    traversalAll(dynamic_cast<AbstractQuantumProgram *>
        (prog.getImplementationPtr().get()));
}

stat_map SingleAmplitudeQVM::getQStat()
{
    run(m_prog);
    if (m_prog_map.isEmptyQProg())
    {
        QCERR("prog is null");
        throw run_fail("prog is null");
    }

    auto qubit_num = m_prog_map.getQubitNum();
    Qnum qubit_vec;
    for (size_t i = 0; i < qubit_num; ++i)
    {
        qubit_vec.emplace_back(i);
    }

    stat_map temp;
    auto vertice = m_prog_map.getVerticeMatrix();
    qubit_vertice_t qubit_vertice_end, qubit_vertice_begin;

    for (size_t i = 0; i < qubit_num; i++)
    {
        auto iter = vertice->getQubitMapIter(qubit_vec[i]);
        auto vertice_map_iter_b = (*iter).begin();
        qubit_vertice_begin.m_qubit_id = i;
        qubit_vertice_begin.m_num = (*vertice_map_iter_b).first;
        TensorEngine::dimDecrementbyValue(m_prog_map, qubit_vertice_begin, 0);
    }

    uint256_t res_size = ((uint256_t)1 << qubit_num);
    for (uint256_t j = 0; j < res_size; ++j)
    {
        auto new_map = new QuantumProgMap(m_prog_map);
        bool is_operater = false;
        for (size_t i = 0; i < qubit_num; i++)
        {
            auto iter = new_map->getVerticeMatrix()->getQubitMapIter(qubit_vec[i]);
            auto vertice_map_iter = (*iter).end();
            vertice_map_iter--;

            size_t value = ((j >> i) & 1) == 0 ? 0 : 1;
            if ((*iter).size() == 0)
            {
                if (value != 0)
                {
                    temp.insert(make_pair(integerToBinary(j,qubit_num),0));
                    is_operater = true;
                }
                continue;
            }
            qubit_vertice_end.m_qubit_id = i;
            qubit_vertice_end.m_num = (*vertice_map_iter).first;
            TensorEngine::dimDecrementbyValue(*new_map, qubit_vertice_end, value);
        }
        if (!is_operater)
        {
            qcomplex_data_t a;
            split(new_map, nullptr, &a);
            temp.insert(make_pair(integerToBinary(j, qubit_num), a));
        }
        delete new_map;
    }

    m_prog_map.clear();
    return temp;
}

qstate_type SingleAmplitudeQVM::PMeasure_bin_index(string index)
{
    run(m_prog);
    if (m_prog_map.isEmptyQProg() || (index.size() > m_prog_map.getQubitNum()))
    {
        QCERR("PMeasure error");
        throw qprog_syntax_error("PMeasure");
    }

    auto vertice = m_prog_map.getVerticeMatrix();
    qubit_vertice_t qubit_vertice_end, qubit_vertice_begin;
    auto size = vertice->getQubitCount();
    for (size_t i = 0; i < size; i++)
    {
        auto iter = vertice->getQubitMapIter(i);
        auto vertice_map_iter_b = (*iter).begin();
        qubit_vertice_begin.m_qubit_id = i;
        qubit_vertice_begin.m_num = (*vertice_map_iter_b).first;
        TensorEngine::dimDecrementbyValue(m_prog_map, qubit_vertice_begin, 0);
    }

    auto check=[](char bin)
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

    for (size_t i = 0; i < size; i++)
    {
        auto iter = m_prog_map.getVerticeMatrix()->getQubitMapIter(i);
        auto vertice_map_iter = (*iter).end();
        if ((*iter).empty())
        {
            continue;
        }
        vertice_map_iter--;
        size_t value = check(index[size - i - 1]);
        qubit_vertice_end.m_qubit_id = i;
        qubit_vertice_end.m_num = (*vertice_map_iter).first;
        TensorEngine::dimDecrementbyValue(m_prog_map, qubit_vertice_end, value);
    }

    qcomplex_data_t a;
    split(&m_prog_map, nullptr, &a);
    return (a.real() * a.real() + a.imag() * a.imag());
}

qstate_type SingleAmplitudeQVM::PMeasure_dec_index(string index)
{
    uint256_t dec_index(index.c_str());
    return PMeasure_bin_index(integerToBinary(dec_index,m_prog_map.getQubitNum()));
}

prob_map SingleAmplitudeQVM::PMeasure(QVec qvec, string select_max)
{
    run(m_prog);
    if (m_prog_map.isEmptyQProg())
    {
        QCERR("prog is null");
        throw run_fail("prog is null");
    }
    
    Qnum  pvec;
    for_each(qvec.begin(), qvec.end(), [&](Qubit *qubit) 
        {pvec.emplace_back(qubit->getPhysicalQubitPtr()->getQubitAddr()); });
    sort(pvec.begin(), pvec.end());
    auto iter = adjacent_find(pvec.begin(), pvec.end());

    uint256_t select_max_size(select_max.c_str());
    auto qubit_num = m_prog_map.getQubitNum();
    Qnum qubit_vec;
    for (size_t i = 0; i < qubit_num; ++i)
    {
        qubit_vec.emplace_back(i);
    }

    if ((pvec.end() != iter) ||
        select_max_size > ((uint256_t)1 << pvec.size()))
    {
        QCERR("PMeasure error");
        throw qprog_syntax_error("PMeasure");
    }

    uint256_t value_size = (uint256_t)1 << (qubit_num - pvec.size());
    prob_map res;
    auto pmeasure_size = pvec.size();
    if (pmeasure_size <= qubit_num)
    {
        auto vertice = m_prog_map.getVerticeMatrix();
        qubit_vertice_t qubit_vertice_end, qubit_vertice_begin;

        for (size_t i = 0; i < qubit_num; ++i)
        {
            auto iter = vertice->getQubitMapIter(i);
            auto vertice_map_iter_b = (*iter).begin();
            qubit_vertice_begin.m_qubit_id = i;
            qubit_vertice_begin.m_num = (*vertice_map_iter_b).first;
            TensorEngine::dimDecrementbyValue(m_prog_map, qubit_vertice_begin, 0);
        }

        for (uint256_t i = 0; i < select_max_size; ++i)
        {
            double temp_value = 0.0;
            for (uint256_t j = 0; j < value_size; ++j)
            {
                auto new_map = new QuantumProgMap(m_prog_map);
                bool is_operater = false;
                uint256_t index = getDecIndex(i, j, pvec, qubit_num);
                for (size_t k = 0; k < qubit_num; ++k)
                {
                    auto iter = new_map->getVerticeMatrix()->getQubitMapIter(qubit_vec[k]);
                    auto vertice_map_iter = (*iter).end();
                    vertice_map_iter--;

                    size_t value = (size_t)((index >> k) & 1);
                    if ((*iter).size() == 0)
                    {
                        if (value != 0)
                        {
                            temp_value += 0;
                            is_operater = true;
                        }
                        continue;
                    }
                    qubit_vertice_end.m_qubit_id = k;
                    qubit_vertice_end.m_num = (*vertice_map_iter).first;
                    TensorEngine::dimDecrementbyValue(*new_map, qubit_vertice_end, value);
                }
                if (!is_operater)
                {
                    qcomplex_data_t a;
                    split(new_map, nullptr, &a);
                    temp_value += (a.real() * a.real() + a.imag() * a.imag());
                }
                delete new_map;
            }

            res.insert(make_pair(integerToString(i),temp_value));
        }

        return res;
    }
    else
    {
        QCERR("PMeasure error");
        throw qprog_syntax_error("PMeasure");
    }
}

prob_map SingleAmplitudeQVM::PMeasure(string select_max)
{
    run(m_prog);
    if (m_prog_map.isEmptyQProg())
    {
        QCERR("prog is null");
        throw run_fail("prog is null");
    }

    auto qubit_num = m_prog_map.getQubitNum();
    uint256_t select_max_size(select_max.c_str());
    if (select_max_size > ((uint256_t)1 << qubit_num))
    {
        QCERR("PMeasure Error");
        throw qprog_syntax_error("PMeasure");
    }

    vector<size_t> qubit_vec;
    for (size_t i = 0; i < qubit_num; ++i)
    {
        qubit_vec.emplace_back(i);
    }

    auto vertice = m_prog_map.getVerticeMatrix();
    qubit_vertice_t qubit_vertice_end,qubit_vertice_begin;

    for (size_t i = 0; i < qubit_num; i++)
    {
        auto iter = vertice->getQubitMapIter(qubit_vec[i]);
        auto vertice_map_iter_b = (*iter).begin();
        qubit_vertice_begin.m_qubit_id = i;
        qubit_vertice_begin.m_num = (*vertice_map_iter_b).first;
        TensorEngine::dimDecrementbyValue(m_prog_map, qubit_vertice_begin, 0);
    }

    prob_map temp;
    for (uint256_t j = 0; j < select_max_size; ++j)
    {
        auto new_map = new QuantumProgMap(m_prog_map);
        bool is_operater = false;
        for (size_t i = 0; i < qubit_num; i++)
        {
            auto iter = new_map->getVerticeMatrix()->getQubitMapIter(qubit_vec[i]);
            auto vertice_map_iter = (*iter).end();
            vertice_map_iter--;
            size_t value = ((j >> i) & 1) == 0 ? 0 : 1;
            if ((*iter).size() == 0)
            {
                if (value != 0)
                {
                    temp.insert(make_pair(integerToString(j),0));
                    is_operater = true;
                }
                continue;
            }
            qubit_vertice_end.m_qubit_id = i;
            qubit_vertice_end.m_num = (*vertice_map_iter).first;
            TensorEngine::dimDecrementbyValue(*new_map, qubit_vertice_end, value);
        }
        if (!is_operater)
        {
            qcomplex_data_t a;
            split(new_map, nullptr, &a);
            auto result = a.real() * a.real() + a.imag() * a.imag();
            temp.insert(make_pair(integerToString(j), result));
        }
        delete new_map;
    }

    m_prog_map.clear();
    return temp;
}

prob_map SingleAmplitudeQVM::getProbDict(QVec qvec, string select_max)
{
    run(m_prog);
    if (m_prog_map.isEmptyQProg())
    {
        QCERR("prog is null");
        throw run_fail("prog is null");
    }

    Qnum  pvec;
    for_each(qvec.begin(), qvec.end(), [&](Qubit *qubit)
    {pvec.emplace_back(qubit->getPhysicalQubitPtr()->getQubitAddr()); });
    sort(pvec.begin(), pvec.end());
    auto iter = adjacent_find(pvec.begin(), pvec.end());

    uint256_t select_max_size(select_max.c_str());
    auto qubit_num = m_prog_map.getQubitNum();
    Qnum qubit_vec;
    for (size_t i = 0; i < qubit_num; ++i)
    {
        qubit_vec.emplace_back(i);
    }

    if ((pvec.end() != iter) ||
        select_max_size > ((uint256_t)1 << pvec.size()))
    {
        QCERR("PMeasure error");
        throw qprog_syntax_error("PMeasure");
    }

    uint256_t value_size = (uint256_t)1 << (qubit_num - pvec.size());
    prob_map res;
    auto pmeasure_size = pvec.size();
    if (pmeasure_size <= m_prog_map.getQubitNum())
    {
        auto vertice = m_prog_map.getVerticeMatrix();
        qubit_vertice_t qubit_vertice_end, qubit_vertice_begin;

        for (size_t i = 0; i < qubit_num; ++i)
        {
            auto iter = vertice->getQubitMapIter(i);
            auto vertice_map_iter_b = (*iter).begin();
            qubit_vertice_begin.m_qubit_id = i;
            qubit_vertice_begin.m_num = (*vertice_map_iter_b).first;
            TensorEngine::dimDecrementbyValue(m_prog_map, qubit_vertice_begin, 0);
        }

        for (uint256_t i = 0; i < select_max_size; ++i)
        {
            double temp_value = 0.0;
            for (uint256_t j = 0; j < value_size; ++j)
            {
                auto new_map = new QuantumProgMap(m_prog_map);
                bool is_operater = false;
                uint256_t index = getDecIndex(i, j, pvec, qubit_num);
                for (size_t k = 0; k < qubit_num; ++k)
                {
                    auto iter = new_map->getVerticeMatrix()->getQubitMapIter(qubit_vec[k]);
                    auto vertice_map_iter = (*iter).end();
                    vertice_map_iter--;

                    size_t value = (size_t)((index >> k) & 1);
                    if ((*iter).size() == 0)
                    {
                        if (value != 0)
                        {
                            temp_value += 0;
                            is_operater = true;
                        }
                        continue;
                    }
                    qubit_vertice_end.m_qubit_id = k;
                    qubit_vertice_end.m_num = (*vertice_map_iter).first;
                    TensorEngine::dimDecrementbyValue(*new_map, qubit_vertice_end, value);
                }
                if (!is_operater)
                {
                    qcomplex_data_t a;
                    split(new_map, nullptr, &a);
                    temp_value += (a.real() * a.real() + a.imag() * a.imag());
                }
                delete new_map;
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

prob_map SingleAmplitudeQVM::probRunDict(QProg &prog, QVec qvec, string select_max)
{
    run(prog);
    return getProbDict(qvec,select_max);
}

void SingleAmplitudeQVM::run(string sFilePath)
{
    auto prog = CreateEmptyQProg();
    transformQRunesToQProg(sFilePath, prog, this);
    run(prog);
}
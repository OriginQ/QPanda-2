#include "QPanda.h"
#include <algorithm>
#include "include/Core/QuantumMachine/SingleAmplitudeQVM.h"
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

QStat SingleAmplitudeQVM::getQStat()
{
    run(m_prog);
    if (m_prog_map.isEmptyQProg())
    {
        QCERR("pQProg is null");
        throw run_fail("pQProg is null");
    }

    auto qubit_num = m_prog_map.getQubitNum();

    vector<size_t> qubit_vec;
    for (size_t i = 0; i < qubit_num; ++i)
    {
        qubit_vec.emplace_back(i);
    }

    vector<complex<double>> temp;
    auto vertice = m_prog_map.getVerticeMatrix();
    qubit_vertice_t qubit_vertice_end, qubit_vertice_begen;

    for (size_t i = 0; i < qubit_num; i++)
    {
        auto iter = vertice->getQubitMapIter(qubit_vec[i]);
        auto vertice_map_iter_b = (*iter).begin();
        qubit_vertice_begen.m_qubit_id = i;
        qubit_vertice_begen.m_num = (*vertice_map_iter_b).first;
        TensorEngine::dimDecrementbyValue(m_prog_map, qubit_vertice_begen, 0);
    }

    for (size_t j = 0; j < 1ull << qubit_num; j++)
    {
        auto new_map = new QuantumProgMap(m_prog_map);
        bool is_operater = false;
        for (size_t i = 0; i < qubit_num; i++)
        {
            auto iter = new_map->getVerticeMatrix()->getQubitMapIter(qubit_vec[i]);
            auto vertice_map_iter = (*iter).end();
            vertice_map_iter--;
            size_t value = j;
            value = (value >> i) & 1;
            if ((*iter).size() == 0)
            {
                if (value != 0)
                {
                    temp.emplace_back(0);
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
            temp.emplace_back(a);
        }
        delete new_map;
    }
    return temp;
}

double SingleAmplitudeQVM::PMeasure_index(size_t index)
{
    run(m_prog);
    if (m_prog_map.isEmptyQProg())
    {
        QCERR("pQProg is null");
        throw run_fail("pQProg is null");
    }

    if (index >=(1ull << m_prog_map.getQubitNum()))
    {
        QCERR("PMeasure error");
        throw qprog_syntax_error("PMeasure");
    }

    vector<pair<size_t, double>> temp;
    auto vertice = m_prog_map.getVerticeMatrix();
    qubit_vertice_t qubit_vertice_end, qubit_vertice_begen;
    auto size = vertice->getQubitCount();
    for (size_t i = 0; i < size; i++)
    {
        auto iter = vertice->getQubitMapIter(i);
        auto vertice_map_iter_b = (*iter).begin();
        qubit_vertice_begen.m_qubit_id = i;
        qubit_vertice_begen.m_num = (*vertice_map_iter_b).first;
        TensorEngine::dimDecrementbyValue(m_prog_map, qubit_vertice_begen, 0);
    }

    for (size_t i = 0; i < size; i++)
    {
        auto iter = m_prog_map.getVerticeMatrix()->getQubitMapIter(i);
        auto vertice_map_iter = (*iter).end();
        vertice_map_iter--;
        size_t value = (index >> i) & 1;
        qubit_vertice_end.m_qubit_id = i;
        qubit_vertice_end.m_num = (*vertice_map_iter).first;
        TensorEngine::dimDecrementbyValue(m_prog_map, qubit_vertice_end, value);
    }

    qcomplex_data_t a;
    split(&m_prog_map, nullptr, &a);
    auto result = a.real() * a.real() + a.imag() * a.imag();
    return result;
}

vector<double> 
SingleAmplitudeQVM::PMeasure(QVec qvec, size_t select_max)
{
    run(m_prog);
    if (m_prog_map.isEmptyQProg())
    {
        QCERR("pQProg is null");
        throw run_fail("pQProg is null");
    }

    vector<size_t>  qubit_vec;
    for_each(qvec.begin(), qvec.end(), [&](Qubit *qubit) 
        {qubit_vec.emplace_back(qubit->getPhysicalQubitPtr()->getQubitAddr()); });

    auto qubit_num = m_prog_map.getQubitNum();
    sort(qubit_vec.begin(), qubit_vec.end());
    auto iter = adjacent_find(qubit_vec.begin(), qubit_vec.end());

    if ((qubit_vec.end() != iter) || 
         select_max >= (1ull << qubit_vec.size()))
    {
        QCERR("PMeasure error");
        throw qprog_syntax_error("PMeasure");
    }

    vector<double> temp(1ull << qubit_vec.size());
    auto vertice = m_prog_map.getVerticeMatrix();
    qubit_vertice_t qubit_vertice_end, qubit_vertice_begen;

    for (size_t i = 0; i < qubit_num; i++)
    {
        auto iter = vertice->getQubitMapIter(qubit_vec[i]);
        auto vertice_map_iter_b = (*iter).begin();
        qubit_vertice_begen.m_qubit_id = i;
        qubit_vertice_begen.m_num = (*vertice_map_iter_b).first;
        TensorEngine::dimDecrementbyValue(m_prog_map, qubit_vertice_begen, 0);
    }

    for (size_t j = 0; j < (1ull << qubit_num); ++j)
    {
        auto new_map = new QuantumProgMap(m_prog_map);
        bool is_operater = false;
        for (size_t i = 0; i < qubit_num; i++)
        {
            auto iter = new_map->getVerticeMatrix()->getQubitMapIter(qubit_vec[i]);
            auto vertice_map_iter = (*iter).end();
            size_t value = j;
            value = (value >> i) & 1;
            if ((*iter).size() == 0)
            {
                if (value != 0)
                {
                    temp[j] = 0;
                    is_operater = true;
                }
                continue;
            }
            vertice_map_iter--;
            qubit_vertice_end.m_qubit_id = i;
            qubit_vertice_end.m_num = (*vertice_map_iter).first;
            TensorEngine::dimDecrementbyValue(*new_map, qubit_vertice_end, value);
        }
        if (!is_operater)
        {
            qcomplex_data_t a;
            split(new_map, nullptr, &a);
            auto result = a.real() * a.real() + a.imag() * a.imag();
            temp[j] = result;
        }
        delete new_map;
    }

    auto pmeasure_size = qubit_vec.size();
    if (pmeasure_size < m_prog_map.getQubitNum())
    {
        size_t bit = 0;
        vector<double> result_vec(1ull << pmeasure_size);
        for (size_t i = 0; i < temp.size(); ++i)
        {
            size_t result_index{ 0 };
            for (size_t j = 0; j < pmeasure_size; ++j)
            {
                bit = ((i & (1 << qubit_vec[j])) >> qubit_vec[j]);
                result_index += bit ? (1ull << j) : 0;
            }
            result_vec[result_index] += temp[i];
        }

        vector<double> res;
        for (size_t i = 0; i < select_max; ++i)
        {
            res.emplace_back(result_vec[i]);
        }
        return res;
    }
    else if (pmeasure_size == m_prog_map.getQubitNum())
    {
        vector<double> res;
        for (size_t i = 0; i < select_max; ++i)
        {
            res.emplace_back(temp[i]);
        }
        return res;
    }
    else
    {
        QCERR("PMeasure error");
        throw qprog_syntax_error("PMeasure");
    }
}

vector<pair<size_t, double>> 
SingleAmplitudeQVM::PMeasure(size_t select_max)
{
    run(m_prog);
    if (m_prog_map.isEmptyQProg())
    {
        QCERR("pQProg is null");
        throw run_fail("pQProg is null");
    }
    auto qubit_num = m_prog_map.getQubitNum();

    if (select_max > (1ull << qubit_num))
    {
        QCERR("PMeasure Error");
        throw qprog_syntax_error("PMeasure");
    }

    vector<size_t> qubit_vec;
    for (size_t i = 0; i < qubit_num; ++i)
    {
        qubit_vec.emplace_back(i);
    }

    vector<pair<qsize_t, double>> temp;
    auto vertice = m_prog_map.getVerticeMatrix();
    qubit_vertice_t qubit_vertice_end;
    qubit_vertice_t qubit_vertice_begen;

    for (size_t i = 0; i < qubit_num; i++)
    {
        auto iter = vertice->getQubitMapIter(qubit_vec[i]);
        auto vertice_map_iter_b = (*iter).begin();
        qubit_vertice_begen.m_qubit_id = i;
        qubit_vertice_begen.m_num = (*vertice_map_iter_b).first;
        TensorEngine::dimDecrementbyValue(m_prog_map, qubit_vertice_begen, 0);
    }

    for (size_t j = 0; j < select_max; j++)
    {
        auto new_map = new QuantumProgMap(m_prog_map);
        bool is_operater = false;
        for (size_t i = 0; i < qubit_num; i++)
        {
            auto iter = new_map->getVerticeMatrix()->getQubitMapIter(qubit_vec[i]);
            auto vertice_map_iter = (*iter).end();
            size_t value = j;
            value = (value >> i) & 1;
            if ((*iter).size() == 0)
            {
                if (value != 0)
                {
                    temp.emplace_back(make_pair(j, 0));
                    is_operater = true;
                }
                continue;
            }
            vertice_map_iter--;
            qubit_vertice_end.m_qubit_id = i;
            qubit_vertice_end.m_num = (*vertice_map_iter).first;
            TensorEngine::dimDecrementbyValue(*new_map, qubit_vertice_end, value);
        }
        if (!is_operater)
        {
            qcomplex_data_t a;
            split(new_map, nullptr, &a);
            auto result = a.real() * a.real() + a.imag() * a.imag();
            temp.emplace_back(make_pair(j, result));
        }
        delete new_map;
    }


    m_prog_map.clear();
    return temp;
}


std::vector<double> 
SingleAmplitudeQVM::getProbList(QVec qvec, size_t select_max)
{
    return PMeasure(qvec, select_max);
}

std::vector<double> 
SingleAmplitudeQVM::probRunList(QProg &prog, QVec qvec, size_t select_max)
{
    run(prog);
    return PMeasure(qvec, select_max);
}


std::map<std::string, double> 
SingleAmplitudeQVM::getProbDict(QVec qvec, size_t select_max)
{
    auto res = PMeasure(qvec, select_max);

    std::map<std::string, double>  result_map;
    size_t size = qvec.size();
    for (size_t i = 0; i < select_max; ++i)
    {
        stringstream ss;
        for (int j = size - 1; j > -1; j--)
        {
            ss << ((i >> j) & 1);
        }
        result_map.insert(make_pair(ss.str(), res[i]));
    }
    return result_map;
}

std::map<std::string, double> 
SingleAmplitudeQVM::probRunDict(QProg &prog, QVec qvec, size_t select_max)    
{
    run(prog);
    return getProbDict(qvec,select_max);
}


std::vector<std::pair<size_t, double>> 
SingleAmplitudeQVM::getProbTupleList(QVec qvec, size_t select_max)
{
    auto res = PMeasure(qvec, select_max);
    std::vector<std::pair<size_t, double>> result_vec;
    for (size_t i = 0; i < res.size(); ++i)
    {
        result_vec.emplace_back(make_pair(i,res[i]));
    }
    return result_vec;
}

std::vector<std::pair<size_t, double>> 
SingleAmplitudeQVM::probRunTupleList(QProg &prog, QVec qvec, size_t select_max)
{
    run(prog);
    return getProbTupleList(qvec, select_max);
}



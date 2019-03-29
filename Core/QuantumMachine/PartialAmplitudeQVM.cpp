#include "Core/QuantumMachine/PartialAmplitudeQVM.h"
using namespace std;
USING_QPANDA

void PartialAmplitudeQVM::getAvgBinary(long long num, size_t QubitNum)
{
    size_t half_qubit = QubitNum / 2;
    long long lower_mask = (1 << half_qubit) - 1;
    low_pos = (num & lower_mask);
    high_pos = (num - low_pos) >> half_qubit;
}

PartialAmplitudeQVM::PartialAmplitudeQVM()
{
    m_prog_map = new MergeMap();
}

PartialAmplitudeQVM::~PartialAmplitudeQVM()
{
    delete  m_prog_map;
}

void PartialAmplitudeQVM::run(QProg& prog)
{
    if (nullptr == m_prog_map)
    {
        QCERR("prog is null");
        throw run_fail("prog is null");
    }
    else
    {
        m_prog_map->clear();
        m_prog_map->traversalAll(dynamic_cast<AbstractQuantumProgram *>
            (prog.getImplementationPtr().get()));
    }
}

QStat PartialAmplitudeQVM::getQStat()
{
    if (nullptr == m_prog_map)
    {
        QCERR("prog is null");
        throw run_fail("prog is null");
    }

    vector<vector<QStat>> calculateMap_vector;
    MergeMap mergeEngine;
    for (unsigned long i = 0; i < m_prog_map->getMapVecSize(); ++i)
    {
        vector<QStat> calculateMap;
        for (size_t j = 0; j < m_prog_map->m_circuit_vec[i].size(); ++j)
        {
            QuantumGateParam* pGateParam = new QuantumGateParam();
            pGateParam->m_qubit_number = m_prog_map->m_qubit_num / 2;
            QPUImpl *pQGate = new CPUImplQPU();
            pQGate->initState(pGateParam);
            mergeEngine.traversalMap(m_prog_map->m_circuit_vec[i][j], pQGate, pGateParam);
            calculateMap.emplace_back(pQGate->getQState());
            delete pGateParam;
            delete pQGate;
        }
        calculateMap_vector.emplace_back(calculateMap);
    }

    QStat addResult_vector(1ull << m_prog_map->m_qubit_num, 0);
    for (size_t i = 0; i < 1ull << m_prog_map->m_qubit_num; ++i)
    {
        complex<double> addResult(0, 0);
        getAvgBinary(i, m_prog_map->m_qubit_num);
        for (size_t j = 0; j < calculateMap_vector.size(); j++)
        {
            complex<double> multiResult(1, 0);
            if (calculateMap_vector[j].size() > 1)
            {
                multiResult = calculateMap_vector[j][0][low_pos] * calculateMap_vector[j][1][high_pos];
            }
            else
            {
                multiResult = calculateMap_vector[j][0][low_pos] * (high_pos == 0 ? 1.0 : 0.0);
            }
            addResult = addResult + multiResult;
        }
        addResult_vector[i] = addResult;
    }

    return addResult_vector;
}

vector<double> 
PartialAmplitudeQVM::PMeasure(QVec qvec, int select_max)
{
    vector<size_t>  qubit_vec;
    for_each(qvec.begin(), qvec.end(), [&](Qubit *qubit)
    {qubit_vec.emplace_back(qubit->getPhysicalQubitPtr()->getQubitAddr()); });

    sort(qubit_vec.begin(), qubit_vec.end());
    auto iter = adjacent_find(qubit_vec.begin(), qubit_vec.end());
    if ((qubit_vec.end() != iter) || select_max > (1ull << qubit_vec.size())
        || *(qubit_vec.end() - 1) > m_prog_map->m_qubit_num)
    {
        QCERR("PMeasure error");
        throw qprog_syntax_error("PMeasure");
    }

    auto addResult_vector = getQStat();
    size_t bit = 0;
    auto pmeasure_size = qubit_vec.size();
    vector<double> result_vec(1ull << pmeasure_size);
    for (size_t i = 0; i < addResult_vector.size(); ++i)
    {
        size_t result_index{ 0 };
        for (size_t j = 0; j < pmeasure_size; ++j)
        {
            bit = ((i & (1 << qubit_vec[j])) >> qubit_vec[j]);
            result_index += bit ? (1ull << j) : 0;
        }
        result_vec[result_index] += addResult_vector[i].real() * addResult_vector[i].real() +
            addResult_vector[i].imag() * addResult_vector[i].imag();
    }

    if (pmeasure_size < m_prog_map->m_qubit_num)
    {
        vector<double> res;
        for (size_t i = 0; i < select_max; ++i)
        {
            res.emplace_back(result_vec[i]);
        }
        return res;
    }
    else
    {
        return result_vec;
    }

}

vector<pair<size_t, double>> 
PartialAmplitudeQVM::PMeasure(int select_max)
{
    if (select_max > (1ull << m_prog_map->m_qubit_num))
    {
        QCERR("PMeasure error");
        throw qprog_syntax_error("PMeasure");
    }

    auto qstate_vec = getQStat();

    vector<pair<size_t, double>> temp;
    auto val_num = select_max < 0 ? qstate_vec.size() : select_max;
    for (auto i = 0; i < val_num; ++i)
    {
        double value = qstate_vec[i].real() * qstate_vec[i].real() +
            qstate_vec[i].imag() * qstate_vec[i].imag();
        temp.emplace_back(make_pair(i, value));
    }

    return temp;
}



std::vector<double> 
PartialAmplitudeQVM::getProbList(QVec  qvec, int select_max)
{
    return PMeasure(qvec, select_max);
}

std::vector<double> 
PartialAmplitudeQVM::probRunList(QProg &prog, QVec qvec, int select_max)
{
    run(prog);
    return PMeasure(qvec, select_max);
}



std::map<std::string, double> 
PartialAmplitudeQVM::getProbDict(QVec qvec, int select_max)
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
PartialAmplitudeQVM::probRunDict(QProg &prog, QVec qvec, int select_max)
{
    run(prog);
    return getProbDict(qvec, select_max);
}



std::vector<std::pair<size_t, double>> 
PartialAmplitudeQVM::getProbTupleList(QVec qvec, int select_max)
{
    auto res = PMeasure(qvec, select_max);
    std::vector<std::pair<size_t, double>> result_vec;
    for (size_t i = 0; i < res.size(); ++i)
    {
        result_vec.emplace_back(make_pair(i, res[i]));
    }
    return result_vec;
}

std::vector<std::pair<size_t, double>> 
PartialAmplitudeQVM::probRunTupleList(QProg &prog, QVec qvec, int select_max)
{
    run(prog);
    return getProbTupleList(qvec, select_max);
}

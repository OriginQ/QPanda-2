#include "QAlg/Shor/Shor.h"

QPANDA_BEGIN
using namespace std;
int ShorAlg::_gcd(int a, int b)
{
    if (0 == b)
        return a;
    return _gcd(b, a % b);
}

int ShorAlg::_continuous_frac_expan(int target, int result)
{
    int Q = 1 << (int)ceil(log2(target)) * 2, d = 0;
    vector<int> cof, cof_d;
    double target_value = result * 1. / Q, tol = 1. / (2. * Q), residual = 1;

    // find the integer of the fraction unless close enough
    cof.emplace_back((1. / target_value));
    residual = 1. / cof[cof.size() - 1];
    while (abs(residual - target_value) > tol)
    {
        residual = target_value;
        residual = accumulate(cof.begin(), cof.end(), residual, [](double x, double y) {return 1. / x - y; });
        cof.emplace_back(1. / residual);

        residual = 0;
        for (int i = cof.size() - 1; i >= 0; i--)
        {
            residual = 1. / (cof[i] + residual);
        }
    }

    // regenerate the denominator
    for (int i = cof.size() - 1; i >= 0; i--)
    {
        int last_d = cof_d.size() > 1 ? cof_d[cof_d.size() - 2] : 1;
        int cur_denominator = cof_d.size() ? cof[i] * cof_d[cof_d.size() - 1] + last_d : cof[i];
        cof_d.emplace_back(cur_denominator);
    }
    d = cof_d[cof_d.size() - 1];

    return d;
}

int ShorAlg::_measure_result_parse(int target, vector<int> max_result)
{
    int r;
    for (int i = 0; i < max_result.size(); i++)
    {
        if (0 == max_result[i])
        {
            max_result[i] = 1;
            continue;
        }
        max_result[i] = _continuous_frac_expan(target, max_result[i]);
    }

    // get the LCM of all results
    r = 1;
    for (auto sample : max_result)
    {
        r = r * sample / _gcd(r, sample);
    }
    return r;
}


int ShorAlg::_period_finding(int base, int target)
{
    int q = ceil(log(target) / log(2)), p = 2 * q, max_prob;
    vector<int> max_result;
    auto qvm = initQuantumMachine();
    QVec cqv = qvm->qAllocMany(p), tqv = qvm->qAllocMany(q), qvec1 = qvm->
        qAllocMany(q), qvec2 = qvm->qAllocMany(q), qvec3 = qvm->qAllocMany(2);
    QProg  qcProg = QProg();

    for (auto i = 0; i < p; i++)
    {
        qcProg << H(cqv[i]);
    }
    qcProg << X(tqv[0]);
    qcProg << constModExp(cqv, tqv, base, target, qvec1, qvec2, qvec3);
    qcProg << QFT(cqv).dagger();
    qvm->directlyRun(qcProg);
    auto result = quickMeasure(cqv, p * p * p);
    destroyQuantumMachine(qvm);

    // get the state with probability larger than max_prob/2
    max_prob = 0;
    for (auto& val : result)
    {
        max_prob = val.second > max_prob ? val.second : max_prob;
    }
    for (auto& val : result) {
        if (0 == stoi(val.first, 0, 2)) continue;
        if (val.second > max_prob / 2) max_result.emplace_back(stoi(val.first, 0, 2));
    }

    // get the LCM of denominators of the state of high probability
    return _measure_result_parse(target, max_result);
}

bool ShorAlg::exec()
{
    for (int i = starter; i < m_target_Num; i++)
    {
        if (_gcd(i, m_target_Num) > 1)
        {
            m_factor_1 = _gcd(i, m_target_Num);
            m_factor_2 = m_target_Num / m_factor_1;
            return true;
        }
        else
        {
            m_factor_2 = _period_finding(i, m_target_Num);
            m_factor_1 = (int)pow(i, m_factor_2 / 2);
            // give up if cannot find a proper result
            if (m_factor_2 >= m_target_Num || 0 != m_factor_2 % 2 || 0 == (m_factor_1 + 1) % m_target_Num)
            {
                continue;
            }
            m_factor_2 = _gcd(m_factor_1 + 1, m_target_Num);
            m_factor_1 = _gcd(m_factor_1 - 1, m_target_Num);
            if (m_factor_1 * m_factor_2 != m_target_Num) {
                return false;
            }
            else {
                return true;;
            }
        }
    }

    throw ("check the input number, its prime factorization cannot be done!");
    return false;
}

std::pair<int, int> ShorAlg::get_results()
{
    return std::make_pair(m_factor_1, m_factor_2);
}

void ShorAlg::set_decomposition_starter(int smallest_base)
{
    starter = smallest_base;
}

ShorAlg::ShorAlg(int target)
{
    if (target <= 1)
    {
        QCERR("number is smaller than 2!");
        throw ("check the input number, it is smaller than 2!");
    }
    m_target_Num = target;
}



std::pair<bool, std::pair<int, int>> Shor_factorization(int target)
{
    ShorAlg sample = ShorAlg(target);
    bool sucess = sample.exec();
    auto factors = sample.get_results();
    return std::make_pair(sucess, factors);
}

QPANDA_END

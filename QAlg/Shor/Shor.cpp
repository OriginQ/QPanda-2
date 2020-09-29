#include "QAlg/Shor/Shor.h"

using namespace QPanda;

int ShorAlg::_gcd(int a, int b)
{
    if (0 == b)
        return a;
    return _gcd(b, a % b);
}

int ShorAlg::_continuous_frac_expan(int target, int result)
{
    int Q = 1 << (int)ceil(log(target) / log(2)) * 2;
    int cof[5] = {}, len = 0, d = 0;
    double target_value = result * 1. / Q, tol = 1. / (2. * Q), residual = 1;

    // find the integer of the fraction unless close enough
    cof[len++] = (int)(1. / target_value);
    residual = 1. / cof[len - 1];
    while (abs(residual - target_value) > tol)
    {
        residual = target_value;
        for (int i = 0; i < len; i++) {
            residual = 1. / residual - cof[i];
        }
        cof[len++] = (int)(1. / residual);

        residual = 0;
        for (int i = len - 1; i >= 0; i--)
        {
            residual = 1. / (cof[i] + residual);
        }
    }

    // regenerate the denominator
    d = cof[len--];
    while (len >= 0)
    {
        d = 0 == d ? cof[len--] : cof[len--] * d + 1;
    }

    return d;
}

int ShorAlg::_measure_result_parse(int target, vector<int> max_result)
{
    int r;
    for (int i = 0; i < 5; i++)
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


int ShorAlg::_perid_finding(int base, int target)
{
    int q = ceil(log(target) / log(2)), p = 2 * q, max_prob;
    vector<int> max_result(5), prob(5);
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
    for (auto val : result)
    {
        max_prob = val.second > max_prob ? val.second : max_prob;
    }
    for (auto val : result) {
        if (0 == stoi(val.first, 0, 2)) continue;
        for (int i = 0; i < 5; i++)
        {
            if (prob[i] < max_prob / 2 && val.second > max_prob / 2)
            {
                max_result[i] = stoi(val.first, 0, 2);
                prob[i] = val.second;
                break;
            }
        }
    }

    // get the LCM of denominators of the state of high probability
    return _measure_result_parse(target, max_result);
}

void ShorAlg::get_Shor_result(int target, int &p, int &q)
{
    for (int i = 2; i < target; i++)
    {
        if (_gcd(i, target) > 1)
        {
            p = i;
            q = target / p;
            return;
        }
        else
        {
            q = _perid_finding(i, target);
            p = (int)pow(i, q / 2);
            // give up if cannot find a proper result
            if (q >= target || 0 != q % 2 || 0 == (p + 1) % target)
            {
                continue;
            }
            q = _gcd(p + 1, target);
            p = _gcd(p - 1, target);
            return;
        }
    }
    QCERR("prime factorization cannot be done!");
    throw ("check the input number, its prime factorization cannot be done!");
    return;
}

ShorAlg::ShorAlg(int target)
{
    if (target <= 1)
    {
        QCERR("number is smaller than 2!");
        throw ("check the input number, it is smaller than 2!");
    }
    m_target_Num = target;
    get_Shor_result(m_target_Num, m_s1, m_s2);
}

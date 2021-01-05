#include <cstring>
#include "Components/Operator/PauliOperator.h"

QPANDA_BEGIN

std::vector<double> kron(
    const std::vector<double>& vec1,
    const std::vector<double>& vec2)
{
    std::vector<double> result;
    result.resize(vec1.size() * vec2.size());
    int cnt = 0;
    for (int i = 0; i < vec1.size(); i++)
    {
        for (int j = 0; j < vec2.size(); j++)
        {
            result[cnt] = vec1[i] * vec2[j];
            cnt++;
        }
    }

    return result;
}

std::vector<double> dot(
    const std::vector<double>& vec1,
    const std::vector<double>& vec2)
{
    if (vec1.size() != vec2.size())
    {
        throw std::runtime_error("vec1 and vec2 size not equal!");
    }

    std::vector<double> result;
    result.resize(vec1.size());
    for (int i = 0; i < vec1.size(); i++)
    {
        result[i] = vec1[i] * vec2[i];
    }

    return result;
}

std::vector<double> operator +(
    const std::vector<double>& vec1,
    const std::vector<double>& vec2)
{
    if (vec1.size() != vec2.size())
    {
        throw std::runtime_error("vec1 and vec2 size not equal!");
    }

    std::vector<double> result;
    result.resize(vec1.size());
    for (int i = 0; i < vec1.size(); i++)
    {
        result[i] = vec1[i] + vec2[i];
    }

    return result;
}

std::vector<double> operator *(
    const std::vector<double>& vec,
    double value)
{
    std::vector<double> result;
    result.resize(vec.size());
    for (int i = 0; i < vec.size(); i++)
    {
        result[i] = vec[i] * value;
    }

    return result;
}

std::vector<double> transPauliOperatorToVec(PauliOperator pauli)
{
    if (!pauli.isAllPauliZorI())
    {
        return {};
    }

    bool ok = true;
    auto hamiltonian = pauli.toHamiltonian(&ok);
    if (!ok)
    {
        return {};
    }

    int max_index = pauli.getMaxIndex();
    std::vector<double> value;
    value.resize((int)(std::pow(2, max_index)));
    memset(value.data(), 0, sizeof(double) * value.size());

    for (auto& item : hamiltonian)
    {
        std::vector<double> tmp_value;
        tmp_value.resize((int)(std::pow(2, max_index)));
        for (int t = 0; t < tmp_value.size(); t++)
        {
            tmp_value[t] = 1;
        }

        if (item.first.empty())
        {
            for (int i = 0; i < tmp_value.size(); i++)
            {
                tmp_value[i] = item.second;
            }
        }
        else
        {
            for (auto term : item.first)
            {
                std::vector<double> cur = { 1, -1 };

                int L_size = term.first;
                int H_size = max_index - 1 - term.first;

                if (L_size > 0)
                {
                    std::vector<double> I_L;
                    I_L.resize((int)(std::pow(2, L_size)));
                    for (int t = 0; t < I_L.size(); t++)
                    {
                        I_L[t] = 1;
                    }

                    cur = kron(cur, I_L);
                }

                if (H_size > 0)
                {
                    std::vector<double> I_H;
                    I_H.resize((int)(std::pow(2, H_size)));
                    for (int t = 0; t < I_H.size(); t++)
                    {
                        I_H[t] = 1;
                    }

                    cur = kron(I_H, cur);
                }

                tmp_value = dot(tmp_value, cur);
            }
            tmp_value = tmp_value * item.second;
        }

        value = value + tmp_value;
    }

    return value;
}

QPANDA_END

#include <cstring>
#include "Components/Operator/PauliOperator.h"

USING_QPANDA

std::vector<double> QPanda::kron(
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

std::vector<double> QPanda::dot(
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

std::vector<double> QPanda::operator +(
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

std::vector<double> QPanda::operator *(
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

std::vector<double> QPanda::transPauliOperatorToVec(PauliOperator pauli)
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



void QPanda::matrix_decompose_hamiltonian(QuantumMachine* qvm, EigenMatrixX& mat, PauliOperator& hamiltonian)
{
    PualiOperatorLinearCombination linear_result;
    matrix_decompose_paulis(qvm, mat, linear_result);

    PauliOperator::PauliMap pauli_map;
    for (auto item : linear_result)
    {
        double val = item.first;
        QCircuit cir = item.second;

        QCircuitToPauliOperator cir_to_opt(val);
        auto pauli_value = cir_to_opt.traversal(cir);

        pauli_map.insert(pauli_value);
    }

    hamiltonian = PauliOperator(pauli_map);
    return;
}

std::vector<complex_d> QPanda::transPauliOperatorToMatrix(const PauliOperator& opt)
{
    auto qubit_pool = OriginQubitPool::get_instance();
    qubit_pool->set_capacity(24);

    auto hamiltonian = opt.toHamiltonian();

    std::vector<QStat> matrix_vector;
    for (const auto &val : hamiltonian)
    {
        QCircuit cir;
        auto term = val.first;

        for (auto& iter : term)
        {
            auto qubit = qubit_pool->allocateQubitThroughPhyAddress(iter.first);
            switch (iter.second)
            {
            case 'X':
                cir << X(qubit);
                break;
            case 'Y':
                cir << Y(qubit);
                break;
            case 'Z':
                cir << Z(qubit);
                break;
            default:
                cir << I(qubit);
                break;
            }
        }

        auto matrix = getCircuitMatrix(cir);
        matrix_vector.emplace_back(matrix);
    }

    auto opt_matrix = matrix_vector.front();
    for (auto i = 1; i < matrix_vector.size(); ++i)
    {
        opt_matrix = (opt_matrix * matrix_vector[i]);
    }

    return opt_matrix;
}


PauliOperator QPanda::x(int index) { return PauliOperator("X" + std::to_string(index), 1); }
PauliOperator QPanda::y(int index) { return PauliOperator("Y" + std::to_string(index), 1); }
PauliOperator QPanda::z(int index) { return PauliOperator("Z" + std::to_string(index), 1); }
PauliOperator QPanda::i(int index) { return PauliOperator(1); }
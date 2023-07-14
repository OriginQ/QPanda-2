#include "Core/Utilities/Tools/Uinteger.h"
#include "Core/VirtualQuantumProcessor/Stabilizer/Clifford.h"

USING_QPANDA


void Clifford::show_tableau()
{
    std::cout << "---tableau and phases---" << std::endl;
    std::cout << std::endl;
    for (size_t i = 0; i < m_tableau.size(); i++)
    {
        for (const auto &val : m_tableau[i].X.get_data())
        {
            auto binary = integerToBinary(val, m_qubits_num);
            std::reverse(binary.begin(), binary.end());

            std::cout << binary;
            std::cout << " ";
        }

        std::cout << "| ";

        for (const auto &val : m_tableau[i].Z.get_data())
        {
            auto binary = integerToBinary(val, m_qubits_num);
            std::reverse(binary.begin(), binary.end());

            std::cout << binary;
            std::cout << " ";
        }

        std::cout << "| ";
        std::cout << m_phases[i] << std::endl;

        if (i == (m_qubits_num - 1))
            std::cout << "----------------" << std::endl;
    }

    std::cout << std::endl;
    return;
}

void Clifford::initialize(uint64_t qubits_num)
{
    //reset tableau and phases instead of allocate sapce
    if (m_qubits_num == qubits_num)
    {
        for (int64_t i = 0; i < static_cast<int64_t>(qubits_num); i++)
        {
            //reset tableaux destabilizers
            m_tableau[i].Z.reset();
            m_tableau[i].X.reset();
            m_tableau[i].X.set_val(1, i);

            //reset tableaux stabilizers
            m_tableau[i + qubits_num].X.reset();
            m_tableau[i + qubits_num].Z.reset();
            m_tableau[i + qubits_num].Z.set_val(1, i);
        }

        m_phases.assign(2 * qubits_num, 0);
    }
    else
    {
        m_tableau.clear();

        // initial state = all zeros
        m_qubits_num = qubits_num;

        // add tableaux destabilizers
        for (int64_t i = 0; i < static_cast<int64_t>(qubits_num); i++)
        {
            PauliGroup Pauli(qubits_num);
            Pauli.X.set_val(1, i);
            m_tableau.push_back(Pauli);
        }

        // add tableaux stabilizers
        for (int64_t i = 0; i < static_cast<int64_t>(qubits_num); i++)
        {
            PauliGroup Pauli(qubits_num);
            Pauli.Z.set_val(1, i);
            m_tableau.push_back(Pauli);
        }

        // Add phases
        m_phases.resize(2 * qubits_num, 0);
    }
   
    return;
}

void Clifford::initialize(const Clifford& clifford)
{
    m_qubits_num = clifford.m_qubits_num;
    m_tableau = clifford.m_tableau;
    m_phases = clifford.m_phases;

    return;
}

void Clifford::append_cx(const uint64_t ctr, const uint64_t tar) 
{
#pragma omp parallel for num_threads(omp_get_max_threads())
    for (int64_t i = 0; i < static_cast<int64_t>(2 * m_qubits_num); i++) 
    {
        m_phases[i] ^= m_tableau[i].X[ctr] && 
                       m_tableau[i].Z[tar] &&
                       m_tableau[i].X[tar] ^ m_tableau[i].Z[ctr] ^ 1;

        m_tableau[i].X.set_val(m_tableau[i].X[tar] ^ m_tableau[i].X[ctr], tar);
        m_tableau[i].Z.set_val(m_tableau[i].Z[tar] ^ m_tableau[i].Z[ctr], ctr);
    }
}

void Clifford::append_cy(const uint64_t control, const uint64_t target)
{
    // CY(0, 1) => H(1) + S(1) + CNOT(0, 1) + S(1)

    append_z(target);
    append_s(target);
    append_cx(control, target);
    append_s(target);

    return;
}

void Clifford::append_cz(const uint64_t control, const uint64_t target)
{
    // CZ(0, 1) => H(1) + CNOT(0, 1) + H(1)

    append_h(target);
    append_cx(control, target);
    append_h(target);

    return;
}

void Clifford::append_h(const uint64_t qubit) 
{
#pragma omp parallel for num_threads(omp_get_max_threads())
    for (int64_t i = 0; i < static_cast<int64_t>(2 * m_qubits_num); i++) 
    {
        m_phases[i] ^= m_tableau[i].X[qubit] && m_tableau[i].Z[qubit];

        // exchange X and Z
        bool temp = m_tableau[i].X[qubit];
        m_tableau[i].X.set_val(m_tableau[i].Z[qubit], qubit);
        m_tableau[i].Z.set_val(temp, qubit);
    }
}

void Clifford::append_s(const uint64_t qubit)
{
#pragma omp parallel for num_threads(omp_get_max_threads())
    for (int64_t i = 0; i < static_cast<int64_t>(2 * m_qubits_num); i++) 
    {
        m_phases[i] ^= m_tableau[i].X[qubit] && m_tableau[i].Z[qubit];

        m_tableau[i].Z.set_val(m_tableau[i].Z[qubit] ^ m_tableau[i].X[qubit], qubit);
    }
}

void Clifford::append_x(const uint64_t qubit)
{
#pragma omp parallel for num_threads(omp_get_max_threads())
    for (int64_t i = 0; i < static_cast<int64_t>(2 * m_qubits_num); i++)
        m_phases[i] ^= m_tableau[i].Z[qubit];
}

void Clifford::append_y(const uint64_t qubit) 
{
#pragma omp parallel for num_threads(omp_get_max_threads())
    for (int64_t i = 0; i < static_cast<int64_t>(2 * m_qubits_num); i++)
        m_phases[i] ^= (m_tableau[i].Z[qubit] ^ m_tableau[i].X[qubit]);
}

void Clifford::append_z(const uint64_t qubit) 
{
#pragma omp parallel for num_threads(omp_get_max_threads())
    for (int64_t i = 0; i < static_cast<int64_t>(2 * m_qubits_num); i++)
        m_phases[i] ^= m_tableau[i].X[qubit];
}

std::pair<bool, uint64_t> Clifford::z_anticommuting(const uint64_t qubit) const 
{
    for (uint64_t i = m_qubits_num; i < 2 * m_qubits_num; i++) 
    {
        if (m_tableau[i].X[qubit])
            return std::make_pair(true, i);
    }

    return std::make_pair(false, 0);
}

std::pair<bool, uint64_t> Clifford::x_anticommuting(const uint64_t qubit) const 
{
    for (uint64_t i = m_qubits_num; i < 2 * m_qubits_num; i++) 
    {
        if (m_tableau[i].Z[qubit])
            return std::make_pair(true, i);
    }

    return std::make_pair(false, 0);
}

bool Clifford::is_deterministic(const uint64_t& qubit)
{
    //measurement of qubit a in standard basis.
    //check whether there exists a p ∈ { n + 1, . . . , 2n }
    //such that x(p,a) in tableau = 1.
    return !z_anticommuting(qubit).first;
}

Qnum Clifford::measure_and_update(const Qnum qubits)
{
    const prob_vec discrete_probs = { 0.5, 0.5 };

    Qnum result;
    for (const auto &qubit : qubits) 
    {
        auto output = m_random.random_discrete(discrete_probs);
        result.push_back(measure_and_update(qubit, output));
    }

    return result;
}

prob_vec Clifford::pmeasure(const Qnum qubits)
{
    prob_vec probs(1ull << qubits.size(), 0.);

    std::function<void(std::string, double)> lambda = [&](std::string binary, double prob)
    {
        auto qubit_flag = -1;

        for (int i = 0; i < qubits.size(); ++i)
        {
            auto qubit = qubits[qubits.size() - i - 1];

            if (binary[i] == 'Q')
            {
                if (is_deterministic(qubit))
                {
                    if (measure_and_update(qubit, 0))
                        binary[i] = '1';
                    else
                        binary[i] = '0';
                }
                else
                {
                    qubit_flag = i;
                }
            }
        }

        if (qubit_flag == -1)
        {
            probs[std::stoull(binary, 0, 2)] = prob;
            return;
        }

        for (auto possible_out = 0; possible_out < 2; possible_out++)
        {
            std::string other_output = binary;

            if (possible_out)
                other_output[qubit_flag] = '1';
            else
                other_output[qubit_flag] = '0';

            Clifford clifford;
            clifford.initialize(*this);
            measure_and_update(qubits[qubits.size() - qubit_flag - 1], possible_out);
            lambda(other_output, 0.5 * prob);
            initialize(clifford);
        }
    };

    std::string output(qubits.size(), 'Q');
    lambda(output, 1.0);
    return probs;
}

bool Clifford::measure_and_update(const uint64_t qubit, const uint64_t random_output)
{
    auto anticom = z_anticommuting(qubit);
    if (anticom.first) 
    {
        bool outcome = (random_output == 1);
        auto row = anticom.second;
        for (uint64_t i = 0; i < 2 * m_qubits_num; i++) 
        {
            if ((m_tableau[i].X[qubit]) && (i != row) && (i != (row - m_qubits_num))) 
                tableau_row_sum(m_tableau[row], m_phases[row], m_tableau[i], m_phases[i]);
        }

        // Update state
        m_tableau[row - m_qubits_num].X = m_tableau[row].X;
        m_tableau[row - m_qubits_num].Z = m_tableau[row].Z;

        m_phases[row - m_qubits_num] = m_phases[row];

        m_tableau[row].X.reset();
        m_tableau[row].Z.reset();
        m_tableau[row].Z.set_val(1, qubit);
        m_phases[row] = outcome;
        return outcome;
    }
    else 
    {
        // is_deterministic measure output
        PauliGroup accum(m_qubits_num);
        int outcome = 0;
        for (uint64_t i = 0; i < m_qubits_num; i++) 
        {
            if (m_tableau[i].X[qubit]) 
            {
                tableau_row_sum(m_tableau[i + m_qubits_num], m_phases[i + m_qubits_num],
                    accum, outcome);
            }
        }
        return outcome;
    }
}

void Clifford::tableau_row_sum(const PauliGroup& row, const int row_phase, 
    PauliGroup &accum, int &accum_phase)
{
    int8_t newr = ((2 * row_phase + 2 * accum_phase) +
        PauliGroup::phase_exponent(row, accum)) % 4;
    // Since we are only using +1 and -1 phases in our Clifford phases
    // the exponent must be 0 (for +1) or 2 (for -1)
    if ((newr != 0) && (newr != 2)) 
    {
        throw std::runtime_error("Clifford: tableau_row_sum error");
    }
    accum_phase = (newr == 2);
    accum.X += row.X;
    accum.Z += row.Z;

    return;
}

  
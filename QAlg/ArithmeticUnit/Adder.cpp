#include "QAlg/ArithmeticUnit/Adder.h"
#include "QPanda.h"

QPANDA_BEGIN

ADDER_MODE operator|(ADDER_MODE lhs, ADDER_MODE rhs)
{
    return static_cast<ADDER_MODE>(
        static_cast<std::underlying_type_t<ADDER_MODE>>(lhs) |
        static_cast<std::underlying_type_t<ADDER_MODE>>(rhs));
}

ADDER_MODE operator&(ADDER_MODE lhs, ADDER_MODE rhs)
{
    return static_cast<ADDER_MODE>(
        static_cast<std::underlying_type_t<ADDER_MODE>>(lhs) &
        static_cast<std::underlying_type_t<ADDER_MODE>>(rhs));
}

QCircuit CDKMRippleAdder::QAdd(QVec a, QVec b, QVec aux, ADDER_MODE mode /* = ADDER_MODE::FULL */)
{
    if ((a.size() == 0) || a.size() != b.size())
    {
        QCERR_AND_THROW(std::invalid_argument, "a and b must be equal, but not equal to 0!");
    }
    QPANDA_ASSERT(aux.size() < 2, "auxiliary bit at least have size 2");
    QPANDA_ASSERT(!(ADDER_MODE::CIN & mode), "CDKMRippleAdder must have c_in");

    int n = a.size();
    /* c_in statu is unknown, but c_out must be |0> */
    Qubit *c_in = aux.front();
    Qubit *c_out = aux.back();
    Qubit *overflow = aux.back();

    QCircuit qc;

    qc << MAJ(c_in, b[0], a[0]);

    for (auto i = 1; i < n; i++)
    {
        /* xor the num carry_out and sign carry_out to determine overflow */
        if ((ADDER_MODE::OF & mode) && i == n - 1)
        {
            qc << CNOT(a[i - 1], overflow);
        }
        qc << MAJ(a[i - 1], b[i], a[i]);
    }
    if (ADDER_MODE::COUT & mode)
    {
        qc << CNOT(a[n - 1], c_out);
    }
    else if (ADDER_MODE::OF & mode)
    {
        qc << CNOT(a[n - 1], overflow);
    }

    for (auto i = n - 1; i > 0; i = i - 1)
    {
        qc << UMA(a[i - 1], b[i], a[i]);
    }

    qc << UMA(c_in, b[0], a[0]);

    return qc;
}

/**
 * @brief Quantum adder MAJ circute.
 *                ┌───┐     
 *  c = c_i: ─────┤ X ├──■── a_i ⊕ c_i
 *           ┌───┐└─┬─┘  │  
 *  b = b_i: ┤ X ├──┼────■── a_i ⊕ b_i = s_i
 *           └─┬─┘  │  ┌─┴─┐
 *  a = a_i: ──■────■──┤ X ├ (a_i ⊕ c_i).(a_i ⊕ b_i) ⊕ a_i = c_i+1
 *                     └───┘
 * 
 * @param c carry in at bit i
 * @param a adder bit at bit i
 * @param b adder bit at bit i
 * @return QCircuit 
 */
QCircuit CDKMRippleAdder::MAJ(Qubit *c, Qubit *b, Qubit *a)
{
    QCircuit circuit;
    circuit << CNOT(a, b) << CNOT(a, c) << X(a).control({c, b});
    return circuit;
}

/**
 * @brief 
 *                       ┌───┐     
 *  c = a_i ⊕ c_i: ──■──┤ X ├──■── c_i
 *                    │  └─┬─┘┌─┴─┐
 *  b = a_1 ⊕ b_i: ──■────┼──┤ X ├ a_1 ⊕ b_i ⊕ c_i = sum_i
 *                  ┌─┴─┐  │  └───┘
 *  a = c_i+1:      ┤ X ├──■─────── a_i
 *                  └───┘
 */
QCircuit CDKMRippleAdder::UMA(Qubit *c, Qubit *b, Qubit *a)
{
    QCircuit circuit;
    circuit << X(a).control({c, b}) << CNOT(a, c) << CNOT(c, b);
    return circuit;
}

QCircuit DraperQFTAdder::QAdd(QVec a, QVec b, QVec aux, ADDER_MODE mode)
{
    if ((a.size() == 0) || a.size() != b.size())
    {
        QCERR_AND_THROW(std::invalid_argument, "a and b must be equal, but not equal to 0!");
    }
    QPANDA_ASSERT(aux.size() < 2, "auxiliary bit at least have size 3");

    Qubit *c_in = aux.front();
    Qubit *c_out = aux.back();
    Qubit *overflow = aux.back();

    if (ADDER_MODE::COUT & mode)
    {
        b += c_out;
    }
    else if (ADDER_MODE::OF & mode)
    {
        b += overflow;
    }

    int b_n = b.size();
    int a_n = a.size();

    /* 
      see [T. G. Draper, Addition on a Quantum Computer, 2000] part 4, how to compute approximate qft threshold
      log2(n) is good enough
    */
    int aqft_threshold = std::floor(std::log2(b_n));

    QCircuit qc = CreateEmptyCircuit();
    qc << AQFT(b, aqft_threshold);

    for (int target_id = b_n - 1; target_id >= 0; target_id--)
    {
        /* align bits */
        int control_id = std::min(target_id, a_n - 1);
        /* use AQFT_THRESHOLD to achieve Approximate QFT */
        for (; control_id >= 0 && target_id - control_id < aqft_threshold; control_id--)
        {
            qc << CR(a[control_id], b[target_id], 2 * PI / (1 << (target_id - control_id + 1)));
            if (control_id == 0 && (mode & ADDER_MODE::CIN))
            {
                qc << CR(c_in, b[target_id], 2 * PI / (1 << (target_id - control_id + 1)));
            }
        }
    }

    qc << inverseAQFT(b, aqft_threshold);
    return qc;
}

QCircuit DraperQFTAdder::AQFT(QVec qvec, int aqft_threshold)
{
    QPANDA_ASSERT(qvec.size() < 1, "qvec is empty");

    int n = qvec.size();
    QCircuit qft = CreateEmptyCircuit();

    for (int target_id = n - 1; target_id >= 0; target_id--)
    {
        qft << H(qvec[target_id]);
        for (int control_id = target_id - 1; control_id >= 0 && target_id - control_id < aqft_threshold; control_id--)
        {
            qft << CR(qvec[control_id],
                      qvec[target_id], 2 * PI / (1 << (target_id - control_id + 1)));
        }
    }
    return qft;
}

QCircuit DraperQFTAdder::inverseAQFT(QVec qvec, int aqft_threshold)
{
    QPANDA_ASSERT(qvec.size() < 1, "qvec is empty");
    int n = qvec.size();

    QCircuit iqft = CreateEmptyCircuit();
    for (int target_id = 0; target_id < n; target_id++)
    {
        int control_id = std::max(target_id - aqft_threshold + 1, 0);
        for (; control_id <= target_id - 1 && target_id - control_id < aqft_threshold; control_id++)
        {
            iqft << CR(qvec[control_id],
                       qvec[target_id], 2 * PI / -(1 << (target_id - control_id + 1)));
        }
        iqft << H(qvec[target_id]);
    }
    return iqft;
}

QCircuit VBERippleAdder::QAdd(QVec a, QVec b, QVec aux, ADDER_MODE mode)
{
    QCircuit qc;
    int n = (int)b.size();
    QVec c(aux.begin(), aux.begin() + n);
    c.push_back(aux.back());
    QPANDA_ASSERT(n < 1, "qubits is empty");
    for (int i = 0; i < n; i++)
    {
        qc << carryModule(c[i], a[i], b[i], c[i + 1]);
    }
    /* 
      no carry out, slightly different with paper
      we skiped circuit follows:

      qc << CNOT(a[n - 1], b[n - 1]);
      qc << sumModule(c[n - 1], a[n - 1], b[n - 1]);
    */

    for (int i = n - 1; i >= 0; i--)
    {
        qc << carryModule(c[i], a[i], b[i], c[i + 1]).dagger()
           << sumModule(c[i], a[i], b[i]);
    }
    return qc;
}

/**
     * @brief 
     * c_i:   ────────────■── c_i
     *                    │
     * a_i:   ──■────■────┼── a_i
     *          │  ┌─┴─┐  │
     * b_i:   ──■──┤ X ├──■── a_i ⊕ b_i = s_i
     *        ┌─┴─┐└───┘┌─┴─┐
     * c_i+1: ┤ X ├─────┤ X ├ (a_i.b_i ⊕ c_i+1) ⊕ (c_i.s_i) 
     *        └───┘     └───┘
     * 
     * @param a 
     * @param b 
     * @param c_in 
     * @param help 
     * @return QCircuit 
     */
QCircuit VBERippleAdder::carryModule(Qubit *c_i, Qubit *a_i, Qubit *b_i, Qubit *c_i_1)
{
    QCircuit qc_carry;
    qc_carry << X(c_i_1).control({a_i, b_i})
             << CNOT(a_i, b_i)
             << X(c_i_1).control({c_i, b_i});
    return qc_carry;
}

/**
     * @brief 
     * c_i:   ──■──────────── c_i
     *          │            
     * a_i:   ──┼────■────■── a_i
     *          │  ┌─┴─┐  │  
     * b_i:   ──■──┤ X ├──■── a_i ⊕ b_i = s_i
     *        ┌─┴─┐└───┘┌─┴─┐
     * c_i+1: ┤ X ├─────┤ X ├ (b_i.c_i ⊕ c_i+1) ⊕ (a_i.s_i)
     *        └───┘     └───┘
     * 
     * @param a 
     * @param b 
     * @param c_in 
     * @param help 
     * @return QCircuit 
     */
// QCircuit VBERippleAdder::carryDaggerModule(Qubit *c_in, Qubit *a, Qubit *b, Qubit *c_out)
// {
//     QCircuit qc_carry_dagger;
//     qc_carry_dagger << X(c_out).control({c_in, b})
//                     << CNOT(a, b)
//                     << X(c_out).control({a, b});
//     return qc_carry_dagger;
// }

/**
     * @brief 
     * c_in: ───────■──
     *              │
     * a:    ──■────┼──
     *       ┌─┴─┐┌─┴─┐
     * b:    ┤ X ├┤ X ├ a_i ⊕ b_i ⊕ c_i = s_i
     *       └───┘└───┘
     * 
     * @return QCircuit 
     */
QCircuit VBERippleAdder::sumModule(Qubit *c_in, Qubit *a, Qubit *b)
{
    QCircuit qc_sum;
    qc_sum << CNOT(a, b)
           << CNOT(c_in, b);
    return qc_sum;
}

DraperQCLAAdder::Propagate::Propagate(QVec b, QVec aux)
{
    for (size_t i = 0; i < b.size(); i++)
    {
        m_propagate[i][i + 1] = b[i];
    }
    m_valide_aux_id = 0;
    m_aux = aux;
}

Qubit *DraperQCLAAdder::Propagate::operator()(int i, int j)
{
    if (!m_propagate.count(i) || !m_propagate.at(i).count(j))
    {
        m_propagate[i][j] = m_aux[m_valide_aux_id];
        m_valide_aux_id++;
    }
    return m_propagate.at(i).at(j);
}

/* QCLA out_of_place mode, may deprecate later */
#if 0
/**
 * @brief DraperQCLAdder out-of-place
 * 
 * @param a 
 * @param b 
 * @param aux 
 * @return QCircuit 
 */
QCircuit DraperQCLAAdder::QAdd(QVec a, QVec b, QVec aux)
{
    QPANDA_ASSERT(a.size() != b.size(), "bit size of a, b should equal");
    QPANDA_ASSERT(0 == b.size(), "bit size of b should bigger than 0");
    QPANDA_ASSERT(aux.size() < b.size(), "auxiliary bits size should bigger than a or b");
    int n = b.size();
    std::stringstream error_msg("auxiliary bits size should at leat 2 * n + 1 - std::floor(std::log2(n)) = ");
    error_msg << int(2 * n + 1 - std::floor(std::log2(n)));
    QPANDA_ASSERT(aux.size() < int(2 * n + 1 - std::floor(std::log2(n))), error_msg.str());
    // size z is n+1
    // size x is n - log n
    QVec z(aux.begin(), aux.begin() + n + 1);
    QVec x(aux.begin() + n + 1, aux.begin() + int(2 * n + 1 - std::floor(std::log2(n))));
    Propagate p(b, x);

    int max_id = b.size() - 1;
    QCircuit qc;
    // step 1
    // z[i+1] = g[i, i + 1]
    // b[i] = p[i, i + 1] for i > 0
    qc << X(z[1]).control({a[0], b[0]});
    for (int i = 1; i < n; i++)
    {
        qc << perasGate(a[i], b[i], z[i + 1]);
    }

    // step 2: P rounds
    //
    // P_t[m] = p[pow(2,t)*m, pow(2,t)*(m+1)]
    // G[m] = g[x,m]     x can be any value
    //
    // for 1 <= t <= floor(log(n)) - 1:
    //   for 1 <= m < floor(n/pow(2,t))):
    //     P_t[m] ⊕= P_t−1[2m]P_t−1[2m + 1]
    for (int t = 1; t <= std::floor(std::log2(n)) - 1; t++)
    {
        for (int m = 1; m < std::floor(n / std::pow(2.0, t)); m++)
        {
            auto p_t_m = p(pow(2, t) * m, pow(2, t) * (m + 1));
            auto p_t_1_2m = p(pow(2, t - 1) * 2 * m, pow(2, t - 1) * (2 * m + 1));
            auto p_t_1_2m_p1 = p(pow(2, t - 1) * (2 * m + 1), pow(2, t - 1) * (2 * m + 2));
            qc << Toffoli(p_t_1_2m, p_t_1_2m_p1, p_t_m);
        }
    }

    // step 2: G rounds
    // for  1 <= t <= floor(log n):
    //   for 0 <= m < floor(n/pow(2,t)):
    //     G[pow(2,t)*m + pow(2,t)] ⊕= G[pow(2,t)*m + pow(2,t−1)]P_t−1[2m + 1]
    for (int t = 1; t <= std::floor(std::log2(n)); t++)
    {
        for (int m = 0; m < std::floor(n / std::pow(2.0, t)); m++)
        {
            auto g_target = z[pow(2, t) * (m + 1)];
            auto g_control = z[pow(2, t - 1) * (2 * m + 1)];
            auto p_t_1 = p(pow(2, t - 1) * (2 * m + 1), pow(2, t - 1) * (2 * m + 2));
            qc << Toffoli(g_control, p_t_1, g_target);
        }
    }

    // step 2: C rounds
    // for floor(log (2n/3)) >= t >= 1:
    //   for 1 <= m <= floor((n-pow(2,t-1))/pow(2,t)):
    //     G[pow(2,t)*m + pow(2,t−1)] ⊕= G[pow(2,t)*m]P_t−1[2m]
    for (int t = floor(log2(2.0 * n / 3.0)); t >= 1; t--)
    {
        for (int m = 1; m <= std::floor((n - std::pow(2.0, t - 1)) / pow(2.0, t)); m++)
        {
            auto g_target = z[pow(2, t - 1) * (2 * m + 1)];
            auto g_control = z[pow(2, t) * m];
            auto p_t_1 = p(pow(2, t - 1) * (2 * m), pow(2, t - 1) * (2 * m + 1));
            qc << Toffoli(g_control, p_t_1, g_target);
        }
    }

    // step 2: reverse P rounds
    // for floor(log n) >= t >= 1:
    //   for 1 <= m < floor(n/pow(2,t)):
    //     P_t[m] ⊕= P_t−1[2m]P_t−1[2m + 1]
    for (int t = floor(log2(n)); t >= 1; t--)
    {
        for (int m = 1; m < std::floor(n / std::pow(2.0, t)); m++)
        {
            auto p_t_m = p(pow(2, t) * m, pow(2, t) * (m + 1));
            auto p_t_1_2m = p(pow(2, t - 1) * 2 * m, pow(2, t - 1) * (2 * m + 1));
            auto p_t_1_2m_p1 = p(pow(2, t - 1) * (2 * m + 1), pow(2, t - 1) * (2 * m + 2));
            qc << Toffoli(p_t_1_2m, p_t_1_2m_p1, p_t_m);
        }
    }

    // step 3:
    qc << CNOT(b[0], z[0]) << CNOT(a[0], z[0]);
    for (int i = 1; i < n; i++)
    {
        qc << CNOT(b[i], z[i]) << CNOT(a[i], b[i]);
    }

    return qc;
}
#endif

/**
 * @brief DraperQCLAdder in_place mode
 * 
 * @param a 
 * @param b 
 * @param aux 
 * @return QCircuit 
 */
QCircuit DraperQCLAAdder::QAdd(QVec a, QVec b, QVec aux, ADDER_MODE mode)
{
    QPANDA_ASSERT(a.size() != b.size(), "bit size of a, b should equal");
    QPANDA_ASSERT(0 == b.size(), "bit size of b should bigger than 0");
    QPANDA_ASSERT(aux.size() < b.size(), "auxiliary bits size should bigger than a or b");

    int n = b.size();
    std::stringstream error_msg("auxiliary bits size should at leat 2 * n + 1 - std::floor(std::log2(n)) = ");
    error_msg << int(2 * n - std::floor(std::log2(n)));
    QPANDA_ASSERT(aux.size() < int(2 * n - std::floor(std::log2(n))), error_msg.str());
    /* sizeof(z) is n, sizeof(x) is n - log n */
    QVec z(aux.begin(), aux.begin() + n);
    QVec x(aux.begin() + n, aux.begin() + int(2 * n - std::floor(std::log2(n))));
    Propagate p(b, x);

    QCircuit qc;
    /*
       z[i+1] = g[i, i + 1]
       b[i] = p[i, i + 1] for i > 0
    */

    /*
      step 1,2:
      for i in [0,n):
        z[i+1] = z[i+1] ⊕(a[i]*b[i])
      for i in [0,n):
        b[i] = b[i] ⊕ a[i] 

      no carry out, so iter i to n-1
    */
    for (int i = 0; i < n - 1; i++)
    {
        qc << perasGate(a.at(i), b.at(i), z.at(i + 1));
    }
    qc << CNOT(a.at(n - 1), b.at(n - 1));

    auto qc_carry = createEmptyCircuit();
    /*
      step 3:
      we skip the highest bit n-1
      
      P rounds
     
      P_t[m] = p[pow(2,t)*m, pow(2,t)*(m+1)]
      G[m] = g[x,m]     x can be any value
     
      for t in [1,floor(log(n)) - 1]:
        for m in [1, floor(n/pow(2,t))):
          P_t[m] ⊕= P_t−1[2m]P_t−1[2m + 1]
    */
    for (int t = 1; t <= std::floor(std::log2(n)) - 1; t++)
    {
        for (int m = 1; m < std::floor(n / std::pow(2.0, t)); m++)
        {
            /* skip the highest generate bit at p[n, x] */
            if (n - 1 == pow(2, t) * m ||
                n - 1 == pow(2, t - 1) * 2 * m ||
                n - 1 == pow(2, t - 1) * (2 * m + 1))
            {
                break;
            }
            auto p_t_m = p(pow(2, t) * m, pow(2, t) * (m + 1));
            auto p_t_1_2m = p(pow(2, t - 1) * 2 * m, pow(2, t - 1) * (2 * m + 1));
            auto p_t_1_2m_p1 = p(pow(2, t - 1) * (2 * m + 1), pow(2, t - 1) * (2 * m + 2));
            qc_carry << Toffoli(p_t_1_2m, p_t_1_2m_p1, p_t_m);
        }
    }

    /*
      step 3:
      G rounds
      for t in [1, floor(log n)]:
        for m in [0, floor(n/pow(2,t))):
          G[pow(2,t)*m + pow(2,t)] ⊕= G[pow(2,t)*m + pow(2,t−1)]P_t−1[2m + 1]
    */
    for (int t = 1; t <= std::floor(std::log2(n)); t++)
    {
        for (int m = 0; m < std::floor(n / std::pow(2.0, t)); m++)
        {
            if (n - 1 == pow(2, t) * (m + 1) - 1 ||
                n - 1 == pow(2, t - 1) * (2 * m + 1) - 1 ||
                n - 1 == pow(2, t - 1) * (2 * m + 1))
            {
                break;
            }
            auto g_target = z[pow(2, t) * (m + 1)];
            auto g_control = z[pow(2, t - 1) * (2 * m + 1)];
            auto p_t_1 = p(pow(2, t - 1) * (2 * m + 1), pow(2, t - 1) * (2 * m + 2));
            qc_carry << Toffoli(g_control, p_t_1, g_target);
        }
    }

    /*
      step 3: 
      C rounds
      for t in [floor(log (2n/3)), 1]:
        for m in [1, floor((n-pow(2,t-1))/pow(2,t))]:
          G[pow(2,t)*m + pow(2,t−1)] ⊕= G[pow(2,t)*m]P_t−1[2m]
    */
    for (int t = floor(log2(2.0 * n / 3.0)); t >= 1; t--)
    {
        for (int m = 1; m <= std::floor((n - std::pow(2.0, t - 1)) / pow(2.0, t)); m++)
        {
            if (n - 1 == pow(2, t - 1) * (2 * m + 1) - 1 ||
                n - 1 == pow(2, t) * m - 1 ||
                n - 1 == pow(2, t - 1) * (2 * m))
            {
                break;
            }
            auto g_target = z[pow(2, t - 1) * (2 * m + 1)];
            auto g_control = z[pow(2, t) * m];
            auto p_t_1 = p(pow(2, t - 1) * (2 * m), pow(2, t - 1) * (2 * m + 1));
            qc_carry << Toffoli(g_control, p_t_1, g_target);
        }
    }

    /*
      step 3: 
      reverse P rounds
      for t in [floor(log n), 1]:
        for m in [1, floor(n/pow(2,t))):
          P_t[m] ⊕= P_t−1[2m]P_t−1[2m + 1]
    */
    for (int t = floor(log2(n)); t >= 1; t--)
    {
        for (int m = 1; m < std::floor(n / std::pow(2.0, t)); m++)
        {
            if (n - 1 == pow(2, t) * m ||
                n - 1 == pow(2, t - 1) * 2 * m ||
                n - 1 == pow(2, t - 1) * (2 * m + 1))
            {
                break;
            }
            auto p_t_m = p(pow(2, t) * m, pow(2, t) * (m + 1));
            auto p_t_1_2m = p(pow(2, t - 1) * 2 * m, pow(2, t - 1) * (2 * m + 1));
            auto p_t_1_2m_p1 = p(pow(2, t - 1) * (2 * m + 1), pow(2, t - 1) * (2 * m + 2));
            qc_carry << Toffoli(p_t_1_2m, p_t_1_2m_p1, p_t_m);
        }
    }

    qc << qc_carry;

    /* 
      step 4:
      b[i] = b[i]⊕z[i] for i in [1, n)
    */
    for (int i = 1; i < n; i++)
    {
        qc << CNOT(z[i], b[i]);
    }
    /*
      step 5:
      flip b[i] for i in [0, n-1)
    */
    for (int i = 0; i < n - 1; i++)
    {
        qc << X(b[i]);
    }
    /*
      step 6:
      b[i] = b[i]⊕a[i] for i in [1, n-1)
    */
    for (int i = 1; i < n - 1; i++)
    {
        qc << CNOT(a[i], b[i]);
    }

    /* step 7: revers step 3 */
    qc << qc_carry.dagger();

    /*
      step 8:
      b[i] = b[i]⊕a[i] for i in [1, n-1)
    */
    for (int i = 1; i < n - 1; i++)
    {
        qc << CNOT(a[i], b[i]);
    }

    /*
      step 9:
      z[i+1] = z[i+1]⊕(a[i]*b[i]) for i in [0, n-1)
    */
    for (int i = 0; i < n - 1; i++)
    {
        qc << Toffoli(a[i], b[i], z[i + 1]);
    }

    /*
      step 10:
      flip b[i] for i in [0, n-1)
    */
    for (int i = 0; i < n - 1; i++)
    {
        qc << X(b[i]);
    }

    return qc;
}

/**
 * @brief 
 * a:    ──■────■── a
 *         │  ┌─┴─┐
 * b:    ──■──┤ X ├ a ⊕ b
 *       ┌─┴─┐└───┘
 * c:    ┤ X ├───── a.b ⊕ c
 *       └───┘
 */
QCircuit DraperQCLAAdder::perasGate(Qubit *a, Qubit *b, Qubit *c)
{
    QCircuit qc;
    qc << X(c).control({a, b})
       << CNOT(a, b);
    return qc;
}

QPANDA_END
#include "QAlg/ArithmeticUnit/Adder.h"
#include "QPanda.h"

QPANDA_BEGIN

ADDER_MODE operator|(ADDER_MODE lhs, ADDER_MODE rhs)
{
    auto bit_mode = static_cast<std::underlying_type_t<ADDER_MODE>>(lhs) |
                    static_cast<std::underlying_type_t<ADDER_MODE>>(rhs);
    /* must op bit on int, don't use ADDER_MODE::operator| combine mode here. otherwise will unstoppable recursively call itself */
    QPANDA_ASSERT((bit_mode & 0b00000110) == 0b00000110, "COUT mode and OF mode can't be both enabled");
    return static_cast<ADDER_MODE>(bit_mode);
}

ADDER_MODE operator&(ADDER_MODE lhs, ADDER_MODE rhs)
{
    return static_cast<ADDER_MODE>(
        static_cast<std::underlying_type_t<ADDER_MODE>>(lhs) &
        static_cast<std::underlying_type_t<ADDER_MODE>>(rhs));
}

/* ---------------------------- CDKMRippleAdder ---------------------------- */
size_t CDKMRippleAdder::auxBitSize(size_t s)
{
    return 3;
}

QCircuit CDKMRippleAdder::QAdd(QVec a, QVec b, QVec aux, ADDER_MODE mode /* = ADDER_MODE::CINOUT */)
{
    if ((a.size() == 0) || a.size() != b.size())
    {
        QCERR_AND_THROW(std::invalid_argument, "a and b must be equal, but not equal to 0!");
    }
    int need_aux_size = 2;
    if (!(mode & ADDER_MODE::CIN))
    {
        need_aux_size = 3;
    }

    QPANDA_ASSERT(aux.size() < need_aux_size, "auxiliary bit at least have size 3");
    int n = a.size();

    auto qc = createEmptyCircuit();

    /*
      qubits layout

      aux |0|0|   ....     |0| highest bit
          | | |  not usde  | |
           ^ ^              ^
        c_in ancil     c_out/overflow
    */
    Qubit *c_in = aux.front();
    Qubit *ancil = *(aux.begin() + 1);
    Qubit *c_out = aux.back();
    Qubit *overflow = aux.back();

    if (mode & ADDER_MODE::CIN)
    {
        qc << addWithCin(a, b, aux, mode);
    }
    else
    {
        /*
         * a_0   ──■────■── a_0
         *         │  ┌─┴─┐
         * b_0   ──■──┤ X ├ a_0 ⊕ b_0 = s_0
         *       ┌─┴─┐└───┘
         * anc   ┤ X ├───── anc ⊕ a_0.b_0 = c_1 (anc should be |0>)
         *       └───┘
         */
        qc << X(ancil).control({a[0], b[0]})
           << CNOT(a[0], b[0]);
        if (n > 1)
        {
            QVec a_1(a.begin() + 1, a.end());
            QVec b_1(b.begin() + 1, b.end());
            QVec new_aux;
            new_aux += ancil;
            new_aux += c_out;
            if (n == 2 && (ADDER_MODE::OF & mode))
            {
                qc << CNOT(ancil, overflow);
            }
            qc << addWithCin(a_1, b_1, new_aux, mode | ADDER_MODE::CIN);
        }
        else
        {
            if (ADDER_MODE::COUT & mode)
            {
                QPANDA_ASSERT(ADDER_MODE::OF & mode, "COUT mode and OF mode can't be both enabled");
                qc << CNOT(a[0], c_out);
            }
            else if (ADDER_MODE::OF & mode)
            {
                QCERR_AND_THROW(std::runtime_error, "1 bit width binary should not have sign, so can not have overflow");
            }
        }
        /*
         *
         *  a_0: ───────■─────── a_0
         *       ┌───┐  │  ┌───┐
         *  b_0: ┤ X ├──■──┤ X ├ a_0 ⊕ b_0 = s_0
         *       └───┘┌─┴─┐└───┘
         *  anc: ─────┤ X ├───── s_0'⊕a_0.b_0 = anc
         *            └───┘
         *
         */
        qc << X(b[0])
           << X(ancil).control({a[0], b[0]})
           << X(b[0]);
    }
    return qc;
}

QCircuit CDKMRippleAdder::addWithCin(QVec a, QVec b, QVec aux, ADDER_MODE mode)
{
    if ((a.size() == 0) || a.size() != b.size())
    {
        QCERR_AND_THROW(std::invalid_argument, "a and b must be equal, but not equal to 0!");
    }
    QPANDA_ASSERT(aux.size() < 2, "auxiliary bit at least have size 2");

    QPANDA_ASSERT(!(ADDER_MODE::CIN & mode), "Must work in ADDER_MODE::CIN mode");

    int n = a.size();

    Qubit *c_in = aux.front();
    Qubit *c_out = aux.back();
    Qubit *overflow = aux.back();
    /*
      qubits layout

      aux |0|    ....     |0| highest bit
          | |   not usde  | |
           ^               ^
       c_in/ancil     c_out/overflow
    */

    QCircuit qc;

    qc << MAJ(c_in, b[0], a[0]);

    for (auto i = 1; i < n; i++)
    {
        /* add the unsign num carry_out to overflow */
        if ((ADDER_MODE::OF & mode) && i == n - 1)
        {
            qc << CNOT(a[i - 1], overflow);
        }
        qc << MAJ(a[i - 1], b[i], a[i]);
    }
    if (ADDER_MODE::COUT & mode)
    {
        QPANDA_ASSERT(ADDER_MODE::OF & mode, "COUT mode and OF mode can't be both enabled");
        qc << CNOT(a[n - 1], c_out);
    }
    else if (ADDER_MODE::OF & mode)
    {
        /* add sign carry_out to overflow */
        QPANDA_ASSERT(ADDER_MODE::COUT & mode, "COUT mode and OF mode can't be both enabled");
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
 *  a = a_i: ──■────■──┤ X ├ (a_i ⊕ c_i).(s_i) ⊕ a_i = c_i+1
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

/* ---------------------------- CDKMRippleAdder ---------------------------- */

/* ---------------------------- DraperQFTAdder  ---------------------------- */
size_t DraperQFTAdder::auxBitSize(size_t s)
{
    return 2;
}

QCircuit DraperQFTAdder::QAdd(QVec a, QVec b, QVec aux, ADDER_MODE mode)
{
    if ((a.size() == 0) || a.size() != b.size())
    {
        QCERR_AND_THROW(std::invalid_argument, "a and b must be equal, but not equal to 0!");
    }
    QPANDA_ASSERT(aux.size() < 2, "auxiliary bit at least have size 2");

    /*
      qubits layout

      aux |0|    ...    |0| highest bit
          | | not usde  | |
           ^             ^
          c_in       c_out/overflow

      QVec c for inner carry catainer
    */

    int b_n = b.size();
    int a_n = a.size();
    Qubit *c_in = aux.front();
    Qubit *c_out = aux.back();
    Qubit *overflow = aux.back();

    /*
      see [T. G. Draper, Addition on a Quantum Computer, 2000] part 4, how to compute approximate qft threshold
      log2(n) is good enough
    */
    int aqft_threshold = (std::max)((int)std::floor(std::log2(b_n)), 6);

    auto qc = createEmptyCircuit();

    if (ADDER_MODE::COUT & mode)
    {
        QPANDA_ASSERT(ADDER_MODE::OF & mode, "COUT mode and OF mode can't be both enabled");
        b += c_out;
    }
    if (ADDER_MODE::OF & mode)
    {
        QPANDA_ASSERT(ADDER_MODE::COUT & mode, "COUT mode and OF mode can't be both enabled");
        qc << carryOverflow(a, b, overflow, aqft_threshold);
        if (mode & ADDER_MODE::CIN)
        {
            qc << CR(c_in, overflow, 2 * PI / (1 << (a_n + 1)));
            qc << CR(c_in, overflow, 2 * PI / (1 << a_n));
        }
    }

    /* do addition */
    qc << AQFT(b, aqft_threshold);

    for (int target_id = b_n - 1; target_id >= 0; target_id--)
    {
        /* align bits */
        int control_id = (std::min)(target_id, a_n - 1);
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
    /* addition finished */

    if (ADDER_MODE::OF & mode)
    {
        QPANDA_ASSERT(ADDER_MODE::COUT & mode, "COUT mode and OF mode can't be both enabled");
        qc << inverseOverflow(a, b, overflow, aqft_threshold);
    }

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
        /* wrap std::max with bracket for avoid use max macro in windef.h */
        int control_id = (std::max)(target_id - aqft_threshold + 1, 0);
        for (; control_id <= target_id - 1 && target_id - control_id < aqft_threshold; control_id++)
        {
            iqft << CR(qvec[control_id],
                       qvec[target_id], 2 * PI / -(1 << (target_id - control_id + 1)));
        }
        iqft << H(qvec[target_id]);
    }
    return iqft;
}

QCircuit DraperQFTAdder::carryOverflow(QVec a, QVec b, Qubit *overflow, int aqft_threshold)
{

    int b_n = b.size();
    int a_n = a.size();
    QPANDA_ASSERT(a_n + b_n == 0, "QVec a, b is empty");
    QPANDA_ASSERT(a_n != b_n, "QVec a, b size should be equal");

    QCircuit qc = CreateEmptyCircuit();

    qc << H(overflow);
    /* sign carry */
    for (int control_id = a_n - 1; control_id >= 0 && control_id < aqft_threshold; control_id--)
    {
        qc << CR(a[control_id], overflow, 2 * PI / (1 << (a_n - control_id + 1)));
        qc << CR(b[control_id], overflow, 2 * PI / (1 << (b_n - control_id + 1)));
    }

    /* usign carry */
    for (int control_id = a_n - 2; control_id >= 0 && control_id < aqft_threshold; control_id--)
    {
        qc << CR(a[control_id], overflow, 2 * PI / (1 << (a_n - control_id)));
        qc << CR(b[control_id], overflow, 2 * PI / (1 << (b_n - control_id)));
    }
    /*
      after operation
      overflow = 0.0an-1...a0 + 0.0bn-1...b0 + 0.0an-2...a0 + 0.0bn-2...b0;
    */

    return qc;
}

QCircuit DraperQFTAdder::inverseOverflow(QVec a, QVec b, Qubit *overflow, int aqft_threshold)
{
    /*
      WARNNING: assume sum is in b
    */
    int b_n = b.size();
    int a_n = a.size();
    QPANDA_ASSERT(a_n + b_n == 0, "QVec a, b is empty");
    QPANDA_ASSERT(a_n != b_n, "QVec a, b size should be equal");

    auto qc = createEmptyCircuit();

    /*
      after addition sum is in b, b_n= s_n
      we take s_n out of aux.back() to recover overflow bit
    */

    for (int i = 0; i < b_n; i++)
    {
        qc << CR(b[i], overflow, 2 * PI / -(1 << (b_n - i + 1)));
    }

    for (int i = 0; i < b_n - 1; i++)
    {
        qc << CR(b[i], overflow, 2 * PI / -(1 << (b_n - i)));
    }

    qc << H(overflow);

    return qc;
}

/* ---------------------------- DraperQFTAdder  ---------------------------- */

/* ---------------------------- VBERippleAdder  ---------------------------- */
size_t VBERippleAdder::auxBitSize(size_t s)
{
    return s + 1;
}

QCircuit VBERippleAdder::QAdd(QVec a, QVec b, QVec aux, ADDER_MODE mode)
{
    if ((a.size() == 0) || a.size() != b.size())
    {
        QCERR_AND_THROW(std::invalid_argument, "a and b must be equal, but not equal to 0!");
    }

    QCircuit qc;
    int n = (int)b.size();

    QPANDA_ASSERT(aux.size() < n + 1, "auxiliary bit at least have size sizeof(b)+1");

    QVec c(aux.begin(), aux.begin() + n);
    /*
      qubits layout

      aux |000 ...|   ....     |0| highest bit
          |   c   |  not usde  | |
           ^                    ^
         c_in              c_out/overflow

      QVec c for inner carry catainer
    */

    Qubit *c_in = c.front();
    Qubit *ancil = c.front();
    QPANDA_ASSERT(c_in != aux.front(), "internal config error, c_in should be lowest bit of aux");
    Qubit *c_out = aux.back();
    Qubit *overflow = aux.back();

    for (int i = 0; i < n - 1; i++)
    {
        if (!(mode & ADDER_MODE::CIN) && i == 0)
        {
            /*
             * a_0   ──■────■── a_0
             *         │  ┌─┴─┐
             * b_0   ──■──┤ X ├ a_0 ⊕ b_0 = s_0
             *       ┌─┴─┐└───┘
             * c_1   ┤ X ├───── c_0 ⊕ a_0.b_0 = c_1 (c_0 should be |0>)
             *       └───┘
             */
            qc << X(c[1]).control({a[0], b[0]})
               << CNOT(a[0], b[0]);
        }
        else
        {
            qc << carryModule(c[i], a[i], b[i], c[i + 1]);
        }
    }

    if (ADDER_MODE::COUT & mode)
    {
        QPANDA_ASSERT(ADDER_MODE::OF & mode, "COUT mode and OF mode can't be both enabled");
        qc << carryModule(c[n - 1], a[n - 1], b[n - 1], c_out);
        qc << CNOT(a[n - 1], b[n - 1]);
        qc << sumModule(c[n - 1], a[n - 1], b[n - 1]);
    }
    else if (ADDER_MODE::OF & mode)
    {
        QPANDA_ASSERT(ADDER_MODE::COUT & mode, "COUT mode and OF mode can't be both enabled");
        QPANDA_ASSERT(a.size() < 2, "to calculate overflow for signed binary, at least 2 bit width binary needed");
        qc << carryModule(c[n - 1], a[n - 1], b[n - 1], overflow);
        qc << CNOT(a[n - 1], b[n - 1]);
        qc << CNOT(c[n - 1], overflow);
        qc << sumModule(c[n - 1], a[n - 1], b[n - 1]);
    }

    for (int i = n - 2; i >= 0; i--)
    {
        if (!(mode & ADDER_MODE::CIN) && i == 0)
        {
            /*
             *
             *  a_0: ───────■─────── a_0
             *       ┌───┐  │  ┌───┐
             *  b_0: ┤ X ├──■──┤ X ├ a_0 ⊕ b_0 = s_0
             *       └───┘┌─┴─┐└───┘
             *  c_1: ─────┤ X ├───── 0
             *            └───┘
             *
             */
            qc << X(b[0])
               << X(c[1]).control({a[0], b[0]})
               << X(b[0]);
        }
        else
        {
            qc << carryModule(c[i], a[i], b[i], c[i + 1]).dagger()
               << sumModule(c[i], a[i], b[i]);
        }
    }
    return qc;
}

/**
 * @brief
 * c_i:   ────────────■── c_i
 *                    │
 * a_i:   ──■────■────┼── a_i
 *          │  ┌─┴─┐  │
 * b_i:   ──■──┤ X ├──■── a_i ⊕ b_i
 *        ┌─┴─┐└───┘┌─┴─┐
 * c_i+1: ┤ X ├─────┤ X ├ (a_i.b_i) ⊕ ((a_i ⊕ b_i ).c_i)
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
 * c_i:    ───────■── c_i
 *                │
 * a_i:    ──■────┼── a_i
 *         ┌─┴─┐┌─┴─┐
 * b_i:    ┤ X ├┤ X ├ a_i ⊕ b_i ⊕a_i⊕ c_i = s_i
 *         └───┘└───┘
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

/* ---------------------------- VBERippleAdder  ---------------------------- */

/* ---------------------------- DraperQCLAAdder ---------------------------- */

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

/* ---------------------------- VBERippleAdder  ---------------------------- */

/* ---------------------------- DraperQCLAAdder  ---------------------------- */

size_t DraperQCLAAdder::auxBitSize(size_t s)
{
    return int(2 * s + 1 - std::floor(std::log2(s)));
}

/**
 * @brief DraperQCLAdder in_place adder, result saved in b
 * out_of_place adder is not implemented, which cost almost half of in_place adder. It will save result in aux and keep a, b
 */
QCircuit DraperQCLAAdder::QAdd(QVec a, QVec b, QVec aux, ADDER_MODE mode)
{
    QPANDA_ASSERT(a.size() != b.size(), "bit size of a, b should equal");
    QPANDA_ASSERT(0 == b.size(), "bit size of b should bigger than 0");
    QPANDA_ASSERT(aux.size() < b.size(), "auxiliary bits size should bigger than a or b");

    int n = b.size();
    std::stringstream error_msg("auxiliary bits size should at leat 2 * n + 1 - std::floor(std::log2(n)) = ");
    error_msg << int(2 * n + 1 - std::floor(std::log2(n)));
    QPANDA_ASSERT(aux.size() < int(2 * n + 1 - std::floor(std::log2(n))), error_msg.str());
    /* sizeof(z) is n, sizeof(x) is n - log n */
    QVec z(aux.begin(), aux.begin() + n);
    QVec x(aux.begin() + n, aux.begin() + int(2 * n - std::floor(std::log2(n))));
    z += aux.back();
    /*
      qubits layout

      aux |000 ...|  |      ..0000|0| highest bit
          |    z  |x |  not usde  |z|
           ^                       ^
         c_in                   c_out/overflow

      z[0,n-1] = aux[0,n-1]
      x[0, x] = aux[n, x-n-1]
      z[n] = auz.back()

      QVec z for inner generate carry container
      QVec x associate QVec b is inner popagate carry container
    */

    Qubit *c_in = z[0];
    QPANDA_ASSERT(c_in != aux.front(), "internal config error, c_in should be lowest bit of aux");
    Qubit *c_out = z.back();
    QPANDA_ASSERT(c_out != aux.back(), "internal config error, c_out should be highest bit of aux");
    Qubit *overflow = z.back();
    QPANDA_ASSERT(overflow != aux.back(), "internal config error, overflow should be highest bit of aux");

    QPANDA_ASSERT(mode & ADDER_MODE::CIN, "DraperQCLAAdder can't support CIN mode now");

    /* Propagate class help pick out proper propagate carry bit */
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
        qc << perasGate(a[i], b[i], z[i + 1]);
    }
    if ((ADDER_MODE::COUT & mode || ADDER_MODE::OF & mode))
    {
        qc << perasGate(a[n - 1], b[n - 1], z[n]);
    }
    else
    {
        qc << CNOT(a[n - 1], b[n - 1]);
    }

    /* step 3 */
    bool skip = !((ADDER_MODE::COUT & mode) || (ADDER_MODE::OF & mode));
    qc << propagateModule(n, z, p, skip);

    /* after step 3, z[i] = c_i from i=1 */
    if (ADDER_MODE::OF & mode)
    {
        QPANDA_ASSERT(ADDER_MODE::COUT & mode, "COUT mode and OF mode can't be both enabled");
        qc << CNOT(z[n - 1], overflow);
    }

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
    qc << propagateModule(n, z, p, true).dagger();

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

QCircuit DraperQCLAAdder::propagateModule(int n, QVec z, Propagate &p, bool skip)
{
    auto qc_carry = createEmptyCircuit();
    /*
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
            if (skip & (n - 1 == pow(2, t) * m ||
                        n - 1 == pow(2, t - 1) * 2 * m ||
                        n - 1 == pow(2, t - 1) * (2 * m + 1)))
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
      G rounds
      for t in [1, floor(log n)]:
        for m in [0, floor(n/pow(2,t))):
          G[pow(2,t)*m + pow(2,t)] ⊕= G[pow(2,t)*m + pow(2,t−1)]P_t−1[2m + 1]
    */
    for (int t = 1; t <= std::floor(std::log2(n)); t++)
    {
        for (int m = 0; m < std::floor(n / std::pow(2.0, t)); m++)
        {
            if (skip & (n - 1 == pow(2, t) * (m + 1) - 1 ||
                        n - 1 == pow(2, t - 1) * (2 * m + 1) - 1 ||
                        n - 1 == pow(2, t - 1) * (2 * m + 1)))
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
      C rounds
      for t in [floor(log (2n/3)), 1]:
        for m in [1, floor((n-pow(2,t-1))/pow(2,t))]:
          G[pow(2,t)*m + pow(2,t−1)] ⊕= G[pow(2,t)*m]P_t−1[2m]
    */
    for (int t = floor(log2(2.0 * n / 3.0)); t >= 1; t--)
    {
        for (int m = 1; m <= std::floor((n - std::pow(2.0, t - 1)) / pow(2.0, t)); m++)
        {
            if (skip & (n - 1 == pow(2, t - 1) * (2 * m + 1) - 1 ||
                        n - 1 == pow(2, t) * m - 1 ||
                        n - 1 == pow(2, t - 1) * (2 * m)))
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
      reverse P rounds
      for t in [floor(log n), 1]:
        for m in [1, floor(n/pow(2,t))):
          P_t[m] ⊕= P_t−1[2m]P_t−1[2m + 1]
    */
    for (int t = floor(log2(n)); t >= 1; t--)
    {
        for (int m = 1; m < std::floor(n / std::pow(2.0, t)); m++)
        {
            if (skip & (n - 1 == pow(2, t) * m ||
                        n - 1 == pow(2, t - 1) * 2 * m ||
                        n - 1 == pow(2, t - 1) * (2 * m + 1)))
            {
                break;
            }
            auto p_t_m = p(pow(2, t) * m, pow(2, t) * (m + 1));
            auto p_t_1_2m = p(pow(2, t - 1) * 2 * m, pow(2, t - 1) * (2 * m + 1));
            auto p_t_1_2m_p1 = p(pow(2, t - 1) * (2 * m + 1), pow(2, t - 1) * (2 * m + 2));
            qc_carry << Toffoli(p_t_1_2m, p_t_1_2m_p1, p_t_m);
        }
    }

    return qc_carry;
}
/* ---------------------------- DraperQCLAAdder ---------------------------- */

QPANDA_END
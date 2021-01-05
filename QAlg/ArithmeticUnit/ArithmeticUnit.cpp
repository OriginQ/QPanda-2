#include "QAlg/ArithmeticUnit/ArithmeticUnit.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/Utilities/Tools/Utils.h"


QPANDA_BEGIN

QCircuit MAJ(Qubit* a, Qubit* b, Qubit* c)
{
    QCircuit circuit;
    circuit << CNOT(c, b) << CNOT(c, a) << X(c).control({ a, b });

    return circuit;
}

QCircuit UMA(Qubit* a, Qubit* b, Qubit* c)
{
    QCircuit circuit;
    circuit << X(c).control({ a, b }) << CNOT(c, a) << CNOT(a, b);

    return circuit;
}

QCircuit MAJ2(QVec &adder1, QVec &adder2, Qubit* c)
{
    if ((adder1.size() == 0) || (adder1.size() != adder2.size()))
    {
        QCERR("adder1 and adder2 must be equal, but not equal to 0!");
        throw ("adder1 and adder2 must be equal, but not equal to 0!");
    }

    int nbit = adder1.size();
    QCircuit circuit;

    circuit << MAJ(c, adder1[0], adder2[0]);

    for (auto i = 1; i < nbit; i++)
    {
        circuit << MAJ(adder2[i - 1], adder1[i], adder2[i]);
    }

    return circuit;

}

QCircuit isCarry(
    QVec &adder1,
    QVec &adder2,
    Qubit* c,
    Qubit* is_carry)
{
    if ((adder1.size() == 0) || (adder1.size() != adder2.size()))
    {
        QCERR("adder1 and adder2 must be equal, but not equal to 0!");
        throw ("adder1 and adder2 must be equal, but not equal to 0!");
    }
    QCircuit circuit;

    circuit << MAJ2(adder1, adder2, c) << CNOT(adder2[adder2.size() - 1], is_carry) << MAJ2(adder1, adder2, c).dagger();

    return circuit;
}

QCircuit QAdder(
    QVec &adder1,
    QVec &adder2,
    Qubit* c,
    Qubit* is_carry)
{
    if ((adder1.size() == 0) || adder1.size() != adder2.size())
    {
        QCERR("adder1 and adder2 must be equal, but not equal to 0!");
        throw run_fail("adder1 and adder2 must be equal, but not equal to 0!");
    }

    int nbit = adder1.size();

    QCircuit circuit;
    circuit << MAJ(c, adder1[0], adder2[0]);

    for (auto i = 1; i < nbit; i++)
    {
        circuit << MAJ(adder2[i - 1], adder1[i], adder2[i]);
    }

    circuit << CNOT(adder2[adder2.size() - 1], is_carry);

    for (auto i = nbit - 1; i > 0; i = i - 1)
    {
        circuit << UMA(adder2[i - 1], adder1[i], adder2[i]);
    }

    circuit << UMA(c, adder1[0], adder2[0]);

    return circuit;

}

QCircuit QAdder(
    QVec &adder1,
    QVec &adder2,
    Qubit* c)
{
    if ((adder1.size() == 0) || (adder1.size() != adder2.size()))
    {
        QCERR("adder1 and adder2 must be equal, but not equal to 0!");
        throw ("adder1 and adder2 must be equal, but not equal to 0!");
    }
    int nbit = adder1.size();
    QCircuit circuit;
    circuit << MAJ(c, adder1[0], adder2[0]);

    for (auto i = 1; i < nbit; i++)
    {
        circuit << MAJ(adder2[i - 1], adder1[i], adder2[i]);
    }

    for (auto i = nbit - 1; i > 0; i = i - 1)
    {
        circuit << UMA(adder2[i - 1], adder1[i], adder2[i]);
    }

    circuit << UMA(c, adder1[0], adder2[0]);

    return circuit;
}

QCircuit QAdd(QVec& a, QVec& b, QVec& k)
{
    auto len = a.size();
    QCircuit circ;
    circ << X(b[len - 1]);
    circ << QSub(a, b, k);
    circ << X(b[len - 1]);
    return circ;
}

QCircuit QComplement(QVec& a, QVec& k)
{
    if (k.size() < a.size() + 2)
    {
        QCERR_AND_THROW_ERRSTR(
            run_fail,
            "Auxiliary qubits is not big enough!");
    }

    int len = a.size();
    auto t = k[len];
    auto q1 = k[len + 1];

    QCircuit circ, circ1;
    for (int i = 0; i < len - 1; i++)
        circ1 << X(a[i]);
    QVec b(k.begin(), k.begin() + len);
    circ1 << X(b[0]);
    circ1 << QAdder(a, b, t);
    circ1 << X(b[0]);

    circ << CNOT(a[len - 1], q1);
    circ << circ1.control(q1);
    circ << CNOT(a[len - 1], q1);

    return circ;
}

QCircuit QSub(QVec& a, QVec& b, QVec& k)
{
    auto len = a.size();
    QVec anc(k.begin(), k.begin() + len + 2);
    auto t = k[len];
    //auto q1 = k[len + 1];
    QCircuit circ, circ1, circ2;

    circ << X(b[len - 1])
        << QComplement(a, anc)
        << QComplement(b, anc)
        << QAdder(a, b, t)
        << QComplement(a, anc)
        << QComplement(b, anc)
        << X(b[len - 1]);

    return circ;
}

/**
* @brief Shift the quantum state one bit to the left
* @ingroup ArithmeticUnit
* @param[in] a  qubits
* @return QCircuit
* @note The result of shift is saved in a.
*/
QCircuit shift(QVec& a)
{
    QCircuit circ;
    auto len = a.size();
    for (auto i = len - 1; i > 0; i--)
    {
        circ << SWAP(a[i], a[i - 1]);
    }
    return circ;
}

QCircuit QMultiplier(QVec& a, QVec& b, QVec& k, QVec& d)
{
    auto len = a.size();
    QVec c(a);
    QVec tem(k.begin(), k.begin() + len);
    c += tem;
    auto t = k[len];
    QCircuit fcirc;

    QCircuit circ;
    circ << QAdder(d, c, t);
    fcirc << circ.control(b[0]);

    for (auto i = 1; i < len; i++)
    {
        QCircuit circ1;
        fcirc << shift(c);
        circ1 << QAdder(d, c, t);
        fcirc << circ1.control(b[i]);
    }
    for (auto i = 1; i < len; i++)
    {
        fcirc << shift(c).dagger();
    }
    return fcirc;
}

QCircuit QMul(QVec& a, QVec& b, QVec& k, QVec& d)
{
    QVec aa(a.begin(), a.end() - 1);
    QVec bb(b.begin(), b.end() - 1);
    QVec dd(d.begin(), d.end() - 1);
    auto len = a.size();
    QCircuit circ, circ1;
    circ1 << X(d[len * 2 - 2]);
    circ << CNOT(a[len - 1], b[len - 1]);
    circ << circ1.control(b[len - 1]);
    circ << CNOT(a[len - 1], b[len - 1]);
    circ << QMultiplier(aa, bb, k, dd);
    return circ;
}

QProg QDivider(QVec& a, QVec& b, QVec& c, QVec& k, ClassicalCondition& t)
{
    auto len = a.size();
    QVec d(k.begin(), k.begin() + len);
    QVec e(k.begin() + len, k.begin() + len * 2 + 2);
    QProg prog;
    prog << X(c[0]) << X(c[len - 1]) << X(d[0]) << X(d[len - 1]);
    QProg prog_in;
    prog_in << QSub(a, b, e) << QSub(c, d, e) << Measure(a[len - 1], t);
    auto qwhile = createWhileProg(t < 1, prog_in);
    prog << qwhile 
        << X(b[len - 1]) 
        << QSub(a, b, e) 
        << X(b[len - 1]) 
        << X(d[0]) 
        << X(d[len - 1]);
    return prog;
}

QProg QDiv(QVec& a, QVec& b, QVec& c, QVec& k, ClassicalCondition& t)
{
    auto len = a.size();
    QProg prog;
    prog << CNOT(a[len - 1], k[len * 2 + 2])
        << CNOT(b[len - 1], k[len * 2 + 3]);
    prog << CNOT(k[len * 2 + 2], a[len - 1])
        << CNOT(k[len * 2 + 3], b[len - 1]);
    prog << QDivider(a, b, c, k, t);
    QCircuit circ;
    circ << X(c[len - 1]);
    prog << CNOT(k[len * 2 + 2], k[len * 2 + 3]);
    prog << circ.control(k[len * 2 + 3]);
    prog << CNOT(k[len * 2 + 2], k[len * 2 + 3]);
    prog << CNOT(k[len * 2 + 2], a[len - 1])
        << CNOT(k[len * 2 + 3], b[len - 1]);
    prog << CNOT(a[len - 1], k[len * 2 + 2])
        << CNOT(b[len - 1], k[len * 2 + 3]);
    return prog;
}

QProg QDivider(
    QVec& a, 
    QVec& b, 
    QVec& c, 
    QVec& k, 
    QVec& f, 
    std::vector<ClassicalCondition>& s)
{
    auto len = a.size(); auto cnt = f.size();
    QVec d(k.begin(), k.begin() + len);
    QVec cc(k.begin() + len, k.begin() + len * 2);
    QVec e(k.begin() + len * 2, k.begin() + len * 3 + 2);
    QVec ee(k.begin() + len * 2, k.begin() + len * 3 + 3);
    QVec aa(a);
    QVec bb(b);
    aa.push_back(k[len * 3 + 3]);
    bb.push_back(k[len * 3 + 4]);
    QProg prog;
    prog << X(c[0]) << X(c[len - 1]) << X(d[0]) << X(d[len - 1]);
    auto& t = s[cnt];
    auto& sum = s[cnt + 1];
    t.set_val(0);
    sum.set_val(0);
    QProg prog_in;
    prog_in << QSub(aa, bb, ee) 
        << QSub(c, d, e) 
        << (sum = sum + 1) 
        << Measure(aa[len], t);
    auto qwhile = createWhileProg(t < 1, prog_in);
    prog << qwhile;

    for (auto i = 0; i < cnt; i++)
    {
        s[i].set_val(0);
        prog << X(bb[len]) << X(cc[0]) << X(cc[len - 1]) << (t = 0);
        prog << QSub(aa, bb, ee);
        prog << X(bb[len]);
        prog << shift(aa);
        QProg prog_t;
        prog_t << QSub(aa, bb, ee) 
            << QSub(cc, d, e) 
            << (s[i] = s[i] + 1) 
            << Measure(aa[len], t);
        auto qwhile_t = createWhileProg(t < 1, prog_t);
        prog << qwhile_t;
        prog << SWAP(cc[0], f[cnt - i - 1]);
    }
    for (auto i = 0; i < cnt; i++)
    {
        prog << X(bb[len]);
        QProg prog_ttt;
        prog_ttt << QSub(aa, bb, ee) << (s[cnt - i - 1] = s[cnt - i - 1] - 1);
        auto qwhile_ttt = createWhileProg(s[cnt - i - 1] > 0, prog_ttt);
        prog << qwhile_ttt;
        prog << shift(aa).dagger();
        prog << X(bb[len]);
        prog << QSub(aa, bb, ee);

    }
    prog << X(bb[len]);
    QProg prog_tt;
    prog_tt << QSub(aa, bb, ee) << (sum = sum - 1);
    auto qwhile_tt = createWhileProg(sum > 0, prog_tt);
    prog << qwhile_tt;
    prog << X(bb[len]);
    prog << X(d[0]) << X(d[len - 1]) << (t = 0);
    return prog;
}

QProg QDiv(
    QVec& a, 
    QVec& b, 
    QVec& c, 
    QVec& k, 
    QVec& f, 
    std::vector<ClassicalCondition>& s)
{
    auto len = a.size();
    QProg prog;
    prog << CNOT(a[len - 1], k[len * 3 + 5])
        << CNOT(b[len - 1], k[len * 3 + 6]);
    prog << CNOT(k[len * 3 + 5], a[len - 1])
        << CNOT(k[len * 3 + 6], b[len - 1]);
    prog << QDivider(a, b, c, k, f, s);
    QCircuit circ;
    circ << X(c[len - 1]);
    prog << CNOT(k[len * 3 + 5], k[len * 3 + 6]);
    prog << circ.control(k[len * 3 + 6]);
    prog << CNOT(k[len * 3 + 5], k[len * 3 + 6]);
    prog << CNOT(k[len * 3 + 5], a[len - 1])
        << CNOT(k[len * 3 + 6], b[len - 1]);
    prog << CNOT(a[len - 1], k[len * 3 + 5])
        << CNOT(b[len - 1], k[len * 3 + 6]);
    return prog;
}

QCircuit bind_data(int value, QVec& qvec)
{
    bool sign_flag = value < 0 ? true,value=-value : false;
    size_t qnum = std::floor(std::log(value) / std::log(2)+1);
    if (qvec.size() < qnum+1)
    {
        QCERR_AND_THROW_ERRSTR(
            run_fail,
            "Qubit register is not big enough to store data!");
    }

    QCircuit circuit;
    int cnt = 0;
    while (value)
    {
        auto v = value % 2;
        if (v == 1)
        {
            circuit << X(qvec[cnt]);
        }

        value = value / 2;
        cnt++;
    }

    if (sign_flag)
    {
        circuit << X(qvec[qvec.size() - 1]);
    }

    return circuit;
}

QCircuit bind_nonnegative_data(size_t value, QVec& qvec)
{
    size_t qnum = std::floor(std::log(value) / std::log(2) + 1);
    if (qvec.size() < qnum)
    {
        QCERR_AND_THROW_ERRSTR(
            run_fail,
            "Qubit register is not big enough to store data!");
    }

    QCircuit circuit;
    int cnt = 0;
    while (value)
    {
        auto v = value % 2;
        if (v == 1)
        {
            circuit << X(qvec[cnt]);
        }

        value = value / 2;
        cnt++;
    }

    return circuit;
}

QCircuit constModAdd(QVec &qvec, int base, int module_Num, QVec &qvec1, QVec &qvec2)
{
    base = base % module_Num;
    QCircuit circuit, tmpcir, tmpcir1;
    int tmpvalue = (1 << qvec.size()) + base - module_Num;
    circuit << bind_nonnegative_data(tmpvalue, qvec1) 
        << isCarry(qvec, qvec1, qvec2[1], qvec2[0]) 
        << bind_nonnegative_data(tmpvalue, qvec1);

    tmpcir << bind_nonnegative_data(tmpvalue, qvec1) 
        << QAdder(qvec, qvec1, qvec2[1]) 
        << bind_nonnegative_data(tmpvalue, qvec1);
    circuit << tmpcir.control(qvec2[0]) << X(qvec2[0]);

    tmpcir1 << bind_nonnegative_data(base, qvec1) 
        << QAdder(qvec, qvec1, qvec2[1])
        << bind_nonnegative_data(base, qvec1);
    circuit << tmpcir1.control(qvec2[0]) << X(qvec2[0]);

    tmpvalue = (1 << qvec.size()) - base;
    circuit << bind_nonnegative_data(tmpvalue, qvec1)
        << isCarry(qvec, qvec1, qvec2[1], qvec2[0]) 
        << bind_nonnegative_data(tmpvalue, qvec1)
        << X(qvec2[0]);

    return circuit;
}

// Euclidean Algorithm to get the inverse element
int modReverse(int a, int b)
{
    a = abs(a);
    b = abs(b);
    // r_{-2},r_{-1}
    int r1 = a, r2 = b;
    // s_{-2},s_{-1}
    int s11 = 1, s12 = 0;
    // t_{-2},t_{-1}
    int t21 = 0, t22 = 1;

    // q_j
    int q = r1 / r2;

    int tempS = s12, tempT = t22, tempR = r1;
    r1 = r2;
    r2 = -q * r2 + tempR;

    while (r2 != 0)
    {
        tempS = s12;
        tempT = t22;
        s12 = (-q) * s12 + s11;
        t22 = (-q) * t22 + t21;
        s11 = tempS;
        t21 = tempT;

        q = r1 / r2;
        tempR = r1;
        r1 = r2;
        r2 = -q * r2 + tempR;
    }

    if (r1 == 1)
    {
        return s12 > 0 ? s12 : s12 + b;
    }
    else
    {
        return -1;
    }
}

QCircuit constModMul(QVec &qvec, int base, int module_Num, QVec &qvec1, QVec &qvec2, QVec &qvec3)
{
    QCircuit  circuit, tmpcir, tmpcir1;
    int tmpvalue, qsize = qvec.size();

    for (int i = 0; i < qsize; i++)
    {
        tmpvalue = (base * (1 << i)) % module_Num;
        circuit << constModAdd(qvec1, tmpvalue, module_Num, qvec2, qvec3).control(qvec[i]);
    }

    for (int i = 0; i < qsize; i++)
    {
        circuit << CNOT(qvec[i], qvec1[i]) << CNOT(qvec1[i], qvec[i]) << CNOT(qvec[i], qvec1[i]);
    }

    int Crev = modReverse(base, module_Num);

    for (int i = 0; i < qsize; i++)
    {
        tmpvalue = (Crev * (1 << i)) % module_Num;
        tmpcir1 << constModAdd(qvec1, tmpvalue, module_Num, qvec2, qvec3).control(qvec[i]);
    }

    circuit << tmpcir1.dagger();

    return circuit;
}

QCircuit constModExp(QVec &qvec, QVec &result, int base, int module_Num, QVec &qvec1, QVec &qvec2, QVec &qvec3)
{
    QCircuit  circuit, tmpcir;
    int tmp = base;
    for (int i = 0; i < qvec.size(); i++)
    {
        circuit << constModMul(result, tmp, module_Num, qvec1, qvec2, qvec3).control(qvec[qvec.size() - 1 - i]);
        tmp = (tmp * tmp) % module_Num;
    }
    return circuit;
}

QPANDA_END

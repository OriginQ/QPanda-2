#include "QAlg/ArithmeticUnit/ArithmeticUnit.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/Utilities/Tools/Utils.h"


USING_QPANDA

QCircuit QPanda::MAJ(Qubit* a, Qubit* b, Qubit* c)
{
    QCircuit circuit;
    circuit << CNOT(c, b) << CNOT(c, a) << X(c).control({ a, b });

    return circuit;
}

QCircuit QPanda::UMA(Qubit* a, Qubit* b, Qubit* c)
{
    QCircuit circuit;
    circuit << X(c).control({ a, b }) << CNOT(c, a) << CNOT(a, b);

    return circuit;
}

QCircuit QPanda::MAJ2(QVec &adder1, QVec &adder2, Qubit* c)
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

QCircuit QPanda::isCarry(
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

QCircuit QPanda::QAdder(
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

QCircuit QPanda::QAdderIgnoreCarry(
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

QCircuit QPanda::BindData(QVec &qvec, int cvec)
{
    QCircuit circuit;
    if (1 << qvec.size() < cvec)
    {
        QCERR("bind data with larger qubit!");
        throw run_fail("qubit register is not big enough to store data!");
    }

    int flag = 0;
    for (int i = 0; i < qvec.size() &&cvec >= 1; i++)
    {
        if (cvec % 2 == 1)
        {
            circuit << X(qvec[i]);
        }
        cvec = cvec >> 1;
    }
    return circuit;
}

QCircuit QPanda::constModAdd(QVec &qvec, int base, int module_Num, QVec &qvec1, QVec &qvec2)
{
    base = base % module_Num;
    QCircuit circuit, tmpcir, tmpcir1;
    int tmpvalue = (1 << qvec.size()) + base - module_Num;
    circuit << BindData(qvec1, tmpvalue) << isCarry(qvec, qvec1, qvec2[1], qvec2[0]) << BindData(qvec1, tmpvalue);

    tmpcir << BindData(qvec1, tmpvalue) << QAdderIgnoreCarry(qvec, qvec1, qvec2[1]) << BindData(qvec1, tmpvalue);
    circuit << tmpcir.control(qvec2[0]) << X(qvec2[0]);

    tmpcir1 << BindData(qvec1, base) << QAdderIgnoreCarry(qvec, qvec1, qvec2[1]) << BindData(qvec1, base);
    circuit << tmpcir1.control(qvec2[0]) << X(qvec2[0]);

    tmpvalue = (1 << qvec.size()) - base;
    circuit << BindData(qvec1, tmpvalue) << isCarry(qvec, qvec1, qvec2[1], qvec2[0]) << BindData(qvec1, tmpvalue) << X(qvec2[0]);

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

QCircuit QPanda::constModMul(QVec &qvec, int base, int module_Num, QVec &qvec1, QVec &qvec2, QVec &qvec3)
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

QCircuit QPanda::constModExp(QVec &qvec, QVec &result, int base, int module_Num, QVec &qvec1, QVec &qvec2, QVec &qvec3)
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
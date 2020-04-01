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

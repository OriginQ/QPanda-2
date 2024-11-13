#pragma once

#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"

QPANDA_BEGIN

class BenchmarkSingleGate
{
public:
    virtual QGate gate(Qubit* qubit) = 0;
    virtual QStat unitary() = 0;
    virtual ~BenchmarkSingleGate() {}
};

class BenchmarkDoubleGate
{
public:
    virtual QGate gate(Qubit* ctr_qubit, Qubit* tar_qubit) = 0;
    virtual QStat unitary() = 0;
    virtual ~BenchmarkDoubleGate() {}
};




class YGate : public BenchmarkSingleGate
{
public:
    YGate() {}
    QGate gate(Qubit* qubit) { return Y(qubit); }
    QStat unitary()
    {
        return{ 0, qcomplex_t(0, -1), qcomplex_t(0, 1), 0 };
    }
};

class XGate : public BenchmarkSingleGate
{
public:
    XGate() {}
    QGate gate(Qubit* qubit) { return X(qubit); }
    QStat unitary()
    {
        return { 0, 1, 1, 0 };
    }
};

class XPowGate : public BenchmarkSingleGate
{
public:
    XPowGate(double phi) { m_angle = PI * phi; }
    QGate gate(Qubit* qubit) { return RX(qubit, m_angle); }

    QStat unitary()
    {
        QStat matrix = { std::cos(m_angle / 2), qcomplex_t(0, -1 * std::sin(m_angle / 2)),
                         qcomplex_t(0, -1 * std::sin(m_angle / 2)), std::cos(m_angle / 2) };
        return matrix;
    }

private:
    double m_angle;
};

class YPowGate : public BenchmarkSingleGate
{
public:
    YPowGate(double phi) { m_angle = PI * phi; }

    QGate gate(Qubit* qubit) { return RY(qubit, m_angle); }
    QStat unitary()
    {
        QStat matrix = { std::cos(m_angle / 2), -std::sin(m_angle / 2),
            std::sin(m_angle / 2), std::cos(m_angle / 2) };
        return matrix;
    }
private:
    double m_angle;
};

//A gate that applies a phase to the |11⟩ state of two qubits.
class CZPowGate : public BenchmarkDoubleGate
{
public:
    CZPowGate(double theta) { m_theta = PI * theta; }

    QGate gate(Qubit* ctr_qubit, Qubit* tar_qubit) 
    { 
        return CR(ctr_qubit, tar_qubit, m_theta);
    }

    QStat unitary()
    {
        QStat matrix = { 1,  0,  0,  0,
                         0,  1,  0,  0,
                         0,  0,  1,  0,
                         0,  0,  0,  std::exp(qcomplex_t(0,1)* m_theta) };
        return matrix;
    }

private:
    double m_theta;
};


/*
    Contains all two qubit interactions that preserve excitations, 
    up to single qubit rotations and global phase.
*/
class FSimGate : public BenchmarkDoubleGate
{
public:

    FSimGate(double theta, double phi) 
    { 
        m_theta = theta;
        m_phi = phi;
    }

    QGate gate(Qubit* ctr_qubit, Qubit* tar_qubit)
    {
        // FSimGate(θ, φ) = ISWAP **(-2θ / π) * CZPowGate(exponent = -φ / π)
        auto CZ_Pow = CZPowGate(-m_phi / PI);
        return iSWAP(ctr_qubit, tar_qubit, -2 * m_theta / PI) * CZ_Pow.gate(ctr_qubit, tar_qubit);
    }

    QStat unitary()
    {
        auto a = std::cos(m_theta);
        auto b = -qcomplex_t(0, 1) * std::sin(m_theta);
        auto c = std::exp(qcomplex_t(0, 1) * m_phi);

        QStat matrix
        { 
            1, 0, 0, 0,
            0, a, b, 0,
            0, b, a, 0,
            0, 0, 0, c
        };

        return matrix;
    }
private:
    double m_theta;
    double m_phi;
};

QPANDA_END

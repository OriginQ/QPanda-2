/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

QGate.cpp
Author: Menghan.Dou
Created in 2018-6-30

Classes for QGate

Update@2018-8-30
Update by code specification
*/

#include "QuantumGate.h"
#include "QGlobalVariable.h"
#include "Utilities/QuantumMetadata.h"
#include "Core/Utilities/QPandaException.h"

using namespace QGATE_SPACE;
using namespace std;
QuantumGate::QuantumGate()
{
    operation_num = 0;
    gate_type = -1;
}

U4::U4(U4 &toCopy)
{
    operation_num = toCopy.operation_num;
    this->alpha = toCopy.alpha;
    this->beta = toCopy.beta;
    this->gamma = toCopy.gamma;
    this->delta = toCopy.delta;
    this->gate_matrix = toCopy.gate_matrix;
}

U4::U4()
{
    operation_num = 1;
    alpha = 0;
    beta = 0;
    gamma = 0;
    delta = 0;
    gate_matrix.push_back(1);
    gate_matrix.push_back(0);
    gate_matrix.push_back(0);
    gate_matrix.push_back(1);
}

U4::U4(double _alpha, double _beta, double _gamma, double _delta)
    : alpha(_alpha), beta(_beta), gamma(_gamma), delta(_delta)
{
    operation_num = 1;
    QStat matrix;
    gate_matrix.push_back(qcomplex_t(cos(alpha - beta / 2 - delta / 2)*cos(gamma / 2),
        sin(alpha - beta / 2 - delta / 2)*cos(gamma / 2)));
    gate_matrix.push_back(qcomplex_t(-cos(alpha - beta / 2 + delta / 2)*sin(gamma / 2),
        -sin(alpha - beta / 2 + delta / 2)*sin(gamma / 2)));
    gate_matrix.push_back(qcomplex_t(cos(alpha + beta / 2 - delta / 2)*sin(gamma / 2),
        sin(alpha + beta / 2 - delta / 2)*sin(gamma / 2)));
    gate_matrix.push_back(qcomplex_t(cos(alpha + beta / 2 + delta / 2)*cos(gamma / 2),
        sin(alpha + beta / 2 + delta / 2)*cos(gamma / 2)));
}
U4::U4(QStat & matrix)
{
    operation_num = 1;
    gate_matrix.resize(4);
    gate_matrix[0] = matrix[0];
    gate_matrix[1] = matrix[1];
    gate_matrix[2] = matrix[2];
    gate_matrix[3] = matrix[3];
    gamma = 2 * acos(abs(gate_matrix[0]));
    if (abs(gate_matrix[0] * gate_matrix[1]) > 1e-20)
    {
        beta = argc(gate_matrix[2] / gate_matrix[0]);
        delta = argc(gate_matrix[3] / gate_matrix[2]);
        alpha = beta / 2 + delta / 2 + argc(gate_matrix[0]);
    }
    else if (abs(gate_matrix[0]) > 1e-10)
    {
        beta = argc(gate_matrix[3] / gate_matrix[0]);
        delta = 0;
        alpha = beta / 2 + argc(gate_matrix[0]);
    }
    else
    {
        beta = argc(gate_matrix[2] / gate_matrix[1]) + PI;
        delta = 0;
        alpha = argc(gate_matrix[1]) + beta / 2 - PI;
    }
}

void U4::getMatrix(QStat & matrix) const
{
    if (gate_matrix.size() != 4)
    {
        QCERR("the size of gate_matrix is error");
        throw invalid_argument("the size of gate_matrix is error");
    }

    for (auto aIter : gate_matrix)
    {
        matrix.push_back(aIter);
    }
}

//RX_GATE gate
X::X()
{
    alpha = PI / 2;
    beta = 0;
    gamma = PI;
    delta = PI;
    gate_matrix[0] = 0;
    gate_matrix[1] = 1;
    gate_matrix[2] = 1;
    gate_matrix[3] = 0;
}


//RY_GATE gate
Y::Y()
{
    alpha = PI / 2;
    beta = 0;
    gamma = PI;
    delta = 0;
    gate_matrix[0] = 0;
    gate_matrix[1].imag(-1);
    gate_matrix[2].imag(1);
    gate_matrix[3] = 0;
}


//PauliZ gate,[1 0;0 -1]
Z::Z()
{
    alpha = PI / 2;
    beta = PI;
    gamma = 0;
    delta = 0;
    gate_matrix[3] = -1;
}

//RX(pi/2) gate
X1::X1()
{
    alpha = PI;
    beta = 3.0 / 2 * PI;
    gamma = PI / 2;
    delta = PI / 2;
    gate_matrix[0] = 1 / SQRT2;
    gate_matrix[1] = qcomplex_t(0, -1 / SQRT2);
    gate_matrix[2] = qcomplex_t(0, -1 / SQRT2);
    gate_matrix[3] = 1 / SQRT2;
}


//RY(pi/2) gate
Y1::Y1()
{
    alpha = 0;
    beta = 0;
    gamma = PI / 2;
    delta = 0;
    gate_matrix[0] = 1 / SQRT2;
    gate_matrix[1] = -1 / SQRT2;
    gate_matrix[2] = 1 / SQRT2;
    gate_matrix[3] = 1 / SQRT2;
}


//RZ(pi/2) gate
Z1::Z1()
{
    alpha = 0;
    beta = PI / 2;
    gamma = 0;
    delta = 0;
    gate_matrix[0] = qcomplex_t(1 / SQRT2, -1 / SQRT2);
    gate_matrix[3] = qcomplex_t(1 / SQRT2, 1 / SQRT2);
}

H::H()
{
    alpha = PI / 2;
    beta = 0;
    gamma = PI / 2;
    delta = PI;
    gate_matrix[0] = 1 / SQRT2;
    gate_matrix[1] = 1 / SQRT2;
    gate_matrix[2] = 1 / SQRT2;
    gate_matrix[3] = -1 / SQRT2;
}

//S:RZ_GATE(pi/2)
S::S()
{
    alpha = PI / 4;
    beta = PI / 2;
    gamma = 0;
    delta = 0;
    gate_matrix[3].real(0);
    gate_matrix[3].imag(1);
}

T::T()
{
    alpha = PI / 8;
    beta = PI / 4;
    gamma = 0;
    delta = 0;
    gate_matrix[3].real(1 / SQRT2);
    gate_matrix[3].imag(1 / SQRT2);
}

RX::RX(double angle)
{
    alpha = PI;
    beta = 3.0 / 2 * PI;
    gamma = angle;
    delta = PI / 2;
    theta = angle;
    gate_matrix[0] = cos(angle / 2);
    gate_matrix[1].imag(-1 * sin(angle / 2));
    gate_matrix[2].imag(-1 * sin(angle / 2));
    gate_matrix[3] = cos(angle / 2);
}

RY::RY(double angle)
{
    alpha = 0;
    beta = 0;
    gamma = angle;
    delta = 0;
    theta = angle;
    gate_matrix[0] = cos(angle / 2);
    gate_matrix[1] = -sin(angle / 2);
    gate_matrix[2] = sin(angle / 2);
    gate_matrix[3] = cos(angle / 2);
}

RZ::RZ(double angle)
{
    alpha = 0;
    beta = angle;
    gamma = 0;
    delta = 0;
    theta = angle;
    gate_matrix[0].real(cos(angle / 2));
    gate_matrix[0].imag(-1 * sin(angle / 2));
    gate_matrix[3].real(cos(angle / 2));
    gate_matrix[3].imag(1 * sin(angle / 2));
}

//U1_GATE=[1 0;0 exp(i*angle)]
U1::U1(double angle)
{
    alpha = angle / 2;
    beta = angle;
    gamma = 0;
    delta = 0;
    theta = angle;
    gate_matrix[3].real(cos(angle));
    gate_matrix[3].imag(1 * sin(angle));
}

QDoubleGate::QDoubleGate()
{
    operation_num = 2;
    gate_matrix.resize(16);
    gate_matrix[0] = 1;
    gate_matrix[5] = 1;
    gate_matrix[10] = 1;
    gate_matrix[15] = 1;
}

QDoubleGate::QDoubleGate(const QDoubleGate & oldDouble)
{
    this->operation_num = oldDouble.operation_num;
    this->gate_matrix = oldDouble.gate_matrix;
}
QDoubleGate::QDoubleGate(QStat & matrix)
{
    operation_num = 2;
    if (matrix.size() != 16)
    {
        QCERR("Given matrix is invalid.");
        throw invalid_argument("Given matrix is invalid.");
    }
    this->gate_matrix = matrix;
}
void QDoubleGate::getMatrix(QStat & matrix) const
{
    if (gate_matrix.size() != 16)
    {
        QCERR("Given matrix is invalid.");
        throw invalid_argument("Given matrix is invalid.");
    }
    matrix = gate_matrix;
}

CU::CU()
{
    operation_num = 2;
    alpha = 0;
    beta = 0;
    gamma = 0;
    delta = 0;
}

CU::CU(CU &toCopy)
{
    operation_num = toCopy.operation_num;
    this->alpha = toCopy.alpha;
    this->beta = toCopy.beta;
    this->gamma = toCopy.gamma;
    this->delta = toCopy.delta;
    this->gate_matrix = toCopy.gate_matrix;
}

CU::CU(double _alpha, double _beta,
    double _gamma, double _delta)
    : alpha(_alpha), beta(_beta), gamma(_gamma), delta(_delta)
{
    operation_num = 2;
    gate_matrix[10] = qcomplex_t(cos(alpha - beta / 2 - delta / 2)*cos(gamma / 2),
        sin(alpha - beta / 2 - delta / 2)*cos(gamma / 2));
    gate_matrix[11] = qcomplex_t(-cos(alpha - beta / 2 + delta / 2)*sin(gamma / 2),
        -sin(alpha - beta / 2 + delta / 2)*sin(gamma / 2));
    gate_matrix[14] = qcomplex_t(cos(alpha + beta / 2 - delta / 2)*sin(gamma / 2),
        sin(alpha + beta / 2 - delta / 2)*sin(gamma / 2));
    gate_matrix[15] = qcomplex_t(cos(alpha + beta / 2 + delta / 2)*cos(gamma / 2),
        sin(alpha + beta / 2 + delta / 2)*cos(gamma / 2));
}

CU::CU(QStat & matrix)
{
    operation_num = 2;
    //QStat matrix;
    gate_matrix.resize(16);
    gate_matrix[0] = 1;
    gate_matrix[5] = 1;
    gate_matrix[10] = matrix[0];
    gate_matrix[11] = matrix[1];
    gate_matrix[14] = matrix[2];
    gate_matrix[15] = matrix[3];
    gamma = 2 * acos(abs(gate_matrix[10]));
    if (abs(gate_matrix[10] * gate_matrix[11]) > 1e-20)
    {
        beta = argc(gate_matrix[14] / gate_matrix[10]);
        delta = argc(gate_matrix[15] / gate_matrix[14]);
        alpha = beta / 2 + delta / 2 + argc(gate_matrix[10]);
    }
    else if (abs(gate_matrix[10]) > 1e-10)
    {
        beta = argc(gate_matrix[15] / gate_matrix[10]);
        delta = 0;
        alpha = beta / 2 + argc(gate_matrix[10]);
    }
    else
    {
        beta = argc(gate_matrix[14] / gate_matrix[11]) + PI;
        delta = 0;
        alpha = argc(gate_matrix[11]) + beta / 2 - PI;
    }
}

CNOT::CNOT()
{
    alpha = PI / 2;
    beta = 0;
    gamma = PI;
    delta = PI;
    gate_matrix[10] = 0;
    gate_matrix[11] = 1;
    gate_matrix[14] = 1;
    gate_matrix[15] = 0;
}

CPhaseGate::CPhaseGate(double angle)
{
    alpha = angle / 2;
    beta = angle;
    gamma = 0;
    delta = 0;
    theta = angle;
    gate_matrix[15] = cos(angle);
    gate_matrix[15].imag(1 * sin(angle));
}

CZ::CZ()
{
    alpha = PI / 2;
    beta = PI;
    gamma = 0;
    delta = 0;
    gate_matrix[15] = -1;
}

ISWAPTheta::ISWAPTheta(double angle)
{
    theta = angle;
    gate_matrix[5] = cos(angle);
    gate_matrix[6].imag(-1 * sin(angle));
    gate_matrix[9].imag(-1 * sin(angle));
    gate_matrix[10] = cos(angle);
}

ISWAP::ISWAP()
{
    theta = PI / 2;
    gate_matrix[5] = 0;
    gate_matrix[6].imag(-1);
    gate_matrix[9].imag(-1);
    gate_matrix[10] = 0;
}

SQISWAP::SQISWAP()
{
    theta = PI / 4;
    gate_matrix[5] = 1 / SQRT2;
    gate_matrix[6].imag(-1 / SQRT2);
    gate_matrix[9].imag(-1 / SQRT2);
    gate_matrix[10] = 1 / SQRT2;
}

SWAP::SWAP()
{
    gate_matrix[5] = 0;
    gate_matrix[6] = 1;
    gate_matrix[9] = 1;
    gate_matrix[10] = 0;
}

void QGateFactory::registClass(string name, CreateGate_cb method)
{
    m_gate_map.insert(pair<string, CreateGate_cb>(name, method));
}

void QGateFactory::registClass(string name, CreateAngleGate_cb method)
{
    m_angle_gate_map.insert(pair<string, CreateAngleGate_cb>(name, method));
}

void QGateFactory::registClass(string name, CreateSingleAndCUGate_cb method)
{
    m_single_and_cu_gate_map.insert(pair<string, CreateSingleAndCUGate_cb>(name, method));
}

void QGateFactory::registClass(string name, CreateGateByMatrix_cb method)
{
    m_double_gate_map.insert(pair<string, CreateGateByMatrix_cb>(name, method));
}

QuantumGate * QGateFactory::getGateNode(const std::string & name)
{
    map<string, CreateGate_cb>::const_iterator iter;
    iter = m_gate_map.find(name);
    if (iter == m_gate_map.end())
    {
        stringstream error;
        error <<"there is no "<< name << " in m_gate_map";
        QCERR(error.str());
        throw QPanda::gate_alloc_fail(error.str());
    }
    else
        return iter->second();
}

QuantumGate * QGateFactory::getGateNode(const std::string & name,const double angle)
{
    map<string, CreateAngleGate_cb>::const_iterator iter;
    iter = m_angle_gate_map.find(name);
    if (iter == m_angle_gate_map.end())
    {
        stringstream error;
        error << "there is no " << name << " in m_angle_gate_map";
        QCERR(error.str());
        throw QPanda::gate_alloc_fail(error.str());
    }
    else
        return iter->second(angle);
}

QuantumGate * QGateFactory::getGateNode(const std::string & name, 
                                        const double alpha,
                                        const double beta,
                                        const double gamma,
                                        const double delta)
{
    map<string, CreateSingleAndCUGate_cb>::const_iterator iter;
    iter = m_single_and_cu_gate_map.find(name);
    if (iter == m_single_and_cu_gate_map.end())
    {
        stringstream error;
        error << "there is no " << name << " in m_single_and_cu_gate_map";
        QCERR(error.str());
        throw QPanda::gate_alloc_fail(error.str());
    }
    else
        return iter->second(alpha, beta, gamma, delta);
}

QuantumGate * QGateFactory::getGateNode(const std::string & name, QStat & matrix)
{

    auto iter = m_double_gate_map.find(name);
    if (iter == m_double_gate_map.end())
    {
        stringstream error;
        error << "there is no " << name << " in m_double_gate_map";
        QCERR(error.str());
        throw QPanda::gate_alloc_fail(error.str());
    }
    else
        return iter->second(matrix);
}

#define REGISTER(className)                                             \
    QuantumGate* objectCreator##className(){                            \
        return new className();                                         \
    }                                                                   \
    RegisterAction g_creatorRegister##className(                        \
        #className,(CreateGate_cb)objectCreator##className)

REGISTER(X);
REGISTER(T);
REGISTER(Y);
REGISTER(Z);
REGISTER(S);
REGISTER(H);
REGISTER(X1);
REGISTER(Y1);
REGISTER(Z1);
REGISTER(CNOT);
REGISTER(CZ);
REGISTER(ISWAP);
REGISTER(SQISWAP);
REGISTER(SWAP);

#define REGISTER_ANGLE(className)                                       \
    QuantumGate* objectCreator##className(double angle){                \
        return new className(angle);}                                    \
    RegisterAction g_angleCreatorRegister##className(                   \
        #className,(CreateAngleGate_cb)objectCreator##className)

#define REGISTER_SINGLE_CU(className)                                   \
    QuantumGate* objectCreator##className(double alpha,                    \
                double beta, double gamma, double delta){                \
        return new className(alpha,beta,gamma,delta);                    \
    }                                                                   \
    RegisterAction g_singleCreatorRegister##className(                  \
        #className,(CreateSingleAndCUGate_cb)objectCreator##className)

#define REGISTER_MATRIX(className)                                      \
    QuantumGate* objectCreator##className(QStat & matrix){                \
        return new className(matrix);                                    \
    }                                                                   \
    RegisterAction g_doubleCreatorRegister##className(                  \
        #className,(CreateGateByMatrix_cb)objectCreator##className)

REGISTER_ANGLE(RX);
REGISTER_ANGLE(RY);
REGISTER_ANGLE(RZ);
REGISTER_ANGLE(U1);
REGISTER_ANGLE(ISWAPTheta);
REGISTER_ANGLE(CPhaseGate);

REGISTER_SINGLE_CU(U4);
REGISTER_SINGLE_CU(CU);

REGISTER_MATRIX(QDoubleGate);
REGISTER_MATRIX(U4);
REGISTER_MATRIX(CU);

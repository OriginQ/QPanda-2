/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "QGate.h"
#include "QGlobalVariable.h"
#include "QPanda/QPandaException.h"

QuantumGate::QuantumGate()
{
    qOpNum = 0;
    gateType = -1;
}


U4::U4(U4 &toCopy)
{
    qOpNum = toCopy.qOpNum;
    this->alpha = toCopy.alpha;
    this->beta = toCopy.beta;
    this->gamma = toCopy.gamma;
    this->delta = toCopy.delta;
    this->gatematrix = toCopy.gatematrix;
}

U4::U4()
{
    qOpNum = 1;
    alpha = 0;
    beta = 0;
    gamma = 0;
    delta = 0;
    gatematrix.push_back(1);
    gatematrix.push_back(0);
    gatematrix.push_back(0);
    gatematrix.push_back(1);

}

U4::U4(double _alpha, double _beta, double _gamma, double _delta)
    : alpha(_alpha), beta(_beta), gamma(_gamma), delta(_delta)
{
    qOpNum = 1;
    QStat matrix;
    gatematrix.push_back(COMPLEX(cos(alpha - beta / 2 - delta / 2)*cos(gamma / 2),
        sin(alpha - beta / 2 - delta / 2)*cos(gamma / 2)));
    gatematrix.push_back(COMPLEX(-cos(alpha - beta / 2 + delta / 2)*sin(gamma / 2),
        -sin(alpha - beta / 2 + delta / 2)*sin(gamma / 2)));
    gatematrix.push_back(COMPLEX(cos(alpha + beta / 2 - delta / 2)*sin(gamma / 2),
        sin(alpha + beta / 2 - delta / 2)*sin(gamma / 2)));
    gatematrix.push_back(COMPLEX(cos(alpha + beta / 2 + delta / 2)*cos(gamma / 2),
        sin(alpha + beta / 2 + delta / 2)*cos(gamma / 2)));
}
U4::U4(QStat & matrix)
{
    qOpNum = 1;
    //QStat matrix;
    gatematrix.resize(4);
    gatematrix[0] = matrix[0];
    gatematrix[1] = matrix[1];
    gatematrix[2] = matrix[2];
    gatematrix[3] = matrix[3];
    gamma = 2 * acos(abs(matrix[0]));
    beta = argc(matrix[2] - matrix[0]);
    delta = argc(matrix[2] - matrix[0]) + PI;
    alpha = argc(matrix[2]) + delta / 2 - beta / 2;
}
;
void U4::getMatrix(QStat & matrix) const
{
    if (gatematrix.size() != 4)
        throw exception();
    for (auto aIter : gatematrix)
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
    gatematrix[0] = 0;
    gatematrix[1] = 1;
    gatematrix[2] = 1;
    gatematrix[3] = 0;
}


//RY_GATE gate
Y::Y()
{
    alpha = PI / 2;
    beta = 0;
    gamma = PI;
    delta = 0;
    gatematrix[0] = 0;
    gatematrix[1].imag( -1);
    gatematrix[2].imag(-1);
    gatematrix[3] = 0;
}


//PauliZ gate,[1 0;0 -1]
Z::Z()
{
    alpha = PI / 2;
    beta = PI;
    gamma = 0;
    delta = 0;
    gatematrix[3] = -1;
}

H::H()
{
    alpha = PI / 2;
    beta = 0;
    gamma = PI / 2;
    delta = PI;
    gatematrix[0] = 1 / SQRT2;
    gatematrix[1] = 1 / SQRT2;
    gatematrix[2] = 1 / SQRT2;
    gatematrix[3] = -1 / SQRT2;
}

//S:RZ_GATE(pi/2)
S::S()
{
    alpha = PI / 4;
    beta = PI / 2;
    gamma = 0;
    delta = 0;
    gatematrix[3].imag(1);
}
T::T()
{
    alpha = PI / 8;
    beta = PI / 4;
    gamma = 0;
    delta = 0;
    gatematrix[3].real(1 / SQRT2);
    gatematrix[3].imag(1 / SQRT2);
}

RX::RX(double angle)
{
    alpha = PI;
    beta = 3.0 / 2 * PI;
    gamma = angle;
    delta = PI / 2;
    theta = angle;
    gatematrix[0] = cos(angle / 2);
    gatematrix[1].imag(-1*sin(angle / 2)) ;
    gatematrix[2].imag (-1*sin(angle / 2));
    gatematrix[3] = cos(angle / 2);
}
RY::RY(double angle)
{
    alpha = 0;
    beta = 0;
    gamma = angle;
    delta = 0;
    theta = angle;
    gatematrix[0] = cos(angle / 2);
    gatematrix[1] = -sin(angle / 2);
    gatematrix[2] = sin(angle / 2);
    gatematrix[3] = cos(angle / 2);
}

RZ::RZ(double angle)
{
    //alpha = angle / 2;
    alpha = 0;
    beta = angle;
    gamma = 0;
    delta = 0;
    theta = angle;
    gatematrix[0].real(cos(angle / 2));
    gatematrix[0].imag(-1*sin(angle / 2));
    gatematrix[3].real(cos(angle / 2));
    gatematrix[3].imag(1*sin(angle / 2));
}
//U1_GATE=[1 0;0 exp(i*angle)]
U1::U1(double angle)
{
    alpha = angle / 2;
    beta = angle;
    gamma = 0;
    delta = 0;
    theta = angle;
    gatematrix[3].real(cos(angle));
    gatematrix[3].imag(1*sin(angle));
}



////////////////////////////////////////////////////////////////

QDoubleGate::QDoubleGate()
{
    qOpNum = 2;
    for (auto i = 0; i < 16; i++)
    {
        gatematrix.push_back(0);
    }
    gatematrix[0] = 1;
    gatematrix[5] = 1;
    gatematrix[10] = 1;
    gatematrix[15] = 1;
}
QDoubleGate::QDoubleGate(const QDoubleGate & oldDouble)
{
    this->qOpNum = oldDouble.qOpNum;
    this->gatematrix = oldDouble.gatematrix;
    //oldDouble.getMatrix(Matrix);

    //for (auto aiter : Matrix)
    //{
    //m_matrix.push_back(aiter);
    //}
}
QDoubleGate::QDoubleGate(QStat & matrix) : qOpNum(2)
{
    if (matrix.size() != 16)
        throw param_error_exception("this param for this function is err", false);
    this->gatematrix = matrix;
    // for (auto aIter : matrix)
    // {
    //     gatematrix.push_back(aIter);
    // }
}
void QDoubleGate::getMatrix(QStat & matrix) const
{
    if (gatematrix.size() != 16)
        throw exception();
    matrix = gatematrix;
    //for (auto aIter : gatematrix)
    // {
    //    matrix.push_back(aIter);
    // }
}

CU::CU()
{
    qOpNum = 2;
    alpha = 0;
    beta = 0;
    gamma = 0;
    delta = 0;
    //matrix
}
CU::CU(CU &toCopy)
{
    qOpNum = toCopy.qOpNum;
    this->alpha = toCopy.alpha;
    this->beta = toCopy.beta;
    this->gamma = toCopy.gamma;
    this->delta = toCopy.delta;
    //matrix
    this->gatematrix = toCopy.gatematrix;
}
CU::CU(double _alpha, double _beta,
    double _gamma, double _delta)
    : alpha(_alpha), beta(_beta), gamma(_gamma), delta(_delta)
{
    qOpNum = 2;
    gatematrix[10] = COMPLEX(cos(alpha - beta / 2 - delta / 2)*cos(gamma / 2),
        sin(alpha - beta / 2 - delta / 2)*cos(gamma / 2));
    gatematrix[11] = COMPLEX(-cos(alpha - beta / 2 + delta / 2)*sin(gamma / 2),
        -sin(alpha - beta / 2 + delta / 2)*sin(gamma / 2));
    gatematrix[14] = COMPLEX(cos(alpha + beta / 2 - delta / 2)*sin(gamma / 2),
        sin(alpha + beta / 2 - delta / 2)*sin(gamma / 2));
    gatematrix[15] = COMPLEX(cos(alpha + beta / 2 + delta / 2)*cos(gamma / 2),
        sin(alpha + beta / 2 + delta / 2)*cos(gamma / 2));
}
CU::CU(QStat & matrix)
{
    qOpNum = 2;
    //QStat matrix;
    gatematrix.resize(4);
    gatematrix[10] = matrix[0];
    gatematrix[11] = matrix[1];
    gatematrix[14] = matrix[2];
    gatematrix[15] = matrix[3];
    gamma = 2 * acos(abs(matrix[0]));
    beta = argc(matrix[2] - matrix[0]);
    delta = argc(matrix[2] - matrix[0]) + PI;
    alpha = argc(matrix[2]) + delta / 2 - beta / 2;
}
;
CNOT::CNOT()
{
    alpha = PI / 2;
    beta = 0;
    gamma = PI;
    delta = PI;
    gatematrix[10] = 0;
    gatematrix[11] = 1;
    gatematrix[14] = 1;
    gatematrix[15] = 0;
}

CPhaseGate::CPhaseGate(double angle)
{
    alpha = angle / 2;
    beta = angle;
    gamma = 0;
    delta = 0;
    gatematrix[15] = cos(angle);
    gatematrix[15].imag(1*sin(angle));
}
CZ::CZ()
{
    alpha = PI / 2;
    beta = PI;
    gamma = 0;
    delta = 0;
    gatematrix[15] = -1;
}

ISWAPTheta::ISWAPTheta(double angle)
{
    theta = angle;
    gatematrix[5] = cos(angle);
    gatematrix[6].imag(-1*sin(angle));
    gatematrix[9].imag(-1*sin(angle));
    gatematrix[10] = cos(angle);
    //matrix
}
ISWAP::ISWAP()
{
    theta = PI / 2;
    gatematrix[5] = 0;
    gatematrix[6].imag(-1);
    gatematrix[9],imag(-1);
    gatematrix[10] = 0;
}
SQISWAP::SQISWAP()
{
    theta = PI / 4;
    gatematrix[5] = 1 / SQRT2;
    gatematrix[6].imag(-1 / SQRT2);
    gatematrix[9] .imag(-1 / SQRT2);
    gatematrix[10] = 1 / SQRT2;
}


void QGateFactory::registClass(string name, CreateGate method)
{
    m_gateMap.insert(pair<string, CreateGate>(name, method));
}

void QGateFactory::registClass(string name, CreateAngleGate method)
{
    m_angleGateMap.insert(pair<string, CreateAngleGate>(name, method));
}

void QGateFactory::registClass(string name, CreateSingleAndCUGate method)
{
    m_singleAndCUGateMap.insert(pair<string, CreateSingleAndCUGate>(name, method));
}

void QGateFactory::registClass(string name, CreateGateByMatrix method)
{
    m_DoubleGateMap.insert(pair<string, CreateGateByMatrix>(name, method));
}


QuantumGate * QGateFactory::getGateNode(std::string & name)
{
    map<string, CreateGate>::const_iterator iter;
    iter = m_gateMap.find(name);
    if (iter == m_gateMap.end())
        return nullptr;
    else
        return iter->second();
}

QuantumGate * QGateFactory::getGateNode(std::string & name, double angle)
{
    map<string, CreateAngleGate>::const_iterator iter;
    iter = m_angleGateMap.find(name);
    if (iter == m_angleGateMap.end())
        return nullptr;
    else
        return iter->second(angle);
}

QuantumGate * QGateFactory::getGateNode(std::string & name, double alpha, double beta, double gamma, double delta)
{
    map<string, CreateSingleAndCUGate>::const_iterator iter;
    iter = m_singleAndCUGateMap.find(name);
    if (iter == m_singleAndCUGateMap.end())
        return nullptr;
    else
        return iter->second(alpha, beta, gamma, delta);
}

QuantumGate * QGateFactory::getGateNode(std::string & name, QStat & matrix)
{

    auto iter = m_DoubleGateMap.find(name);
    if (iter == m_DoubleGateMap.end())
        return nullptr;
    else
        return iter->second(matrix);
}

#define REGISTER(className)                                             \
    QuantumGate* objectCreator##className(){                              \
        return new className();                                          \
    }                                                                   \
    RegisterAction g_creatorRegister##className(                        \
        #className,(CreateGate)objectCreator##className)

REGISTER(X);
REGISTER(Y);
REGISTER(Z);
REGISTER(S);
REGISTER(H);
REGISTER(CNOT);
REGISTER(CZ);
REGISTER(ISWAP);
REGISTER(SQISWAP);

#define REGISTER_ANGLE(className)                                             \
    QuantumGate* objectCreator##className(double angle){                              \
        return new className(angle);                                          \
    }                                                                   \
    RegisterAction g_angleCreatorRegister##className(                        \
        #className,(CreateAngleGate)objectCreator##className)

#define REGISTER_SINGLE_CU(className)                                             \
    QuantumGate* objectCreator##className(double alpha,double beta,double gamma,double delta){      \
        return new className(alpha,beta,gamma,delta);                    \
    }                                                                   \
    RegisterAction g_singleCreatorRegister##className(                        \
        #className,(CreateSingleAndCUGate)objectCreator##className)

#define REGISTER_MATRIX(className)                                             \
    QuantumGate* objectCreator##className(QStat & matrix){      \
        return new className(matrix);                    \
    }                                                                   \
    RegisterAction g_doubleCreatorRegister##className(                        \
        #className,(CreateGateByMatrix)objectCreator##className)


REGISTER_ANGLE(RX);
REGISTER_ANGLE(RY);
REGISTER_ANGLE(RZ);
REGISTER_ANGLE(ISWAPTheta);

REGISTER_SINGLE_CU(U4);
REGISTER_SINGLE_CU(CU);

REGISTER_MATRIX(QDoubleGate);
REGISTER_MATRIX(U4);
REGISTER_MATRIX(CU);






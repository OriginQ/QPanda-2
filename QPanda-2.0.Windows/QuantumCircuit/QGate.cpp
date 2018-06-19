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
}
//RX gate
XGate::XGate()
{
    alpha = PI/2;
    beta = 0;
    gamma = PI;
    delta = PI;

}

XGate::XGate(double angle)
{
    alpha = PI;
    beta = 3.0/2*PI;
    gamma = angle;
    delta = PI/2 ;
}
//RY gate
YGate::YGate()
{
    alpha = PI/2;
    beta = 0;
    gamma = PI;
    delta = 0;
}

YGate::YGate(double angle)
{
    alpha = 0;
    beta = 0;
    gamma = angle;
    delta = 0;
}
//RZ gate
ZGate::ZGate()
{
    alpha = PI / 2;
    beta = PI;
    gamma = 0;
    delta = 0;
}
//not [exp(-i*theta/2) 0;0 exp(i*theta/2)]
//[1 0;0 exp(i*theta)]
ZGate::ZGate(double angle)
{
    alpha = angle/2;
    beta = angle;
    gamma = 0;
    delta = 0;
}

HadamardGate::HadamardGate()
{
    alpha = PI / 2;
    beta = 0;
    gamma = PI/2;
    delta = PI;
}

//SGate:RZ(pi/2)
SGate::SGate()
{
    alpha = PI / 4;
    beta = PI/2;
    gamma = 0;
    delta = 0;
}

QSingleGate::QSingleGate(QSingleGate &toCopy)
{
	qOpNum = toCopy.qOpNum;
	this->alpha = toCopy.alpha;
	this->beta = toCopy.beta;
	this->gamma = toCopy.gamma;
	this->delta = toCopy.delta;
}

QSingleGate::QSingleGate()
{
	qOpNum = 1;
    alpha = 0;
    beta = 0;
    gamma = 0;
    delta = 0;
}

QSingleGate::QSingleGate(double _alpha,double _beta,double _gamma,double _delta)
	: alpha(_alpha),beta(_beta),gamma(_gamma),delta(_delta)
{
    qOpNum = 1;
};

////////////////////////////////////////////////////////////////

CU::CU()
{
    qOpNum = 2;
    alpha = 0;
    beta = 0;
    gamma = 0;
    delta = 0;
}
CU::CU(CU &toCopy)
{
    qOpNum = toCopy.qOpNum;
    this->alpha = toCopy.alpha;
    this->beta = toCopy.beta;
    this->gamma = toCopy.gamma;
    this->delta = toCopy.delta;
}
CU::CU(double _alpha, double _beta,
    double _gamma, double _delta)
    : alpha(_alpha), beta(_beta), gamma(_gamma), delta(_delta)
{
    qOpNum = 2;
};
CNOTGate::CNOTGate()
{
    alpha = PI / 2;
    beta = 0;
    gamma = PI;
    delta = PI ;
}
CZGate::CZGate()
{
    alpha = PI / 2;
    beta = PI ;
    gamma = 0;
    delta = 0;
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

void QGateFactory::registClass(string name, CreateDoubleGate method)
{
    m_DoubleGateMap.insert(pair<string, CreateDoubleGate>(name, method));
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

QuantumGate * QGateFactory::getGateNode(std::string & name,double alpha, double beta, double gamma, double delta)
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

REGISTER(XGate);
REGISTER(YGate);
REGISTER(ZGate);
REGISTER(SGate);
REGISTER(HadamardGate);
REGISTER(CNOTGate);
REGISTER(CZGate);
REGISTER(iSwapGate);

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

#define REGISTER_DOUBLE(className)                                             \
    QuantumGate* objectCreator##className(QStat & matrix){      \
        return new className(matrix);                    \
    }                                                                   \
    RegisterAction g_doubleCreatorRegister##className(                        \
        #className,(CreateDoubleGate)objectCreator##className)


REGISTER_ANGLE(XGate);
REGISTER_ANGLE(YGate);
REGISTER_ANGLE(ZGate);

REGISTER_SINGLE_CU(QSingleGate);
REGISTER_SINGLE_CU(CU);

REGISTER_DOUBLE(QDoubleGate);

QDoubleGate::QDoubleGate(const QDoubleGate & oldDouble) 
{
    this->qOpNum = ISWAP;
    QStat Matrix(16,0);
    oldDouble.getMatrix(Matrix);
    
    for (auto aiter : Matrix)
    {
        m_matrix.push_back(aiter);
    }

}

QDoubleGate::QDoubleGate(QStat & matrix) : qOpNum(3)
{
    if (matrix.size() <= 0)
        throw param_error_exception("this param for this function is err", false);
    for (auto aIter : matrix)
    {
        m_matrix.push_back(aIter);
    }
}

void QDoubleGate::getMatrix(QStat & matrix) const
{
    if (m_matrix.size() <= 0)
        throw exception();
    for (auto aIter : m_matrix)
    {
        matrix.push_back(aIter);
    }
}

iSwapGate::iSwapGate() : m_matrix(16,0)
{
    m_matrix[0] = 1;
    m_matrix[6].imag(-1);
    m_matrix[9].imag(-1);
    m_matrix[15] = 1;
    qOpNum = ISWAP;
}

void iSwapGate::getMatrix(QStat & matrix) const
{
    if (m_matrix.size() <= 0)
        throw exception();
    for (auto aIter : m_matrix)
    {
        matrix.push_back(aIter);
    }
}

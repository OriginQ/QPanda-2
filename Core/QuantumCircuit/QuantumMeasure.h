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

#ifndef _QUANTUM_MEASURE_H
#define _QUANTUM_MEASURE_H

#include "QuantumMachine/QubitFactory.h"
#include "QNode.h"
#include "ClassicalConditionInterface.h"
QPANDA_BEGIN
class AbstractQuantumMeasure
{
public:
    virtual Qubit * getQuBit() const =0;
    virtual CBit * getCBit()const = 0;
    virtual ~AbstractQuantumMeasure() {}
};

class QMeasure : public QNode, public AbstractQuantumMeasure
{
private:
    AbstractQuantumMeasure * m_pQuantumMeasure;
    qmap_size_t m_stPosition;
    QMeasure();

public:
    QMeasure(const QMeasure &);
    QMeasure(Qubit *, CBit *);
    ~QMeasure();
    Qubit * getQuBit() const;
    CBit * getCBit()const;

    NodeType getNodeType() const;
    qmap_size_t getPosition() const;
private:
    void setPosition(qmap_size_t) {};
};

typedef AbstractQuantumMeasure * (*CreateMeasure)(Qubit *, CBit *);
class QuantunMeasureFactory
{
public:
    void registClass(std::string name, CreateMeasure method);
    AbstractQuantumMeasure * getQuantumMeasure(std::string &, Qubit *, CBit *);

    static QuantunMeasureFactory & getInstance()
    {
        static QuantunMeasureFactory  s_Instance;
        return s_Instance;
    }
private:
    std::map<std::string, CreateMeasure> m_measureMap;
    QuantunMeasureFactory() {};

};

class QuantumMeasureRegisterAction {
public:
    QuantumMeasureRegisterAction(std::string className, CreateMeasure ptrCreateFn) {
         QuantunMeasureFactory::getInstance().registClass(className, ptrCreateFn);
    }

};

#define REGISTER_MEASURE(className)                                             \
    AbstractQuantumMeasure* objectCreator##className(Qubit * pQubit, CBit * pCBit){      \
        return new className(pQubit,pCBit);                    \
    }                                                                   \
    QuantumMeasureRegisterAction g_measureCreatorRegister##className(                        \
        #className,(CreateMeasure)objectCreator##className)

class OriginMeasure : public QNode ,public AbstractQuantumMeasure
{
public:
    OriginMeasure (Qubit *, CBit *);
    ~OriginMeasure() {};
    
    Qubit * getQuBit() const;
    CBit * getCBit()const;

    NodeType getNodeType() const;
    qmap_size_t getPosition() const;
    void setPosition(qmap_size_t stPositio);
private:
    OriginMeasure();
    OriginMeasure(OriginMeasure &);
    NodeType m_iNodeType;
    Qubit * m_pTargetQubit;
    CBit * m_pCBit;
    qmap_size_t m_stPosition;
     
};

QMeasure Measure(Qubit * targetQuBit, ClassicalCondition );



QPANDA_END

#endif // !_QUANTUM_MEASURE_H



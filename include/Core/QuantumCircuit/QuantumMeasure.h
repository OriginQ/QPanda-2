/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.

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
/*! \file QuantumMeasure.h */
#ifndef _QUANTUM_MEASURE_H
#define _QUANTUM_MEASURE_H

#include "Core/QuantumMachine/QubitFactory.h"
#include "Core/QuantumCircuit/QNode.h"
#include "Core/QuantumCircuit/ClassicalConditionInterface.h"

QPANDA_BEGIN

/**
* @class AbstractQuantumMeasure
* @brief Quantum Measure basic abstract class
* @ingroup QuantumCircuit
*/
class AbstractQuantumMeasure
{
public:
	/**
     * @brief Get measure node qubit address
	 * @return Qubit *
     */
    virtual Qubit * getQuBit() const =0;
	
	/**
     * @brief  Get measure node cbit address
	 * @return CBit *
     */
    virtual CBit * getCBit()const = 0;
    virtual ~AbstractQuantumMeasure() {}
};

/**
* @class QMeasure
* @brief Quantum Measure  basic  class
* @ingroup QuantumCircuit
*/
class QMeasure : public AbstractQuantumMeasure
{
private:
    std::shared_ptr<AbstractQuantumMeasure> m_measure;
public:
    QMeasure(const QMeasure &);
    QMeasure(Qubit *, CBit *);
    QMeasure(std::shared_ptr<AbstractQuantumMeasure> node);
    std::shared_ptr<AbstractQuantumMeasure> getImplementationPtr();
    ~QMeasure();
    Qubit * getQuBit() const;
    CBit * getCBit()const;
    NodeType getNodeType() const;
private:
    QMeasure();
};

typedef AbstractQuantumMeasure * (*CreateMeasure)(Qubit *, CBit *);

/**
 * @brief Factory for class AbstractQuantumMeasure
 * @ingroup QuantumCircuit
 */
class QuantumMeasureFactory
{
public:
    void registClass(std::string name, CreateMeasure method);
    AbstractQuantumMeasure * getQuantumMeasure(std::string &, Qubit *, CBit *);

	/**
     * @brief Get the static instance of factory 
	 * @return QuantumMeasureFactory &
     */
    static QuantumMeasureFactory & getInstance()
    {
        static QuantumMeasureFactory  s_Instance;
        return s_Instance;
    }
private:
    std::map<std::string, CreateMeasure> m_measureMap;
    QuantumMeasureFactory() {};

};

/**
* @brief QMeasure program register action
* @note Provide QuantumMeasureFactory class registration interface for the outside
 */
class QuantumMeasureRegisterAction {
public:
    QuantumMeasureRegisterAction(std::string className, CreateMeasure ptrCreateFn) {
         QuantumMeasureFactory::getInstance().registClass(className, ptrCreateFn);
    }

};

#define REGISTER_MEASURE(className)                                             \
    AbstractQuantumMeasure* objectCreator##className(Qubit * pQubit, CBit * pCBit){      \
        return new className(pQubit,pCBit);                    \
    }                                                                   \
    QuantumMeasureRegisterAction g_measureCreatorRegister##className(                        \
        #className,(CreateMeasure)objectCreator##className)

/**
* @class  OriginMeasure
* @brief  Qrigin quantum measure basic class
*/
class OriginMeasure : public QNode ,public AbstractQuantumMeasure
{
public:
    OriginMeasure (Qubit *, CBit *);
    ~OriginMeasure() {};
    
    /**
    * @brief  Get measure node qubit address
    * @return    QPanda::Qubit*  QuBit address
    */
    Qubit * getQuBit() const;
    /**
    * @brief  Get measure node cbit address
    * @return    QPanda::CBit*  cBit address
    */
    CBit * getCBit()const;
    /**
    * @brief  Get current node type
    * @return     NodeType  current node type
    * @see  NodeType
    */
    NodeType getNodeType() const;
private:
    OriginMeasure();
    OriginMeasure(OriginMeasure &);

    NodeType m_node_type;
    Qubit * m_target_qubit;
    CBit * m_target_cbit;
     
};

/**
* @brief  QPanda2 basic interface for creating a quantum measure node
* @param[in]  Qubit*   qubit address
* @param[in]  ClassicalCondition  cbit
* @return     QPanda::QMeasure  quantum measure node
* @ingroup Core
*/
QMeasure Measure(Qubit * , ClassicalCondition );
QPANDA_END

#endif // !_QUANTUM_MEASURE_H



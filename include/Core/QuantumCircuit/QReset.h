/*
Copyright (c) 2017-2023 Origin Quantum Computing. All Right Reserved.

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
/*! \file QReset.h */
#pragma once

#include "Core/QuantumMachine/QubitFactory.h"
#include "Core/QuantumCircuit/QNode.h"
#include "Core/QuantumCircuit/ClassicalConditionInterface.h"
QPANDA_BEGIN

/**
* @class AbstractQuantumReset
* @brief Quantum Reset basic abstract class
* @ingroup QuantumCircuit
*/
class AbstractQuantumReset
{
public:
	/**
     * @brief Get the reset qubit
	 * @return Qubit *
     */
	virtual Qubit * getQuBit() const = 0;
	virtual ~AbstractQuantumReset() {}
};

/**
* @class QReset
* @brief Quantum Reset  basic  class
* @ingroup QuantumCircuit
*/
class QReset : public AbstractQuantumReset
{
private:
	std::shared_ptr<AbstractQuantumReset> m_reset;
public:
	QReset() = delete;
	QReset(const QReset &);
	QReset(Qubit *);
	QReset(std::shared_ptr<AbstractQuantumReset> node);
	std::shared_ptr<AbstractQuantumReset> getImplementationPtr();
	~QReset();

	/**
	* @brief  Get reset node qubit address
	* @return    QPanda::Qubit*  QuBit address
	*/
	Qubit * getQuBit() const;

	/**
	* @brief  Get current node type
	* @return     NodeType  current node type
	* @see  NodeType
	*/
	NodeType getNodeType() const;
};

typedef AbstractQuantumReset * (*CreateReset)(Qubit *);

/**
* @brief   Factory for class AbstractQuantumReset
* @ingroup QuantumCircuit
*/
class QResetFactory
{
public:
	void registClass(std::string name, CreateReset method);
	AbstractQuantumReset * getQuantumReset(std::string &, Qubit *);

	/**
     * @brief Get the static instance of factory 
	 * @return QResetFactory &
     */
	static QResetFactory & getInstance()
	{
		static QResetFactory  s_Instance;
		return s_Instance;
	}
private:
	std::map<std::string, CreateReset> m_reset_map;
	QResetFactory() {};

};

/**
* @brief Quantum reset register action
* @note Provide QResetFactory class registration interface for the outside
 */
class QuantumResetRegisterAction {
public:
	QuantumResetRegisterAction(std::string className, CreateReset ptrCreateFn) {
		QResetFactory::getInstance().registClass(className, ptrCreateFn);
	}

};

#define REGISTER_RESET(className)                                             \
    AbstractQuantumReset* objectCreator##className(Qubit * pQubit){      \
        return new className(pQubit);                    \
    }                                                                   \
    QuantumResetRegisterAction g_resetCreatorRegister##className(                        \
        #className,(CreateReset)objectCreator##className)

/**
* @brief Implementation  class of QReset
* @ingroup QuantumCircuit
*/
class OriginReset : public QNode, public AbstractQuantumReset
{
public:
	OriginReset(Qubit *);
	~OriginReset() {};

	/**
	* @brief  Get reset node qubit address
	* @return    QPanda::Qubit*  QuBit address
	*/
	Qubit * getQuBit() const;
	
	/**
	* @brief  Get current node type
	* @return     NodeType  current node type
	* @see  NodeType
	*/
	NodeType getNodeType() const;

private:
	OriginReset();
	OriginReset(OriginReset &);

	NodeType m_node_type;
	Qubit * m_target_qubit;
};

/**
* @brief  QPanda2 basic interface for creating a quantum Reset node
* @param[in]  Qubit*   Qubit pointer
* @return     QPanda::QReset quantum reset node
* @ingroup QuantumCircuit
*/
QReset Reset(Qubit *);


/**
* @brief  QPanda2 basic interface for creating a quantum Reset node
* @param[in]  int qubit phy addr
* @return QPanda::QReset quantum reset node
* @ingroup QuantumCircuit
*/
QReset Reset(int qaddr);

QPANDA_END
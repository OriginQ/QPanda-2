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

#ifndef QUBIT_FACTORY_H
#define QUBIT_FACTORY_H


#include "Core/QuantumMachine/PhysicalQubitFactory.h"
#include "Core/QuantumCircuit/CExprFactory.h"
#include <functional>
#include <stack>
#include <map>
#include <memory>
QPANDA_BEGIN

/**
* @brief Qubit abstract class
* @ingroup QuantumMachine
*/
class Qubit
{
public:
	/**
     * @brief Get physical qubit pointer
	 * @return PhysicalQubit *
     */
    virtual PhysicalQubit* getPhysicalQubitPtr() = 0;

	/**
	 * @brief get the occupancy status of this qubit
	 * @return PhysicalQubit *
     */
    virtual bool getOccupancy() = 0;
    virtual ~Qubit() {}
};

/**
* @brief QubitReferenceInterface abstract class
* @ingroup QuantumMachine
*/
class QubitReferenceInterface
{
public:
	virtual std::shared_ptr<CExpr>  getExprPtr() = 0;
};

#define REGISTER_QUBIT(classname) \
Qubit* classname##_Constructor(PhysicalQubit* physQ)\
{\
    return new classname(physQ);\
}\
 QubitFactoryHelper _Qubit_Factory_Helper_##classname(\
    #classname, \
    classname##_Constructor\
)



/**
* @brief Factory for class Qubit
* @ingroup QuantumMachine
*/
class QubitFactory
{
    QubitFactory();
public:
	/**
     * @brief Get the static instance of factory 
	 * @return QubitFactory &
     */
    static QubitFactory & GetFactoryInstance();
    Qubit* GetInstance(PhysicalQubit*);
    typedef std::function<Qubit*(PhysicalQubit*)> constructor_t;
    typedef std::map<std::string, constructor_t> constructor_Map_t;
    void registerclass(std::string &, constructor_t constructor);
    // the constructor stack
    constructor_Map_t _Qubit_Constructor;
};

/**
 * @brief Qubit Factory helper
 * Provide QubitFactory class registration interface for the outside
 * @ingroup QuantumMachine
 */
class QubitFactoryHelper
{
    typedef QubitFactory::constructor_t constructor_t;
public:
    QubitFactoryHelper(std::string, constructor_t);
};
QPANDA_END
#endif
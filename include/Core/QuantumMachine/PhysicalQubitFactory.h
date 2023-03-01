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

#ifndef PHYSICAL_QUBIT_FACTORY_H
#define PHYSICAL_QUBIT_FACTORY_H
#include <functional>
#include <stack>
#include <map>
#include "Core/Utilities/QPandaNamespace.h"

QPANDA_BEGIN

/**
* @brief Physical Qubit abstract class
* @ingroup QuantumMachine
*/
class PhysicalQubit
{
    // Interface for the PhysicalQubit

public:
	/**
    * @brief    get qubit address
    * @return   size_t 
    */
    virtual size_t getQubitAddr() = 0;

	/**
    * @brief    set qubit address
    * @param[in]   size_t   qubit address
    */
    virtual void setQubitAddr(size_t) = 0;
    //virtual void setAdjacentQubit(const vector<PhysicalQubit*>&) = 0;
    //virtual vector<PhysicalQubit*> getAdjacentQubit() const = 0;

	/**
    * @brief    get the occupancy status of this qubit
	* @return bool ture: occupancy
    */
    virtual bool getOccupancy() const = 0;

	/**
	* @brief    set the occupancy status of this qubit
	* @param[in] bool  occupancy status
	*/
    virtual void setOccupancy(bool) = 0;
    virtual ~PhysicalQubit() {}
};


/**
* @brief Factory for class PhysicalQubit
* @ingroup QuantumMachine
*/
class PhysicalQubitFactory
{
    // Factory for class PhysicalQubit
    PhysicalQubitFactory();
public:
	/**
     * @brief Get the static instance of factory 
	 * @return PhysicalQubitFactory &
     */
    static PhysicalQubitFactory & GetFactoryInstance();

    PhysicalQubit* GetInstance();
    typedef std::function<PhysicalQubit*()> constructor_t;
    typedef std::map<std::string, constructor_t> constructor_Map_t;
    void registerclass(std::string &, constructor_t constructor);
    // the constructor stack
    constructor_Map_t _Physical_Qubit_Constructor;
};


/**
 * @brief Physical Qubit Factory helper
 * Provide PhysicalQubitFactory class registration interface for the outside
 * @ingroup QuantumMachine
 */
class PhysicalQubitFactoryHelper
{
    typedef PhysicalQubitFactory::constructor_t constructor_t;
public:
    PhysicalQubitFactoryHelper(std::string, constructor_t);
};

#define REGISTER_PHYSICAL_QUBIT(classname)  \
PhysicalQubit * classname##_Constructor()\
{\
    return new classname();\
}\
 PhysicalQubitFactoryHelper  _Physical_Qubit_Factory_Helper_##classname(\
    #classname, \
    classname##_Constructor \
) 
QPANDA_END
#endif

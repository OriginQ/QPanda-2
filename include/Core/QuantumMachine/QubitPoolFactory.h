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

#ifndef QUBIT_POOL_FACTORY_H
#define QUBIT_POOL_FACTORY_H

#include "Core/QuantumMachine/QubitFactory.h"
#include <functional>
#include <stack>
QPANDA_BEGIN

/**
* @brief QubitPool abstract class
*   It is the container of the PhysicalQubit
* @ingroup QuantumMachine
*/
class QubitPool
{
    // Get the static information of the pool
public:
	/**
     * @brief get size of the PhysicalQubit vector
	 * @return size_t
     */
    virtual size_t getMaxQubit() const = 0;
	
	/**
     * @brief Gets the largest address in the used physical qubit 
	 * @return size_t
     */
	virtual size_t get_max_usedqubit_addr() const = 0;

	/**
     * @brief get size of the idle position
	 * @return size_t
     */
    virtual size_t getIdleQubit() const = 0;
	
	/**
     * @brief allocate a Qubit 
	 * @return Qubit*
     */
    virtual Qubit* allocateQubit() = 0;

	/**
     * @brief allocate a Qubit  through physical address
	 * @return Qubit*
     */
    virtual Qubit* allocateQubitThroughPhyAddress(size_t) = 0;
	
	/**
     * @brief allocate a Qubit  through virtual address
	 * @return Qubit*
     */
    virtual Qubit* allocateQubitThroughVirAddress(size_t qubit_num) = 0; 
	
	/**
     * @brief free a Qubit  
     */
    virtual void Free_Qubit(Qubit*) = 0;
	
	/**
     * @brief  clear the PhysicalQubit vector
     */
    virtual void clearAll() = 0;
    virtual ~QubitPool() {}
	
	/**
     * @brief get physical qubit address
	 * @param[in] Qubit*
	 * @return size_t
     */
    virtual size_t getPhysicalQubitAddr(Qubit*) = 0;

	/**
     * @brief get virtual qubit address
	 * @param[in] Qubit*
	 * @return size_t
     */
    virtual size_t getVirtualQubitAddress(Qubit*) const = 0;

    /**
     * @brief get allocate qubits 
     * @param[out] QVec&
     * @return size_t
     */
    virtual size_t get_allocate_qubits(std::vector<Qubit*>&) const = 0;
};


#define REGISTER_QUBIT_POOL_SIZE_(classname) \
QubitPool* classname##_Constructor(size_t size)\
{\
    return new classname(size);\
}\
static QubitPoolFactoryHelper _Qubit_Pool_Factory_Helper_##classname(\
    #classname,\
    classname##_Constructor\
)


/**
* @brief Factory for class QubitPool
* @ingroup QuantumMachine
*/
class QubitPoolFactory
{
    QubitPoolFactory();
public:
	/**
     * @brief Get the static instance of factory 
	 * @return QubitPoolFactory &
     */
    static QubitPoolFactory& GetFactoryInstance();
    typedef std::function<QubitPool*(size_t)> size_constructor_t;
    typedef std::map<std::string, size_constructor_t> size_constructor_stack_t;
    size_constructor_stack_t _Qubit_Pool_Constructor;
    QubitPool* GetPoolWithoutTopology(size_t);
    void registerclass_size_(std::string &, size_constructor_t constructor);
};


/**
 * @brief Qubit Pool  Factory helper
 * Provide QubitPoolFactory class registration interface for the outside
 * @ingroup QuantumMachine
 */
class QubitPoolFactoryHelper
{
    typedef QubitPoolFactory::
        size_constructor_t size_constructor_t;
public:
    QubitPoolFactoryHelper(std::string, size_constructor_t);
};
QPANDA_END
#endif
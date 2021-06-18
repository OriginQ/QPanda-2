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

#ifndef CMEM_FACTORY_H
#define CMEM_FACTORY_H

#include "Core/QuantumMachine/CBitFactory.h"
#include "Core/Utilities/QPandaNamespace.h"
#include <functional>
#include <stack>

QPANDA_BEGIN

/**
* @brief CMem abstract class, this class is considered as the container of the CBit
* @ingroup QuantumMachine
*/
class CMem
{
public:
	/**
     * @brief allocate a CBit
	 * @return CBit*
     */
    virtual CBit* Allocate_CBit() = 0;
	
	/**
     * @brief allocate a CBit by bit address
	 * @return CBit*
     */
    virtual CBit* Allocate_CBit(size_t) = 0;

	/**
     * @brief get size of the CBit vector
	 * @return size_t
     */
    virtual size_t getMaxMem() const = 0;

	/**
     * @brief get size of the idle position
	 * @return size_t
     */
    virtual size_t getIdleMem() const = 0;

	/**
     * @brief  free a CBit
	 * @param[in] CBit*
     */
    virtual void Free_CBit(CBit*) = 0;
	
	/**
     * @brief  clear the CBit vector
     */
    virtual void clearAll() = 0;

    /**
    * @brief  get allocate cbits
    * @param[out] std::vector<CBit *>&
    * @return size_t  allocate cbits size
    */
    virtual size_t get_allocate_cbits(std::vector<CBit *>&) = 0;

    virtual ~CMem() {}
};
#define REGISTER_CMEM_SIZE_(classname) \
CMem* classname##_Constructor(size_t size)\
{\
    return new classname(size);\
}\
static CMemFactoryHelper _CMem_Factory_Helper_##classname(\
    #classname,\
    classname##_Constructor\
)

/**
 * @brief Factory for class CMem
 * @ingroup QuantumMachine
 */
class CMemFactory
{
    CMemFactory();
public:
    typedef std::function<CMem*(size_t)> size_constructor_t;
    typedef std::map<std::string, size_constructor_t> size_constructor_stack_t;
    size_constructor_stack_t _CMem_Constructor;
    CMem* GetInstanceFromSize(size_t);
    void registerclass_size_(std::string &, size_constructor_t);
	
	/**
     * @brief Get the static instance of factory 
	 * @return CMemFactory &
     */
    static CMemFactory& GetFactoryInstance();
};

/**
 * @brief CMem factory helper
 * Provide CMemFactory class registration interface for the outside
 * @ingroup QuantumMachine
 */
class CMemFactoryHelper
{
    typedef CMemFactory::size_constructor_t
        size_constructor_t;
public:
    CMemFactoryHelper(std::string, size_constructor_t _Constructor);
};
QPANDA_END
#endif
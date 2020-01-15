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

#ifndef CBIT_FACTORY_H
#define CBIT_FACTORY_H

#include "Core/Utilities/QPandaNamespace.h"
#include <functional>
#include <stack>
#include <map>

QPANDA_BEGIN

typedef long long cbit_size_t;

/**
* @brief CBit abstract class
* @ingroup QuantumMachine
*/
class CBit
{
public:
	/**
     * @brief get the occupancy status of this bit
	 * @return bool ture: occupancy
     */
    virtual bool getOccupancy() const = 0; 
	
	/**
     * @brief get the name of this bit
	 * @return std::string
     */
    virtual std::string getName() const = 0;

	/**
     * @brief set the occupancy status of this bit
	 * @param[in] bool  occupancy status
     */
    virtual void setOccupancy(bool) = 0;

	/**
	 * @brief get the value of this bit
	 * @return cbit_size_t
	 */
    virtual cbit_size_t getValue() const noexcept = 0;
	
	/**
     * @brief set the value of this bit
	 * @param[in] cbit_size_t  value
     */
    virtual void setValue(const cbit_size_t) noexcept = 0;
    virtual ~CBit() {}
};

#define REGISTER_CBIT_NAME_(classname) \
CBit* classname##_Constructor(std::string name)\
{\
    return new classname(name);\
}\
static CBitFactoryHelper _CBit_Factory_Helper_##classname(\
    #classname, \
    classname##_Constructor\
)


/**
 * @brief Factory for class CBit
 * @ingroup QuantumMachine
 */
class CBitFactory
{
    CBitFactory();
public:
	/**
     * @brief Get the static instance of factory 
	 * @return CBitFactory &
     */
    static CBitFactory & GetFactoryInstance();
    typedef std::function<CBit*(std::string)> name_constructor_t;
    typedef std::map<std::string, name_constructor_t> name_constructor_stack_t;
    name_constructor_stack_t _CBit_Constructor;
    void registerclass_name_(std::string&, name_constructor_t constructor);
    CBit* CreateCBitFromName(std::string);
};


/**
 * @brief CBit factory helper
 * Provide CBitFactory class registration interface for the outside
 * @ingroup QuantumMachine
 */
class CBitFactoryHelper
{
    typedef CBitFactory::name_constructor_t
        name_constructor_t;
public:
    CBitFactoryHelper(std::string, name_constructor_t);
};
QPANDA_END
#endif
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

#ifndef QRESULT_FACTORY_H
#define QRESULT_FACTORY_H

#include <functional>
#include <stack>
#include <map>
#include "Core/Utilities/QPandaNamespace.h"
QPANDA_BEGIN

/**
* @brief QResult abstract class, this class contains the result of the quantum measurement
* @ingroup QuantumMachine
*/
class QResult
{
public:
    virtual std::map<std::string, bool> getResultMap() const = 0;
    virtual void append(std::pair<std::string, bool>) = 0;
    virtual ~QResult() {}
};

#define REGISTER_QRES_NAME(classname) \
QResult* classname##_Constructor()\
{\
    return new classname();\
}\
static QResultFactoryHelper _QRes_Factory_Helper_##classname(\
    #classname,\
    classname##_Constructor\
)


/**
* @brief Factory for class QResult
* @ingroup QuantumMachine
*/
class QResultFactory
{
    QResultFactory();
public:
    typedef std::function<QResult*()> constructor_t;
    typedef std::map<std::string, constructor_t> constructor_Map_t;
    constructor_Map_t _QResult_Constructor;
    QResult* GetEmptyQResult();
    void registerclass(std::string &, constructor_t);
	
	/**
     * @brief Get the static instance of factory 
	 * @return QResultFactory &
     */
    static QResultFactory& GetFactoryInstance();
};

/**
 * @brief QResult Factory helper
 * Provide QResultFactory class registration interface for the outside
 * @ingroup QuantumMachine
 */
class QResultFactoryHelper
{
    typedef QResultFactory::constructor_t
        constructor_t;
public:
    QResultFactoryHelper(std::string, constructor_t _Constructor);
};

QPANDA_END
#endif
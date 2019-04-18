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
class CBit
{
public:
    virtual bool getOccupancy() const = 0; // get the occupancy status of this bit
    virtual std::string getName() const = 0;
    virtual void setOccupancy(bool) = 0;
    virtual cbit_size_t getValue() const noexcept = 0;
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

/* CBit Factory */
class CBitFactory
{
    CBitFactory();
public:
    static CBitFactory & GetFactoryInstance();
    typedef std::function<CBit*(std::string)> name_constructor_t;
    typedef std::map<std::string, name_constructor_t> name_constructor_stack_t;
    name_constructor_stack_t _CBit_Constructor;
    void registerclass_name_(std::string&, name_constructor_t constructor);
    CBit* CreateCBitFromName(std::string);
};

class CBitFactoryHelper
{
    typedef CBitFactory::name_constructor_t
        name_constructor_t;
public:
    CBitFactoryHelper(std::string, name_constructor_t);
};
QPANDA_END
#endif
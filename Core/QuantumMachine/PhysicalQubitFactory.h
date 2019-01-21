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

#ifndef PHYSICAL_QUBIT_FACTORY_H
#define PHYSICAL_QUBIT_FACTORY_H

#include "QPandaNamespace.h"
#include <functional>
#include <stack>
#include<map>
QPANDA_BEGIN
class PhysicalQubit
{
    // Interface for the PhysicalQubit

public:
    virtual size_t getQubitAddr() = 0;
    virtual void setQubitAddr(size_t) = 0;
    //virtual void setAdjacentQubit(const vector<PhysicalQubit*>&) = 0;
    //virtual vector<PhysicalQubit*> getAdjacentQubit() const = 0;
    virtual bool getOccupancy() const = 0;
    virtual void setOccupancy(bool) = 0;
    virtual ~PhysicalQubit() {}
};

/*  Physical Qubit Factory*/
class PhysicalQubitFactory
{
    // Factory for class PhysicalQubit
    PhysicalQubitFactory();
public:

    static PhysicalQubitFactory & GetFactoryInstance();

    PhysicalQubit* GetInstance();
    typedef std::function<PhysicalQubit*()> constructor_t;
    typedef std::map<std::string, constructor_t> constructor_Map_t;
    void registerclass(std::string &, constructor_t constructor);
    // the constructor stack
    constructor_Map_t _Physical_Qubit_Constructor;
};

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
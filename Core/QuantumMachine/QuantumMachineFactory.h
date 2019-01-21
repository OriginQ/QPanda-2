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

#ifndef QUANTUM_MACHINE_FACTORY_H
#define QUANTUM_MACHINE_FACTORY_H

#include "QuantumMachineInterface.h"
#include "QPandaNamespace.h"
#include <functional>
#include <stack>

QPANDA_BEGIN

#define REGISTER_QUANTUM_MACHINE(classname) \
QuantumMachine* classname##_Constructor()\
{\
    return new classname();\
}\
static QuantumMachineFactoryHelper _Quantum_Machine_Factory_Helper_##classname(\
    #classname,\
    classname##_Constructor\
)

/* 2. Quantum Machine Status Factory */
class QMachineStatusFactory
{
public:
    static QMachineStatus* GetQMachineStatus();
};

/* Quantum Machine Factory */
class QuantumMachineFactory
{
    // Factory for class QuantumMachine
    QuantumMachineFactory();
public:
    static QuantumMachineFactory & GetFactoryInstance();
    typedef std::function<QuantumMachine*()> constructor_t;
    typedef std::map<std::string, constructor_t> constructor_map_t;

    // create the quantum machine by its class name
    QuantumMachine* CreateByName(std::string);

    // a constructor std::map
    constructor_map_t _Quantum_Machine_Constructor;

    // class register std::function
    void registerclass(std::string name, constructor_t constructor);
};

class QuantumMachineFactoryHelper
{
    typedef QuantumMachineFactory::constructor_t constructor_t;
public:
    QuantumMachineFactoryHelper(std::string, constructor_t);
};
QPANDA_END
#endif
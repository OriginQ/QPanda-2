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

#ifndef FACTORY_H
#define FACTORY_H

#include "Core/QuantumMachine/QuantumMachineFactory.h"
#include "Core/QuantumMachine/QubitPoolFactory.h"
#include "Core/QuantumMachine/CBitFactory.h"
#include "Core/QuantumMachine/CMemFactory.h"
#include "Core/QuantumMachine/QResultFactory.h"
#include "Core/QuantumMachine/QuantumMachineInterface.h"
QPANDA_BEGIN
class QMachineTypeTarnfrom
{
public:
    static QMachineTypeTarnfrom &getInstance()
    {
        static QMachineTypeTarnfrom temp;
        return temp;
    }
    ~QMachineTypeTarnfrom() {};
    std::string operator [](QMachineType type)
    {
        auto iter =m_qmachine_type_map.find(type);
        if (iter == m_qmachine_type_map.end())
        {
            QCERR("quantum machine type error ");
            throw std::invalid_argument("quantum machine type error");
        }
        return iter->second;
    }
    QMachineType operator [](std::string gate_name)
    {
        for (auto & aiter : m_qmachine_type_map)
        {
            if (gate_name == aiter.second)
            {
                return aiter.first;
            }
        }
        QCERR("quantum machine type error ");
        throw std::invalid_argument("quantum machine type error");
    }
private:
    std::map<QMachineType,std::string> m_qmachine_type_map;
    QMachineTypeTarnfrom &operator=(const QMachineTypeTarnfrom &);
    QMachineTypeTarnfrom()
    {
        m_qmachine_type_map = { {QMachineType::CPU,"CPUQVM"},
        {QMachineType::CPU_SINGLE_THREAD,"CPUSingleThreadQVM"},
        {QMachineType::GPU,"GPUQVM"},
        {QMachineType::NOISE,"NoiseQVM"} };

    }
    QMachineTypeTarnfrom(const QMachineTypeTarnfrom &);
};

QPANDA_END
#endif
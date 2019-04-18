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

#ifndef _QUANTUM_GATE_PARAM_H
#define _QUANTUM_GATE_PARAM_H
#include <mutex>
#include <iostream>
#include <map>
#include <complex>
#include <algorithm>
#include <vector>
#include "Core/QuantumMachine/QubitFactory.h"
#include "Core/Utilities/QPandaNamespace.h"

#define iunit qcomplex_t(0,1)
/*****************************************************************************************************************
QuantumGateParam:quantum gate param 
*****************************************************************************************************************/
class QuantumGateParam
{
 public:
    size_t m_qubit_number; /* quantum bit number */
    std::map<std::string , bool> m_return_value; /* MonteCarlo result */
    bool m_is_dagger;
    std::vector<QPanda::Qubit *> m_control_qubit_vector;

    QuantumGateParam()
    {
        m_qubit_number = 0;
        m_is_dagger = false;
    }
};

struct QGateParam
{
    QGateParam() {};
    QGateParam(int qn) :qVec(qn, 0), qstate(1ull << qn, 0), qubitnumber(qn)
    {
        for (auto i = 0; i < qubitnumber; i++)
        {
            qVec[i] = i;
        }
        qstate[0] = 1;
    }
    Qnum qVec;
    QStat qstate;
    int qubitnumber;
    bool enable = true;

};

#endif

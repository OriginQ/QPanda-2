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
#ifndef _BERNSTEIN_VAZIRANI_ALGORITHM_H
#define _BERNSTEIN_VAZIRANI_ALGORITHM_H

#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/Utilities/Tools/Utils.h"
#include <vector>
#include <bitset>

QPANDA_BEGIN

using BV_Oracle = Oracle<QVec, Qubit *>;

/**
* @brief  Bernstein Vazirani algorithm
* @ingroup B_V_Algorithm
* @param[in] std::string Hidden string, is equal of "a" in "f(x)=a*x+b"
* @param[in] bool is equal of "b" in "f(x)=a*x+b"
* @param[in] QuantumMachine* Quantum machine ptr
* @param[in] BV_Oracle Deutsch Jozsa algorithm oracle
* @return    QProg
* @note
*/
QProg bernsteinVaziraniAlgorithm(std::string stra, bool b, QuantumMachine * qvm, BV_Oracle oracle);
QPANDA_END
#endif

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

#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/Utilities/Tools/Utils.h"
#include "Core/Utilities/QPandaNamespace.h"
#include <vector>
QPANDA_BEGIN
using DJ_Oracle = Oracle<QVec, Qubit*>;

/**
* @brief  Deutsch Jozsa algorithm
* @ingroup DJ_Algorithm
* @param[in] std::vector<bool> boolean_function{f(0)= (0/1)?, f(1)=(0/1)?}
* @param[in] QuantumMachine* Quantum machine ptr
* @param[in] DJ_Oracle Deutsch Jozsa algorithm oracle
* @return    QProg
* @note    In the Deutsch-Jozsa problem, we are given a black box quantum computer 
           known as an oracle that implements some function f:{0,1}^{n}->{0,1}.
           The function takes n-digit binary values as input and produces either
		   a 0 or a 1 as output for each such value. We are promised that the 
		   function is either constant (0 on all outputs or 1 on all outputs) or 
		   balanced (returns 1 for half of the input domain and 0 for the 
		   other half); the task then is to determine if f is constant or 
		   balanced by using the oracle.
*/
QProg deutschJozsaAlgorithm(std::vector<bool> boolean_function, QuantumMachine * qvm, DJ_Oracle oracle);
QPANDA_END
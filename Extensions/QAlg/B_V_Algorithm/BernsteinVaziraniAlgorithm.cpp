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
#include "Core/Utilities/Tools/Utils.h"
#include "QAlg/B_V_Algorithm/BernsteinVaziraniAlgorithm.h"
#include "Core/Core.h"
#include <bitset>


using namespace std;
using namespace QPanda;


QProg BV_QProg(QVec qVec, vector<ClassicalCondition> cVec, vector<bool>& a,
    BV_Oracle & oracle)
{

    if (qVec.size() != (a.size()+1))
    {
        QCERR("param error");
        throw invalid_argument("param error");
    }
    size_t length = qVec.size();
    QProg  bv_qprog = CreateEmptyQProg();
    bv_qprog << X(qVec[length - 1])
             << apply_QGate(qVec, H)
             << oracle(qVec, qVec[length - 1]);
             
    qVec.pop_back();

    bv_qprog << apply_QGate(qVec, H) << MeasureAll(qVec, cVec);
    return bv_qprog;
}

QProg QPanda::bernsteinVaziraniAlgorithm(std::string stra,bool b,QuantumMachine * qvm, BV_Oracle oracle)
{
	vector<bool> a;
	for (auto iter = stra.begin(); iter != stra.end(); iter++)
	{
		if (*iter == '0')
		{
			a.push_back(0);
		}
		else
		{
			a.push_back(1);
		}
	}

	size_t qubitnum = a.size();
	vector<Qubit*> qVec = qvm->allocateQubits(qubitnum + 1);
	auto cVec = qvm->allocateCBits(qubitnum);
	auto bvAlgorithm = BV_QProg(qVec, cVec, a, oracle);
	return bvAlgorithm;
}

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

#include "QAlg/Utils/Utilities.h"
USING_QPANDA
QProg  QPanda::Reset_Qubit_Circuit(Qubit *q, ClassicalCondition& cbit, bool setVal)
{
    auto prog = CreateEmptyQProg();
    prog << Measure(q, cbit);
    auto resetcircuit = CreateEmptyCircuit();
    resetcircuit << X(q);
    auto no_reset = CreateEmptyCircuit();
    if (setVal == false)
        prog << CreateIfProg(cbit, &resetcircuit, &no_reset);
    else
        prog << CreateIfProg(cbit, &no_reset, &resetcircuit);
    return prog;
}

QProg QPanda::Reset_Qubit(Qubit* q, bool setVal)
{
    auto cbit = cAlloc();
    auto aTmep = Reset_Qubit_Circuit(q, cbit, setVal);
    return aTmep;
}

QProg QPanda::Reset_All(std::vector<Qubit*> qubit_vector, bool setVal)
{

    QProg temp;

    for_each(qubit_vector.begin(),
        qubit_vector.end(),
        [setVal,&temp](Qubit* q) {temp <<Reset_Qubit(q, setVal); });

    return temp;
}

QCircuit QPanda::parity_check_circuit(std::vector<Qubit*> qubit_vec)
{
    QCircuit circuit;
    for (auto i = 0; i < qubit_vec.size() - 1; i++)
    {
        circuit << CNOT(qubit_vec[i], qubit_vec[qubit_vec.size() - 1]);
    }
    return circuit;
}

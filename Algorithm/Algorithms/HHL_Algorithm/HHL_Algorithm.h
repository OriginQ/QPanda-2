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

#ifndef HHL_ALGORITHM_H
#define HHL_ALGORITHM_H

#include "QPanda.h"

QPANDA_BEGIN
//solve linaer system equation Ax=b
//matrix A=[1.5 0.5;0.5 1.5]
USING_QPANDA
void HHL_Algorithm();
void new_HHL_Algorithm();
void new_HHL_Algorithm_PMeasure();
map<string, bool> hhlalgorithm();

int HHL_Test(int);
QProg hhl(vector<Qubit*> qVec, vector<CBit*> cVec);
QProg hhl_no_measure(vector<Qubit*> qVec, vector<CBit*> cVec);
QCircuit hhlPse(vector<Qubit*> qVec);
QCircuit CRotate(vector<Qubit*> qVec);

QPANDA_END
#endif
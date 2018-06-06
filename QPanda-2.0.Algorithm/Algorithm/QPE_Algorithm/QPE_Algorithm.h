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

/******************************************************************************
  There is no 
  "OriginQCircuit & operator << (const QMeasureNode &);"
  function yet, QPE algorithm doesnot work.
  Mr.Dou is adding this function.
******************************************************************************/
#ifndef QPE_ALGORITHM_H
#define QPE_ALGORITHM_H

#include "QPanda.h"
#include "../../QAlgorithm.h"
#define QGEN  function<OriginQCircuit&(vector<Qubit*>)> 
OriginQCircuit& QFT(vector<Qubit*> qvec);
OriginQCircuit& QFTdagger(vector<Qubit*> qvec);
OriginQCircuit& unitarypower(vector<Qubit*> qvec, size_t min, QGEN qc);
OriginQCircuit& QPE(vector<Qubit*> controlqvec, vector<Qubit*> targetqvec,QGEN);
OriginQCircuit& Hadamard(vector<Qubit*> qvec);
void QPE_AlgorithmTest();
#endif
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
//Bernstein Vazirani Algorithm: f(x)=ax+b mod 2;
//                            a,x:{0,1}^n,b:{0,1}
//questions: find a and b;
#ifndef SIMONALGORITHM_H
#define SIMONALGORITHM_H

#include "QPanda.h"
#include "../../QAlgorithm.h"

//4-qubit Simon's Algorithm
QProg Simon_QProg(vector<Qubit*> qVec, vector<CBit*> cVec, vector<int> funvalue);
QCircuit oraclefunc(vector<Qubit*> qVec, vector<int> funvalue);
QCircuit controlfunc(vector<Qubit*> qVec, size_t index, int value);
void simonTest();
//int simonExecution(vector<Qubit*> qVec, vector<CBit*> cVec, QProg pro);
//typedef vector<vector<bool>> vvb;
//void SimonAlgorithm();
//
////QProg BV_Algorithm(vector<Qubit*> qVec, vector<CBit*> cVec,vector<bool> & ,bool );
//QCircuit SimonOracle(vector<Qubit*> qVec, vector<bool> oracle);
//QProg Simon_QProg(vector<Qubit*> qVec, vector<Qubit*> cVec, vvb& func);

#endif
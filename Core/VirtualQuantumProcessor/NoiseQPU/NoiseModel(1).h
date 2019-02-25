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
#ifndef NOISE_MODEL_H
#define NOISE_MODEL_H
#include "Core/VirtualQuantumProcessor/QPUImpl.h"
#include "Core/VirtualQuantumProcessor/NoiseQPU/NoisyQPU.h"
#include "Core/Utilities/QPandaNamespace.h"
#define NoiseOp std::vector<std::vector<qcomplex_t>>
QStat matrix_tensor(const QStat &matrix_left, const QStat &matrix_right);
bool damping_kraus_operator(double probability, NoiseOp & noise);
bool dephasing_kraus_operator(double probability, NoiseOp & noise);
bool decoherence_kraus_operator(double t1, double t2, double gate_time, NoiseOp & noise);
bool pauli_kraus_map(std::vector<double> probability, NoiseOp & noise);


#endif  // ! NOISE_MODEL_H
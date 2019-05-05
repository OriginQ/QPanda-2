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
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/VirtualQuantumProcessor/NoiseQPU/NoiseCPUImplQPU.h"
#include "ThirdParty/rapidjson/document.h"

enum NOISE_MODEL
{
    DAMPING_KRAUS_OPERATOR,
    DEPHASING_KRAUS_OPERATOR,
    DECOHERENCE_KRAUS_OPERATOR,
    PAULI_KRAUS_MAP,
    DOUBLE_DAMPING_KRAUS_OPERATOR,
    DOUBLE_DEPHASING_KRAUS_OPERATOR,
    DOUBLE_DECOHERENCE_KRAUS_OPERATOR
};

#define NoiseOp std::vector<std::vector<qcomplex_t>>
QStat matrix_tensor(const QStat &matrix_left, const QStat &matrix_right);
bool damping_kraus_operator(rapidjson::Value &, NoiseOp & noise);
bool dephasing_kraus_operator(rapidjson::Value &, NoiseOp & noise);
bool decoherence_kraus_operator(rapidjson::Value &, NoiseOp & noise);
bool double_damping_kraus_operator(rapidjson::Value &, NoiseOp & noise);
bool double_decoherence_kraus_operator(rapidjson::Value &, NoiseOp & noise);
bool pauli_kraus_map(rapidjson::Value &, NoiseOp & noise);

typedef bool(*noise_mode_function)(rapidjson::Value &, NoiseOp &);
class NoiseModeMap
{
public:
    static NoiseModeMap &getInstance();
    ~NoiseModeMap() {};
    noise_mode_function operator [](NOISE_MODEL);
private:
    std::map<NOISE_MODEL, noise_mode_function> m_function_map;
    NoiseModeMap &operator=(const NoiseModeMap &) = delete;
    NoiseModeMap();
    NoiseModeMap(const NoiseModeMap &) = delete;
};
#endif  // ! NOISE_MODEL_H

/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.

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


/**
* @brief noise model type
*/
enum NOISE_MODEL
{
	DAMPING_KRAUS_OPERATOR,
	DEPHASING_KRAUS_OPERATOR,
    DECOHERENCE_KRAUS_OPERATOR_P1_P2,
    BITFLIP_KRAUS_OPERATOR,
    DEPOLARIZING_KRAUS_OPERATOR,
    BIT_PHASE_FLIP_OPRATOR,
    PHASE_DAMPING_OPRATOR,
	DECOHERENCE_KRAUS_OPERATOR,
	PAULI_KRAUS_MAP,

	KRAUS_MATRIX_OPRATOR,
	MIXED_UNITARY_OPRATOR,
};

#define NoiseOp std::vector<std::vector<qcomplex_t>>
QStat matrix_tensor(const QStat &matrix_left, const QStat &matrix_right);
bool damping_kraus_operator(rapidjson::Value &, NoiseOp & noise);
bool dephasing_kraus_operator(rapidjson::Value &, NoiseOp & noise);
bool decoherence_kraus_operator(rapidjson::Value &, NoiseOp & noise);
bool double_damping_kraus_operator(rapidjson::Value &, NoiseOp & noise);
bool double_decoherence_kraus_operator(rapidjson::Value &, NoiseOp & noise);
bool pauli_kraus_map(rapidjson::Value &, NoiseOp & noise);

bool decoherence_kraus_operator_p1_p2(rapidjson::Value &value, NoiseOp & noise);
bool bitflip_kraus_operator(rapidjson::Value &value, NoiseOp & noise);
bool depolarizing_kraus_operator(rapidjson::Value &value, NoiseOp & noise);

/**
* @brief  Get Noise model bit-phase flip matrix
* @ingroup VirtualQuantumProcessor
* @param[in]  rapidjson::Value  Noise model and probability
* @param[out]  NoiseOp  Noise model matrix: E1 = sqrt(1-p){1,0,0,1}, E2 = sqrt(p) {0,-i,i,0}
* @return  bool true:get matrix success,  false:get matrix failed
* @note    Use this at the SingleGateNoiseModeMap constructor
*/
bool bit_phase_flip_operator(rapidjson::Value &value, NoiseOp & noise);

/**
* @brief  Get Noise model bit-phase flip matrix
* @ingroup VirtualQuantumProcessor
* @param[in]  rapidjson::Value  Noise model and probability
* @param[out]  NoiseOp  Noise model matrix: E1 = {1,0,0,sqrt(1-p)} , E2 = {0,0,0,sqrt(p)}
* @return  bool true:get matrix success,  false:get matrix failed
* @note    Use this at the SingleGateNoiseModeMap constructor
*/
bool phase_damping_oprator(rapidjson::Value &value, NoiseOp &noise);

bool double_decoherence_kraus_operator_p1_p2(rapidjson::Value &value, NoiseOp & noise);
bool double_bitflip_kraus_operator(rapidjson::Value &value, NoiseOp & noise);
bool double_depolarizing_kraus_operator(rapidjson::Value &value, NoiseOp & noise);

/**
* @brief  Get Noise model bit-phase flip matrix
* @ingroup VirtualQuantumProcessor
* @param[in]  rapidjson::Value  Noise model and probability
* @param[out]  NoiseOp  Noise model matrix: E1 = sqrt(1-p){1,0,0,1}, E2 = sqrt(p) {0,-i,i,0}
* @return  bool true:get matrix success,  false:get matrix failed
* @note    Use this at the DoubleGateNoiseModeMap constructor
*/
bool double_bit_phase_flip_operator(rapidjson::Value &value, NoiseOp & noise);
/**
* @brief  Get Noise model bit-phase flip matrix
* @ingroup VirtualQuantumProcessor
* @param[in]  rapidjson::Value  Noise model and probability
* @param[out]  NoiseOp  Noise model matrix: E1 = {1,0,0,sqrt(1-p)}, E2 = {0,0,0,sqrt(p)}
* @return  bool true:get matrix success,  false:get matrix failed
* @note    Use this at the DoubleGateNoiseModeMap constructor
*/
bool double_phase_damping_oprator(rapidjson::Value &value, NoiseOp &noise);

bool kraus_matrix_oprator(rapidjson::Value &value, NoiseOp &noise);

typedef bool(*noise_mode_function)(rapidjson::Value &, NoiseOp &);

/**
* @brief  Single gate noise mode map
* @ingroup VirtualQuantumProcessor
*/
class SingleGateNoiseModeMap
{
public:
    static SingleGateNoiseModeMap &getInstance();
    ~SingleGateNoiseModeMap() {};
    noise_mode_function operator [](NOISE_MODEL);
private:
    std::map<NOISE_MODEL, noise_mode_function> m_function_map;
	SingleGateNoiseModeMap &operator=(const SingleGateNoiseModeMap &) = delete;
	SingleGateNoiseModeMap();
	SingleGateNoiseModeMap(const SingleGateNoiseModeMap &) = delete;
};

/**
* @brief  Double gate noise mode map
* @ingroup VirtualQuantumProcessor
*/
class DoubleGateNoiseModeMap
{
public:
	static DoubleGateNoiseModeMap &getInstance();
	~DoubleGateNoiseModeMap() {};
	noise_mode_function operator [](NOISE_MODEL);
private:
	std::map<NOISE_MODEL, noise_mode_function> m_function_map;
	DoubleGateNoiseModeMap &operator=(const DoubleGateNoiseModeMap &) = delete;
	DoubleGateNoiseModeMap();
	DoubleGateNoiseModeMap(const DoubleGateNoiseModeMap &) = delete;
};

#endif  // ! NOISE_MODEL_H

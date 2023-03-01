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

#ifndef _RANDOM_CIRCUIT_H
#define _RANDOM_CIRCUIT_H

#include <vector>
#include <functional>
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumMachine/QuantumMachineInterface.h"
#include "Core/QuantumCircuit/QCircuit.h"
QPANDA_BEGIN


/**
* @class RandomCircuit
* @ingroup Utilities
* @brief Generate random quantum circuit
*/
class RandomCircuit
{
public:
	struct QubitInformation
	{
		int x = 0;
		int y = 0;
		bool has_T = false;
		int gate_type = 0;
	};

	using LayerInfo = std::vector<std::vector<QubitInformation > >;

	using SetLayerFunc = std::function< bool(int, int, LayerInfo&) > ;

private:
	QVec &m_qv;
	QuantumMachine *m_qvm;

	QProg m_prog;  /**< out quantum prog */
	std::string m_originr; /**< out originIR */

	std::vector<SetLayerFunc> m_set_layer_func_vec;

public:
	RandomCircuit(QuantumMachine *qvm, QVec &qv);

	~RandomCircuit();

	void  random_circuit(int qbitRow, int qbitColumn, int depth);

	std::string get_random_originir();

	QProg get_random_qprog();

private:
	int get_middle_qubit(int qbitRow, int qbitColumn);

	bool is_greater_than_middle(int target_1, int target_2, int middle);

	bool is_need_break_up(int middle, int qbitRow, int qbitColumn, LayerInfo& layer);

	bool set_layer_type_1(int qbitRow, int qbitColumn, LayerInfo& layer);
	bool set_layer_type_2(int qbitRow, int qbitColumn, LayerInfo& layer);
	bool set_layer_type_3(int qbitRow, int qbitColumn, LayerInfo& layer);
	bool set_layer_type_4(int qbitRow, int qbitColumn, LayerInfo& layer);
	bool set_layer_type_5(int qbitRow, int qbitColumn, LayerInfo& layer);
	bool set_layer_type_6(int qbitRow, int qbitColumn, LayerInfo& layer);
	bool set_layer_type_7(int qbitRow, int qbitColumn, LayerInfo& layer);
	bool set_layer_type_8(int qbitRow, int qbitColumn, LayerInfo& layer);

	void set_layer_one_qubit_gate(int qbitRow, int qbitColumn, LayerInfo & layer, LayerInfo& previous_layer);

	void set_all_hadamard(int qbitRow, int qbitColumn, LayerInfo& layer);

	void create_one_layer(int qbitRow, int qbitColumn, LayerInfo &grid);

	void generate_circuit_info(int qbitRow, int qbitColumn, int depth, std::vector<LayerInfo>& qubit_information_vector);

	void generate_random_circuit(std::vector<LayerInfo> circuit,  int qbitRow, int qbitColumn);
};

/**
* @brief   Generate random OriginIR
* @ingroup Utilities
* @param[in]  int qubit row
* @param[in]  int qubit column
* @param[in]  int depth
* @param[in]  QuantumMachine* quantum machine
* @param[out]  QVec qubit vector
* @return  std::string   OriginIR instruction set
*/
std::string random_originir(int qubitRow, int qubitCol, int depth, QuantumMachine *qvm, QVec &qv);

/**
* @brief  Generate random quantum program
* @ingroup Utilities
* @param[in]  int qubit row
* @param[in]  int qubit column
* @param[in]  int depth
* @param[in]  QuantumMachine* quantum machine
* @param[out]  QVec qubit vector
* @return QProg	  Quantum Program
*/
QProg random_qprog(int qubitRow, int qubitCol, int depth, QuantumMachine *qvm, QVec &qv);

/**
* @brief  Generate random quantum circuit
* @ingroup Utilities
* @param[in]  const QVec& target qubit-vector
* @param[in]  const std::vector<std::string>& Custom logic gate type, The support types include: 
              "X", "Y", "Z", "RX", "RY", "RZ", "S", "T", "H", "CNOT", "CZ"
			  Note: All types are used by default
* @param[in]  int depth, default 100
* @return QCircuit	random quantum circuit
*/
QCircuit random_qcircuit(const QVec& qv, int depth = 100, const std::vector<std::string>& gate_type = {});


QPANDA_END

#endif // !_RANDOM_CIRCUIT_H
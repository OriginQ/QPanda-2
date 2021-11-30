#ifndef  _GROVER_FRAME_H
#define  _GROVER_FRAME_H

#include <vector>
#include "Core/Core.h"
#include "Core/Utilities/Tools/Utils.h"
#include "QAlg/Grover/GroverAlgorithm.h"
#include "QAlg/Grover/QuantumWalkGroverAlg.h"
#include "QAlg/Base_QCircuit/AmplitudeEncode.h"

QPANDA_BEGIN

/**
* @brief  Grover Algorithm
* @ingroup grover amplify operate
* @param[in] QVec&  amplify quantum number
* @param[in] QCircuit input circuit
* @return    QCircuit of amplify operate
* @note
*/
QCircuit grover_amplify_operate(QVec& q_us,
	QCircuit in_operate)
{
	QVec controller(q_us.begin(), q_us.end() - 1);

	QCircuit cir_diff;
	int num = q_us.size();
	if (num == 0)
	{
		std::cout << "Error: Working qubits size." << std::endl;
	}
	else if (num == 1)
	{
		cir_diff << X(q_us);
		cir_diff << Z(q_us);
		cir_diff << X(q_us);
	}
	else
	{
		cir_diff << X(q_us);
		cir_diff << Z(q_us.back()).control(controller);
		cir_diff << X(q_us);
	}

	QCircuit cir_amp;

	cir_amp << in_operate.dagger();
	cir_amp << cir_diff;
	cir_amp << in_operate;

	return cir_amp;
}

/**
* @brief  Grover Algorithm
* @ingroup grover mark data flip operate
* @param[in] QVec& flip operate quantum number
* @param[in] std::vector<std::string> mark data
* @return    QCircuit
* @note
*/
QCircuit grover_mark_data_flip(QVec& flip_qubit,
	std::vector<std::string> mark_data)
{
	int n_len = end(mark_data) - begin(mark_data);
	QCircuit cir_flip;
	QVec controller(flip_qubit.begin(), flip_qubit.end() - 1);

	for (int i = 0; i < n_len; i = i + 1)
	{
		//string data = mark_data[i];

		int  num = mark_data[i].size();
		for (int j = 0; j < num; j = j + 1)
		{
			if ('0' == mark_data[i][j])
			{
				int n = num - 1 - j;
				cir_flip << X(flip_qubit[n]);
			}
		}

		cir_flip << Z(flip_qubit.back()).control(controller);

		for (int j = 0; j < num; j = j + 1)
		{
			if ('0' == mark_data[i][j])
			{
				int n = num - 1 - j;
				cir_flip << X(flip_qubit[n]);
			}
		}
	}

	return cir_flip;
}


/**
* @brief  Grover Algorithm
* @ingroup grover_search
* @param[in] QCircuit input circuit
* @param[in] std::vector<std::string> mark data
* @param[in] const QVec&  input quantum number
* @param[in] QuantumMachine* the quantum virtual machine
* @return    QProg
* @note
*/

QProg grover_search(QCircuit in_operate,
	QCircuit state_operate,
	std::vector<std::string> mark_data,
	const QVec& data_qubits,
	const QVec& in_qubits,
	QuantumMachine* qvm)
{
	int n_len = end(mark_data) - begin(mark_data);

	// initial input qubits/flip qubits/ amplify qubits
	QVec q_input = in_qubits;
	QVec q_flip = data_qubits;
	QVec q_us = in_qubits;


	//initial state prepare   
	QCircuit circuit_prepare;
	if (in_operate.is_empty())
		circuit_prepare = H(in_qubits);
	else
		circuit_prepare << in_operate;

	// initial mark data
	QCircuit cir_flip;

	cir_flip = grover_mark_data_flip(q_flip, mark_data);

	// Amplify Operate

	if (cir_flip.is_empty())
		cir_flip << Z(q_flip.back());

	QCircuit cir_amp = grover_amplify_operate(q_input, circuit_prepare);


	//repeat  
	QProg grover_prog;

	double search_num = n_len;
	int Num = in_qubits.size();
	const double max_repeat = floor(PI * sqrt((2 << Num) / (search_num)) / 4.0);
	// cout << max_repeat << endl;

		/*QProg grover;
		for (size_t i = 0; i < max_repeat; ++i)
		{
			grover << cir_flip << cir_amp;
		}

		grover_prog << circuit_prepare << grover;*/


	grover_prog << circuit_prepare << state_operate << cir_flip << state_operate.dagger() << cir_amp;

	return grover_prog;

}



QPANDA_END

#endif
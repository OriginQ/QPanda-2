#include "Core/Core.h"
#include "Core/Utilities/Tools/Utils.h"
#include "QAlg/Base_QCircuit/AmplitudeEncode.h"
#include "QAlg/Base_QCircuit/base_circuit.h"


USING_QPANDA
using namespace std;
using namespace QPanda;

QCircuit grover_amplify_operate(QVec& q_us,
	QCircuit in_operate)
{
	QVec controller(q_us.begin(), q_us.end() - 1);

	QCircuit cir_diff;
	int num = q_us.size();
	if (num == 0)
	{
		cout << "Error: Working qubits size." << endl;
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


QCircuit grover_mark_data_flip(QVec& flip_qubit,
	std::vector<std::string> mark_data)
{
	int n_len = end(mark_data) - begin(mark_data);
	QCircuit cir_flip;
	QVec controller(flip_qubit.begin(), flip_qubit.end() - 1);

	for (int i = 0; i < n_len; i = i + 1)
	{
		string data = mark_data[i];

		int  num = data.size();
		for (int j = 0; j < num; j = j + 1)
		{
			if ('0' == data[j])
			{
				int n = num - 1 - j;
				cir_flip << X(flip_qubit[n]);
			}
			else
			{
				int n = num - 1 - j;
				cir_flip << BARRIER(flip_qubit[n]);
			}
		}

		cir_flip << Z(flip_qubit.back()).control(controller);

		for (int j = 0; j < num; j = j + 1)
		{
			if ('0' == data[j])
			{
				int n = num - 1 - j;
				cir_flip << X(flip_qubit[n]);
			}
			else
			{
				int n = num - 1 - j;
				cir_flip << BARRIER(flip_qubit[n]);
			}
		}
	}

	return cir_flip;
}


QProg grover_search(QCircuit in_operate,
	QCircuit state_operate,
	std::vector<std::string> mark_data,
	const QVec& in_qubits,
	QuantumMachine* qvm)
{
	int n_len = end(mark_data) - begin(mark_data);

	// initial input qubits/flip qubits/ amplify qubits
	QVec q_input = in_qubits;
	QVec q_flip = in_qubits;
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
	const double max_repeat = floor(PI * sqrt((1 << Num) / (search_num)) / 4.0);

	 QProg grover;
	for (size_t i = 0; i < max_repeat; ++i)
	{
        grover << state_operate << cir_flip << state_operate.dagger() << cir_amp;
	}

	grover_prog << circuit_prepare << grover;
	return grover_prog;
}


vector<double> nor_data_operate(vector<double>& data)
{
	int n = data.size();
	vector<double> res(n, 0);
	vector<double> nor_data(n, 0);
	double sum = 0;
	for (int i = 0; i < n; ++i)
	{
		res[i] = data[i] * data[i];
		sum = sum + res[i];
	}

	for (int i = 0; i < n; ++i)
	{
		double num;
		num = res[i] / sum;
		nor_data[i] = sqrt(num);
	}

	return nor_data;
}

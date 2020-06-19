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

#ifndef _QSVM_ALGORITHM_H
#define _QSVM_ALGORITHM_H

#include <vector>
#include <string>
#include <map>
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumMachine/QuantumMachineInterface.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"
QPANDA_BEGIN


/**
* @brief  Quantum Support Vector Machines
* @ingroup QAlg
*/
class QSVM
{
private:
	CPUQVM *m_qvm;
	QVec m_qv;
	std::vector<ClassicalCondition> m_cv;

	std::vector<double>  m_y;
	std::vector<std::vector<double > > m_x;

	std::vector<double> m_u_coefficient;
	std::vector<std::vector<double > > m_u_vector;

	int m_oracle_qubits;
	int m_swap_qubits;
	int m_psi_position;
	int m_phi_position;
	int m_swap_position;

public:
	QSVM(std::vector<std::vector<double > > data);

	~QSVM();

	bool run(std::vector<double> query_x);

private:

	std::vector<double > solve(std::vector<std::vector<double > > x, std::vector<double >y);

	double construct_qcircuit(QVec qv, std::vector<ClassicalCondition> cv, std::vector<double> a, std::vector<double> b);

	std::vector<std::vector<double > > get_x_vector(std::vector<std::vector<double > > matrix);

	std::vector<double> get_coefficient(std::vector<std::vector<double > > matrix);

	QCircuit get_number_circuit(QVec qlist, int position, int number, int qubit_number);

	std::vector<std::vector<double > > encode_matrix(std::vector<double >  flat_matrix);

	QCircuit prepare_state(QVec qlist, int position, std::vector<double> values);

	QCircuit training_data_oracle(QVec qlist, int position, std::vector<double> coe_vector, std::vector<std::vector<double > > x_vector);

	QCircuit construct_state_psi(QVec qlist, int position, std::vector<double> u_coefficient, std::vector<std::vector<double > >u_vector,
		std::vector<double > x_coefficient, std::vector<std::vector<double > >x_vector, int oracle_qubits);

	QCircuit construct_state_phi(QVec qlist, int position);

	QCircuit swap_test_p(QVec qlist, int position, int swap_qubits);

	QCircuit construct_circuit(QVec qlist, std::vector<double> x_coefficient, std::vector<std::vector<double>> x_vector);

	void preprocess_input_x(std::vector<double> query_x, std::vector<std::vector<double>> & extend_x, std::vector<double> &extend_x_coefficient);

	double predict(QVec qlist, std::vector<ClassicalCondition> clist, std::vector<double>  query_x);

};

/**
* @brief  Quantum Support Vector Machines Algorithm
* @ingroup QAlg
* @param[in] std::vector<std::vector<double >> training sample data
* @param[in] std::vector<double> query data
* @return bool  true : belongs to the first category; false :Belongs to the second category
*/
bool qsvm_algorithm(std::vector<std::vector<double > > data, std::vector<double> query_x);


QPANDA_END


#endif // !_QSVM_ALGORITHM_H
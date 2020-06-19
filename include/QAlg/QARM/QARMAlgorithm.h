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

#ifndef _QARM_ALGORITHM_H
#define _QARM_ALGORITHM_H

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
* @brief  Quantum Association Rules Mining 
* @ingroup QAlg
*/
class QARM
{
private:
	std::vector <std::vector<std::string> > transaction_data;
	int transaction_number;
	int items_length;
	std::vector<int> items;
	std::map<int, std::string> items_dict;
	std::vector <std::vector <int > > transaction_matrix;
	int items_qubit_number;
	int transaction_qubit_number;
	int index_qubit_number;
	int digit_qubit_number;
	CPUQVM *m_qvm;

public:
	QARM(std::vector <std::vector<std::string> > data);

	~QARM();

	std::map<std::string, double> run();

private:
	int get_qubit_number(int number);

	QCircuit get_number_circuit(QVec qlist, int position, int number, int qubit_number);

	std::vector<int> create_c1(std::vector <std::vector <int > > data);

	QCircuit encode_circuit(QVec qlist, int position, int index_qubit_number, int items_length, int transaction_number);

	QCircuit query_circuit(QVec qlist, int position, int target_number);

	QCircuit transfer_to_phase(QVec qlist, int position);

	QCircuit oracle_cir(QVec qlist, int position, int locating_number);

	QCircuit coin_cir(QVec qlist, int position);

	QCircuit gk_cir(QVec qlist, int position, int locating_number);

	prob_dict iter_cir(QVec qlist, std::vector<ClassicalCondition> clist, int position, int locating_number, int iter_number);

	int iter_number();

	std::vector<std::vector<int>> get_result(QVec qlist, std::vector<ClassicalCondition> clist, int position, int locating_number, int iter_number);

	std::vector<std::vector<int>> get_index(std::vector<int> index);

	void find_f1(QVec qlist, std::vector<ClassicalCondition> clist, int position, std::vector<int> c1, double min_support,
		std::vector<std::vector<int> > &f1, std::map<std::vector<int>, std::pair<std::vector<int>, double> > &f1_dict);

	void find_fk(int k, std::vector<std::vector<int > > &fk, std::map<std::vector<int>, std::pair<std::vector<int>, double> > &fk_dict, double min_support = 0.4);

	void fk_result(QVec qlist, std::vector<ClassicalCondition> clist, int position, double min_support,
		std::vector<std::vector<std::vector<int > > > &fn, std::map<std::vector<int>, std::pair<std::vector<int>, double> > &fn_dict);

	double conf_x_y(double supp_xy, double supp_x);

	std::map<std::string, double>  get_all_conf(QVec qlist, std::vector<ClassicalCondition> clist, int position, double min_conf);

	std::string get_conf_key(std::vector<int> cause, std::vector<int> effect);
};



/**
* @brief  Quantum Association Rules Mining Algorithm
* @ingroup QAlg
* @param[in] std::vector<std::vector<std::string>> Transaction set data
* @return  confidence coefficient result
*/
std::map<std::string, double> qarm_algorithm(std::vector<std::vector<std::string>> data);

QPANDA_END


#endif // !_QARM_ALGORITHM_H
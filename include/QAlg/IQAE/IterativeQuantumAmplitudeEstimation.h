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

#ifndef _ITERATIVE_QUAMTUM_AMPLITUDE_ESTIMATION_H_
#define _ITERATIVE_QUAMTUM_AMPLITUDE_ESTIMATION_H_

#include <vector>
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/Utilities/Tools/Utils.h"
#include "Core/Utilities/QPandaNamespace.h"


QPANDA_BEGIN
/**
* @brief  IterativeAmplitudeEstimation Algorthm
* Estimate the probability corresponding to the ground state |1> of the last bit
* @note
*/
class IterativeAmplitudeEstimation
{
public:
	/**
   *@initialize the IterativeAmplitudeEstimation and get the result.
   *@cir: Quantum Circuit
   *@qnumber:  Number of bits used by A.
   *@epsilon: Estimate the accuracy of 'a' (the amplitude of the ground state |1>)..
   *@alpha: Confidence is 1-alpha.
   *@confint_method: Statistical method for estimating confidence interval Chernoff-Hoeffding.
   *@min_ratio: Find the minimum magnification of the next K.
   *@QType: Quantum virtual machine type, currently only CPU is provided, other types of virtual machine types can be added later.
   */
	IterativeAmplitudeEstimation(
		const QCircuit& cir, // Quantum Circuit. 
		const int qnumber, // number of bits used by A.
		const double epsilon, // estimate the accuracy of 'a' (the probability of the ground state |1>).
		const double alpha = 0.05,  // confidence is 1-alpha.
		const std::string confint_method = "CH", // statistical method for estimating confidence interval Chernoff-Hoeffding.
		const double min_ratio = 2.0, // find the minimum magnification of the next K.
		const QMachineType QType = CPU// Quantum virtual machine type, currently only CPU is provided, other types of virtual machine types can be added later.
	);
	/**
   *@Set the name of the function to save the data, the default is "IterNsum_a.json" .
   */
	void setFileName(const std::string fileName) {
		m_filename_json = fileName;
	}

	/**
   *@Save the number of iterative measurements in each round of the IterativeAmplitudeEstimation process and the amplitude of the last qubit basis vector as |1>.
   *@b: b is true, save the data; otherwise, do not save.
   */
	void save_Nsum_a(bool b);
	/**
   *@Exec. IterativeAmplitudeEstimation.
   *@estimated prob. corresponding to the |1> of the last qubit and the total number of measurements.
   */
	std::pair<double, int> exec();
	/**
   *@freeQVM
   */
	void freeQVM()
	{
		m_qvm->finalize();
	}

	double get_result()
	{
		return m_result;
	}

	~IterativeAmplitudeEstimation();

protected:
	QCircuit grover_operator(QCircuit& cir, const QVec& qubits);
	std::pair<double, double> set_confidence_intervals_CH(double val, int max_round, int shots_num, double alpha);
	QCircuit _Gk_A_QC(const QCircuit& cir, const QCircuit& G, const QVec& qubits, int k);
	int _QAE_in_QMachine(QCircuit& cir, const QVec& qubits, const int k, const int N);
	std::pair<int, bool> find_nextK(int k_i, double theta_l, double theta_u, bool up_i);
	bool write_basedata(const std::vector<std::pair<int, double>>& result);


private:
	QCircuit m_cir;
	int m_qnumber;
	double m_epsilon;
	double m_alpha;
	double m_min_ratio{ 2.0 };
	QMachineType m_QType;
	double  m_N_max;
	int m_round_max;
	double m_L_max;
	double m_theta_l;
	double m_theta_u;
	int m_NumShotsAll;
	int m_N_shots;
	double m_result;
	QuantumMachine* m_qvm;
	QVec m_qubits;
	std::vector<ClassicalCondition> m_cbits;
	std::string m_confint_method;
	std::string m_filename_json{ "IterNsum_a.json" };
	bool m_isWriteData{ false };
};


/**
* @brief  iterative_amplitude_estimation
* Estimate the probability corresponding to the ground state |1> of the last bit
* * @param[in] const QCircuit& cir£ºthe source circuit
* * @param[in] const QVec& qvec : the QVec
* * @param[in] double epsilon: the epsilon of iterative_amplitude_estimation Alg
* * @param[in] double confidence: the confidence of iterative_amplitude_estimation Alg
* * @return the probability corresponding to the ground state |1> of the last bit
*/
double iterative_amplitude_estimation(
	const QCircuit& cir,
	const QVec& qvec,
	const double epsilon = 0.0001,
	const double confidence = 0.01
);

QPANDA_END
#endif // _ITERATIVE_QUAMTUM_AMPLITUDE_ESTIMATION_H_

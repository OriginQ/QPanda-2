/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

Quanatum Phase estimation

*/

#ifndef  QPE_H
#define  QPE_H

#include "EigenUnsupported/Eigen/MatrixFunctions"
#include "Core/Core.h"
#include "QAlg/Base_QCircuit/base_circuit.h"
#include "Core/Utilities//Tools/ThreadPool.h"
#include <atomic>
#include <chrono>

QPANDA_BEGIN

#define PRINT_TRACE 0
#if PRINT_TRACE
#define PTrace printf
#define PTraceCircuit(cir) (std::cout << cir << endl)
#define PTraceCircuitMat(cir) { auto m = getCircuitMatrix(cir); std::cout << m << endl; }
#define PTraceMat(mat) (std::cout << (mat) << endl)
#else
#define PTrace
#define PTraceCircuit(cir)
#define PTraceCircuitMat(cir)
#define PTraceMat(mat)
#endif

class QPEAlg
{
public:
	using generate_cir_U = std::function<QCircuit(QVec)>;

public:
	QPEAlg(const QVec& control_qubits, const QVec& target_qubits, const QStat& matrix)
		:m_control_qubits(control_qubits), m_target_qubits(target_qubits)
	{
		if (is_unitary_matrix(matrix))
		{
			m_unitary_mat = matrix;
#if PRINT_TRACE
			auto start = chrono::system_clock::now();
#endif
			m_unitary_mat_cir = matrix_decompose(target_qubits, matrix); 
			//m_unitary_mat_cir = Householder_qr_matrix_decompose(target_qubits, matrix);
#if PRINT_TRACE
			auto end = chrono::system_clock::now();
			auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
			cout << "The matrix decomposition takes "
				<< double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den
				<< "seconds" << endl;
			cout << "There are " << getQGateNum(m_unitary_mat_cir) << " gates in the decomposed-circuit." << endl;
#endif
		}
		else if (matrix == dagger_c(matrix))
		{
			m_hermitian_mat = matrix;
		}
		else
		{
			QCERR_AND_THROW_ERRSTR(invalid_argument, "Error: The input matrix for QPE must be a unitary matrix or Hermitian N*N matrix with N=2^n.");
		}

		m_thread_pool.init_thread_pool(2);
	}

	QPEAlg(const QVec& control_qubits, const QVec& target_qubits, generate_cir_U cir_fun)
		:m_control_qubits(control_qubits), m_target_qubits(target_qubits), m_cir_fun(cir_fun)
	{}

	~QPEAlg() {}

	QCircuit QPE() {
		m_qpe_cir << apply_QGate(m_control_qubits, H);
		const size_t control_qubit_cnt = m_control_qubits.size();
		m_job_cnt = 0;
		for (auto i = 0; i < control_qubit_cnt; ++i)
		{
			//m_qpe_cir << control_unitary_power(m_control_qubits[control_qubit_cnt - 1 - i], 1 << (i));
			m_thread_pool.append(std::bind(&QPEAlg::control_unitary_power, this, m_control_qubits[control_qubit_cnt - 1 - i], 1 << (i), i));
		}
		//Wait for all threads to complete the task
		while (m_job_cnt != control_qubit_cnt) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); }

		sort(m_control_unitary_circuit_vec.begin(), m_control_unitary_circuit_vec.end(),
			[](const std::pair<int, QCircuit>& a, const std::pair<int, QCircuit>& b) {return a.first < b.first; });

		for (const auto& item : m_control_unitary_circuit_vec)
		{
			m_qpe_cir << item.second;
		}

		m_qpe_cir << QFT_dagger(m_control_qubits);
		return m_qpe_cir;
	}

	QCircuit quantum_eigenvalue_estimation() {
		PTrace("On quantum_eigenvalue_estimation.\n");
		m_qpe_cir << apply_QGate(m_control_qubits, H);
		const size_t control_qubit_cnt = m_control_qubits.size();
		m_job_cnt = 0;
		for (auto i = 0; i < control_qubit_cnt; ++i)
		{
			//m_qpe_cir << control_unitary_power(m_control_qubits[control_qubit_cnt - 1 - i], 1 << (control_qubit_cnt - i), i);
			m_thread_pool.append(std::bind(&QPEAlg::control_unitary_power, this,
				m_control_qubits[control_qubit_cnt - 1 - i], 1 << (control_qubit_cnt - i), i));
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			PTrace("append task %d.\n", i);
		}
		PTrace("wait for threads to complete the task.\n");
		//Wait for all threads to complete the task
		while (m_job_cnt != control_qubit_cnt) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); }

		sort(m_control_unitary_circuit_vec.begin(), m_control_unitary_circuit_vec.end(),
			[](const std::pair<int, QCircuit>& a, const std::pair<int, QCircuit>& b) {return a.first < b.first; });

		for (const auto& item : m_control_unitary_circuit_vec)
		{
			m_qpe_cir << item.second;
		}

		m_qpe_cir << QFT_dagger(m_control_qubits);
		return m_qpe_cir;
	}

	QCircuit get_qpe_circuit(){
		return m_qpe_cir;
	}

protected:
	QCircuit unitary_power(size_t min){
		QCircuit cir_u  = CreateEmptyCircuit();
		PTrace("on unitary power.\n");

		QCircuit cir_swap_qubits_b;
		for (size_t i = 0; (i * 2) < (m_target_qubits.size() - 1); ++i)
		{
			cir_swap_qubits_b << SWAP(m_target_qubits[i], m_target_qubits[m_target_qubits.size() - 1 - i]);
		}

		if (m_cir_fun)
		{
			for (auto i = 0; i < min; ++i)
			{
				cir_u << m_cir_fun(m_target_qubits);
			}
		}
		else if (m_unitary_mat.size() != 0)
		{
			cir_u << cir_swap_qubits_b;
			const size_t t = (1 << m_control_qubits.size()) / min;
			for (auto i = 0; i < t; ++i)
			{
				cir_u << m_unitary_mat_cir;
			}
			cir_u << cir_swap_qubits_b;
		}
		else if (m_hermitian_mat.size() != 0)
		{
			PTrace("Deal with hermitian mat.\n"); 
			auto tmp_A = m_hermitian_mat;
			for (auto& item : tmp_A)
			{
				item *= qcomplex_t(0, m_t0 / min);
			}

			EigenMatrixXc eigen_mat = QStat_to_Eigen(tmp_A);
			auto exp_matrix = eigen_mat.exp().eval();

			PTrace("On matrix decompose: %llu.\n", min);
			QCircuit decomposed_cir = matrix_decompose(m_target_qubits, exp_matrix);
			//QCircuit decomposed_cir = Householder_qr_matrix_decompose(m_target_qubits, exp_matrix);
			PTrace("Finished matrix decompose: %llu.\n", min);
			cir_u << cir_swap_qubits_b << decomposed_cir << cir_swap_qubits_b;
		}
		else
		{
			QCERR_AND_THROW_ERRSTR(run_fail, "Error: unknow QPE error.");
		}

		return cir_u;
	}

	QCircuit control_unitary_power(Qubit *ControlQubit, const size_t min, const int index){
		PTrace("Start control unitary power on: %llu.\n", index);
		QCircuit qCircuit = unitary_power(min);
		qCircuit.setControl({ ControlQubit });

		m_queue_mutex.lock();
		m_control_unitary_circuit_vec.push_back(std::make_pair(index, qCircuit));
		m_queue_mutex.unlock();
		++m_job_cnt;
		PTrace("Finished control_unitary_power on: %llu.\n", index);
		return qCircuit;
	}

	QCircuit QFT_dagger(QVec qvec){
		QCircuit qft = QFT(qvec);
		return qft.dagger();
	}

private:
	QVec m_control_qubits;
	QVec m_target_qubits;
	QStat m_hermitian_mat;
	QStat m_unitary_mat;
	QCircuit m_unitary_mat_cir;
	generate_cir_U m_cir_fun;
	QCircuit m_qpe_cir;
	const double m_t0 = PI * 2.0;
	threadPool m_thread_pool;
	std::atomic<size_t> m_job_cnt;
	std::mutex m_queue_mutex;
	std::vector<std::pair<int, QCircuit>> m_control_unitary_circuit_vec;
};

/**
* @brief  build QPE quantum circuit
* @ingroup QAlg
* @param[in]  QVec& the control qubits
* @param[in]  QVec& the target qubits
* @param[in]  Template parameters support the following types:
              1) QStat& a unitary matrix or Hermitian N*N matrix with N=2^n
              2) QPEAlg::generate_cir_U Generating function of corresponding circuit of unitary matrix
* @param[in]  bool Estimate eigenvalue or not
* @return    QCircuit Quantum Phase Estimation circuit
*/
template<typename T>
QCircuit build_QPE_circuit(const QVec& control_qubits, const QVec& target_qubits, T&& alg_para, bool b_estimate_eigenvalue = false) {
	QPEAlg qalg(control_qubits, target_qubits, alg_para);
	QCircuit ret_cir;
	if (b_estimate_eigenvalue)
	{
		ret_cir = qalg.quantum_eigenvalue_estimation();
	}
	else
	{
		ret_cir = qalg.QPE();
	}

	return ret_cir;
}

QPANDA_END

#endif
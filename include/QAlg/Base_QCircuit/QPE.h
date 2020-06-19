/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

Quanatum Phase estimation

*/

#ifndef  QPE_H
#define  QPE_H

#include "EigenUnsupported/Eigen/MatrixFunctions"
#include "Core/Core.h"
#include "QAlg/Base_QCircuit/QFT.h"

QPANDA_BEGIN

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
			m_unitary_mat_cir = matrix_decompose(target_qubits, matrix);
		}
		else if (matrix == dagger_c(matrix))
		{
			m_hermitian_mat = matrix;
		}
		else
		{
			QCERR_AND_THROW_ERRSTR(invalid_argument, "Error: The input matrix for QPE must be a unitary matrix or Hermitian N*N matrix with N=2^n.");
		}
	}

	QPEAlg(const QVec& control_qubits, const QVec& target_qubits, generate_cir_U cir_fun)
		:m_control_qubits(control_qubits), m_target_qubits(target_qubits), m_cir_fun(cir_fun)
	{}

	~QPEAlg() {}

	QCircuit QPE() {
		m_qpe_cir << apply_QGate(m_control_qubits, H);

		for (auto i = 0; i < m_control_qubits.size(); i++)
		{
			m_qpe_cir << control_unitary_power(m_control_qubits[m_control_qubits.size() - 1 - i], 1 << (i));
		}

		m_qpe_cir << QFT_dagger(m_control_qubits);
		return m_qpe_cir;
	}

	QCircuit quantum_eigenvalue_estimation() {
		m_qpe_cir << apply_QGate(m_control_qubits, H);

		for (auto i = 0; i < m_control_qubits.size(); i++)
		{
			m_qpe_cir << control_unitary_power(m_control_qubits[m_control_qubits.size() - 1 - i], 1 << (m_control_qubits.size() - i));
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

		if (m_cir_fun)
		{
			for (auto i = 0; i < min; ++i)
			{
				cir_u << m_cir_fun(m_target_qubits);
			}
		}
		else if (m_unitary_mat.size() != 0)
		{
			for (auto i = 0; i < min; ++i)
			{
				cir_u << m_unitary_mat_cir;
			}
		}
		else if (m_hermitian_mat.size() != 0)
		{
			auto tmp_A = m_hermitian_mat;
			for (auto& item : tmp_A)
			{
				item *= qcomplex_t(0, m_t0 / min);
			}

			EigenMatrixXc eigen_mat = QStat_to_Eigen(tmp_A);
			auto exp_matrix = eigen_mat.exp().eval();

			QCircuit cir_swap_qubits_b;
			for (size_t i = 0; (i * 2) < (m_target_qubits.size() - 1); ++i)
			{
				cir_swap_qubits_b << SWAP(m_target_qubits[i], m_target_qubits[m_target_qubits.size() - 1 - i]);
			}

			QCircuit decomposed_cir = matrix_decompose(m_target_qubits, exp_matrix);
			cir_u << cir_swap_qubits_b << decomposed_cir << cir_swap_qubits_b;
		}
		else
		{
			QCERR_AND_THROW_ERRSTR(run_fail, "Error: unknow QPE error.");
		}

		return cir_u;
	}

	QCircuit control_unitary_power(Qubit *ControlQubit, size_t min){
		QCircuit qCircuit = unitary_power(min);
		qCircuit.setControl({ ControlQubit });

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
	double m_t0 = PI * 2.0;
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
	auto qalg = QPEAlg(control_qubits, target_qubits, alg_para);
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
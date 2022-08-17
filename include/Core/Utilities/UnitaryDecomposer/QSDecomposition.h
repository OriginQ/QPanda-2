#ifndef  _QS_DECOMPOSITION_H_
#define  _QS_DECOMPOSITION_H_

#include "ThirdParty/Eigen/Eigen"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/Utilities/Tools/MatrixDecomposition.h"

QPANDA_BEGIN

/**
* @brief  Quantum Shannon Decomposition.
* @note  The decomposition is based on 'Synthesis of Quantum Logic Circuits'
				Reference: https://arxiv.org/pdf/quant-ph/0406176.pdf
* @ingroup Utilities
*/
class QSDecomposition
{
public:
	QSDecomposition() = default;

	~QSDecomposition()= default;

	QCircuit synthesize_qcircuit(
		const Eigen::MatrixXcd& in_matrix,
		const QVec& qv, DecompositionMode type, bool is_positive_seq);

private:

	void _cosine_sine_decomposition(const Eigen::MatrixXcd& U,
		Eigen::MatrixXcd& u1,
		Eigen::MatrixXcd& u2,
		std::vector<double>& vtheta,
		Eigen::MatrixXcd& v1,
		Eigen::MatrixXcd& v2);

	QCircuit _demultiplex(const std::vector<Eigen::MatrixXcd>& um_vec,
		const QVec& qv);

	QCircuit _qs_demultiplex(const std::vector<Eigen::MatrixXcd>& um_vec,
		const QVec& qv);

	QCircuit _cs_demultiplex(const std::vector<Eigen::MatrixXcd>& um_vec,
		const QVec& qv);

	QCircuit _decompose(const Eigen::MatrixXcd& in_matrix, const QVec& qv);

private:
	DecompositionMode m_dec_type{ DecompositionMode::QSD };
	bool m_is_positive_seq{ false };

};

/**
* @brief  unitary matrix decomposition 
* @ingroup Utilities
* @param[in]  const Eigen::MatrixXcd&  unitary matrix
* @param[in]  const QVec&  qubits vector
* @param[in]  DecompositionMode decomposition mode, default is QSD
* @param[in]  const bool true for positive sequence(q0q1q2), false for inverted order(q2q1q0), default is true
* @return			QCircuit The quantum circuit for target matrix
*/
QCircuit unitary_decomposer_nq(const Eigen::MatrixXcd& in_matrix, const QVec& qv, 
	DecompositionMode type = DecompositionMode::QSD, 
	bool is_positive_seq = false);


/**
* @brief  unitary matrix decomposition
* @ingroup Utilities
* @param[in]  const QSta  unitary matrix by vector<complex<double>>
* @param[in]  const QVec&  qubits vector
* @param[in]  DecompositionMode decomposition mode, default is QSD
* @param[in]  const bool true for positive sequence(q0q1q2), false for inverted order(q2q1q0), default is true
* @return			QCircuit The quantum circuit for target matrix
*/
QCircuit unitary_decomposer_nq(const QStat& in_matrix, const QVec& qv,
	DecompositionMode type = DecompositionMode::QSD,
	bool is_positive_seq = false);


QPANDA_END
#endif   // !_QS_DECOMPOSITION_H_
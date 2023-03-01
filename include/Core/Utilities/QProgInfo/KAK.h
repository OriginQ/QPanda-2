#ifndef  _KAK_H_
#define  _KAK_H_
#include <vector>
#include "ThirdParty/Eigen/Eigen"
#include "Core/QuantumCircuit/QCircuit.h"
#include "include/Core/Utilities/Tools/QMatrixDef.h"

QPANDA_BEGIN

/**
* @brief	Kak description of an arbitrary two-qubit operation.
*				U = g x (Gate A1 Gate A0) x exp(i(xXX + yYY + zZZ))x(Gate b1 Gate b0)
*					A global phase factor
*					Two single-qubit operations (before): Gate b0, b1
*					The Exp() circuit specified by 3 coefficients (x, y, z)
*					Two single-qubit operations (after): Gate a0, a1
* @ingroup Utilities
*/

struct KakDescription
{
	Eigen::Matrix4cd in_matrix;
	std::complex<double> global_phase;
	Eigen::Matrix2cd b0;
	Eigen::Matrix2cd b1;
	Eigen::Matrix2cd a0;
	Eigen::Matrix2cd a1;
	double x;
	double y;
	double z;

	QCircuit to_qcircuit(Qubit* in_bit1, Qubit* in_bit2) const;
	Eigen::MatrixXcd to_matrix() const;
};

/**
* @brief  KAK decomposition via *Magic* Bell basis transformation
* @note  Reference: https://arxiv.org/pdf/quant-ph/0211002.pdf
* @ingroup Utilities
*/
class KAK
{
public:
	KAK(double _tol=1e-7):m_tol(_tol)
	{
	}

	KakDescription decompose(const Eigen::Matrix4cd& in_matrix);

private:
	double m_tol{ 1e-7 };

private:
	/**
	* @brief 	 Returns a canonicalized interaction plus before and after corrections.
	*/
	KakDescription canonicalize(double x, double y, double z) const;

	/**
	* @brief 	 Finds orthogonal matrices L, R such that L x in_matrix x R is diagonal
	*/
	void bidiagonalize(const Eigen::Matrix4cd& in_matrix,
		Eigen::Matrix4cd& out_left, std::vector<std::complex<double>>& diagVec, Eigen::Matrix4cd& out_right) const;

	/**
	* @brief Joint diagonalize two symmetric real matrices
	*          4x4 factor into kron(A, B) where A and B are 2x2
	*/
	void kron_factor(const Eigen::Matrix4cd& in_matrix, std::complex<double>& out_g, Eigen::Matrix2cd& out_f1, Eigen::Matrix2cd& out_f2) const;

	/**
	* @brief Decompose an input matrix in the *Magic* Bell basis
	*/
	void so4_to_magic_su2s(const Eigen::Matrix4cd& in_matrix, Eigen::Matrix2cd& f1, Eigen::Matrix2cd& f2) const;


	/**
	* @brief Returns an orthogonal matrix that diagonalizes the given matrix
	*			diagonalize real symmetric matrix
	*/
	Eigen::MatrixXd diagonalize_rsm(const Eigen::MatrixXd& in_mat) const;


	/**
	* @brief 	 Returns an orthogonal matrix P such that
	*				 P^-1 x symmetric_matrix x P is diagonal
	*				 and P^-1 @ diagonal_matrix @ P = diagonal_matrix
	*		diagonalize real symmetric and sorted diagonal matrices
	*/
	Eigen::MatrixXd diagonalize_rsm_sorted_diagonal(const Eigen::MatrixXd& in_symMat, const Eigen::MatrixXd& in_diagMat) const;


	/**
	* @brief 	 Finds orthogonal matrices that diagonalize both in_mat1 and in_mat2
	*		bidiagonalize real matrix pair with symmetric products
	*/
	void bidiagonalize_rsm_products(const Eigen::Matrix4d& in_mat1, const Eigen::Matrix4d& in_mat2,
		Eigen::Matrix4d& left, Eigen::Matrix4d& right) const;


};


QCircuit random_kak_qcircuit(Qubit* in_qubit1, Qubit* in_qubit2);


/**
* @brief  4*4 unitary matrix decomposition
* @ingroup Utilities
* @param[in]  const Eigen::Matrix4cd&  unitary matrix
* @param[in]  const QVec&  qubits vector
* @param[in]  const bool true for positive sequence(q0q1q2), false for inverted order(q2q1q0), default is true
* @return			QCircuit The quantum circuit for target matrix
*/
QCircuit unitary_decomposer_1q(const Eigen::Matrix2cd& in_mat, Qubit* in_qubit);

/**
* @brief  4*4 unitary matrix decomposition
* @ingroup Utilities
* @param[in]  const Eigen::Matrix4cd&  unitary matrix
* @param[in]  const QVec&  qubits vector
* @param[in]  const bool true for positive sequence(q0q1q2), false for inverted order(q2q1q0), default is true
* @return			QCircuit The quantum circuit for target matrix
*/

QCircuit unitary_decomposer_2q(const Eigen::Matrix4cd& in_mat, const QVec& qv, bool is_positive_seq = true, double _tol = 1e-7);


QPANDA_END
#endif // !_KAK_H_
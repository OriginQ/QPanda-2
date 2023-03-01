#ifndef  _ISOMETRY_DECOMPOSITION_H_
#define  _ISOMETRY_DECOMPOSITION_H_
#include "ThirdParty/Eigen/Eigen"
#include "Core/QuantumCircuit/QCircuit.h"

QPANDA_BEGIN
enum class IsoScheme
{
	CCD = 0,
	KNILL = 1
	//CSD
};

/**
* @brief   Decompose an isometry from m to n qubits.
* @note   It decomposes unitaries on n qubits(m = n) or prepare a quantum state on n qubits(m = 0).
*				see https://arxiv.org/abs/1501.06911
*				an isometry from m to n qubits (n>=2 and m<=n),
*				i.e., a complex 2^n x 2^m array with orthonormal columns.
* @ingroup Utilities
*/
class IsometryDecomposition
{
public:
	IsometryDecomposition() = default;

	~IsometryDecomposition() = default;

	QCircuit decompose(const Eigen::MatrixXcd& isometry, const QVec& qv, IsoScheme scheme );

private:
	Eigen::MatrixXcd _extend_to_unitary(const Eigen::MatrixXcd& iso, size_t log_lines, size_t log_cols);

	QCircuit _knill(const  Eigen::MatrixXcd& iso, const QVec& qv, size_t log_lines, size_t log_cols);

	QCircuit _decompose_column(const QVec& qv, size_t log_lines, size_t col_idx, Eigen::MatrixXcd& iso);

	QCircuit _ccd(const Eigen::MatrixXcd& iso_mat, const QVec& qv, size_t log_lines, size_t log_cols);

	Eigen::MatrixXcd  _mc_unitary(const Eigen::MatrixXcd& iso, size_t col_idx, size_t bit_idx);

	Eigen::MatrixXcd _unitary(const Eigen::MatrixXcd& iso, int basis);

	std::vector<Eigen::MatrixXcd> _uc_unitaries(const  Eigen::MatrixXcd& iso, size_t n_qubits, size_t col_idx, size_t bit_idx);

	QCircuit _mc_gate(const Eigen::MatrixXcd& unitary, const QVec& qv,  
		const std::vector<size_t>& ctrl, const size_t& target, const std::string& k_bin);

	void _update_isometry(const QCircuit& mcg_circ, const QVec& qv, Eigen::MatrixXcd& iso);

	inline size_t _k_s(size_t col_idx, size_t bit_idx)
	{
		// returns the bit value at bit_index of col_index (k in the paper).
		size_t tmp = pow(2, bit_idx);
		return (col_idx & tmp) / tmp;
	}

	inline size_t _a (size_t col_idx, size_t bit_idx)
	{
		// returns int representing n-bit_index most
		return std::floor(col_idx / pow(2, bit_idx));
	}

	inline size_t _b (size_t col_idx, size_t bit_idx)
	{
		// returns int representing bit_index less significant bits.
		return col_idx - (_a(col_idx, bit_idx) * pow(2, bit_idx));
	}
};

/**
* @brief decompose an isometry from m to n qubits
* @ingroup Utilities
* @param[in]  const Eigen::MatrixXcd&   a complex 2^n x 2^m array with orthonormal columns 	 (matrix size : 2^m x 2^n)
* @param[in]  const QVec&  qubits vector ( qubit size : n)
* @param[in]  IsoScheme  isometry decomposition scheme
* @return	 QCircuit a quantum circuit with the isometry attached.
*/
QCircuit isometry_decomposition(const Eigen::MatrixXcd& isometry, const QVec& qv, IsoScheme scheme = IsoScheme::CCD);


QPANDA_END
#endif // !_ISOMETRY_DECOMPOSITION_H_
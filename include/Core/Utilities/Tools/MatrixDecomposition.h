#ifndef MATRIX_DECOMPOSITION_H
#define MATRIX_DECOMPOSITION_H

#include "ThirdParty/Eigen/Dense"
#include "ThirdParty/Eigen/Sparse"
#include "Core/Utilities/Tools/QStatMatrix.h"

QPANDA_BEGIN

enum class MatrixUnit
{
	SINGLE_P0,
	SINGLE_P1,
	SINGLE_I2,
	SINGLE_V2
};

enum DecompositionMode
{
	QR = 0,
	HOUSEHOLDER_QR
};

/**
* @brief  matrix decomposition
* @ingroup Utilities
* @param[in]  QVec& the used qubits
* @param[in]  QStat& The target matrix
* @param[in]  DecompositionMode decomposition mode, default is HOUSEHOLDER_QR
* @return    QCircuit The quantum circuit for target matrix
* @see
*/
QCircuit matrix_decompose(QVec qubits, const QStat& src_mat, DecompositionMode de_mode = HOUSEHOLDER_QR);
QCircuit matrix_decompose(QVec qubits, EigenMatrixXc& src_mat, DecompositionMode de_mode = HOUSEHOLDER_QR);
QCircuit diagonal_matrix_decompose(const QVec& qubits, const QStat& src_mat);

QPANDA_END
#endif // MATRIX_DECOMPOSITION_H
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

/**
* @brief  Decomposition of quantum gates by Chi Kwong Li and Diane Christine Pelejo
* @ingroup Utilities
* @param[in]  QVec& the used qubits
* @param[in]  QStat& The target matrix
* @return    QCircuit The quantum circuit for target matrix
* @see   Un·Un-1···U1·U = I
*/
QCircuit matrix_decompose(QVec qubits, const QStat& src_mat);
QCircuit matrix_decompose(QVec qubits, EigenMatrixXc& src_mat);

QPANDA_END
#endif // MATRIX_DECOMPOSITION_H
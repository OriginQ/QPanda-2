#ifndef _HOUSEHOLDER_DECOMPOSE_H_
#define _HOUSEHOLDER_DECOMPOSE_H_

#include "ThirdParty/Eigen/Dense"
#include "ThirdParty/Eigen/Sparse"
#include "Core/Utilities/Tools/QStatMatrix.h"
#include "Core/QuantumCircuit/QCircuit.h"

QPANDA_BEGIN

/**
* @brief  matrix decomposition (Householder QR)
* @ingroup Utilities
* @param[in]  QVec& the used qubits
* @param[in]  QStat& The target matrix
* @param[in]  const bool True for positive sequence(q0q1q2); False for inverted order(q2q1q0), 
              default is true
* @return    QCircuit The quantum circuit for target matrix
* @see <<Quantum circuits synthesis using Householder transformations>>(https://arxiv.org/abs/2004.07710v1)
*/
QCircuit matrix_decompose_householder(QVec qubits, const QStat& src_mat, bool b_positive_seq = true);
QCircuit matrix_decompose_householder(QVec qubits, const QMatrixXcd& src_mat, bool b_positive_seq = true);

QPANDA_END

#endif

#ifndef  _UNIFORMLY_CONTROLLED_GATES_H_
#define  _UNIFORMLY_CONTROLLED_GATES_H_
#include "ThirdParty/Eigen/Eigen"
#include "Core/QuantumCircuit/QCircuit.h"
QPANDA_BEGIN

QCircuit ucry_circuit(QVec controls, Qubit* target, prob_vec params);

QCircuit ucry_decomposition(QVec controls, Qubit* target, prob_vec params);

QCircuit ucrz_decomposition(QVec controls, Qubit* target, prob_vec params);

QCircuit diagonal_decomposition(QVec qv, std::vector<qcomplex_t> diag_vec);

QCircuit uc_decomposition(QVec ctrl_qv, Qubit* target_q,
	const std::vector<Eigen::MatrixXcd>& um_vec, bool up_to_diagonal = false);


QPANDA_END
#endif // !_UNIFORMLY_CONTROLLED_GATES_H_


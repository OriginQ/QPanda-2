#ifndef HHL_H
#define HHL_H

#include <vector>
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/Utilities/Tools/QStatMatrix.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"

QPANDA_BEGIN

class HHLAlg
{
public:
	HHLAlg(const QStat& A, const std::vector<double>& b, QuantumMachine *qvm);
	virtual ~HHLAlg();

	QCircuit get_hhl_circuit();

	std::string check_QPE_result();

	/**
    * @brief  Extending linear equations to N dimension, N=2^n
    * @ingroup QAlg
    * @param[in] QStat& the source matrix, which will be extend to N*N, N=2^n
    * @param[in] std::vector<double>& the source vector b, which will be extend to 2^n
    * @return 
    * @note
    */
	static void expand_linear_equations(QStat& A, std::vector<double>& b);

protected:
	QCircuit build_CR_cir(QVec& controlqvec, Qubit* target_qubit, double r = 6.0);
	std::vector<double> get_max_eigen_val(const QStat& A);
	EigenMatrixX to_real_matrix(const EigenMatrixXc& c_mat);
	QCircuit build_cir_b(QVec qubits, const std::vector<double>& b);
	void init_qubits();
	bool is_hermitian_matrix();
	void transform_hermitian_to_unitary_mat(QStat& src_mat);

private:
	const QStat& m_A;
	const std::vector<double>& m_b;
	QuantumMachine& m_qvm;
	Qubit* m_ancillary_qubit;
	QVec m_qubits_for_qft;
	QVec m_qubits_for_b;
	QCircuit m_cir_b;
	QCircuit m_cir_qpe;
	QCircuit m_cir_cr;
	QCircuit m_hhl_cir;
	size_t m_qft_cir_used_qubits_cnt;
	size_t m_mini_qft_qubits;
};

/**
* @brief  build the quantum circuit for HHL algorithm to solve the target linear systems of equations: Ax=b
* @ingroup QAlg
* @param[in] QStat& a unitary matrix or Hermitian N*N matrix with N=2^n
* @param[in] std::vector<double>& a given vector
* @return  QCircuit The whole quantum circuit for HHL algorithm
* @note
*/
QCircuit build_HHL_circuit(const QStat& A, const std::vector<double>& b, QuantumMachine *qvm);

/**
* @brief  Use HHL algorithm to solve the target linear systems of equations: Ax=b
* @ingroup QAlg
* @param[in] QStat& a unitary matrix or Hermitian N*N matrix with N=2^n
* @param[in] std::vector<double>& a given vector
* @return  QStat The solution of equation, i.e. x for Ax=b
* @note
*/
QStat HHL_solve_linear_equations(const QStat& A, const std::vector<double>& b);

QPANDA_END

#endif
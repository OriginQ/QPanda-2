#ifndef HHL_H
#define HHL_H

#include <vector>
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/Utilities/Tools/QStatMatrix.h"
#include "Core/Utilities/Tools/SharedMemory.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"

QPANDA_BEGIN

struct HHL_HOLD
{
	size_t m_thread;
	std::mutex m_mutex;
};

class HHLAlg
{
public:
	virtual ~HHLAlg();
	HHLAlg(QuantumMachine* qvm);

	std::string check_QPE_result();
	QCircuit get_hhl_circuit(const QStat& A, const std::vector<double>& b, const uint32_t& precision_cnt);

	/**
	 * @brief  Extending linear equations to N dimension, N=2^n
	 * @ingroup QAlg
	 * @param[in] QStat& the source matrix, which will be extend to N*N, N=2^n
	 * @param[in] std::vector<double>& the source vector b, which will be extend to 2^n
	 * @return
	 * @note
	 */
	static void expand_linear_equations(QStat& A, std::vector<double>& b);

	const double& get_amplification_factor() const { return m_amplification_factor; }

	/*Qubit* get_ancillary_qubit() const
	{
		return m_ancillary_qubit;
	}*/
    QVec get_ancillary_qubit() const
    {
        QVec qv{ m_ancillary_qubit };
        //qv.emplace_back(m_ancillary_qubit);
        return qv;
    }

	const QVec& get_qubit_for_b() const
	{
		return m_qubits_for_b;
	}

	const QVec& get_qubit_for_QFT() const
	{
		return m_qubits_for_qft;
	}

	uint32_t query_uesed_qubit_num() const { return m_hhl_qubit_cnt; }

protected:
	bool is_hermitian_matrix(const QStat& A);
	QMatrixXd to_real_matrix(const QMatrixXcd& c_mat);
	std::vector<double> get_max_eigen_val(const QStat& A);
	void transform_hermitian_to_unitary_mat(QStat& src_mat);
	QCircuit build_cir_b(QVec qubits, const std::vector<double>& b);
	QCircuit build_CR_cir(QVec& controlqvec, Qubit* target_qubit, double r = 6.0);
	void init_qubits(const QStat& A, const std::vector<double>& b, const uint32_t& precision_cnt);

private:
	QCircuit m_cir_b;
	QCircuit m_cir_cr;
	QCircuit m_cir_qpe;
	QCircuit m_hhl_cir;
	QVec m_qubits_for_b;
	QuantumMachine* m_qvm;
	QVec m_qubits_for_qft;
	Qubit* m_ancillary_qubit;
    uint32_t m_hhl_qubit_cnt;
    double m_amplification_factor; /**< For eigenvalue amplification. */
    size_t m_qft_cir_used_qubits_cnt;

	/*static SharedMemory* m_share;
	static struct HHL_HOLD* m_hold;
	static void abort(int signals);*/
};

/**
* @brief  build the quantum circuit for HHL algorithm to solve the target linear systems of equations: Ax=b
* @ingroup QAlg
* @param[in] QStat& a unitary matrix or Hermitian N*N matrix with N=2^n
* @param[in] std::vector<double>& a given vector
* @param[in] uint32_t The count of digits after the decimal point,
			 default is 0, indicates that there are only integer solutions
* @return  QCircuit The whole quantum circuit for HHL algorithm
* @note The higher the precision is, the more qubit number and circuit-depth will be,
		for example: 1-bit precision, 4 additional qubits are required,
		for 2-bit precision, we need 7 additional qubits, and so on.
*/
QCircuit build_HHL_circuit(const QStat& A, const std::vector<double>& b, QuantumMachine* qvm, const uint32_t precision_cnt = 0);

/**
* @brief  Use HHL algorithm to solve the target linear systems of equations: Ax=b
* @ingroup QAlg
* @param[in] QStat& a unitary matrix or Hermitian N*N matrix with N=2^n
* @param[in] std::vector<double>& a given vector
* @param[in] uint32_t The count of digits after the decimal point,
			 default is 0, indicates that there are only integer solutions.
* @return  QStat The solution of equation, i.e. x for Ax=b
* @note The higher the precision is, the more qubit number and circuit-depth will be,
		for example: 1-bit precision, 4 additional qubits are required,
		for 2-bit precision, we need 7 additional qubits, and so on.
*/
QStat HHL_solve_linear_equations(const QStat& A, const std::vector<double>& b, const uint32_t precision_cnt = 0);

QPANDA_END

#endif

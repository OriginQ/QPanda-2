#include "QAlg/HHL/HHL.h"
#include "QAlg/Base_QCircuit/AmplitudeEncode.h"
#include "EigenUnsupported/Eigen/MatrixFunctions"
#include "Core/Utilities/Tools/MatrixDecomposition.h"
#include "Core/Utilities//Tools/QStatMatrix.h"
#include <functional>
#include "QAlg/Base_QCircuit/QPE.h"
#include <sstream>

USING_QPANDA
using namespace std;

#define PRINT_TRACE 0
#if PRINT_TRACE
#define PTrace printf
#define PTraceCircuit(cir) (std::cout << cir << endl)
#define PTraceCircuitMat(cir) { auto m = getCircuitMatrix(cir); std::cout << m << endl; }
#define PTraceMat(mat) (std::cout << (mat) << endl)
#else
#define PTrace
#define PTraceCircuit(cir)
#define PTraceCircuitMat(cir)
#define PTraceMat(mat)
#endif

#define T0  2*PI
#ifndef MAX_PRECISION
#define MAX_PRECISION 0.000001
#endif // !MAX_PRECISION

HHLAlg::HHLAlg(const QStat& A, const std::vector<double>& b, QuantumMachine *qvm)
	:m_A(A), m_b(b), m_qvm(*qvm), m_qft_cir_used_qubits_cnt(0), m_mini_qft_qubits(0)
{
	if (b.size() < 2)
	{
		QCERR_AND_THROW_ERRSTR(init_fail, "Error: error size of b for HHL.");
	}

	if (!(is_hermitian_matrix()) && (!is_unitary_matrix(m_A)))
	{
		QCERR_AND_THROW_ERRSTR(init_fail, "Error: The input matrix for HHL must be a Hermitian sparse N*N matrix.");
	}
}

HHLAlg::~HHLAlg()
{}

QCircuit HHLAlg::index_to_circuit(size_t index, vector<Qubit*> &controlqvec)
{
	QCircuit ret_cir;
	size_t data_qubits_cnt = controlqvec.size();
	for (size_t i = 0; i < data_qubits_cnt; ++i)
	{
		if (0 == index % 2)
		{
			ret_cir << X(controlqvec[i]);
		}

		index /= 2;
	}

	return ret_cir;
}

QCircuit HHLAlg::build_CR_cir(vector<Qubit*>& controlqvec, Qubit* target_qubit, double/* r = 6.0*/)
{
	QCircuit  circuit = CreateEmptyCircuit();
	size_t ctrl_qubits_cnt = controlqvec.size();
	double lambda = pow(2, ctrl_qubits_cnt);
	const int s = (1 << (ctrl_qubits_cnt - 1));
	double thet = 0;
	for (int i = 1; i < lambda; ++i)
	{
		if (s > i)
		{
			thet = 2 * asin(1.0 / ((double)(i)));

		}
		else
		{
			int tmp_i = ~(i - 1);
			int v = -1 * (tmp_i & ((1 << controlqvec.size()) - 1));
			thet = 2 * asin(1.0 / ((double)(v)));
		}

		auto gate = RY(target_qubit, thet).control(controlqvec);
		auto index_cir = index_to_circuit(i, controlqvec);
		circuit << index_cir << gate << index_cir;
	}

	return circuit;
}

EigenMatrixX HHLAlg::to_real_matrix(const EigenMatrixXc& c_mat)
{
	size_t rows = c_mat.rows();
	size_t cols = c_mat.cols();
	EigenMatrixX real_matrix(rows, cols);

	for (size_t i = 0; i < rows; ++i)
	{
		for (size_t j = 0; j < cols; ++j)
		{
			real_matrix(i, j) = c_mat(i, j).real();
		}
	}

	return real_matrix;
}

double HHLAlg::get_max_eigen_val(const QStat& A)
{
	auto e_mat_A = QStat_to_Eigen(A);
	EigenMatrixX real_eigen_A = to_real_matrix(e_mat_A);

	Eigen::EigenSolver<EigenMatrixX> eigen_solver(real_eigen_A);
	auto eigen_vals = eigen_solver.eigenvalues();

	double max_eigen_val = 0.0;
	for (size_t i = 0; i < eigen_vals.rows(); ++i)
	{
		for (size_t j = 0; j < eigen_vals.cols(); ++j)
		{
			const auto &m = abs(eigen_vals(i, j).real());
			if (m > max_eigen_val)
			{
				max_eigen_val = m;
			}
		}
	}

	return max_eigen_val;
}

QCircuit HHLAlg::build_cir_b(QVec qubits, const std::vector<double>& b)
{
	//check parameter b
	double tmp_sum = 0.0;
	for (const auto& i : b)
	{
		tmp_sum += (i*i);
	}
	if (abs(1.0 - tmp_sum) > MAX_PRECISION)
	{
		QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input vector B must satisfy the normalization condition.");
	}

	QCircuit cir_b;
	cir_b = amplitude_encode(qubits, b);

	return cir_b;
}

string HHLAlg::check_QPE_result()
{
	QProg qpe_prog;
	qpe_prog << m_cir_b << m_cir_qpe;
	auto qpe_result_quantum_state = probRunDict(qpe_prog, m_qubits_for_qft);

#define QUAN_STATE_PRECISION 0.0001
	for (auto &val : qpe_result_quantum_state)
	{
		val.second = abs(val.second) < QUAN_STATE_PRECISION ? 0.0 : val.second;
	}

	stringstream ss;
	for (auto &val : qpe_result_quantum_state)
	{
		ss << val.first << ", " << val.second << std::endl;
	}

	ss << "QPE over." << endl;
	return ss.str();
}

void HHLAlg::init_qubits()
{
	double max_eigen_val = get_max_eigen_val(m_A);
	size_t ex_qubits = ceil(log2(max_eigen_val)) + 1;

	size_t b_cir_used_qubits_cnt = ceil(log2(m_b.size()));
	m_qubits_for_b = m_qvm.allocateQubits(b_cir_used_qubits_cnt);

	m_mini_qft_qubits = ceil(log2(sqrt(m_A.size()))) + 3; //For eigenvalue amplification
	m_qft_cir_used_qubits_cnt = (m_mini_qft_qubits + ex_qubits);
	m_qubits_for_qft = m_qvm.allocateQubits(m_qft_cir_used_qubits_cnt);
	PTrace("Total need qubits number: %d, mini_qft_qubits=%d, ex_qubits=%d\n", 
		m_qft_cir_used_qubits_cnt, m_mini_qft_qubits, ex_qubits);

	m_ancillary_qubit = m_qvm.allocateQubit();
}

bool HHLAlg::is_hermitian_matrix()
{
	const auto tmp_A = dagger_c(m_A);
	return (tmp_A == m_A);
}

QCircuit HHLAlg::get_hhl_circuit()
{
	init_qubits();

	auto tmp_A = m_A;
	if (is_hermitian_matrix())
	{
		const size_t cc = 1 << ((size_t)(m_mini_qft_qubits));
		for (auto& i : tmp_A)
		{
			i *= cc;
		}
	}

	m_cir_b = build_cir_b(m_qubits_for_b, m_b);

	//QPE
	m_cir_qpe = build_QPE_circuit(m_qubits_for_qft, m_qubits_for_b, tmp_A, true);
	PTrace("qpe_gate_cnt: %d\n", getQGateNum(m_cir_qpe));

	m_cir_cr = build_CR_cir(m_qubits_for_qft, m_ancillary_qubit, m_qft_cir_used_qubits_cnt);
	PTrace("cr_cir_gate_cnt: %d\n", getQGateNum(m_cir_cr));

	m_hhl_cir << m_cir_b << m_cir_qpe << m_cir_cr << m_cir_qpe.dagger();
	PTrace("hhl_cir_gate_cnt: %d\n", getQGateNum(m_hhl_cir));

	return m_hhl_cir;
}

QCircuit QPanda::build_HHL_circuit(const QStat& A, const std::vector<double>& b, QuantumMachine *qvm)
{
	HHLAlg hhl_alg(A, b, qvm);

	return hhl_alg.get_hhl_circuit();
}

QStat QPanda::HHL_solve_linear_equations(const QStat& A, const std::vector<double>& b)
{
	std::vector<double> tmp_b = b;
	double norm_coffe = 0.0;
	for (const auto& item : tmp_b)
	{
		norm_coffe += (item*item);
	}
	norm_coffe = sqrt(norm_coffe);
	for (auto& item : tmp_b)
	{
		item = item / norm_coffe;
	}

	//build HHL quantum program
	auto machine = initQuantumMachine(CPU);
	auto prog = QProg();
	QCircuit hhl_cir = build_HHL_circuit(A, tmp_b, machine);
	prog << hhl_cir;
	//PTraceCircuit(prog);

	PTrace("HHL quantum circuit is running ...\n");
	directlyRun(prog);
	auto stat = machine->getQState();
	machine->finalize();
	stat.erase(stat.begin(), stat.begin() + (stat.size() / 2));

	// normalise
	double norm = 0.0;
	for (auto &val : stat)
	{
		norm += ((val.real() * val.real()) + (val.imag() * val.imag()));
	}
	norm = sqrt(norm);

	QStat stat_normed;
	for (auto &val : stat)
	{
		stat_normed.push_back(val / qcomplex_t(norm, 0));
	}

	for (auto &val : stat_normed)
	{
		qcomplex_t tmp_val((abs(val.real()) < MAX_PRECISION ? 0.0 : val.real()), (abs(val.imag()) < MAX_PRECISION ? 0.0 : val.imag()));
		val = tmp_val;
	}

	//get solution
	QStat result;
	for (size_t i = 0; i < b.size(); ++i)
	{
		result.push_back(stat_normed.at(i));
	}

	return result;
}
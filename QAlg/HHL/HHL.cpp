#include "QAlg/HHL/HHL.h"
#include "QAlg/Base_QCircuit/AmplitudeEncode.h"
#include "EigenUnsupported/Eigen/MatrixFunctions"
#include "Core/Utilities/Tools/MatrixDecomposition.h"
#include "Core/Utilities//Tools/QStatMatrix.h"
#include <functional>
#include "QAlg/Base_QCircuit/QPE.h"
#include <sstream>
#include "Core/Utilities/Tools/QCircuitOptimize.h"
#include "Core/Utilities/Tools/QProgFlattening.h"
#include <chrono>

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
#define MAX_PRECISION 1e-10


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

QCircuit HHLAlg::build_CR_cir(QVec& controlqvec, Qubit* target_qubit, double/* r = 6.0*/)
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

		if (1 == i)
		{
			QCircuit first_index_cir = index_to_circuit(i, controlqvec);
			circuit << first_index_cir;
		}
		else
		{
			QCircuit index_cir = index_to_circuit(i, controlqvec, i - 1, true);
			circuit << index_cir;
		}

		circuit << gate;
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

std::vector<double> HHLAlg::get_max_eigen_val(const QStat& A)
{
	auto e_mat_A = QStat_to_Eigen(A);
	EigenMatrixX real_eigen_A = to_real_matrix(e_mat_A);

	Eigen::EigenSolver<EigenMatrixX> eigen_solver(real_eigen_A);
	auto eigen_vals = eigen_solver.eigenvalues();

	std::vector<double> eigen_vec(2);
	double max_eigen_val = 0.0;
	double min_eigen_val = 0XEFFFFFFF;
	for (size_t i = 0; i < eigen_vals.rows(); ++i)
	{
		for (size_t j = 0; j < eigen_vals.cols(); ++j)
		{
			const auto &m = abs(eigen_vals(i, j).real());
			if (m > max_eigen_val)
			{
				max_eigen_val = m;
			}

			if ((m < min_eigen_val) && (m > 0.0001))
			{
				min_eigen_val = m;
			}
		}
	}

	eigen_vec[0] = max_eigen_val;
	eigen_vec[1] = min_eigen_val;
	return eigen_vec;
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
		if (abs(tmp_sum) < MAX_PRECISION)
		{
			QCERR("Error: The input vector b is zero.");
			return QCircuit();
		}

		QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input vector b must satisfy the normalization condition.");
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
	const std::vector<double> max_and_min_eigen_val = get_max_eigen_val(m_A);
	PTrace("The max-eigen-val = %f, min-eigen-val = %f\n", max_and_min_eigen_val[0], max_and_min_eigen_val[1]);
	size_t ex_qubits = ceil(log2(max_and_min_eigen_val[0])) + 1;

	size_t b_cir_used_qubits_cnt = ceil(log2(m_b.size()));
	m_qubits_for_b = m_qvm.allocateQubits(b_cir_used_qubits_cnt);

	//m_mini_qft_qubits = ceil(log2(sqrt(m_A.size()))) + 3; //For eigenvalue amplification
	m_mini_qft_qubits = 1;
	if (abs(max_and_min_eigen_val[1]) < 1)
	{
		auto f = 1.0 / max_and_min_eigen_val[1];
		m_mini_qft_qubits += ceil(log2(f));
	}

	m_qft_cir_used_qubits_cnt = (m_mini_qft_qubits + ex_qubits);
	m_qubits_for_qft = m_qvm.allocateQubits(m_qft_cir_used_qubits_cnt);
	printf("Total need qubits number: %d, qft_qubits=%d=%d+%d\n", 
		(m_qft_cir_used_qubits_cnt + b_cir_used_qubits_cnt + 1), m_qft_cir_used_qubits_cnt, ex_qubits, m_mini_qft_qubits);

	m_ancillary_qubit = m_qvm.allocateQubit();
}

bool HHLAlg::is_hermitian_matrix()
{
	const auto tmp_A = dagger_c(m_A);
	return (tmp_A == m_A);
}

void HHLAlg::transform_hermitian_to_unitary_mat(QStat& src_mat)
{
	for (auto& item : src_mat)
	{
		item *= qcomplex_t(0, PI * 2.0 / (1 << m_qft_cir_used_qubits_cnt));
	}

	EigenMatrixXc eigen_mat = QStat_to_Eigen(src_mat);
	auto exp_matrix = eigen_mat.exp().eval();
	src_mat = Eigen_to_QStat(exp_matrix);
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

	//transfer to unitary matrix
	//transform_hermitian_to_unitary_mat(tmp_A);

	//QPE
	m_cir_qpe = build_QPE_circuit(m_qubits_for_qft, m_qubits_for_b, tmp_A, true);
	PTrace("qpe_gate_cnt: %llu\n", getQGateNum(m_cir_qpe));

	m_cir_cr = build_CR_cir(m_qubits_for_qft, m_ancillary_qubit, m_qft_cir_used_qubits_cnt);
	m_hhl_cir << m_cir_b << m_cir_qpe << m_cir_cr << m_cir_qpe.dagger();
	printf("^^^^^^^^^^^^^^^^^whole hhl_cir_gate_cnt: %llu ^^^^^^^^^^^^^^^^^^^^\n", getQGateNum(m_hhl_cir));

	return m_hhl_cir;
}

void HHLAlg::expand_linear_equations(QStat& A, std::vector<double>& b)
{
	size_t dimension = sqrt(A.size());
	double e = ceil(log2(dimension));
	const double expand_dimension = pow(2, e) - (double)dimension;
	if ((expand_dimension - 0.0) < 0.000001)
	{
		return;
	}

	for (size_t i = 0; i < expand_dimension; ++i)
	{
		b.push_back(0);
	}

	size_t new_dimension = dimension + expand_dimension;
	QStat new_A;
	new_A.resize(pow(new_dimension, 2), 0);
	const auto src_size = A.size() - 1;
	for (size_t i = 0; i < dimension; ++i)
	{
		for (size_t j = 0; j < dimension; ++j)
		{
			new_A[i*new_dimension + j] = A[i*dimension + j];
		}
	}

	A.swap(new_A);
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

	if (abs(norm_coffe) < MAX_PRECISION)
	{
		QStat r;
		r.resize(b.size(), 0);
		return r;
	}

	norm_coffe = sqrt(norm_coffe);
	for (auto& item : tmp_b)
	{
		item = item / norm_coffe;
	}

	//build HHL quantum program
	auto machine = initQuantumMachine(CPU);
	machine->setConfigure({ 64,64 });
	auto prog = QProg();
	QCircuit hhl_cir = build_HHL_circuit(A, tmp_b, machine);
	prog << hhl_cir;
	//PTraceCircuit(prog);

	printf("HHL quantum circuit is running ...\n");
        auto start = chrono::system_clock::now();
	directlyRun(prog);
	auto stat = machine->getQState();
        auto end = chrono::system_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        cout << "run HHL used: "
             << double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den
             << " s" << endl;

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
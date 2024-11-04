#include "QAlg/QAlg.h"
#include "QAlg/Base_QCircuit/SzegedyWalk.h"
#include <cmath>

using namespace std;
using namespace QPanda;

typedef qcomplex_t CX;

static QMatrixXcd get_swap_matrix(int nqubit)
{
	assert(nqubit > 0);
	int size = pow(2, 2 * nqubit);
	int n = pow(2, nqubit);
	QMatrixXcd m = QMatrixXcd::Zero(size, size);
	for (int col = 0; col < size; ++col) {
		int row = (col / n) + n * (col % n);
		m(row, col) = CX(1, 0);
	}
	return m;
}

static QMatrixXcd get_adjacency_matrix_for_cyclic_graph(int n)
{
	assert(n > 0);
	QMatrixXcd adjacency_mat = QMatrixXcd::Zero(n, n);

	int col_index1 = 1, col_index2 = n - 1;
	for (int i = 0; i < n; ++i) {
		adjacency_mat(i, col_index1) = 1;
		adjacency_mat(i, col_index2) = 1;
		col_index1 = (col_index1 + 1) % n;
		col_index2 = (col_index2 + 1) % n;
	}
	return adjacency_mat;
}

static QMatrixXcd get_isometry_for_cycle_graph(int n)
{
	auto a = get_adjacency_matrix_for_cyclic_graph(n);
	QMatrixXcd t = QMatrixXcd::Zero(n * n, n);
	for (int col = 0; col < n; ++col) {
		for (int row = col * n; row < col * n + n; ++row) {
			t(row, col) = a(row % n, col);
		}
	}
	return t;
}

static int sgn(int i)
{
	if (i > 0)
		return 1;
	else if (i < 0)
		return -1;
	else
		return 0;
}

static bool is_real_negative(const CX& c)
{
	return c.real() < 1e-15 && abs(c.imag()) < 1e-15;
}

QMatrixXd kroneckerProduct(int j, int k, int nqubit)
{
	int dim = 1 << nqubit;
	QMatrixXd jk = QMatrixXd::Zero(dim * dim, 1);
	jk(j * dim + k, 0) = 1;
	return jk;
}

QMatrixXd computational_basis_state_ket(int j, int nqubit)
{
	QMatrixXd b = QMatrixXd::Zero(1 << nqubit, 1);
	b(j, 0) = 1;
	return b;
}

QMatrixXd computational_basis_state_bra(int j, int nqubit)
{
	QMatrixXd b = QMatrixXd::Zero(1, 1 << nqubit);
	b(0, j) = 1;
	return b;
}

static QMatrixXcd complexify(const QMatrixXd& m)
{
	QMatrixXcd result(m.rows(), m.cols());
	for (int i = 0; i < result.rows(); ++i)
	{
		for (int j = 0; j < result.cols(); ++j)
		{
			result(i, j) = CX(m(i, j), 0);
		}
	}
	return result;
}

QMatrixXcd SzegedyWalk::expm(QMatrixXcd H)
{
	// assert H is Hermitian
	const int rows = H.rows();
	const int cols = H.cols();
	int nqubits = log2(rows);
	int qubits_per_reg = nqubits + 1;

	//H *= PI;

	bool has_negative_diagonal_entry = false;
	double lambda_max = 0.0;
	for (int i = 0; i < rows; ++i)
	{
		if (is_real_negative(H(i, i)))
		{
			has_negative_diagonal_entry = true;
			lambda_max = max(lambda_max, sqrt(std::norm(H(i, i))));
		}
	}

	auto H_prime = H;
	if (has_negative_diagonal_entry) {
		H_prime += lambda_max * QMatrixXcd::Identity(rows, cols);
	}

	QMatrixXd absH_prime(rows, cols);
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			absH_prime(i, j) = sqrt(std::norm(H_prime(i, j)));
		}
	}
	//cout << "absH_prime: " << endl;
	//cout << absH_prime << endl;

	auto H_prime1Norm = absH_prime.rowwise().sum().maxCoeff();
	auto H_primemaxNorm = absH_prime.maxCoeff();
	auto sqrtAbsH_prime = absH_prime.cwiseSqrt();
	//cout << "H_prime1Norm: " << H_prime1Norm << endl;
	//cout << "H_primemaxNorm: " << H_primemaxNorm << endl;
	//cout << "sqrtAbsH_prime: " << endl << sqrtAbsH_prime << endl;

	QMatrixXcd sqrtH_prime_conjugate(rows, cols);
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			if (is_real_negative(H_prime(i, j))) {
				sqrtH_prime_conjugate(i, j) = CX(0, sgn(i - j) * sqrtAbsH_prime(i, j));
			}
			else {
				sqrtH_prime_conjugate(i, j) = sqrt(CX(H_prime(i, j).real(), -H_prime(i, j).imag()));
			}
		}
	}
	//cout << "sqrtH_prime_conjugate: \n" << sqrtH_prime_conjugate << endl;

	int tau = 1e5 * H_prime1Norm * sqrt(1 + (0.5 * PI - 1) * H_prime1Norm);
	double epsilon = H_prime1Norm / tau; // lazy quantum walk parameter
	double epsilon_H = epsilon / H_prime1Norm;
	//cout << "tau: " << tau << endl;
	//cout << "epsilon: " << epsilon << endl;

	auto rowSum = absH_prime.rowwise().sum();
	//cout << "row sums: " << rowSum.transpose() << endl;
	vector<QMatrixXcd> psi_js(cols);
	for (int j = 0; j < cols; ++j) {
		psi_js[j] = sqrt(1 - epsilon_H * rowSum(j, 0)) *
			complexify(kroneckerProduct(j, cols, qubits_per_reg));
		for (int k = 0; k < cols; ++k) {
			psi_js[j] += sqrt(epsilon_H) * sqrtH_prime_conjugate(j, k) *
				complexify(kroneckerProduct(j, k, qubits_per_reg));
		}
	}
	QMatrixXcd T = QMatrixXcd::Zero(1 << (2 * qubits_per_reg), 1 << qubits_per_reg);
	for (int i = 0; i < cols; ++i) {
		T += psi_js[i] * complexify(computational_basis_state_bra(i, qubits_per_reg));
	}

	auto T_dagger = T.adjoint();
	auto S = get_swap_matrix(qubits_per_reg);
	auto IS = S;
	for (auto i = 0; i < IS.rows(); ++i) {
		for (auto j = 0; j < IS.cols(); ++j) {
			IS(i, j) = CX(-S(i, j).imag(), S(i, j).real());
		}
	}
	auto I = QMatrixXcd::Identity(S.rows(), S.cols());
	auto U = IS * (2 * T * T_dagger - I);
	auto sim = 0.5 * T_dagger * (I - IS) * U.pow(tau) * (I + IS) * T;
	auto sim_block = sim.block(0, 0, H.rows(), H.cols());
	return sim_block;
	//cout << Eigen_to_QStat(sim_block) << endl;

	//cout << "sim: \n" << sim_block << endl;


	//auto iH = H; // actually -iH
	//for (int i = 0; i < iH.rows(); ++i) {
	//	for (int j = 0; j < iH.cols(); ++j) {
	//		iH(i, j) = CX(H(i, j).imag(), -H(i, j).real());
	//	}
	//}
	//auto result_eigen = iH.exp().eval();
	//cout << "result using Eigen: \n" << Eigen_to_QStat(result_eigen) << endl;
}

QMatrixXcd SzegedyWalk::expm(QMatrixXd H)
{
	auto H_con = complexify(H);
	return expm(H_con);
}

QMatrixXcd SzegedyWalk::expm_i(QMatrixXcd H)
{
	auto iH = H;
	for (int i = 0; i < iH.rows(); ++i) {
		for (int j = 0; j < iH.cols(); ++j) {
			iH(i, j) = CX(-H(i, j).imag(), H(i, j).real());
		}
	}
	return expm(iH);
}

QMatrixXcd SzegedyWalk::expm_i(QMatrixXd H)
{
	auto H_con = complexify(H);
	return expm_i(H_con);
}

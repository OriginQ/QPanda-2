#include "Core/Utilities/UnitaryDecomposer/QSDecomposition.h"
#include "Core/Utilities/QProgInfo/KAK.h"
#include "Core/Utilities/UnitaryDecomposer/MatrixUtil.h"
#include "Core/Utilities/Tools/MultiControlGateDecomposition.h"
#include "ThirdParty/EigenUnsupported/Eigen/KroneckerProduct"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include "Core/Utilities/Tools/MatrixDecomposition.h"
#include "Core/Utilities/UnitaryDecomposer/UniformlyControlledGates.h"

USING_QPANDA
using namespace std;
using namespace Eigen;

static void _check_demultiplex(const Eigen::MatrixXcd& um1,
	const Eigen::MatrixXcd& um2,
	const QCircuit& circ,
	const bool& is_positive,
	double _tol = tolerance)
{
	Eigen::MatrixXcd test_U = Eigen::MatrixXcd::Zero(um1.rows() + um2.rows(), um1.cols() + um2.cols());
	test_U.block(0, 0, um1.rows(), um1.cols()) = um1;
	test_U.block(um1.rows(), um1.cols(), um2.rows(), um2.cols()) = um2;
	auto mat_1 = eigen2qstat(test_U);
	QProg prog;
	prog << circ;
	auto mat_2 = getCircuitMatrix(prog, is_positive);

	if (mat_compare(mat_1, mat_2, _tol))
	{
		std::cout << "test_U : \n" << test_U << std::endl;
		std::cout << "mat_1 : \n" << mat_1 << std::endl;
		std::cout << "mat_2 : \n" << mat_2 << std::endl;
		throw std::runtime_error("demultiplex fail!");
	}
}

static void _check_ucry(QCircuit ucry_circ,
	std::vector<double> vtheta,
	bool is_positive_seq, double _tol = tolerance)
{
	auto circ_mat = getCircuitMatrix(ucry_circ, is_positive_seq);
	auto _temp_vec = vtheta;
	int n = vtheta.size();
	Eigen::MatrixXcd c(n, n);
	Eigen::MatrixXcd s(n, n);
	Eigen::Map<Eigen::VectorXd> vtheta_1(_temp_vec.data(), _temp_vec.size());
	Eigen::MatrixXcd mcry = Eigen::MatrixXcd::Zero(n + n, n + n);
	mcry.topLeftCorner(n, n).diagonal() = Eigen::cos(vtheta_1.array());
	mcry.bottomLeftCorner(n, n).diagonal() = Eigen::sin(vtheta_1.array());
	mcry.topRightCorner(n, n).diagonal() = -1 * Eigen::sin(vtheta_1.array());
	mcry.bottomRightCorner(n, n).diagonal() = Eigen::cos(vtheta_1.array());

	auto _mat = eigen2qstat(mcry);
	if (mat_compare(circ_mat, _mat, _tol))
	{
		QCERR("ucry decompose fail!");
		throw std::runtime_error("decompose fail!");
	}
}

static void _check_ucrz(QCircuit ucry_circ,
	std::vector<double> vtheta,
	bool is_positive_seq, double _tol = tolerance)
{
	auto circ_mat = getCircuitMatrix(ucry_circ, is_positive_seq);
	int n = vtheta.size();

	auto rz_mat = [&](double angle) {
		Matrix2cd  mat;
		qcomplex_t val0(cos(angle / 2), -1 * sin(angle / 2));
		qcomplex_t val3(cos(angle / 2), 1 * sin(angle / 2));
		mat << val0, 0, 0, val3;
		return mat;
	};

	Eigen::MatrixXcd mcrz_mat = Eigen::MatrixXcd::Zero(2 * n, 2 * n);
	int start_row_idx = 0;
	int  start_cos_idx = 0;
	for (auto& val : vtheta)
	{
		mcrz_mat.block(start_row_idx, start_cos_idx, 2, 2) = rz_mat(val);
		start_row_idx += 2;
		start_cos_idx += 2;
	}

	auto _mat = eigen2qstat(mcrz_mat);
	if (mat_compare(circ_mat, _mat, _tol))
	{
		QCERR("ucrz decompose fail!");
		throw std::runtime_error("ucrz decompose fail!");
	}
}


static void _check_uc(QCircuit uc_circ,
	const std::vector<Eigen::MatrixXcd>& um_vec,
	bool is_positive_seq, double _tol = tolerance)
{
	auto uc_mat = getCircuitMatrix(uc_circ, is_positive_seq);

	Eigen::MatrixXcd total_mat = Eigen::MatrixXcd::Zero(2 * um_vec.size(), 2 * um_vec.size());
	for (int i = 0; i < um_vec.size(); i++)
	{
		int idx = 2 * i;
		total_mat.block(idx, idx, 2, 2) = um_vec[i];
	}
	auto _mat = eigen2qstat(total_mat);
	if (mat_compare(uc_mat, _mat, _tol))
	{
		QCERR("ucrz decompose fail!");
		throw std::runtime_error("ucrz decompose fail!");
	}
}


#if 0
// Decomposition is fast, but occasionally fail.  need debug.
void QSDecomposition::_cosine_sine_decomposition(const Eigen::MatrixXcd& U,
	Eigen::MatrixXcd& u1,
	Eigen::MatrixXcd& u2,
	std::vector<double>& vtheta,
	Eigen::MatrixXcd& v1,
	Eigen::MatrixXcd& v2
)
{
	// Cosine sine decomposition
	// U =	[q0, q1] = [u1    ] [ c  s] [v1    ]
	//			[q2, q3] = [    u2] [-s c ] [   v2]

	int n = U.rows();

	Eigen::BDCSVD<Eigen::MatrixXcd> svd(n / 2, n / 2);
	svd.compute(U.topLeftCorner(n / 2, n / 2), Eigen::ComputeThinU | Eigen::ComputeThinV);

	int p = n / 2;
	Eigen::MatrixXcd c(svd.singularValues().reverse().asDiagonal());

	u1.noalias() = svd.matrixU().rowwise().reverse();
	v1.noalias() = svd.matrixV().rowwise().reverse();
	Eigen::MatrixXcd q2 = U.bottomLeftCorner(p, p) * v1;

	int k = 0;
	for (int j = 1; j < p; j++)
	{
		if (c(j, j).real() <= 0.70710678119)
		{
			k = j;
		}
	}
	Eigen::HouseholderQR<Eigen::MatrixXcd> qr(p, k + 1);
	qr.compute(q2.block(0, 0, p, k + 1));
	u2 = qr.householderQ();
	Eigen::MatrixXcd s = u2.adjoint() * q2;
	if (k < p - 1)
	{
		//std::cout << "k is smaller than size of U00 = " << p << ", adjustments will be made, k = " << k << std::endl;
		k = k + 1;
		Eigen::BDCSVD<Eigen::MatrixXcd> svd2(p - k, p - k);
		svd2.compute(s.block(k, k, p - k, p - k), Eigen::ComputeThinU | Eigen::ComputeThinV);
		s.block(k, k, p - k, p - k) = svd2.singularValues().asDiagonal();
		c.block(0, k, p, p - k) = c.block(0, k, p, p - k) * svd2.matrixV();
		u2.block(0, k, p, p - k) = u2.block(0, k, p, p - k) * svd2.matrixU();
		v1.block(0, k, p, p - k) = v1.block(0, k, p, p - k) * svd2.matrixV();

		Eigen::HouseholderQR<Eigen::MatrixXcd> qr2(p - k, p - k);

		qr2.compute(c.block(k, k, p - k, p - k));
		c.block(k, k, p - k, p - k) = qr2.matrixQR().triangularView<Eigen::Upper>();
		u1.block(0, k, p, p - k) = u1.block(0, k, p, p - k) * qr2.householderQ();
	}

	std::vector<int> c_ind;
	std::vector<int> s_ind;
	for (int j = 0; j < p; j++)
	{
		if (c(j, j).real() < 0)
		{
			c_ind.push_back(j);
		}
		if (s(j, j).real() < 0)
		{
			s_ind.push_back(j);
		}
	}

	auto deal_func = [](Eigen::MatrixXcd& mat,
		const std::vector<int>& rows_idx, const std::vector<int>& cols_idx)
	{
		for (const auto& ridx : rows_idx)
		{
			for (const auto& cidx : cols_idx)
			{
				mat(ridx, cidx) = -mat(ridx, cidx);
			}
		}

	};

	auto all_idx_func = [](
		const Eigen::MatrixXcd& mat, bool is_row_idx = true)
	{
		std::vector<int> all_idx;
		int _size = mat.rows();
		if (!is_row_idx)
		{
			_size = mat.cols();
		}
		for (int i = 0; i < _size; i++)
		{
			all_idx.emplace_back(i);
		}
		return all_idx;
	};

	deal_func(c, c_ind, c_ind);
	deal_func(u1, all_idx_func(u1), c_ind);

	deal_func(s, s_ind, s_ind);
	deal_func(u2, all_idx_func(u2), s_ind);


	if (!is_approx(U.topLeftCorner(p, p), u1 * c * v1.adjoint()))
	{
		std::cout << "q0 is not correct! need reconstructed q0 to u1 * c * v1.adjoint()\n";
	}

	if (!is_approx(U.bottomLeftCorner(p, p), u2 * s * v1.adjoint()))
	{
		std::cout << "q2 is not correct! need reconstructed q2 to u2 * s * v1.adjoint()\n";
	}

	v1.adjointInPlace();
	s = -s;

	Eigen::MatrixXcd tmp_s = u1.adjoint() * U.topRightCorner(p, p);
	Eigen::MatrixXcd tmp_c = u2.adjoint() * U.bottomRightCorner(p, p);

	for (int i = 0; i < p; i++)
	{
		if (abs(s(i, i)) > abs(c(i, i)))
		{
			v2.row(i).noalias() = tmp_s.row(i) / s(i, i);
		}
		else
		{
			v2.row(i).noalias() = tmp_c.row(i) / c(i, i);
		}
	}

	// check  
	Eigen::MatrixXcd tmp(n, n);
	tmp.topLeftCorner(p, p) = u1 * c * v1;
	tmp.bottomLeftCorner(p, p) = -u2 * s * v1;
	tmp.topRightCorner(p, p) = u1 * s * v2;
	tmp.bottomRightCorner(p, p) = u2 * c * v2;
	_ASSERT(is_approx(tmp, U, 10e-3), "csd fail");

	Eigen::VectorXd tmp_theta = -Eigen::asin(s.diagonal().array()).real();
	vtheta = std::vector<double>(tmp_theta.data(), tmp_theta.data() + tmp_theta.size());
}


#endif

QCircuit QSDecomposition::synthesize_qcircuit(
	const Eigen::MatrixXcd& in_matrix,
	const QVec& qv, DecompositionMode type, bool is_positive_seq)
{
	QCircuit cir;

	QVec tmp_qv;
	tmp_qv += qv;
	bool check = (type == DecompositionMode::CSD
		|| type == DecompositionMode::QSD);

	int qnum = (int)log2(in_matrix.rows());
	_ASSERT(check, "supports qsd or csd");
	_ASSERT(tmp_qv.size() == qv.size(), "input repeated qubits, need check");
	_ASSERT(qnum == qv.size(), "matrix dim not matching qubits size");
	_ASSERT(pow(2,qnum) == in_matrix.rows(), "matrix dim need 2**qnum");
	_ASSERT(is_unitary(in_matrix), "is not a unitary matrix");

	if (!is_positive_seq) {
		std::reverse(tmp_qv.begin(), tmp_qv.end());
	}
	m_dec_type = type;
	m_is_positive_seq = is_positive_seq;
	return _decompose(in_matrix, tmp_qv);
}

QCircuit QSDecomposition::_decompose(const Eigen::MatrixXcd& in_matrix, const QVec& qv)
{
	QCircuit circ;
	int dim = in_matrix.rows();

	Eigen::MatrixXcd identity = Eigen::MatrixXcd::Identity(dim, dim);
	if (is_approx(in_matrix, identity))
	{
		//std::cout << "is identity,  return null circuit" << std::endl;
		return circ;
	}

	if (dim == 2){
		circ = unitary_decomposer_1q(in_matrix, qv[0]);
	}
	else if (dim == 4){
		circ = unitary_decomposer_2q(in_matrix, qv, true);
	}
	else
	{
		int n = in_matrix.rows() / 2;
		constexpr double prec = 1e-8;
		if (in_matrix.bottomLeftCorner(n, n).isZero(prec) && in_matrix.topRightCorner(n, n).isZero(prec))
		{
			std::vector<Eigen::MatrixXcd> um_vec;
			um_vec.emplace_back(in_matrix.topLeftCorner(n, n));
			um_vec.emplace_back(in_matrix.bottomRightCorner(n, n));
			circ = _demultiplex(um_vec, qv);
		}
		else
		{
			Eigen::MatrixXcd u1(n, n);
			Eigen::MatrixXcd u2(n, n);
			Eigen::MatrixXcd v1h(n, n);
			Eigen::MatrixXcd v2h(n, n);
			std::vector<double> vtheta;
			_cosine_sine_decomposition(in_matrix, u1, u2, vtheta, v1h, v2h);
			std::vector<Eigen::MatrixXcd> left_um = { v1h,v2h };
			std::vector<Eigen::MatrixXcd>  right_um = { u1 , u2 };

			auto left_circ = _demultiplex(left_um, qv);
			//_check_demultiplex(v1h, v2h, left_circ, m_is_positive_seq);

			std::vector<double> ucry_vtheta;
			for (const auto& val : vtheta) {
				ucry_vtheta.emplace_back(2 * val);
			}
			auto middle_cir = ucry_decomposition(qv - qv.back(), qv.back(), ucry_vtheta);
			//_check_ucry(middle_cir, vtheta, m_is_positive_seq);

			auto right_circ = _demultiplex(right_um, qv);
			//_check_demultiplex(u1, u2, right_circ, m_is_positive_seq);

			circ << left_circ
				<< middle_cir
				<< right_circ
				;
		}
	}

	return circ;
}

/**
 *  Cosine-sine decomposition.
 * 	 U =	[u00, u01] = [u1    ] [c  -s] [v1   ]
 *			[u10, u11] = [    u2] [s  c ] [   v2]
 */
void QSDecomposition::_cosine_sine_decomposition(
	const Eigen::MatrixXcd& U,
	Eigen::MatrixXcd& u1,
	Eigen::MatrixXcd& u2,
	std::vector<double>& vtheta,
	Eigen::MatrixXcd& v1,
	Eigen::MatrixXcd& v2)
{
	unsigned n = U.rows() / 2;
	Eigen::MatrixXcd u00 = U.topLeftCorner(n, n);
	Eigen::MatrixXcd u01 = U.topRightCorner(n, n);
	Eigen::MatrixXcd u10 = U.bottomLeftCorner(n, n);
	Eigen::MatrixXcd u11 = U.bottomRightCorner(n, n);
	Eigen::MatrixXcd v1_dag;
	
	Eigen::JacobiSVD<Eigen::MatrixXcd, Eigen::NoQRPreconditioner> svd(
		u00, Eigen::ComputeFullU | Eigen::ComputeFullV);
	//Eigen::BDCSVD<Eigen::MatrixXcd> svd(n, n);
	//svd.compute(u00, Eigen::ComputeThinU | Eigen::ComputeThinV);

	u1 = svd.matrixU().rowwise().reverse();
	v1_dag = svd.matrixV().rowwise().reverse();
	Eigen::MatrixXd c = svd.singularValues().reverse().asDiagonal();
	v1 = v1_dag.adjoint();

	Eigen::HouseholderQR<Eigen::MatrixXcd> qr = (u10 * v1_dag).householderQr();
	u2 = qr.householderQ();
	Eigen::MatrixXcd S = qr.matrixQR().triangularView<Eigen::Upper>();
	if (!S.imag().isZero())
	{
		for (int j = 0; j < n; j++)
		{
			std::complex<double> z = S(j, j);
			double r = std::abs(z);
			if (r > 1e-11)
			{
				std::complex<double> w = std::conj(z) / r;
				S(j, j) *= w;
				u2.col(j) /= w;
			}
		}
	}

	// make all entries in s non-negative.
	Eigen::MatrixXd s = S.real();
	for (int j = 0; j < n; j++)
	{
		if (s(j, j) < 0)
		{
			s(j, j) = -s(j, j);
			u2.col(j) = -u2.col(j);
		}
	}

	v2 = Eigen::MatrixXcd::Zero(n, n);
	for (int i = 0; i < n; i++)
	{
		if (s(i, i) > c(i, i)) {
			v2.row(i) = -(u1.adjoint() * u01).row(i) / s(i, i);
		}
		else {
			v2.row(i) = (u2.adjoint() * u11).row(i) / c(i, i);
		}
	}

	for(int i=0; i < s.rows(); i++){
		vtheta.push_back(atan2(s(i, i), c(i, i)));
	}

	// check  
	Eigen::MatrixXcd tmp(2 * n, 2 * n);
	tmp.topLeftCorner(n, n) = u1 * c * v1;
	tmp.bottomLeftCorner(n, n) = u2 * s * v1;
	tmp.topRightCorner(n, n) = u1 * -s * v2;
	tmp.bottomRightCorner(n, n) = u2 * c * v2;
	if (!is_approx(tmp, U, 10e-3))
	{
		QCERR("csd fail!");
		throw std::runtime_error("csd fail!");
	}
}

QCircuit QSDecomposition::_demultiplex(const std::vector<Eigen::MatrixXcd>& um_vec,
	const QVec& qv)
{
	if (m_dec_type == DecompositionMode::QSD){
		return _qs_demultiplex(um_vec, qv);
	}
	else if (m_dec_type == DecompositionMode::CSD)
	{
		if (!um_vec.empty() && um_vec[0].rows() == 2){
			return uc_decomposition(qv - qv.front(), qv.front(), um_vec);
		}

		return _cs_demultiplex(um_vec, qv);
	}
}

/*
*	Cosine sine  Demultiplex
*/
QCircuit QSDecomposition::_cs_demultiplex(
	const std::vector<Eigen::MatrixXcd>& um_vec,
	const QVec& qv)
{
	QCircuit circ;
	std::vector<Eigen::MatrixXcd> left_um_vec;
	std::vector<double> ucry_vtheta;
	std::vector<Eigen::MatrixXcd> right_um_vec;

	for (const auto& mat : um_vec)
	{
		int n = mat.rows() / 2;
		Eigen::MatrixXcd u1(n, n);
		Eigen::MatrixXcd u2(n, n);
		Eigen::MatrixXcd v1h(n, n);
		Eigen::MatrixXcd v2h(n, n);
		std::vector<double> vtheta;
		_cosine_sine_decomposition(mat, u1, u2, vtheta, v1h, v2h);
		left_um_vec.emplace_back(v1h);
		left_um_vec.emplace_back(v2h);
		for (const auto& val : vtheta) {
			ucry_vtheta.emplace_back(2 * val);
		}
		right_um_vec.emplace_back(u1);
		right_um_vec.emplace_back(u2);
	}

	circ << _demultiplex(left_um_vec, qv);

	int target_idx = qv.size() - (int)log2(left_um_vec.size());
	auto target_q = qv[target_idx];
	QVec ctrl_qv = qv - target_q;
	circ << ucry_decomposition(ctrl_qv, target_q, ucry_vtheta);

	circ << _demultiplex(right_um_vec, qv);

	return circ;
}

/*
*  Quantum Shannon Demultiplex
* 	[um0 0 ]  = [V 0][D 0 ][W 0]
*	[0  um1]     [0 V][0 D*][0 W]
*/
QCircuit QSDecomposition::_qs_demultiplex(
	const std::vector<Eigen::MatrixXcd>& um_vec,
	const QVec& qv)
{
	_ASSERT(um_vec.size() == 2, "um_vec size != 2 ");

	Eigen::MatrixXcd V;
	Eigen::MatrixXcd W;
	Eigen::VectorXcd D;
	Eigen::MatrixXcd um0 = um_vec[0];
	Eigen::MatrixXcd um1 = um_vec[1];
	Eigen::MatrixXcd um0um1 = um0 * um1.adjoint();

    //Eigen::ComplexSchur<Eigen::MatrixXcd> schur_solver(um0um1);
    //D = schur_solver.matrixT().diagonal().cwiseSqrt();
    //V = schur_solver.matrixU();

	if (um0um1 == um0um1.adjoint())
	{
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> esolver(um0um1);
		D = esolver.eigenvalues();
		V = esolver.eigenvectors();

        for (uint64_t i = 0; i < D.size(); ++i)
            D(i) = std::sqrt(D(i));
	}
	else
	{
		Eigen::ComplexSchur<Eigen::MatrixXcd> schur_solver(um0um1);
		D = schur_solver.matrixT().diagonal().cwiseSqrt();
		V = schur_solver.matrixU();
	}

	if (!is_unitary(V))
	{
		std::cout << "need closest unitary\n";
		Eigen::BDCSVD<Eigen::MatrixXcd> svd3(V, Eigen::ComputeThinU | Eigen::ComputeThinV);
		V = svd3.matrixV() * svd3.matrixU();
	}
	W = D.asDiagonal() * V.adjoint() * um1;

	// check demultiplexing
	Eigen::MatrixXcd Dtemp = D.asDiagonal();
	if (!is_unitary(V) || !is_unitary(W)
		|| !is_approx(um0, V * Dtemp * W, 10e-3)
		|| !is_approx(um1, V * Dtemp.adjoint() * W, 10e-3))
	{
		QCERR("Demultiplexing of unitary not correct!");
		throw std::runtime_error("Demultiplexing of unitary not correct!");
	}

	auto left_circ = _decompose(W, qv - qv.back());

	std::vector<double> angles_vec;
	for (int i = 0; i < D.size(); i++)
	{
		auto val = D(i);
		angles_vec.push_back(-2.0 * atan2(val.imag(), val.real()));
	}
	auto middle_circ = ucrz_decomposition(qv - qv.back(), qv.back(), angles_vec);
	//_check_ucrz(middle_circ, angles_vec, m_is_positive_seq);

	auto right_circ = _decompose(V, qv - qv.back());

	QCircuit total_circ;
	total_circ << left_circ
		<< middle_circ
		<< right_circ;

	return total_circ;
}

QCircuit unitary_decomposer_nq(const QStat& in_matrix, const QVec& qv,
	DecompositionMode type = DecompositionMode::QSD,
	bool is_positive_seq = false)
{
	auto _mat = qstat2eigen(in_matrix);
	return unitary_decomposer_nq(_mat, qv, type, is_positive_seq);
}


QCircuit QPanda::unitary_decomposer_nq(const Eigen::MatrixXcd& in_matrix,
	const QVec& qv, DecompositionMode type, bool is_positive_seq)
{
	QCircuit circ;
	switch (type)
	{
	case DecompositionMode::QR:
	case DecompositionMode::HOUSEHOLDER_QR:
		QCERR_AND_THROW(std::runtime_error, "QR or HOUSEHOLDER_QR is not supported");

	case DecompositionMode::QSD:
	case DecompositionMode::CSD:
	{
		QSDecomposition uder;
		circ = uder.synthesize_qcircuit(in_matrix, qv, type, is_positive_seq);
	}
	break;
	default:
		throw std::runtime_error("DecompositionMode error");
		break;
	}
	return circ;
}
#ifndef  _MATRIX_UTIL_H_
#define  _MATRIX_UTIL_H_

#include "ThirdParty/Eigen/Eigen"
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"

QPANDA_BEGIN

#define _ASSERT(con, argv)    do{		\
										if (!(con)){		\
											QCERR_AND_THROW_ERRSTR(run_fail, argv); \
										}	                \
									}while(0)

constexpr double tolerance = 1e-6;  // see tolerance 

inline QStat eigen2qstat(const Eigen::MatrixXcd & in_matrix)
{
	Eigen::MatrixXcd _mat = in_matrix;
	_mat.transposeInPlace();
	QStat _tmp(_mat.data(), _mat.data() + _mat.size());
	return _tmp;
}

inline Eigen::MatrixXcd qstat2eigen(const QStat & qstat)
{
	int _msize = (int)sqrt(qstat.size());
	QStat _tmp(qstat);
	Eigen::Map<Eigen::MatrixXcd> matrix(_tmp.data(), _msize, _msize);
	matrix.transposeInPlace();
	return matrix;
}


// Using QR decomposition to generate a dim*dim random unitary
inline Eigen::MatrixXcd random_unitary(int dim, unsigned int seed = time(NULL))
{
	std::srand(seed);
	Eigen::MatrixXcd mat = Eigen::MatrixXcd::Random(dim, dim);
	auto QR = mat.householderQr();
	Eigen::MatrixXcd unitary_mat = QR.householderQ() * Eigen::MatrixXcd::Identity(dim, dim);
	if (!unitary_mat.isUnitary(1e-11))
	{
		QCERR_AND_THROW(std::runtime_error, "is not unitary!!");
	}
	return unitary_mat;
}

inline bool is_square(const Eigen::MatrixXcd& _mat)
{
	return _mat.rows() == _mat.cols();
}

inline bool is_approx(const Eigen::MatrixXcd& _mat1, const Eigen::MatrixXcd& _mat2, double _tol = tolerance)
{
	if (!_mat1.allFinite() || !_mat2.allFinite()){
		return false;
	}

	if (_mat1.rows() == _mat2.rows() && _mat1.cols() == _mat2.cols())
	{
		for (int i = 0; i < _mat1.rows(); ++i)
		{
			for (int j = 0; j < _mat1.cols(); ++j)
			{
				if (std::abs(_mat1(i, j) - _mat2(i, j)) > _tol){
					return false;
				}
			}
		}
		return true;
	}

	return false;
}

inline bool is_hermitian(const Eigen::MatrixXcd& _mat, double _tol = tolerance)
{
	if (!is_square(_mat) || !_mat.allFinite())
		return false;

	return is_approx(_mat, _mat.adjoint(), _tol);
}

inline bool is_orthogonal(const Eigen::MatrixXcd& _mat, double _tol = tolerance)
{
	if (!is_square(_mat) || !_mat.allFinite())
		return false;

	// is real 
	for (int i = 0; i < _mat.rows(); ++i)
	{
		for (int j = 0; j < _mat.cols(); ++j)
		{
			if (std::abs(_mat(i, j).imag()) > _tol)
				return false;
		}
	}
	// its transpose is its inverse
	return is_approx(_mat.inverse(), _mat.transpose(), _tol);
}

inline bool is_special_orthogonal(const Eigen::MatrixXcd& _mat, double _tol = tolerance)
{
	// is orthogonal and determinant == 1
	return is_orthogonal(_mat) && (std::abs(std::abs(_mat.determinant()) - 1.0) < _tol);
}

inline bool is_diagonal(const Eigen::MatrixXcd& _mat, double _tol = tolerance)
{
	if (!_mat.allFinite()){
		return false;
	}

	for (int i = 0; i < _mat.rows(); ++i) 
	{
		for (int j = 0; j < _mat.cols(); ++j) 
		{
			if (i != j) 
			{
				if (std::abs(_mat(i, j)) > _tol) {
					return false;
				}
			}
		}
	}

	return true;
}

inline bool is_unitary(const Eigen::MatrixXcd& in_mat, double _tol = tolerance)
{
	if (!is_square(in_mat)) 
		return false;
	Eigen::MatrixXcd identity = Eigen::MatrixXcd::Identity(in_mat.rows(), in_mat.cols());
	return  is_approx(in_mat.adjoint() * in_mat, identity, _tol);
}

QPANDA_END

#endif // !_MATRIX_UTIL_H_
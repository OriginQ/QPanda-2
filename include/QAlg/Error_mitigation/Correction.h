/*
Copyright (c) 2017-2023 Origin Quantum Computing. All Right Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef  CORRECTION_H
#define  CORRECTION_H
#include <vector>
#include "QAlg/Encode/Encode.h"
#include "ThirdParty/Eigen/Eigen"
#include "include/QAlg/Error_mitigation/Sample.h"
#include <ThirdParty/Eigen/Core>
#include <ThirdParty/Eigen/Dense>
#include "ThirdParty/EigenUnsupported/Eigen/IterativeSolvers"
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/Utilities/Tools/RandomCircuit.h"
#include <bitset>
class MatrixReplacement;
namespace Eigen {
	namespace internal {
		// MatrixReplacement looks-like a SparseMatrix, so let's inherits its traits:
		template<>
		struct traits<MatrixReplacement> : public Eigen::internal::traits<Eigen::SparseMatrix<double> >
		{};
	}
}
class MatrixReplacement : public Eigen::EigenBase<MatrixReplacement> {
public:
	// Required typedefs, constants, and method:
	typedef double Scalar;
	typedef double RealScalar;
	typedef int StorageIndex;
	enum {
		ColsAtCompileTime = Eigen::Dynamic,
		MaxColsAtCompileTime = Eigen::Dynamic,
		IsRowMajor = false
	};

	Index rows() const { return mp_mat->rows(); }
	Index cols() const { return mp_mat->cols(); }

	template<typename Rhs>
	Eigen::Product<MatrixReplacement, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs>& x) const {
		return Eigen::Product<MatrixReplacement, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
	}

	// Custom API:
	MatrixReplacement() : mp_mat(0) {}

	void attachMyMatrix(const Eigen::SparseMatrix<double>& mat) {
		mp_mat = &mat;
	}
	const Eigen::SparseMatrix<double> my_matrix() const { return *mp_mat; }

private:
	const Eigen::SparseMatrix<double>* mp_mat;
};
namespace Eigen {
	namespace internal {

		template<typename Rhs>
		struct generic_product_impl<MatrixReplacement, Rhs, SparseShape, DenseShape, GemvProduct> // GEMV stands for matrix-vector
			: generic_product_impl_base<MatrixReplacement, Rhs, generic_product_impl<MatrixReplacement, Rhs> >
		{
			typedef typename Product<MatrixReplacement, Rhs>::Scalar Scalar;

			template<typename Dest>
			static void scaleAndAddTo(Dest& dst, const MatrixReplacement& lhs, const Rhs& rhs, const Scalar& alpha)
			{
				// This method should implement "dst += alpha * lhs * rhs" inplace,
				// however, for iterative solvers, alpha is always equal to 1, so let's not bother about it.
				assert(alpha == Scalar(1) && "scaling is not implemented");
				EIGEN_ONLY_USED_FOR_DEBUG(alpha);

				// Here we could simply call dst.noalias() += lhs.my_matrix() * rhs,
				// but let's do something fancier (and less efficient):
				for (Index i = 0; i < lhs.cols(); ++i)
					dst += rhs(i) * lhs.my_matrix().col(i);
			}
		};

	}
}
QPANDA_BEGIN
inline double kl_divergence(const std::vector<double>& input_data, const std::vector<double>& output_data) {
	int size = input_data.size();
	double result = 0.0;
	for (int i = 0; i < size; ++i) {
		if (input_data[i] - 0.0 > 1e-6 && output_data[i] - 0.0 > 1e-6) {
			result += input_data[i] * std::log(input_data[i] / output_data[i]);
		}
	}
	return abs(result);

}
inline size_t binary2ull(const std::string& str) {
	return std::stoull(str, nullptr, 2);
}
inline std::string ull2binary(const size_t& value, const size_t& length) {
	std::bitset<64> temp(value);
	std::string str = temp.to_string();
	std::string str_back(str.begin() + (64 - length), str.end());
	return str_back;
}
enum sample_method {
	Full = 0,
	Independ = 1,
	Balance = 2,
	BFA=3,
};


template <typename Scalar>
using dyn_col_vect = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
template <typename Scalar>
using dyn_mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

template<typename T>
std::vector<T> inv_correct(const dyn_mat<T>&A, const std::vector<T> &b)
{
	std::vector<T> miti_prob(b.size());
	dyn_mat<T> inv_A = A.inverse;
	dyn_col_vect<T> v2(b.size());
	int cnt = 0;
	for (T i : b) {
		v2(cnt++) = i;
	}
	dyn_col_vect<T> prob = inv_A * v2;
	for (int i = 0; i < b.size(); ++i) {
		miti_prob[i] = prob(i);
	}
	return miti_prob;
};

inline std::vector<double> gmres_correct(const Eigen::SparseMatrix<double> &A, const std::vector<double>& b)
{
	std::vector<double> miti_prob(b.size());

	dyn_col_vect<double> v2(b.size());
	int cnt = 0;
	for (double i : b) {
		v2(cnt++) = i;
	}
	MatrixReplacement A_;
	A_.attachMyMatrix(A);
	Eigen::GMRES<MatrixReplacement, Eigen::IdentityPreconditioner> gmres;
	gmres.compute(A_);
	dyn_col_vect<double> x = gmres.solve(v2);
	for (int i = 0; i < b.size(); ++i) {
		miti_prob[i] = x(i);
	}
	return miti_prob;
};

template<typename T>
std::vector<double> square_correct(const dyn_mat<T>& A, const std::vector<T>& b)
{
	std::vector<T> miti_prob(b.size());
	dyn_col_vect<T> v2(b.size());
	int cnt = 0;
	for (T i : b) {
		v2(cnt++) = i;
	}

	dyn_col_vect<T> prob = (A.transpose() * A).inverse() * A.transpose() * v2;

	prob.normalize();
	for (int i = 0; i < b.size(); ++i) {
		miti_prob[i] = prob(i)*prob(i);
	}
	return miti_prob;
};
class Mitigation 
{
public:
	Mitigation(const QVec &q);
	Mitigation(const QVec& q,QuantumMachine* qvm,NoiseModel &noise,size_t shots);
	void readout_error_mitigation(const sample_method &method,const std::vector<double>& m_unmiti_prob);
	void zne_error_mitigation(QCircuit& circuit, const std::vector<size_t>& order);
	void quasi_probability(QCircuit& circuit, const std::tuple<double, std::vector<std::pair<double, GateType>>>& representation, const int &num_samples);
	std::vector<double> get_miti_result() {
		return m_miti_prob;
	}
protected:
	std::vector<Eigen::Matrix2d> calc_circ_balance(const int& shots);
	std::vector<Eigen::Matrix2d> calc_circ_independ(const int& shots);
	std::vector<Eigen::Matrix2d> calc_circ_bfa(const int& shots);
	Eigen::MatrixXd prob2density(const std::map<std::string, size_t>& res, const size_t& size);
	double get_expection(Eigen::MatrixXd& A, Eigen::MatrixXd& Ham);
	std::vector<int> sample_circuit(const std::tuple<double, std::vector<std::pair<double, GateType>>>& representation, const int& num_samples, std::vector<QCircuit>& circuits, const QVec& q);
private:
	QVec m_qubits;
	size_t m_shots;
	QuantumMachine* m_qvm;
	NoiseModel m_noise;
	std::vector<double> m_miti_prob;

	Sample m_sample;
};


QPANDA_END

#endif
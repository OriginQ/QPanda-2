
#include "Core/Utilities/QProgInfo/KAK.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include "Core/Utilities/Tools/QStatMatrix.h"
#include "ThirdParty/EigenUnsupported/Eigen/KroneckerProduct"
#include "ThirdParty/EigenUnsupported/Eigen/MatrixFunctions"

constexpr std::complex<double> CPX_I{ 0.0, 1.0 };
const  double PRECISION = 1e-9;
#define  SQRT1_2  0.707106781186547524401  // 1/sqrt(2)

#define _ASSERT(con, argv)    do{											\
                                        if (!con)												\
                                        {														\
                                            throw std::runtime_error(argv);	\
                                        }														\
                                    }while(0)


USING_QPANDA
using namespace std;

// Define some special matrices
const static Eigen::MatrixXcd& MAGIC()
{
	static Eigen::MatrixXcd MAGIC(4, 4);
	static bool init = false;
	if (!init)
	{
		MAGIC << 1, 0, 0, CPX_I,
			0, CPX_I, 1, 0,
			0, CPX_I, -1, 0,
			1, 0, 0, -CPX_I;
		MAGIC = MAGIC * std::sqrt(0.5);
		init = true;
	}

	return MAGIC;
}

const static Eigen::MatrixXcd& MAGIC_DAG()
{
	static Eigen::MatrixXcd MAGIC_DAG = MAGIC().adjoint();
	return MAGIC_DAG;
}

const static Eigen::MatrixXcd& GAMMA()
{
	static Eigen::MatrixXcd GAMMA(4, 4);
	static bool init = false;
	if (!init)
	{
		GAMMA << 1, 1, 1, 1,
			1, 1, -1, -1,
			-1, 1, -1, 1,
			1, -1, -1, 1;

		GAMMA = 0.25 * GAMMA;
		init = true;
	}
	return GAMMA;
}

Eigen::Matrix4cd generate_random_unitary()
{
	// Using QR decomposition to generate a random unitary
	Eigen::Matrix4cd mat = Eigen::Matrix4cd::Random();
	auto QR = mat.householderQr();
	Eigen::Matrix4cd qMat = QR.householderQ() * Eigen::Matrix4cd::Identity();
	return qMat;
}

// Splits i = 0...length into approximate equivalence classes
// determine by the predicate
std::vector<std::pair<int, int>> contiguous_groups(int in_length, std::function<bool(int, int)> in_predicate)
{
	int start = 0;
	std::vector<std::pair<int, int>> result;
	while (start < in_length)
	{
		auto past = start + 1;
		while ((past < in_length) && in_predicate(start, past))
		{
			past++;
		}
		result.emplace_back(start, past);
		start = past;
	}
	return result;
}

Eigen::MatrixXd block_diagonal(const Eigen::MatrixXd& in_first, const Eigen::MatrixXd& in_second)
{
	Eigen::MatrixXd bdm = Eigen::MatrixXd::Zero(in_first.rows() + in_second.rows(), in_first.cols() + in_second.cols());
	bdm.block(0, 0, in_first.rows(), in_first.cols()) = in_first;
	bdm.block(in_first.rows(), in_first.cols(), in_second.rows(), in_second.cols()) = in_second;
	return bdm;
}

inline bool is_square(const Eigen::MatrixXcd& in_mat)
{
	return in_mat.rows() == in_mat.cols();
}

bool is_approx(const Eigen::MatrixXcd& in_mat1, const Eigen::MatrixXcd& in_mat2, double perc = 1e-9)
{
	if (!in_mat1.allFinite() || !in_mat2.allFinite())
		return false;

	if (in_mat1.rows() == in_mat2.rows() && in_mat1.cols() == in_mat2.cols())
	{
		for (int i = 0; i < in_mat1.rows(); ++i)
		{
			for (int j = 0; j < in_mat1.cols(); ++j)
			{
				if (std::abs(in_mat1(i, j) - in_mat2(i, j)) > perc)
					return false;
			}
		}
		return true;
	}
	return false;
}

bool is_hermitian(const Eigen::MatrixXcd& in_mat)
{
	if (!is_square(in_mat) || !in_mat.allFinite())
		return false;

	return is_approx(in_mat, in_mat.adjoint());
}

bool is_orthogonal(const Eigen::MatrixXcd& in_mat, double perc = 1e-9)
{
	if (!is_square(in_mat) || !in_mat.allFinite())
		return false;

	// Is real 
	for (int i = 0; i < in_mat.rows(); ++i)
	{
		for (int j = 0; j < in_mat.cols(); ++j)
		{
			if (std::abs(in_mat(i, j).imag()) > perc)
				return false;
		}
	}
	// its transpose is its inverse
	return is_approx(in_mat.inverse(), in_mat.transpose());
}

// Is Orthogonal and determinant == 1
bool is_special_orthogonal(const Eigen::MatrixXcd& in_mat, double perc = 1e-9)
{
	return is_orthogonal(in_mat) && (std::abs(std::abs(in_mat.determinant()) - 1.0) < perc);
}

bool is_canonicalized(double x, double y, double z, double perc = 1e-9)
{
	// 0 ¡Ü abs(z) ¡Ü y ¡Ü x ¡Ü pi/4 , if x = pi/4, z >= 0
	if (std::abs(z) >= 0 && y >= std::abs(z) && x >= y && x <= (PI / 4.0 + perc))
	{
		if (std::abs(x - PI / 4.0) < perc)
			return (z >= 0);

		return true;
	}
	return false;
}

bool is_diagonal(const Eigen::MatrixXcd& in_mat, double perc = 1e-9)
{
	if (!in_mat.allFinite())
		return false;

	for (int i = 0; i < in_mat.rows(); ++i)
	{
		for (int j = 0; j < in_mat.cols(); ++j)
		{
			if (i != j)
			{
				if (std::abs(in_mat(i, j)) > perc)
					return false;
			}
		}
	}
	return true;
}




// Compute exp(i(x XX + y YY + z ZZ)) matrix
Eigen::Matrix4cd exp_xyz_matrix(double x, double y, double z)
{
	Eigen::MatrixXcd X{ Eigen::MatrixXcd::Zero(2, 2) };
	Eigen::MatrixXcd Y{ Eigen::MatrixXcd::Zero(2, 2) };
	Eigen::MatrixXcd Z{ Eigen::MatrixXcd::Zero(2, 2) };
	X << 0, 1, 1, 0;
	Y << 0, -CPX_I, CPX_I, 0;
	Z << 1, 0, 0, -1;
	auto XX = Eigen::kroneckerProduct(X, X);
	auto YY = Eigen::kroneckerProduct(Y, Y);
	auto ZZ = Eigen::kroneckerProduct(Z, Z);
	Eigen::MatrixXcd herm = x * XX + y * YY + z * ZZ;
	herm = CPX_I * herm;
	Eigen::MatrixXcd unitary = herm.exp();
	return unitary;
}


QCircuit ZYZ_decomposition(double zAngleBefore, double yAngle, double zAngleAfter, Qubit * in_qubit)
{
	auto zExpBefore = zAngleBefore / PI - 0.5;
	auto  middleExp = yAngle / PI;
	QGate (*GATE_FUNC)(Qubit*, double)  = RX;

	auto  zExpAfter = zAngleAfter / PI + 0.5;

	const auto is_near_zeromod = [](double a, double period) -> bool {
		const auto halfPeriod = period / 2;
		const double perc = 1e-8;
		return std::abs(fmod(a + halfPeriod, period) - halfPeriod) < perc;
	};

	const auto to_quarter_turns = [](double in_exp) -> int {
		return static_cast<int>(round(2 * in_exp)) % 4;
	};

	const auto is_clifford_rotation = [&](double in_exp) -> bool {
		return is_near_zeromod(in_exp, 0.5);
	};

	const auto is_quarter_turn = [&](double in_exp) -> bool {
		return (is_clifford_rotation(in_exp) && to_quarter_turns(in_exp) % 2 == 1);
	};

	const auto is_half_turn = [&](double in_exp) -> bool {
		return (is_clifford_rotation(in_exp) && to_quarter_turns(in_exp) == 2);
	};

	const auto is_no_turn = [&](double in_exp) -> bool {
		return (is_clifford_rotation(in_exp) && to_quarter_turns(in_exp) == 0);
	};

	// Clean up angles
	if (is_clifford_rotation(zExpBefore))
	{
		if ((is_quarter_turn(zExpBefore) || is_quarter_turn(zExpAfter)) != (is_half_turn(middleExp) && is_no_turn(zExpBefore - zExpAfter)))
		{
			zExpBefore += 0.5;
			zExpAfter -= 0.5;
			GATE_FUNC = RY;
		}
		if (is_half_turn(zExpBefore) || is_half_turn(zExpAfter))
		{
			zExpBefore -= 1;
			zExpAfter += 1;
			middleExp = -middleExp;
		}
	}
	if (is_no_turn(middleExp))
	{
		zExpBefore += zExpAfter;
		zExpAfter = 0;
	}
	else if (is_half_turn(middleExp))
	{
		zExpAfter -= zExpBefore;
		zExpBefore = 0;
	}

	QCircuit cir;
	if (!is_no_turn(zExpBefore))
	{
		cir << RZ(in_qubit, zExpBefore * PI);
	}
	if (!is_no_turn(middleExp))
	{
		cir << GATE_FUNC(in_qubit, middleExp * PI);
	}
	if (!is_no_turn(zExpAfter))
	{
		cir << RZ(in_qubit, zExpAfter * PI);
	}
	return cir;
}

QCircuit generate_single_gate(const Eigen::Matrix2cd& in_mat, Qubit * in_qubit)
{
	// Use Z-Y decomposition of Nielsen and Chuang (Theorem 4.1).
	// An arbitrary one qubit gate matrix can be written as
	// U = [ exp(j*(a-b/2-d/2))*cos(c/2), -exp(j*(a-b/2+d/2))*sin(c/2)
	//       exp(j*(a+b/2-d/2))*sin(c/2), exp(j*(a+b/2+d/2))*cos(c/2)]
	// where a,b,c,d are real numbers.
	auto matrix = in_mat;
	double a = 0.0, bHalf = 0.0, cHalf = 0.0, dHalf = 0.0;

	if (!is_approx(matrix, Eigen::Matrix2cd::Identity()))
	{
		const auto checkParams = [&matrix](double a, double bHalf, double cHalf, double dHalf) {
			Eigen::Matrix2cd U;
			U << std::exp(CPX_I*(a - bHalf - dHalf))*std::cos(cHalf),
				-std::exp(CPX_I*(a - bHalf + dHalf))*std::sin(cHalf),
				std::exp(CPX_I*(a + bHalf - dHalf))*std::sin(cHalf),
				std::exp(CPX_I*(a + bHalf + dHalf))*std::cos(cHalf);

			return is_approx(U, matrix);
		};

		if (std::abs(matrix(0, 1)) < PRECISION)
		{
			auto two_a = fmod(std::arg(matrix(0, 0)*matrix(1, 1)), 2 * PI);
			a = (std::abs(two_a) < PRECISION || std::abs(two_a) > 2 * PI - PRECISION) ? 0 : two_a / 2.0;
			auto dHalf = 0.0;
			auto b = std::arg(matrix(1, 1)) - std::arg(matrix(0, 0));
			std::vector<double> possibleBhalf{ fmod(b / 2.0, 2 * PI), fmod(b / 2.0 + PI, 2.0 * PI) };
			std::vector<double> possibleChalf{ 0.0, PI };
			bool found = false;
			for (int i = 0; i < possibleBhalf.size(); ++i)
			{
				for (int j = 0; j < possibleChalf.size(); ++j)
				{
					bHalf = possibleBhalf[i];
					cHalf = possibleChalf[j];
					if (checkParams(a, bHalf, cHalf, dHalf))
					{
						found = true;
						break;
					}
				}
				if (found)
					break;
			}
			_ASSERT(found, "not found");
		}
		else if (std::abs(matrix(0, 0)) < PRECISION)
		{
			auto two_a = fmod(std::arg(-matrix(0, 1)*matrix(1, 0)), 2 * PI);
			a = (std::abs(two_a) < PRECISION || std::abs(two_a) > 2 * PI - PRECISION) ? 0 : two_a / 2.0;
			dHalf = 0;
			auto b = std::arg(matrix(1, 0)) - std::arg(matrix(0, 1)) + PI;
			std::vector<double> possibleBhalf{ fmod(b / 2., 2 * PI), fmod(b / 2. + PI, 2 * PI) };
			std::vector<double> possibleChalf{ PI / 2., 3. / 2.*PI };
			bool found = false;
			for (int i = 0; i < possibleBhalf.size(); ++i)
			{
				for (int j = 0; j < possibleChalf.size(); ++j)
				{
					bHalf = possibleBhalf[i];
					cHalf = possibleChalf[j];
					if (checkParams(a, bHalf, cHalf, dHalf))
					{
						found = true;
						break;
					}
				}
				if (found)
				{
					break;
				}
			}
			_ASSERT(found, "not found");
		}
		else
		{
			auto two_a = fmod(std::arg(matrix(0, 0)*matrix(1, 1)), 2 * PI);
			a = (std::abs(two_a) < PRECISION || std::abs(two_a) > 2 * PI - PRECISION) ? 0 : two_a / 2.0;
			auto two_d = 2.*std::arg(matrix(0, 1)) - 2.*std::arg(matrix(0, 0));
			std::vector<double> possibleDhalf{ fmod(two_d / 4., 2 * PI),
							  fmod(two_d / 4. + PI / 2., 2 * PI),
							  fmod(two_d / 4. + PI, 2 * PI),
							  fmod(two_d / 4. + 3. / 2.*PI, 2 * PI) };
			auto two_b = 2.*std::arg(matrix(1, 0)) - 2.*std::arg(matrix(0, 0));
			std::vector<double> possibleBhalf{ fmod(two_b / 4., 2 * PI),
							  fmod(two_b / 4. + PI / 2., 2 * PI),
							  fmod(two_b / 4. + PI, 2 * PI),
							  fmod(two_b / 4. + 3. / 2.*PI, 2 * PI) };
			auto tmp = std::acos(std::abs(matrix(1, 1)));
			std::vector<double> possibleChalf{ fmod(tmp, 2 * PI),
							  fmod(tmp + PI, 2 * PI),
							  fmod(-1.*tmp, 2 * PI),
							  fmod(-1.*tmp + PI, 2 * PI) };
			bool found = false;
			for (int i = 0; i < possibleBhalf.size(); ++i)
			{
				for (int j = 0; j < possibleChalf.size(); ++j)
				{
					for (int k = 0; k < possibleDhalf.size(); ++k)
					{
						bHalf = possibleBhalf[i];
						cHalf = possibleChalf[j];
						dHalf = possibleDhalf[k];
						if (checkParams(a, bHalf, cHalf, dHalf))
						{
							found = true;
							break;
						}
					}
					if (found)
						break;
				}
				if (found)
					break;

			}
			_ASSERT(found, "not found");
		}

		_ASSERT(checkParams(a, bHalf, cHalf, dHalf), "check param error");
	}
	// Final check:

	// Validate U = exp(j*a) Rz(b) Ry(c) Rz(d).
	const auto validate = [](const Eigen::Matrix2cd& in_mat, double a, double b, double c, double d) {
		Eigen::Matrix2cd Rz_b, Ry_c, Rz_d;
		Rz_b << std::exp(-CPX_I * b / 2.0), 0, 0, std::exp(CPX_I*b / 2.0);
		Rz_d << std::exp(-CPX_I * d / 2.0), 0, 0, std::exp(CPX_I*d / 2.0);
		Ry_c << std::cos(c / 2), -std::sin(c / 2), std::sin(c / 2), std::cos(c / 2);
		auto mat = std::exp(CPX_I*a)*Rz_b*Ry_c*Rz_d;
		return is_approx(in_mat, mat);
	};
	// Validate the *raw* decomposition
	_ASSERT(validate(in_mat, a, 2 * bHalf, 2 * cHalf, 2 * dHalf),  "not validate decomposition");

	// Simplify/optimize the sequence:
	std::vector < std::pair<std::string, double > > test_gates;
	QCircuit cir = ZYZ_decomposition(2 * dHalf, 2 * cHalf, 2 * bHalf, in_qubit);

	QStat mat_1 = getCircuitMatrix(cir);
	QStat  mat_2 = Eigen_to_QStat(in_mat);
	if (mat_1 != mat_2)
	{
		QCERR("mat_1  mat_2  not unequal !");
		throw invalid_argument("mat_1  mat_2  not unequal !");
	}
	return cir;
}

Eigen::MatrixXcd KakDescription::to_matrix() const
{
	auto before = Eigen::kroneckerProduct(b1, b0);
	auto after = Eigen::kroneckerProduct(a1, a0);
	Eigen::MatrixXcd unitary = exp_xyz_matrix(x, y, z);
	auto total = global_phase * after * unitary * before;
	return total;
}

QCircuit KakDescription::to_qcircuit(Qubit *  in_bit1, Qubit * in_bit2) const
{
	QCircuit interaction_gates;
	const double perc = 1e-8;
	auto target = exp_xyz_matrix(x, y, z);

	if (std::abs(z) >= perc)   	// Full decomposition is required
	{
		const double xAngle = PI * (x * -2 / PI + 0.5);
		const double yAngle = PI * (y * -2 / PI + 0.5);
		const double zAngle = PI * (z * -2 / PI + 0.5);
		interaction_gates << H(in_bit1)
			<< CZ(in_bit2, in_bit1)
			<< H(in_bit1)
			<< RZ(in_bit1, zAngle)
			<< RX(in_bit1, PI / 2.0)
			<< H(in_bit2)
			<< CZ(in_bit1, in_bit2)
			<< H(in_bit2)
			<< RY(in_bit1, yAngle)
			<< RX(in_bit2, xAngle)
			<< H(in_bit1)
			<< CZ(in_bit1, in_bit2)
			<< H(in_bit1)
			<< RX(in_bit2, -PI / 2.0)
			;
	}
	else if (y >= perc) 	// ZZ interaction is near zero: only XX and YY
	{
		const double xAngle = -2 * x;
		const double yAngle = -2 * y;
		interaction_gates << RX(in_bit1, PI / 2.0)
			<< H(in_bit1)
			<< CZ(in_bit2, in_bit1)
			<< H(in_bit1)
			<< RY(in_bit1, yAngle)
			<< RX(in_bit2, xAngle)
			<< H(in_bit1)
			<< CZ(in_bit1, in_bit2)
			<< H(in_bit1)
			<< RX(in_bit2, -PI / 2.0)
			;
	}
	else  // only XX is significant
	{
		const double xAngle = -2 * x;
		interaction_gates << H(in_bit1)
			<< CZ(in_bit2, in_bit1)
			<< RX(in_bit2, xAngle)
			<< CZ(in_bit1, in_bit2)
			<< H(in_bit1)
			;
	}

	auto mat_1 = getCircuitMatrix(interaction_gates);
	auto mat_2 = Eigen_to_QStat(target);
	if (mat_1 != mat_2)
	{
		QCERR("  decomposition error !");
		throw invalid_argument("  decomposition error !");
	}

	auto a0_gates = generate_single_gate(a0, in_bit2);
	auto a1_gates = generate_single_gate(a1, in_bit1);
	auto b0_gates = generate_single_gate(b0, in_bit2);
	auto b1_gates = generate_single_gate(b1, in_bit1);

	// U = g x (Gate A1 Gate A0) x exp(i(xXX + yYY + zZZ))x(Gate b1 Gate b0)
	QCircuit total_qcir;
	total_qcir << b0_gates
		<< b1_gates
		<< interaction_gates
		<< a0_gates
		<< a1_gates
		;

	return total_qcir;
}

KakDescription KAK::decompose(const Eigen::Matrix4cd& in_matrix)
{
	_ASSERT(in_matrix.isUnitary(), "not unitary");
	Eigen::MatrixXcd mInMagicBasis = MAGIC_DAG() * in_matrix * MAGIC();
	Eigen::Matrix4cd left, right;
	std::vector<std::complex<double>>  diag;

	bidiagonalize(mInMagicBasis, left, diag, right);

	Eigen::Matrix2cd a1, a0, b1, b0;
	so4_to_magic_su2s(left.transpose(), a1, a0);
	so4_to_magic_su2s(right.transpose(), b1, b0);
	_ASSERT(a0.isUnitary(), "not unitary");
	_ASSERT(a1.isUnitary(), "not unitary");
	_ASSERT(b0.isUnitary(), "not unitary");
	_ASSERT(b1.isUnitary(), "not unitary");

	Eigen::Vector4cd angles;
	for (size_t i = 0; i < 4; ++i)
	{
		angles(i) = std::arg(diag[i]);
	}
	auto factors = GAMMA() * angles;
	KakDescription result;
	{
		result.in_matrix = in_matrix;
		result.global_phase = std::exp(CPX_I * factors(0));
		result.a0 = a0;
		result.a1 = a1;
		result.b0 = b0;
		result.b1 = b1;
		result.x = factors(1).real();
		_ASSERT(std::abs(factors(1).imag()) < 1e-9, "error");
		result.y = factors(2).real();
		_ASSERT(std::abs(factors(2).imag()) < 1e-9, "error");
		result.z = factors(3).real();
		_ASSERT(std::abs(factors(3).imag()) < 1e-9, "error");
	}

	// Failed to validate
	if (!is_approx(result.to_matrix(), in_matrix))
	{
		QCERR("validateMatrix error !");
		throw invalid_argument("validateMatrix  error !");
	}

	auto canon = canonicalize(result.x, result.y, result.z);

	// Combine the single-qubit blocks:
	result.b1 = canon.b1 * result.b1;
	result.b0 = canon.b0 * result.b0;
	result.a1 = result.a1 * canon.a1;
	result.a0 = result.a0 * canon.a0;
	result.global_phase = result.global_phase * canon.global_phase;
	result.x = canon.x;
	result.y = canon.y;
	result.z = canon.z;

	_ASSERT(is_canonicalized(result.x, result.y, result.z), "not canonicalized");
	_ASSERT(is_approx(result.to_matrix(), in_matrix), "not approx");
	return result;
}

void KAK::bidiagonalize(const Eigen::Matrix4cd& in_matrix,
	Eigen::Matrix4cd&out_left, std::vector<std::complex<double>> & diagVec, Eigen::Matrix4cd &out_right) const
{
	Eigen::Matrix4d realMat;
	Eigen::Matrix4d imagMat;
	for (int row = 0; row < in_matrix.rows(); ++row)
	{
		for (int col = 0; col < in_matrix.cols(); ++col)
		{
			realMat(row, col) = in_matrix(row, col).real();
			imagMat(row, col) = in_matrix(row, col).imag();
		}
	}
	// Assert A X B.T and A.T X B are hermitian
	//_ASSERT(is_hermitian(realMat * imagMat.transpose()), "not hermitian" );
	//_ASSERT(is_hermitian(realMat.transpose() * imagMat), "not hermitian");

	Eigen::Matrix4d left;
	Eigen::Matrix4d right;
	bidiagonalize_rsm_products(realMat, imagMat, left, right);

	// Convert to special orthogonal w/o breaking diagonalization.
	if (left.determinant() < 0)
	{
		for (int i = 0; i < left.cols(); ++i)
		{
			left(0, i) = -left(0, i);
		}
	}
	if (right.determinant() < 0)
	{
		for (int i = 0; i < right.rows(); ++i)
		{
			right(i, 0) = -right(i, 0);
		}
	}

	auto diag = left * in_matrix * right;
	// Validate:
	_ASSERT(is_diagonal(diag), "not diagonal" );

	for (int i = 0; i < diag.rows(); ++i)
	{
		diagVec.emplace_back(diag(i, i));
	}
	out_left = left;
	out_right = right;
}

void KAK::kron_factor(const Eigen::Matrix4cd& in_matrix, std::complex<double> &out_g, Eigen::Matrix2cd& out_f1, Eigen::Matrix2cd &out_f2) const
{
	Eigen::Matrix2cd f1 = Eigen::Matrix2cd::Zero();
	Eigen::Matrix2cd f2 = Eigen::Matrix2cd::Zero();

	// Get row and column of the max element
	size_t a = 0;
	size_t b = 0;
	double maxVal = std::abs(in_matrix(a, b));
	for (int row = 0; row < in_matrix.rows(); ++row)
	{
		for (int col = 0; col < in_matrix.cols(); ++col)
		{
			if (std::abs(in_matrix(row, col)) > maxVal)
			{
				a = row;
				b = col;
				maxVal = std::abs(in_matrix(a, b));
			}
		}
	}

	// Extract sub-factors touching the reference cell.
	for (int i = 0; i < 2; ++i)
	{
		for (int j = 0; j < 2; ++j)
		{
			f1((a >> 1) ^ i, (b >> 1) ^ j) = in_matrix(a ^ (i << 1), b ^ (j << 1));
			f2((a & 1) ^ i, (b & 1) ^ j) = in_matrix(a ^ i, b ^ j);
		}
	}

	// Rescale factors to have unit determinants.
	f1 /= (std::sqrt(f1.determinant()));
	f2 /= (std::sqrt(f2.determinant()));

	//Determine global phase.
	std::complex<double> g = in_matrix(a, b) / (f1(a >> 1, b >> 1) * f2(a & 1, b & 1));
	if (g.real() < 0.0)
	{
		f1 *= -1;
		g = -g;
	}

	Eigen::Matrix4cd testMat = g * Eigen::kroneckerProduct(f1, f2);
	_ASSERT(is_approx(testMat, in_matrix), "not approx");

	out_g = g;
	out_f1 = f1;
	out_f2 = f2;
}

void KAK::so4_to_magic_su2s(const Eigen::Matrix4cd& in_matrix, Eigen::Matrix2cd &f1, Eigen::Matrix2cd &f2) const
{
	_ASSERT(is_special_orthogonal(in_matrix), "not pecial_orthogonal" );
	auto matInMagicBasis = MAGIC() * in_matrix * MAGIC_DAG();
	std::complex<double> g;
	kron_factor(matInMagicBasis, g, f1, f2);
}

Eigen::MatrixXd KAK::diagonalize_rsm(const Eigen::MatrixXd& in_mat) const
{
	_ASSERT(is_hermitian(in_mat), "not hermitian" );
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(in_mat);
	Eigen::MatrixXd p = solver.eigenvectors();

	// Orthogonal basis (Hermitian/symmetric matrix)  
	_ASSERT(is_orthogonal(p), "not orthogonal");

	// An orthogonal matrix P such that PT x matrix x P is diagonal.
	_ASSERT(is_diagonal(p.transpose() * in_mat * p), "not diagonal");
	return p;
}

Eigen::MatrixXd KAK::diagonalize_rsm_sorted_diagonal(const Eigen::MatrixXd& in_symMat, const Eigen::MatrixXd& in_diagMat) const
{
	_ASSERT(is_diagonal(in_diagMat), "not diagonal");
	_ASSERT(is_hermitian(in_symMat), "not hermitian");
	const auto similarSingular = [&in_diagMat](int i, int j) {
		return std::abs(in_diagMat(i, i) - in_diagMat(j, j)) < 1e-5;
	};

	const auto ranges = contiguous_groups(in_diagMat.rows(), similarSingular);
	Eigen::MatrixXd p = Eigen::MatrixXd::Zero(in_symMat.rows(), in_symMat.cols());

	for (auto iter : ranges)
	{
		int start = iter.first;
		int end = iter.second;

		const int blockSize = end - start;

		Eigen::MatrixXd block = Eigen::MatrixXd(blockSize, blockSize);
		for (int i = 0; i < blockSize; ++i)
		{
			for (int j = 0; j < blockSize; ++j)
			{
				block(i, j) = in_symMat(i + start, j + start);
			}
		}
		auto block_diag = diagonalize_rsm(block);

		for (int i = 0; i < blockSize; ++i)
		{
			for (int j = 0; j < blockSize; ++j)
			{
				p(i + start, j + start) = block_diag(i, j);
			}
		}
	}

	// P.T x symmetric_matrix x P is diagonal
	_ASSERT(is_diagonal(p.transpose() * in_symMat * p), "not diagonal");

	// and P.T x diagonal_matrix x P = diagonal_matrix
	_ASSERT(is_approx(p.transpose() * in_diagMat * p, in_diagMat), "not approx");
	return p;
}


void KAK::bidiagonalize_rsm_products(const Eigen::Matrix4d& in_mat1, const Eigen::Matrix4d& in_mat2,
	Eigen::Matrix4d &left, Eigen::Matrix4d &right) const
{
	// Use SVD to bi-diagonalize the first matrix.
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(in_mat1, Eigen::ComputeThinU | Eigen::ComputeThinV);
	auto baseLeft = svd.matrixU();
	auto baseDiagVec = svd.singularValues();
	auto baseRight = svd.matrixV().adjoint();

	Eigen::MatrixXd baseDiag = Eigen::MatrixXd::Zero(baseDiagVec.size(), baseDiagVec.size());
	for (int i = 0; i < baseDiagVec.size(); ++i)
	{
		baseDiag(i, i) = baseDiagVec(i);
	}

	// Determine where we switch between diagonalization-fixup strategies.
	const auto dim = baseDiag.rows();
	auto rank = dim;
	while (rank > 0 && std::abs(baseDiag(rank - 1, rank - 1) < 1e-5))
	{
		rank--;
	}
	Eigen::MatrixXd baseDiagTrim = Eigen::MatrixXd::Zero(rank, rank);
	for (int i = 0; i < rank; ++i)
	{
		for (int j = 0; j < rank; ++j)
		{
			baseDiagTrim(i, j) = baseDiag(i, j);
		}
	}

	// Try diagonalizing the second matrix with the same factors as the first.
	auto semiCorrected = baseLeft.transpose() * in_mat2 * baseRight.transpose();

	Eigen::MatrixXd overlap = Eigen::MatrixXd::Zero(rank, rank);
	for (int i = 0; i < rank; ++i)
	{
		for (int j = 0; j < rank; ++j)
		{
			overlap(i, j) = semiCorrected(i, j);
		}
	}

	auto overlapAdjust = diagonalize_rsm_sorted_diagonal(overlap, baseDiagTrim);

	const auto extraSize = dim - rank;
	Eigen::MatrixXd extra(extraSize, extraSize);
	for (int i = 0; i < extraSize; ++i)
	{
		for (int j = 0; j < extraSize; ++j)
		{
			extra(i, j) = semiCorrected(i + rank, j + rank);
		}
	}
	Eigen::MatrixXd extraLeftAdjust;
	Eigen::VectorXd extraDiag;
	Eigen::MatrixXd extraRightAdjust;
	if (dim > rank)
	{
		Eigen::JacobiSVD<Eigen::MatrixXd> svd2(extra, Eigen::ComputeThinU | Eigen::ComputeThinV);
		extraLeftAdjust = svd2.matrixU();
		extraDiag = svd2.singularValues();
		extraRightAdjust = svd2.matrixV().adjoint();
	}

	auto leftAdjust = block_diagonal(overlapAdjust, extraLeftAdjust);
	auto rightAdjust = block_diagonal(overlapAdjust.transpose(), extraRightAdjust);
	left = leftAdjust.transpose() * baseLeft.transpose();
	right = baseRight.transpose() * rightAdjust.transpose();
	// L x mat1 x R and L x mat2 x R are diagonal matrices.
	_ASSERT(is_diagonal(left * in_mat1 * right), "not diagonal");
	_ASSERT(is_diagonal(left * in_mat2 * right), "not diagonal");
}

KakDescription KAK::canonicalize(double x, double y, double z) const
{
	// Accumulated global phase.
	std::complex<double> phase = 1.0;
	//Per-qubit left factors.
	std::vector<Eigen::Matrix2cd> left{ Eigen::Matrix2cd::Identity(), Eigen::Matrix2cd::Identity() };
	// Per-qubit right factors.
	std::vector<Eigen::Matrix2cd> right{ Eigen::Matrix2cd::Identity(), Eigen::Matrix2cd::Identity() };
	// Remaining XX/YY/ZZ interaction vector.
	std::vector<double> v{ x, y, z };

	std::vector<Eigen::Matrix2cd> flippers{
	  (Eigen::Matrix2cd() << 0, CPX_I, CPX_I, 0).finished(),
	  (Eigen::Matrix2cd() << 0, 1, -1, 0).finished(),
	  (Eigen::Matrix2cd() << CPX_I, 0, 0, -CPX_I).finished()
	};

	std::vector<Eigen::Matrix2cd> swappers{
	  (Eigen::Matrix2cd() << CPX_I * SQRT1_2, SQRT1_2, -SQRT1_2, -CPX_I * SQRT1_2).finished(),
	  (Eigen::Matrix2cd() << CPX_I * SQRT1_2, CPX_I*SQRT1_2, CPX_I*SQRT1_2, -CPX_I * SQRT1_2).finished(),
	  (Eigen::Matrix2cd() << 0, CPX_I*SQRT1_2 + SQRT1_2, CPX_I*SQRT1_2 - SQRT1_2, 0).finished()
	};

	const auto shift = [&](int k, int step) {
		v[k] += step * PI / 2.0;
		phase *= std::pow(CPX_I, step);
		const auto expFact = ((step % 4) + 4) % 4;
		const Eigen::Matrix2cd mat = flippers[k].array().pow(expFact);
		right[0] = mat * right[0];
		right[1] = mat * right[1];
	};

	const auto negate = [&](int k1, int k2) {
		v[k1] *= -1;
		v[k2] *= -1;
		phase *= -1;
		const auto& s = flippers[3 - k1 - k2];
		left[1] = left[1] * s;
		right[1] = s * right[1];
	};

	const auto swap = [&](int k1, int k2) {
		std::iter_swap(v.begin() + k1, v.begin() + k2);
		const auto& s = swappers[3 - k1 - k2];
		left[0] = left[0] * s;
		left[1] = left[1] * s;
		right[0] = s * right[0];
		right[1] = s * right[1];
	};

	const auto canonical_shift = [&](int k) {
		while (v[k] <= -PI / 4.0)
		{
			shift(k, +1);
		}
		while (v[k] > PI / 4.0)
		{
			shift(k, -1);
		}
	};

	const auto sort = [&]() {
		if (std::abs(v[0]) < std::abs(v[1]))
		{
			swap(0, 1);
		}
		if (std::abs(v[1]) < std::abs(v[2]))
		{
			swap(1, 2);
		}
		if (std::abs(v[0]) < std::abs(v[1]))
		{
			swap(0, 1);
		}
	};

	canonical_shift(0);
	canonical_shift(1);
	canonical_shift(2);
	sort();

	if (v[0] < 0)
	{
		negate(0, 2);
	}
	if (v[1] < 0)
	{
		negate(1, 2);
	}
	canonical_shift(2);

	if ((v[0] > PI / 4.0 - 1e-9) && (v[2] < 0))
	{
		shift(0, -1);
		negate(0, 2);
	}

	_ASSERT(is_canonicalized(v[0], v[1], v[2]), "not canonicalized");
	KakDescription result;
	{
		result.global_phase = phase;
		result.a0 = left[1];
		result.a1 = left[0];
		result.b0 = right[1];
		result.b1 = right[0];
		result.x = v[0];
		result.y = v[1];
		result.z = v[2];
	}

	_ASSERT(is_approx(result.to_matrix(), exp_xyz_matrix(x, y, z)), "not approx");
	return result;
}

QCircuit QPanda::random_kak_qcircuit(Qubit *in_qubit1, Qubit *in_qubit2)
{
	KAK kak;
	Eigen::Matrix4cd mat = generate_random_unitary();
	KakDescription kak_desc = kak.decompose(mat);
	return kak_desc.to_qcircuit(in_qubit1, in_qubit2);
}


#include "Core/Utilities/UnitaryDecomposer/UniformlyControlledGates.h"
#include "Core/Utilities/UnitaryDecomposer/MatrixUtil.h"
#include "Core/Utilities/QProgInfo/KAK.h"
#include "Core/Utilities/Tools/Uinteger.h"
USING_QPANDA

static Eigen::Matrix2cd _rz(double angle)
{
	Eigen::Matrix2cd   mat;
	auto val0 = exp(qcomplex_t(0, 1) * angle / 2.0);
	auto val3 = exp(qcomplex_t(0, -1) * angle / 2.0);
	mat << val0, 0, 0, val3;
	return mat;
}

static Eigen::Matrix2cd _h()
{
	Eigen::Matrix2cd mat;
	mat << 1, 1, 1, -1;
	mat = mat * sqrt(2) / 2.0;
	return mat;
}

static std::string _int2binarystr(int num)
{
	std::string res;
	while (true)
	{
		res += std::to_string(num % 2);
		num = num / 2;
		if (num == 0)
			break;
	}
	std::reverse(res.begin(), res.end());
	return res;
}

void _dec_uc_sqg(const std::vector<Eigen::MatrixXcd>& um_vec,
	std::vector <Eigen::MatrixXcd>& single_qubit_gates,
	Eigen::VectorXcd& diag)
{
	const double TOL = 1e-9;
	int num_contr = log2(um_vec.size());
	single_qubit_gates = um_vec;
	diag = Eigen::VectorXcd::Ones(pow(2, num_contr + 1));
	for (int dec_step = 0; dec_step < num_contr; dec_step++)
	{
		int num_ucgs = pow(2, dec_step);
		for (int ucg_idx = 0; ucg_idx < num_ucgs; ucg_idx++)
		{
			int len_ucg = pow(2, (num_contr - dec_step));
			for (int i = 0; i < (int)len_ucg / 2; i++)
			{
				int shift = ucg_idx * len_ucg;
				auto a = single_qubit_gates[shift + i];
				auto b = single_qubit_gates[shift + (int)std::floor(len_ucg / 2) + i];
				auto x = a * b.adjoint();
				auto det_x = x.determinant();
				auto x11 = x(0, 0) / sqrt(det_x);
				auto phi = std::arg(det_x);
				auto r1 = std::exp(qcomplex_t(0, 1) / 2.0 * (PI / 2.0 - phi / 2.0 - std::arg(x11)));
				auto r2 = std::exp(qcomplex_t(0, 1) / 2.0 * (PI / 2.0 - phi / 2.0 + std::arg(x11) + PI));
				Eigen::Matrix2cd r;
				r << r1, 0, 0, r2;

				Eigen::ComplexEigenSolver<Eigen::Matrix2cd> esolver(r * x * r);
				Eigen::Vector2cd d = esolver.eigenvalues();
				Eigen::Matrix2cd u = esolver.eigenvectors();
				if (std::abs(d[0] + qcomplex_t(0, 1)) < TOL)
				{
					d.colwise().reverseInPlace();
					u.rowwise().reverseInPlace();
				}
				Eigen::MatrixXcd dmat = d.cwiseSqrt().asDiagonal();
				auto v = dmat * u.adjoint() * r.adjoint() * b;

				//std::cout << "v :\n" << v << std::endl;
				//std::cout << "u :\n" << u << std::endl;
				//std::cout << "r :\n" << r << std::endl;

				single_qubit_gates[shift + i] = v;
				single_qubit_gates[shift + (int)std::floor(len_ucg / 2) + i] = u;
				if (ucg_idx < num_ucgs - 1)
				{
					auto k = shift + len_ucg + i;
					single_qubit_gates[k] = single_qubit_gates[k].eval() * r.adjoint() * (_rz(PI / 2)(0, 0));
					k = k + (int)std::floor(len_ucg / 2);
					single_qubit_gates[k] = single_qubit_gates[k].eval() * r * (_rz(PI / 2)(1, 1));
				}
				else
				{
					for (int ucg_idx_2 = 0; ucg_idx_2 < num_ucgs; ucg_idx_2++)
					{
						int shift_2 = ucg_idx_2 * len_ucg;
						auto k = 2 * (i + shift_2);
						diag[k] = (diag[k] * r.adjoint()(0, 0) * _rz(PI / 2.0)(0, 0));
						diag[k + 1] = (diag[k + 1] * r.adjoint()(1, 1) * _rz(PI / 2.0)(0, 0));
						k = k + len_ucg;
						diag[k] *= r(0, 0) * _rz(PI / 2.0)(1, 1);
						diag[k + 1] *= r(1, 1) * _rz(PI / 2.0)(1, 1);
					}
				}
			}
		}
	}
}

std::vector<int> _gray_code(int n)
{
	std::vector<int> ret(1 << n);

#pragma omp parallel for num_threads(omp_get_max_threads())
	for (int i = 0; i < ret.size(); i++)
		ret[i] = (i >> 1) ^ i;

	return ret;
}

int _matrix_M_entry(const int row, const int col)
{
	int b_and_g = row & ((col >> 1) ^ col);
	int sum_of_ones = 0;
	while (b_and_g > 0) {
		if (b_and_g & 0b1) {
			sum_of_ones += 1;
		}
		b_and_g = b_and_g >> 1;
	}

	return  sum_of_ones & 1 == 1 ? -1 : 1;
}

std::vector<double> _compute_theta(prob_vec alpha)
{
	int ln = alpha.size();
	int k = log2(ln);
	std::vector<double>theta(ln);

#pragma omp parallel for num_threads(omp_get_max_threads())
	for (int i = 0; i < ln; ++i)
	{
		double angle = 0.0;
#pragma omp parallel for num_threads(omp_get_max_threads())
		for (int j = 0; j < ln; ++j) {
			angle += alpha[j] * _matrix_M_entry(j, i);
		}
		theta[i] = angle / (1 << k);
	}
	return theta;
}

/*
*  Decomposes a diagonal matrix into elementary gates
*  see https://arxiv.org/pdf/quant-ph/0406176.pdfin Theorem 7
*/
QCircuit QPanda::diagonal_decomposition(QVec qv, std::vector<qcomplex_t> diag_vec)
{
	QCircuit circ;
	int qnum = log2(diag_vec.size());
	_ASSERT(pow(2, qnum) == diag_vec.size(), "diag size must be 2^k");

	std::vector<double> diag_phases;
	for (const auto& val : diag_vec)
	{
		if (std::abs(std::abs(val) - 1) > 1e-10) {
			QCERR_AND_THROW(std::runtime_error, "val error");
		}
		diag_phases.push_back(std::arg(val));
	}
	auto n = diag_phases.size();
	while (n >= 2)
	{
		std::vector<double> rz_angles;
		for (int i = 0; i < n; i += 2)
		{
			auto phase = (diag_phases[i + 1] + diag_phases[i]) / 2.0;
			auto angle = diag_phases[i + 1] - diag_phases[i];

			diag_phases[std::floor(i / 2)] = phase;
			rz_angles.emplace_back(angle);
		}
		QVec ctrl_qv;
		auto act_qnum = (int)log2(n);
		for (int j = (qnum - act_qnum + 1); j < qnum; j++) {
			ctrl_qv.emplace_back(qv[j]);
		}
		auto target_q = qv[qnum - act_qnum];
		circ << ucrz_decomposition(ctrl_qv, target_q, rz_angles);
		n = std::floor(n / 2);
	}
	return circ;
}

/*
* Uniformly controlled gate ,also called multiplexed gate.
* These gates can have several control qubits and a single target qubit.
* If the k control qubits are in the state |i> (in the computational basis),
* a single-qubit unitary U_i is applied to the target qubit.
* This gate is represented by a block-diagonal matrix, where each block is a
* 2x2 unitary:
						U_0, 0,   ....        0,
						0,   U_1, ...,        0,
								.
									.
						0,   0,  ..., U_(2^k-1)
* Decomposition see: https://arxiv.org/pdf/quant-ph/0410066.pdf
*/
QCircuit QPanda::uc_decomposition(QVec ctrl_qv, Qubit* target_q,
	const std::vector<Eigen::MatrixXcd>& um_vec)
{
	QCircuit circ;
	QVec qv = QVec(target_q) + ctrl_qv;
	auto num_contr = (int)log2(um_vec.size());
	_ASSERT(pow(2, num_contr) == um_vec.size(), "need power of 2");
	_ASSERT(num_contr == ctrl_qv.size(), "not correspond  ctrl qubits size");

	for (auto& um : um_vec)
	{
		if (um.rows() != 2 || um.cols() != 2 || !is_unitary(um))
		{
			QCERR("uc_circuit not valid matrix!");
			throw std::runtime_error("uc_circuit not valid matrix!");
		}
	}

	if (ctrl_qv.empty()) {
		return unitary_decomposer_1q(um_vec[0], target_q);
	}

	std::vector <Eigen::MatrixXcd> sq_gates;
	Eigen::VectorXcd _diag;
	_dec_uc_sqg(um_vec, sq_gates, _diag);
	for (int i = 0; i < sq_gates.size(); i++)
	{
		auto gate = sq_gates[i];
		Eigen::Matrix2cd squ;
		if (i == 0) {
			squ = _h() * gate;
		}
		else if (i == sq_gates.size() - 1) {
			squ = gate * _rz(PI / 2.0) * _h();
		}
		else {
			squ = _h() * (gate * _rz(PI / 2.0)) * _h();
		}
		circ << unitary_decomposer_1q(squ, target_q);
		std::string binary_rep = _int2binarystr(i + 1);
		int rstrip_0_cnt = binary_rep.size();
		for (int j = binary_rep.size() - 1; j >= 0; j--)
		{
			if (binary_rep[j] != '0') {
				break;
			}

			rstrip_0_cnt--;
		}

		auto ctrl_idx = binary_rep.size() - rstrip_0_cnt;
		if (i != sq_gates.size() - 1) {
			circ << CNOT(ctrl_qv[ctrl_idx], target_q);
		}
	}
	std::vector<qcomplex_t> diag_vec(_diag.data(), _diag.data() + _diag.size());
	circ << diagonal_decomposition(qv, diag_vec);

	return circ;
}


QCircuit QPanda::ucry_circuit(QVec controls, Qubit* target, prob_vec params)
{
	auto control_num = controls.size();
	auto indices_num = 1ull << control_num;
    std::sort(controls.begin(), controls.end(), [&](Qubit* a, Qubit* b)
    {
        return a->get_phy_addr() > b->get_phy_addr();
    });

	std::vector<std::string> binary_indices;
	for (size_t i = 0; i < indices_num; ++i)
		binary_indices.emplace_back(integerToBinary(i, control_num));

	QCircuit ucry_circuit;
	for (auto index = 0; index < indices_num; ++index)
	{
		auto binary_string = binary_indices[index];

		QCircuit filp_circuit;
		for (auto i = 0; i < binary_string.size(); ++i)
		{
			if (binary_string[i] == '0')
				filp_circuit << X(controls[i]);
		}

		ucry_circuit << filp_circuit;
		ucry_circuit << RY(target, params[index]).control(controls);
		ucry_circuit << filp_circuit;
	}

	return ucry_circuit;
}

QCircuit QPanda::ucry_decomposition(QVec controls, Qubit* target, prob_vec params)
{
	const double TOL = 1e-10;
	QCircuit ucry_result;
	if (controls.empty())
	{
		if (std::abs(params[0]) > TOL) {
			ucry_result << RY(target, params[0]);
		}
		return ucry_result;
	}
	auto control_num = controls.size();
	auto indices_num = 1ull << control_num;

	std::vector<double> theta = _compute_theta(params);

	std::vector<int> code = _gray_code(control_num);

	int num_selections = code.size();
	std::vector<int>control_indices(num_selections);

#pragma omp parallel for num_threads(omp_get_max_threads())
	for (int i = 0; i < num_selections; ++i)
		control_indices[i] = log2(code[i] ^ (code[(i + 1) % num_selections]));

	for (int i = 0; i < control_indices.size(); ++i)
	{
		if (std::abs(theta[i]) > TOL) {
			ucry_result << RY(target, theta[i]);
		}

		ucry_result << CNOT(controls[control_indices[i]], target);
	}

	return ucry_result;
}


QCircuit QPanda::ucrz_decomposition(QVec controls, Qubit* target, prob_vec params)
{
	const double TOL = 1e-10;
	QCircuit ucrz_result;
	if (controls.empty())
	{
		if (std::abs(params[0]) > TOL) {
			ucrz_result << RZ(target, params[0]);
		}
		return ucrz_result;
	}
	auto control_num = controls.size();
	auto indices_num = 1ull << control_num;

	std::vector<double> theta = _compute_theta(params);

	std::vector<int> code = _gray_code(control_num);

	int num_selections = code.size();
	std::vector<int>control_indices(num_selections);

#pragma omp parallel for num_threads(omp_get_max_threads())
	for (int i = 0; i < num_selections; ++i)
		control_indices[i] = log2(code[i] ^ (code[(i + 1) % num_selections]));

	for (int i = 0; i < control_indices.size(); ++i)
	{
		if (std::abs(theta[i]) > TOL) {
			ucrz_result << RZ(target, theta[i]);
		}

		ucrz_result << CNOT(controls[control_indices[i]], target);
	}

	return ucrz_result;
}

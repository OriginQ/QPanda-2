
#include "Core/Utilities/UnitaryDecomposer/IsometryDecomposition.h"
#include "Core/Utilities/UnitaryDecomposer/UniformlyControlledGates.h"
#include "Core/Utilities/UnitaryDecomposer/MatrixUtil.h"
#include "QAlg/Base_QCircuit/AmplitudeEncode.h"
#include "ThirdParty/EigenUnsupported/Eigen/KroneckerProduct"
#include "Core/Utilities/UnitaryDecomposer/QSDecomposition.h"
USING_QPANDA
using namespace Eigen;


static bool _is_isometry(const MatrixXcd& isometry, size_t log_cols, double _tol = tolerance)
{
	MatrixXcd _mat = isometry.adjoint() * isometry;
	return _mat.isIdentity(_tol);
}


MatrixXcd IsometryDecomposition::_extend_to_unitary(const MatrixXcd& iso, size_t lines, size_t cols)
{
	auto _kernel_cod = [](const MatrixXcd &mat) {
		Eigen::CompleteOrthogonalDecomposition<MatrixXcd>cod;
		cod.compute(mat);
		unsigned rk = cod.rank();
		MatrixXcd P = cod.colsPermutation();
		MatrixXcd V = cod.matrixZ().transpose();
		MatrixXcd kernel = P * V.block(0, rk, V.rows(), V.cols() - rk);
		return kernel;
	};

	MatrixXcd unitary;
	if (lines == cols)
	{
		unitary = iso;
	}
	else
	{
		MatrixXcd null_space = _kernel_cod(iso.transpose()).conjugate();
		//size_t lines = 1<<log_lines;
		//size_t cols = 1<<log_cols;
		unitary = MatrixXcd::Zero(lines, lines);
		unitary.block(0, 0, lines, cols) = iso;
		unitary.block(0, cols, lines, lines - cols) = null_space;
	}

	return unitary;
}

QCircuit IsometryDecomposition::_knill(const MatrixXcd& iso, const QVec & qv, size_t log_lines, size_t log_cols)
{
	_ASSERT(log_lines > 1, "Knill decomposition does not work on a 1 qubit isometry (N=2)");
	auto unitary = _extend_to_unitary(iso, log_lines, log_cols);
	/*ComplexEigenSolver<MatrixXcd> esolver(unitary);
	VectorXcd eigval = esolver.eigenvalues();
	MatrixXcd  eigvec = esolver.eigenvectors();
	std::vector<double> arg;
	for (int i = 0; i < eigval.size(); i++){
		arg.push_back(atan2(eigval(i).imag(), eigval(i).real()));
	}

	QCircuit circ;
	for (int i = 0; i < pow(2, log_lines); i++)
	{
		if (std::abs(arg[i]) > 1e-7)
		{
			MatrixXcd  state = eigvec.col(i);
			Encode encode;
			auto data = eigen2qstat(state);
			encode.amplitude_encode(qv, data);
			auto sub_circ = encode.get_circuit();
			circ << sub_circ.dagger();
			circ << X(qv);

			QVec ctrl_qv;
			for (int j = 0; j < log_lines - 1; j++){
				ctrl_qv.push_back(qv[j]);
			}
			circ << P(qv[log_lines-1], arg[i]).control(ctrl_qv);
			circ << X(qv);
			circ << sub_circ;
		}
	}*/
	QCircuit circuit = unitary_decomposer_nq(unitary, qv, DecompositionMode::QSD, true);
	return circuit;
}

Eigen::MatrixXcd IsometryDecomposition::_unitary(const Eigen::MatrixXcd& iso, int basis = 0)
{
	if (basis > 1 || basis < 0){
		QCERR_AND_THROW_ERRSTR(run_fail, "basis 0 or 1");
	}
	Eigen::MatrixXcd  iden = Eigen::MatrixXcd::Identity(2,2);
	auto iso_norm = iso.col(0).norm();
	if (std::abs(iso_norm) < 1e-6){
		return iden;
	}

	auto psi = iso / iso_norm;
	auto psi_dagger = psi.adjoint();

	auto val1 = (-psi_dagger * iden.col(1))(0);
	auto val2 = (psi_dagger * iden.col(0))(0);
	auto phi = val1 * iden.col(0) + val2 * iden.col(1);
	auto phi_dagger = phi.adjoint();

	Eigen::MatrixXcd  unitary = kroneckerProduct(iden.col(basis), psi_dagger)
		+ kroneckerProduct(iden.col(1 - basis), phi_dagger);

	//std::cout << "unitary  : \n" << unitary << std::endl;

	return unitary;
}

Eigen::MatrixXcd  IsometryDecomposition::_mc_unitary(const Eigen::MatrixXcd& iso, size_t col_idx, size_t bit_idx)
{
	auto col_mat = iso.col(col_idx);
	auto idx1 = 2 * _a(col_idx, bit_idx + 1) * pow(2, bit_idx) + _b(col_idx, bit_idx + 1);
	auto idx2 = (2 * _a(col_idx, bit_idx + 1) + 1) * pow(2, bit_idx) + _b(col_idx, bit_idx + 1);
	Eigen::MatrixXcd mat(2, 1);
	mat << col_mat[idx1], col_mat[idx2];
	return _unitary(mat);
}

QCircuit IsometryDecomposition::_mc_gate(const MatrixXcd& unitary, const QVec& qv,
	const vector<size_t>& ctrl, const size_t& target, const std::string& k_bin)
{
	QVec ctrl_qv;
	for (const auto& i : ctrl)
	{
		if (k_bin[i] == '1'){
			ctrl_qv.emplace_back(qv[i]);
		}
	}
	reverse(ctrl_qv.begin(), ctrl_qv.end());

	auto iden = MatrixXcd::Identity(2, 2);
	std::vector<MatrixXcd> um_vec{ iden, unitary };
	QCircuit  ucg_circ = uc_decomposition(ctrl_qv, qv[target], um_vec, true);
	return ucg_circ;
}

void IsometryDecomposition::_update_isometry(const QCircuit& mcg_circ, const QVec & qv, MatrixXcd& iso)
{
	QProg temp_prog;
	temp_prog << I(qv) << mcg_circ;
	auto circ_mat = getCircuitMatrix(temp_prog, true);
	auto  mat = qstat2eigen(circ_mat);
	iso = mat * iso;
	//std::cout << "_update_isometry : \n" << iso << std::endl;
}


std::vector<Eigen::MatrixXcd> 
IsometryDecomposition::_uc_unitaries(const  Eigen::MatrixXcd& iso, size_t n_qubits, size_t col_idx, size_t bit_idx)
{
	auto start = _a(col_idx, bit_idx + 1) + 1;
	if (_b(col_idx, bit_idx + 1) == 0){
		start = _a(col_idx, bit_idx + 1);
	}
	std::vector<MatrixXcd> gates;
	for (size_t i = 0; i < start; i++){
		gates.emplace_back(MatrixXcd::Identity(2, 2));
	}
	auto col_mat = iso.col(col_idx);
	size_t _one = 1;
	size_t end = _one << (n_qubits - bit_idx - 1);
	for (size_t i = start; i < end; i++)
	{
		auto idx1 = 2 * i * (_one << bit_idx) + _b(col_idx, bit_idx);
		auto idx2 = (2 * i + 1) * (_one << bit_idx) + _b(col_idx, bit_idx);
 
		Eigen::MatrixXcd temp_mat(2, 1);
		temp_mat << col_mat[idx1], col_mat[idx2];
		gates.emplace_back(_unitary(temp_mat, _k_s(col_idx, bit_idx)));
	}

	return gates;
}



QCircuit IsometryDecomposition::_decompose_column(const QVec& qv, size_t log_lines, size_t col_idx, MatrixXcd& iso)
{
	QCircuit circ;
	std::string col_idx_bin = integerToBinary(col_idx, log_lines);

	for (size_t i = 0; i < log_lines; i++)
	{
		size_t target = log_lines - i - 1;
		std::vector<size_t>  ctrl_ancilla;
		QVec ctrl_qv;
		for (size_t j = 0; j < target; j++) {
			ctrl_qv.emplace_back(qv[j]);
			ctrl_ancilla.emplace_back(j);
		}
		for (size_t j = target + 1; j < log_lines; j++) {
			ctrl_ancilla.emplace_back(j);
		}

		if (_k_s(col_idx, i) == 0 && _b(col_idx, i + 1) != 0)
		{
			auto unitary = _mc_unitary(iso, col_idx, i);
			auto mcg = _mc_gate(unitary, qv, ctrl_ancilla, target, col_idx_bin);
			_update_isometry(mcg, qv, iso);
			circ << mcg;
		}

		auto unitaries = _uc_unitaries(iso, log_lines, col_idx, i);
		auto ucg = uc_decomposition(ctrl_qv, qv[target], unitaries, true);
		_update_isometry(ucg, qv, iso);
		circ << ucg;
	}
	return circ;
}

QCircuit IsometryDecomposition::_ccd(const MatrixXcd& iso_mat, const QVec& qv, size_t log_lines, size_t log_cols)
{
	QCircuit circ;
	MatrixXcd iso = iso_mat;
	auto reverse_qv = qv;
	std::reverse(reverse_qv.begin(), reverse_qv.end());
	auto cols = 1 << log_cols;
	for (size_t i = 0; i < cols; i++){
		circ << _decompose_column(reverse_qv, log_lines, i, iso);
	}

	if (log_cols > 0)
	{
		auto temp = iso.block(0, 0, cols, cols).diagonal();
		std::vector<qcomplex_t> diag;
		for (int i = 0; i < temp.size(); i++) {
			auto phase =  atan2(temp(i).imag(), temp(i).real());
			diag.emplace_back(std::exp(qcomplex_t(0, -1) * phase));
		}
		circ << diagonal_decomposition(qv, diag);
	}

	return circ.dagger();
}

QCircuit IsometryDecomposition::decompose(const Eigen::MatrixXcd& isometry, const QVec& qv, IsoScheme scheme)
{
	auto lines = isometry.rows();
	auto cols = isometry.cols();
	size_t log_lines = std::log2(lines);
	size_t log_cols = std::log2(cols);

	_ASSERT((lines == pow(2, log_lines)) && log_lines != 0, "The number of rows of the isometry is not a non negative power of 2.");
	//_ASSERT((cols == pow(2, log_cols)) && log_cols != 0, "The number of columns of the isometry is not a non negative power of 2.");
	_ASSERT(log_cols <= log_lines, "The input matrix has more columns than rows.");
	_ASSERT(_is_isometry(isometry, log_cols), "The input matrix has non orthonormal columns.");
	_ASSERT(qv.size() == log_lines, "The input qubti size equal to log2 number of rows of the isometry.");

	QCircuit circ;
	if (scheme == IsoScheme::CCD){
		circ = _ccd(isometry, qv, log_lines, log_cols);
	}
	else if (scheme == IsoScheme::KNILL){
		circ = _knill(isometry, qv, lines, cols);
	}
	else{
		QCERR_AND_THROW_ERRSTR(run_fail, "type error!");
	}
	return circ;
}


QCircuit QPanda::isometry_decomposition(const Eigen::MatrixXcd& isometry, const QVec& qv, IsoScheme scheme)
{
	IsometryDecomposition iso_dec;
	return iso_dec.decompose(isometry, qv, scheme);
}

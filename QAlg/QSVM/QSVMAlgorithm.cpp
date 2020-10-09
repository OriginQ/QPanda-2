#include "QAlg/QSVM/QSVMAlgorithm.h"
#include "Core/Utilities/Tools/Utils.h"
#include "QAlg/HHL/HHL.h"
#include <cfloat>
#include <queue>

USING_QPANDA
using namespace std;


QSVM::QSVM(std::vector<std::vector<double > > data)
{
	m_qvm = new CPUQVM();
	m_qvm->init();
	m_qv = m_qvm->qAllocMany(20);
	m_cv = m_qvm->cAllocMany(20);
	int row_size = data.size();
	std::vector<double> x_data_min(data[0].size(), FLT_MAX);
	for (int i = 0; i < row_size; i++)
	{
		int col_size = data[i].size();
		std::vector<double> x_data;
		for (int j = 0; j < col_size - 1; j++)
		{
			x_data.push_back(data[i][j]);
			if (x_data_min[j] > data[i][j])
			{
				x_data_min[j] = data[i][j];
			}
		}
		m_x.push_back(x_data);
		m_y.push_back(data[i][col_size - 1]);
	}

	for (int i = 0; i < row_size; i++)
	{
		int col_size = data[i].size();
		for (int j = 0; j < col_size - 1; j++)
		{
			m_x[i][j] -= x_data_min[j];
			m_x[i][j] = round(m_x[i][j] * 100) / 100;
		}
	}

	// data size
	int m = m_x.size();

	// length of vector
	int n = m_x[0].size();

	// the number of input line data
	int M = m + 1;

	// the number of qubits in a single vector of input data
	int vector_qubits = (int)ceil(log2(n));

	// training data oracle the number of qubits in the coefficient
	int number = (int)ceil(log2(M));

	// add a unit vector to the x vector
	std::vector<std::vector<double > > x_vector = get_x_vector(m_x);

	// training data oracle  coefficient of the x vector
	std::vector<double> coefficient = get_coefficient(x_vector);

	// training data oracle qubit number
	m_oracle_qubits = number + vector_qubits * M + 1;

	// swap-test qubit number
	m_swap_qubits = 2 * m_oracle_qubits + 1;

	// b, a solution result
	std::vector<double > b_a = solve(m_x, m_y);

	// the coefficient of u
	if (coefficient.size() != b_a.size())
	{
		QCERR("coefficient  error!");
		throw run_fail("coefficient  error!");
	}

	for (int i = 0; i < coefficient.size(); i++)
		m_u_coefficient.push_back(coefficient[i] * b_a[i]);

	// u vector
	m_u_vector = x_vector;

	// psi circuit starting position
	m_psi_position = 2;

	// phi circuit starting position
	m_phi_position = 1;

	// swap circuit starting position
	m_swap_position = 0;
}

QSVM::~QSVM()
{
	m_qvm->finalize();
	delete m_qvm;
}

std::vector<double > QSVM::solve(std::vector<std::vector<double > > x, std::vector<double >y)
{
	if (x.empty() || x[0].empty())
	{
		QCERR("x param error!");
		throw run_fail("x param error!");
	}

	std::vector<double> temp;
	for (int i = 0; i < x.size(); i++)
	{
		double row_squ_sum = 0;
		for (int j = 0; j < x[i].size(); j++)
		{
			row_squ_sum += x[i][j] * x[i][j];
		}
		if (row_squ_sum < 1e-9)
			row_squ_sum = 0.0000001;
		temp.push_back(row_squ_sum);
	}

	std::vector<std::vector<double > > norm_x;
	for (int i = 0; i < x.size(); i++)
	{
		std::vector<double> norm_row_x;
		for (int j = 0; j < x[i].size(); j++)
		{
			double norm_val = x[i][j] / sqrt(temp[i]);
			norm_row_x.push_back(norm_val);
		}
		norm_x.push_back(norm_row_x);
	}

	//get the F matrix
	std::vector<std::vector<double > > F_matrix;
	std::vector<double> tmp(norm_x.size() + 1, 1);
	F_matrix.push_back(tmp);
	for (auto _a : norm_x)
	{
		temp.clear();
		temp.push_back(1);
		for (auto _b : norm_x)
		{
			double p0 = construct_qcircuit(m_qv, m_cv, _a, _b);
			if (p0 < 0.5)
			{
				p0 = 0.5 + 0.00000000001;
			}
			p0 = round(sqrt(p0 * 2 - 1) * 100000) / 100000;
			temp.push_back(p0);
		}
		F_matrix.push_back(temp);
	}
	F_matrix[0][0] = 0;


	if (F_matrix.size() != F_matrix[0].size())
	{
		QCERR("F_matrix param error!");
		throw run_fail("F_matrix param error!");
	}

	// normalization of matrices
	double trace = 0;
	for (int i = 0; i < F_matrix.size(); i++)
	{
		for (int j = 0; j < F_matrix.size(); j++)
		{
			if (i == j)
				trace += F_matrix[i][j];
		}
	}

	std::vector<std::vector<double > > norm_F;
	for (int i = 0; i < F_matrix.size(); i++)
	{
		temp.clear();
		for (int j = 0; j < F_matrix.size(); j++)
			temp.push_back(F_matrix[i][j] / trace);

		norm_F.push_back(temp);
	}

	if (y.size() > norm_F.size())
	{
		QCERR("norm_F param error!");
		throw run_fail("norm_F param error!");
	}
	std::vector<double> b(norm_F.size() - y.size(), 0);
	b.insert(b.end(), y.begin(), y.end());

	QStat A;
	for (int i = 0; i < norm_F.size(); i++)
	{
		for (int j = 0; j < norm_F[i].size(); j++)
			A.push_back(norm_F[i][j]);
	}

	QStat b_a = HHL_solve_linear_equations(A, b);

	std::vector<double > b_a_result;
	for (auto data : b_a)
	{
		if (data.imag() > 1e-6)
		{
			QCERR("b_a_result error!");
			throw run_fail("b_a_result error!");
		}
		b_a_result.push_back(data.real());
	}
	return b_a_result;
}

// Constructing quantum circuits SWAP-test
double QSVM::construct_qcircuit(QVec qv, std::vector<ClassicalCondition> cv, std::vector<double> a, std::vector<double> b)
{
	auto prog = QProg();
	auto swap_gate = SWAP(qv[1], qv[2]);

	prog << H(qv[0])
		<< RY(qv[1], acos(pow(a[0], 2) - pow(a[1], 2)))
		<< RY(qv[2], acos(pow(b[0], 2) - pow(b[1], 2)))
		<< swap_gate.control({ qv[0] })
		<< H(qv[0]);

	prob_dict result = m_qvm->probRunDict(prog, { qv[0] }, -1);
	double p0 = result["0"];
	return p0;
}


std::vector<std::vector<double > > QSVM::get_x_vector(std::vector<std::vector<double > > matrix)
{
	int len = matrix[0].size();
	std::vector<double> unit_vector(len, 0);
	unit_vector[0] = 1;
	matrix.insert(matrix.begin(), unit_vector);
	auto x_vector = matrix;
	return x_vector;
}

// get the coefficients on the vector
std::vector<double> QSVM::get_coefficient(std::vector<std::vector<double > > matrix)
{
	// each row of norm
	std::vector < double > coefficient;
	for (int i = 0; i < matrix.size(); i++)
	{
		double norm = 0;
		for (int j = 0; j < matrix[i].size(); j++)
		{
			norm += (matrix[i][j] * matrix[i][j]);
		}
		norm = sqrt(norm);
		coefficient.push_back(norm);
	}
	return coefficient;
}

QCircuit QSVM::get_number_circuit(QVec qlist, int position, int number, int qubit_number)
{
	auto cir = QCircuit();

	std::string str_bin = dec2bin(number, qubit_number);
	reverse(str_bin.begin(), str_bin.end());

	for (int i = 0; i < str_bin.size(); i++)
	{
		if (str_bin[i] == '0')
			cir << X(qlist[position + i]);
	}
	return cir;
}

// encode the amplitude of the matrix
std::vector<std::vector<double > > QSVM::encode_matrix(std::vector<double >  flat_matrix)
{
	int len = flat_matrix.size();
	int n = (int)ceil(log2(len));
	int zero_num = pow(n, 2) - len;
	for (int i = 0; i < zero_num; i++)
	{
		flat_matrix.push_back(0);
	}

	std::queue<std::vector<double>> q;
	std::vector<std::vector<double>> vector_list;  // œÚ¡ø
	std::vector<double> temp;
	q.push(flat_matrix);
	while (q.size())
	{
		temp = q.front();
		q.pop();
		vector_list.push_back(temp);
		int offset = temp.size() / 2;
		std::vector<double> left(temp.begin(), temp.begin() + offset);
		std::vector<double> right(temp.begin() + offset, temp.end());

		if (left.size() > 1)
			q.push(left);

		if (right.size() > 1)
			q.push(right);
	}

	std::vector<std::vector<double>> theata_list;
	int idx = 0;
	temp.clear();
	for (auto vec : vector_list)
	{
		int offset = vec.size() / 2;
		std::vector<double> left(vec.begin(), vec.begin() + offset);
		std::vector<double> right(vec.begin() + offset, vec.end());
		double left_sum = 0, right_sum = 0, theata = 0;
		for (auto it : left)
			left_sum += it;

		for (auto it : right)
			right_sum += it;

		if (left_sum < 1e-6 && right_sum < 1e-6)
			theata = 0;
		else
			theata = 2 * acos(left_sum / sqrt(pow(left_sum, 2) + pow(right_sum, 2)));

		temp.push_back(theata);
		if (temp.size() == (idx * idx))
		{
			theata_list.push_back(temp);
			temp.clear();
			idx += 1;
		}
	}
	return  theata_list;
}

// amplitude encoding
QCircuit QSVM::prepare_state(QVec qlist, int position, std::vector<double> values)
{
	// prepare RY's rotation angle matrix
	std::vector<std::vector<double>> theata_list = encode_matrix(values);

	auto cir = QCircuit();
	QVec control_position;
	for (int i = 0; i < theata_list.size(); i++)
	{
		for (int j = 0; j < theata_list[i].size(); j++)
		{
			double theata = theata_list[i][j];
			if (i == 0)
			{
				cir << RY(qlist[position + i], theata);
			}
			else
			{
				QGate  gate = RY(qlist[position + i], theata);
			
				QCircuit temp_cir = get_number_circuit(qlist, position, j, i);
				cir << temp_cir;
				cir << gate.control(control_position);
				cir << temp_cir;
			}
		}
		control_position.push_back(qlist[position + i]);
	}

	return cir;
}

// build training data oracle
QCircuit QSVM::training_data_oracle(QVec qlist, int position, std::vector<double> coe_vector, std::vector<std::vector<double > > x_vector)
{
	int M = m_x.size() + 1;
	int n = m_x[0].size();
	int number = (int)ceil(log2(M));
	int vector_qubits = (int)ceil(log2(n));

	int coe_position = position + 1;

	// the starting bit of vector encoding
	int vector_position = coe_position + number;

	// the amplitude coding of the coefficients
	QCircuit coe_cir = prepare_state(qlist, coe_position, coe_vector);

	// controlled coefficient X gate coding
	QCircuit control_cir = QCircuit();
	QVec control_qubits;
	for (int i = 0; i < number; i++)
	{
		control_cir << X(qlist[position + 1 + i]);
		control_qubits.push_back(qlist[position + 1 + i]);
	}

	QCircuit x_cir = QCircuit();
	for (int i = 0; i < M; i++)
	{
		QCircuit temp_cir = prepare_state(qlist, vector_position + i * vector_qubits, x_vector[i]);
		x_cir << temp_cir;
	}

	auto cir = QCircuit();
	// insert coefficient code lines
	cir << coe_cir;

	// insert a zero-controlled X-gate circuit
	cir << control_cir;
	cir << X(qlist[position]).control(control_qubits);
	cir << control_cir;
	// insert amplitude coded circuit
	cir << X(qlist[position]);
	cir << x_cir.control({ qlist[position] });
	cir << X(qlist[position]);
	// insert a zero-controlled X-gate circuit
	cir << control_cir;
	cir << X(qlist[position]).control(control_qubits);
	cir << control_cir;

	return cir;
}

// Construction circuits ¶◊ = 1/sqrt(2)(|0>|u> + |1>|x>)
QCircuit QSVM::construct_state_psi(QVec qlist, int position, std::vector<double> u_coefficient, std::vector<std::vector<double > >u_vector,
	std::vector<double > x_coefficient, std::vector<std::vector<double > >x_vector, int oracle_qubits)
{
	// u circuit
	int u_cir_position = position + 1;
	auto u_cir = training_data_oracle(qlist, u_cir_position, u_coefficient, u_vector);

	// x circuit
	int x_cir_position = position + 1 + oracle_qubits;
	auto x_cir = training_data_oracle(qlist, x_cir_position, x_coefficient, x_vector);

	// CNOT gate 
	auto cir_copy = QCircuit();
	for (int i = 0; i < oracle_qubits; i++)
		cir_copy << CNOT(qlist[u_cir_position + i], qlist[x_cir_position + i]);

	// insert CNOT gate 
	auto cir = QCircuit();
	// insert H and X gate on the auxiliary qubit 
	cir << H(qlist[position]);
	cir << X(qlist[position]);
	//  insert u circuit
	cir << u_cir;
	//  insert CNOT gate
	cir << cir_copy.control({ qlist[position] });
	cir << X(qlist[position]);
	//  insert x circuit
	cir << x_cir;

	return cir;
}

// Construction circuits  ¶’ = 1 / sqrt(2)(| 0 > -| 1 > )
QCircuit QSVM::construct_state_phi(QVec qlist, int position)
{
	auto cir = QCircuit();
	cir << X(qlist[position]);
	cir << H(qlist[position]);
	return cir;
}

// By measuring the auxiliary bit 0 state probability to backstepping |<¶◊|¶’>|^ ,  P(|0>) = 1/2 + 1/2(|<¶◊|¶’>|^2)
QCircuit QSVM::swap_test_p(QVec qlist, int position, int swap_qubits)
{
	auto cir = QCircuit();
	cir << H(qlist[position]);
	int phi_position = position + 1;
	int psi_position = phi_position + 1;

	auto swap_cir = QCircuit();
	for (int i = 0; i < swap_qubits; i++)
		swap_cir << SWAP(qlist[phi_position], qlist[psi_position + i]);

	cir << swap_cir.control({ qlist[position] });
	cir << H(qlist[position]);

	return cir;
}

// construction master circuit
QCircuit QSVM::construct_circuit(QVec qlist, std::vector<double> x_coefficient, std::vector<std::vector<double>> x_vector)
{
	// psi circuit
	auto psi_cir = construct_state_psi(qlist, m_psi_position, m_u_coefficient, m_u_vector, x_coefficient, x_vector, m_oracle_qubits);

	// phi circuit
	auto phi_cir = construct_state_phi(qlist, m_phi_position);

	// swap-test circuit
	auto swap_cir = swap_test_p(qlist, m_swap_position, m_swap_qubits);

	// construction circuit
	auto cir = QCircuit();
	cir << phi_cir;
	cir << psi_cir;
	cir << swap_cir;

	return cir;
}

void QSVM::preprocess_input_x(std::vector<double> query_x, std::vector<std::vector<double>> & extend_x, std::vector<double> &extend_x_coefficient)
{
	extend_x.push_back(query_x);
	extend_x.push_back(query_x);
	extend_x.push_back(query_x);
	extend_x = get_x_vector(extend_x);
	extend_x_coefficient = get_coefficient(extend_x);
}

// predict
double QSVM::predict(QVec qlist, std::vector<ClassicalCondition> clist, std::vector<double>  query_x)
{
	std::vector<std::vector<double>> x_vector;
	std::vector<double> 	x_coefficient;
	preprocess_input_x(query_x, x_vector, x_coefficient);

	auto prog = QProg();
	auto	cir = construct_circuit(qlist, x_coefficient, x_vector);
	prog << cir;
	prob_dict measure_result = m_qvm->probRunDict(prog, { qlist[0] }, -1);
	double p0 = measure_result["0"];
	double  p = p0 * 2 - 1;
	return p;

}

bool QSVM::run(std::vector<double> query_x)
{
	bool ret = false;
	double p  = predict(m_qv, m_cv, query_x);
	if (p < 0.5)
		ret = true;
	else
		ret = false;

	return ret;
}


bool QPanda::qsvm_algorithm(std::vector<std::vector<double > > data, std::vector<double> query_x)
{
	if (query_x.size() != 2)
	{
		QCERR("query_x param error!");
		throw run_fail("query_x param error!");
	}

	QSVM qsvm = QSVM(data);
	return qsvm.run(query_x);
}
#include "HHL.h"
#include <chrono>
#include <csignal>
#include <sstream>
#include <fstream>
#include <functional>
#include "QAlg/Base_QCircuit/QPE.h"
#include "ThirdParty/rapidjson/rapidjson.h"
#include "Core/Utilities/Tools/QStatMatrix.h"
#include "QAlg/Base_QCircuit/AmplitudeEncode.h"
#include "Core/Utilities/Tools/QProgFlattening.h"
#include "Core/Utilities/Tools/QCircuitOptimize.h"
#include "Core/Utilities/Tools/MatrixDecomposition.h"
#include "ThirdParty/EigenUnsupported/Eigen/MatrixFunctions"

#if defined(__linux__) ||  defined(__unix__) || defined(__FreeBSD__) || defined(__APPLE__)
#include <unistd.h>
#endif

using namespace std;

USING_QPANDA
#define PRINT_TRACE
#ifdef PRINT_TRACE
#define PTrace(_msg)                                               \
	{                                                              \
		std::ostringstream ss;                                     \
		ss << _msg;                                                \
		std::cout << __FUNCTION__ << ":" << ss.str() << std::endl; \
	}
#define PTraceCircuit(cir) (std::cout << cir << endl)
#define PTraceCircuitMat(cir)           \
	{                                   \
		auto m = getCircuitMatrix(cir); \
		std::cout << m << endl;         \
	}
#define PTraceMat(mat) (std::cout << (mat) << endl)
#else
#define PTrace(_msg)
#define PTraceCircuit(cir)
#define PTraceCircuitMat(cir)
#define PTraceMat(mat)
#endif

#define T0 2 * PI
#define MAX_PRECISION 1e-10

std::string noise_json;
//SharedMemory* HHLAlg::m_share = { nullptr };
//struct HHL_HOLD* HHLAlg::m_hold = { nullptr };

HHLAlg::HHLAlg(QuantumMachine* qvm) :
    m_qvm(qvm), m_qft_cir_used_qubits_cnt(0), m_amplification_factor(0), m_hhl_qubit_cnt(0), m_ancillary_qubit(nullptr)
{
}

HHLAlg::~HHLAlg()
{
	/*if (m_share != nullptr)
	{
		if ((m_hold->m_thread -= 1) == 0)
		{
			m_share->memory_delete();
		}
		delete m_share, m_share = nullptr;
	}*/
}

//inline void HHLAlg::abort(int signals)
//{
//	if (m_share != nullptr)
//	{
//		if ((m_hold->m_thread -= 1) == 0)
//		{
//			m_share->memory_delete();
//		}
//		delete m_share, m_share = nullptr;
//	}
//	exit(0);
//}

QCircuit HHLAlg::build_CR_cir(QVec& controlqvec, Qubit* target_qubit, double /* r = 6.0*/)
{
	QCircuit circuit = CreateEmptyCircuit();
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

QMatrixXd HHLAlg::to_real_matrix(const QMatrixXcd& c_mat)
{
	size_t rows = c_mat.rows();
	size_t cols = c_mat.cols();
	QMatrixXd real_matrix(rows, cols);

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
	QMatrixXd real_eigen_A = to_real_matrix(e_mat_A);

	Eigen::EigenSolver<QMatrixXd> eigen_solver(real_eigen_A);
	auto eigen_vals = eigen_solver.eigenvalues();

	std::vector<double> eigen_vec(2);
	double max_eigen_val = 0.0;
	double min_eigen_val = 0XEFFFFFFF;
	for (size_t i = 0; i < eigen_vals.rows(); ++i)
	{
		for (size_t j = 0; j < eigen_vals.cols(); ++j)
		{
			const auto& m = abs(eigen_vals(i, j).real());
			if (m > max_eigen_val)
			{
				max_eigen_val = m;
			}

			if (m < min_eigen_val)
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
	// check parameter b
	double tmp_sum = 0.0;
	for (const auto& i : b)
	{
		tmp_sum += (i * i);
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
	for (auto& val : qpe_result_quantum_state)
	{
		val.second = abs(val.second) < QUAN_STATE_PRECISION ? 0.0 : val.second;
	}

	stringstream ss;
	for (auto& val : qpe_result_quantum_state)
	{
		ss << val.first << ", " << val.second << std::endl;
	}

	ss << "QPE over." << endl;
	return ss.str();
}

void HHLAlg::init_qubits(const QStat& A, const std::vector<double>& b, const uint32_t& precision_cnt)
{
	const std::vector<double> max_and_min_eigen_val = get_max_eigen_val(A);
	//PTrace("The max-eigen-val = " << max_and_min_eigen_val[0] << ", min-eigen-val = " << max_and_min_eigen_val[1]);
	const uint32_t ex_qubits = ceil(log2(max_and_min_eigen_val[0] + 1)) + 1; /**< Need a qubit to represent the sign */
	size_t b_cir_used_qubits_cnt = ceil(log2(b.size()));
	m_qubits_for_b = m_qvm->allocateQubits(b_cir_used_qubits_cnt);

	size_t eigen_val_amplification_qubits = ceil(log2(pow(10, precision_cnt)));
	if ((std::abs(max_and_min_eigen_val[1]) > 1e-10) && (std::abs(max_and_min_eigen_val[1]) < 1))
	{
		const auto f = 1.0 / max_and_min_eigen_val[1];
		eigen_val_amplification_qubits += ceil(log2(f));
	}

	const uint32_t _min_qtf_qubit_num = 3;
	if (_min_qtf_qubit_num >= (ex_qubits + eigen_val_amplification_qubits))
	{
		eigen_val_amplification_qubits += (_min_qtf_qubit_num - ex_qubits);
	}

	m_qft_cir_used_qubits_cnt = (eigen_val_amplification_qubits + ex_qubits);
	m_qubits_for_qft = m_qvm->allocateQubits(m_qft_cir_used_qubits_cnt);
	m_hhl_qubit_cnt = m_qft_cir_used_qubits_cnt + b_cir_used_qubits_cnt + 1;
	//PTrace("Total need qubits number: " << m_hhl_qubit_cnt
	//	<< ", qft_qubits: " << m_qft_cir_used_qubits_cnt
	//	<< "=" << ex_qubits << "+" << eigen_val_amplification_qubits);

	m_ancillary_qubit = m_qvm->allocateQubit();

	m_amplification_factor = 1 << eigen_val_amplification_qubits;
}

bool HHLAlg::is_hermitian_matrix(const QStat& A)
{
	const auto tmp_A = dagger_c(A);
	return (tmp_A == A);
}

void HHLAlg::transform_hermitian_to_unitary_mat(QStat& src_mat)
{
	for (auto& item : src_mat)
	{
		item *= qcomplex_t(0, PI * 2.0 / (1 << m_qft_cir_used_qubits_cnt));
	}

	QMatrixXcd eigen_mat = QStat_to_Eigen(src_mat);
	auto exp_matrix = eigen_mat.exp().eval();
	src_mat = Eigen_to_QStat(exp_matrix);
}

QCircuit HHLAlg::get_hhl_circuit(const QStat& A, const std::vector<double>& b, const uint32_t& precision_cnt)
{
	if (b.size() < 2)
	{
		QCERR_AND_THROW_ERRSTR(init_fail, "Error: error size of b for HHL.");
	}

    auto is_square_matrix = [&]() {
        size_t dimension = sqrt(A.size());
        double e = ceil(log2(dimension));
        const double expand_dimension = pow(2, e) - (double)dimension;
        return ((expand_dimension - 0.0) < 0.000001);
    };

	if ((!is_square_matrix())
        || ((!is_hermitian_matrix(A)) && (!is_unitary_matrix(A))))
	{
		QCERR_AND_THROW_ERRSTR(init_fail, "Error: The input matrix for HHL must be a Hermitian sparse N*N matrix.");
	}

	/*std::signal(SIGFPE, abort);
	std::signal(SIGILL, abort);
	std::signal(SIGINT, abort);
	std::signal(SIGABRT, abort);
	std::signal(SIGSEGV, abort);
	std::signal(SIGTERM, abort);
	m_share = m_share == nullptr ? new SharedMemory(sizeof(struct HHL_HOLD), "HHL_HOLD") : m_share;
	m_hold = (struct HHL_HOLD*&)m_share->memory();
	if ((m_hold->m_thread += 1) == 1)
	{
		std::mutex mm_mutex;
		memcpy(&m_hold->m_mutex, &mm_mutex, sizeof(std::mutex));
	}
#if defined(_WIN32) || defined(_WIN64)
	while (!m_hold->m_mutex.try_lock())
	{
		Sleep(1);
	}
#elif defined(__linux__) ||  defined(__unix__) || defined(__FreeBSD__) || defined(__APPLE__)
	while (!m_hold->m_mutex.try_lock())
	{
		usleep(1);
	}
#endif*/

	init_qubits(A, b, precision_cnt);

	auto tmp_A = A;
	for (auto& i : tmp_A)
	{
		i *= m_amplification_factor;
	}

	m_cir_b = build_cir_b(m_qubits_for_b, b);

	// transfer to unitary matrix
	// transform_hermitian_to_unitary_mat(tmp_A);

	// QPE
	m_cir_qpe = build_QPE_circuit(m_qubits_for_qft, m_qubits_for_b, tmp_A, true);
	//PTrace("qpe_gate_cnt: " << getQGateNum(m_cir_qpe));

	m_cir_cr = build_CR_cir(m_qubits_for_qft, m_ancillary_qubit, m_qft_cir_used_qubits_cnt);
	m_hhl_cir << m_cir_b << m_cir_qpe << m_cir_cr << m_cir_qpe.dagger();
	//PTrace("^^^^^^^^^^^^^^^^^whole hhl_cir_gate_cnt: " << getQGateNum(m_hhl_cir) << " ^^^^^^^^^^^^^^^^^^^^");

	//m_hold->m_mutex.unlock();
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
			new_A[i * new_dimension + j] = A[i * dimension + j];
		}
	}

	A.swap(new_A);
}

QCircuit QPanda::build_HHL_circuit(const QStat& A, const std::vector<double>& b,
	QuantumMachine* qvm, const uint32_t precision_cnt /*= 0*/)
{
	HHLAlg hhl_alg(qvm);

	return hhl_alg.get_hhl_circuit(A, b, precision_cnt);
}

void Get_Noise_Modle(NoiseModel& noise)
{
	int i = 0;
	double prob;
	GateType type;
	NOISE_MODEL model;
	std::string json_str;
	std::string line_str;
	fstream noise_fstream;
	rapidjson::Document json;
	noise_fstream.open(noise_json, std::ios::in | std::ios::out);
	while (!noise_fstream.eof())
	{
		noise_fstream >> line_str;
		json_str += line_str;
	}
	noise_fstream.close();
	json.Parse(json_str.c_str());
	if (!json["noise_model"].Empty())
	{
		try
		{
			if (json["noise_model"].IsArray())
			{
				rapidjson::Value& noise_array = json["noise_model"];
				for (rapidjson::Value::ConstValueIterator iter = noise_array.Begin(); iter != noise_array.End(); iter++)
				{
					i = 0;
					if (iter->IsArray())
					{
						for (rapidjson::Value::ConstValueIterator str = iter->Begin(); str != iter->End(); str++)
						{
							switch (i)
							{
							case 0:
							{
								json_str = str->GetString();
								if (json_str == "BITFLIP_KRAUS_OPERATOR")
								{
									model = NOISE_MODEL::BITFLIP_KRAUS_OPERATOR;
								}
								else if (json_str == "BIT_PHASE_FLIP_OPRATOR")
								{
									model = NOISE_MODEL::BIT_PHASE_FLIP_OPRATOR;
								}
								else if (json_str == "DAMPING_KRAUS_OPERATOR")
								{
									model = NOISE_MODEL::DAMPING_KRAUS_OPERATOR;
								}
								else if (json_str == "DECOHERENCE_KRAUS_OPERATOR")
								{
									model = NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR;
								}
								else if (json_str == "DEPHASING_KRAUS_OPERATOR")
								{
									model = NOISE_MODEL::DEPHASING_KRAUS_OPERATOR;
								}
								else if (json_str == "DEPOLARIZING_KRAUS_OPERATOR")
								{
									model = NOISE_MODEL::DEPOLARIZING_KRAUS_OPERATOR;
								}
								else if (json_str == "KRAUS_MATRIX_OPRATOR")
								{
									model = NOISE_MODEL::KRAUS_MATRIX_OPRATOR;
								}
								else if (json_str == "MIXED_UNITARY_OPRATOR")
								{
									model = NOISE_MODEL::MIXED_UNITARY_OPRATOR;
								}
								else if (json_str == "PAULI_KRAUS_MAP")
								{
									model = NOISE_MODEL::PAULI_KRAUS_MAP;
								}
								else if (json_str == "PHASE_DAMPING_OPRATOR")
								{
									model = NOISE_MODEL::PHASE_DAMPING_OPRATOR;
								}
								else
								{
									throw(std::runtime_error("error noise model:" + json_str));
								}
								break;
							}
							case 1:
							{
								json_str = str->GetString();
								if (json_str == "GATE_NOP")
								{
									type = GateType::GATE_NOP;
								}
								else if (json_str == "GATE_UNDEFINED")
								{
									type = GateType::GATE_UNDEFINED;
								}
								else if (json_str == "P0_GATE")
								{
									type = GateType::P0_GATE;
								}
								else if (json_str == "P1_GATE")
								{
									type = GateType::P1_GATE;
								}
								else if (json_str == "PAULI_X_GATE")
								{
									type = GateType::PAULI_X_GATE;
								}
								else if (json_str == "PAULI_Y_GATE")
								{
									type = GateType::PAULI_Y_GATE;
								}
								else if (json_str == "PAULI_Z_GATE")
								{
									type = GateType::PAULI_Z_GATE;
								}
								else if (json_str == "X_HALF_PI")
								{
									type = GateType::X_HALF_PI;
								}
								else if (json_str == "Y_HALF_PI")
								{
									type = GateType::Y_HALF_PI;
								}
								else if (json_str == "Z_HALF_PI")
								{
									type = GateType::Z_HALF_PI;
								}
								else if (json_str == "P_GATE")
								{
									type = GateType::P_GATE;
								}
								else if (json_str == "HADAMARD_GATE")
								{
									type = GateType::HADAMARD_GATE;
								}
								else if (json_str == "T_GATE")
								{
									type = GateType::T_GATE;
								}
								else if (json_str == "S_GATE")
								{
									type = GateType::S_GATE;
								}
								else if (json_str == "RX_GATE")
								{
									type = GateType::RX_GATE;
								}
								else if (json_str == "RY_GATE")
								{
									type = GateType::RY_GATE;
								}
								else if (json_str == "RZ_GATE")
								{
									type = GateType::RZ_GATE;
								}
								else if (json_str == "RPHI_GATE")
								{
									type = GateType::RPHI_GATE;
								}
								else if (json_str == "U1_GATE")
								{
									type = GateType::U1_GATE;
								}
								else if (json_str == "U2_GATE")
								{
									type = GateType::U2_GATE;
								}
								else if (json_str == "U3_GATE")
								{
									type = GateType::U3_GATE;
								}
								else if (json_str == "U4_GATE")
								{
									type = GateType::U4_GATE;
								}
								else if (json_str == "CU_GATE")
								{
									type = GateType::CU_GATE;
								}
								else if (json_str == "CNOT_GATE")
								{
									type = GateType::CNOT_GATE;
								}
								else if (json_str == "CZ_GATE")
								{
									type = GateType::CZ_GATE;
								}
								else if (json_str == "CP_GATE")
								{
									type = GateType::CP_GATE;
								}
								else if (json_str == "RYY_GATE")
								{
									type = GateType::RYY_GATE;
								}
								else if (json_str == "RXX_GATE")
								{
									type = GateType::RXX_GATE;
								}
								else if (json_str == "RZZ_GATE")
								{
									type = GateType::RZZ_GATE;
								}
								else if (json_str == "RZX_GATE")
								{
									type = GateType::RZX_GATE;
								}
								else if (json_str == "CPHASE_GATE")
								{
									type = GateType::CPHASE_GATE;
								}
								else if (json_str == "ISWAP_THETA_GATE")
								{
									type = GateType::ISWAP_THETA_GATE;
								}
								else if (json_str == "ISWAP_GATE")
								{
									type = GateType::ISWAP_GATE;
								}
								else if (json_str == "SQISWAP_GATE")
								{
									type = GateType::SQISWAP_GATE;
								}
								else if (json_str == "SWAP_GATE")
								{
									type = GateType::SWAP_GATE;
								}
								else if (json_str == "TWO_QUBIT_GATE")
								{
									type = GateType::TWO_QUBIT_GATE;
								}
								else if (json_str == "P00_GATE")
								{
									type = GateType::P00_GATE;
								}
								else if (json_str == "P11_GATE")
								{
									type = GateType::P11_GATE;
								}
								else if (json_str == "TOFFOLI_GATE")
								{
									type = GateType::TOFFOLI_GATE;
								}
								else if (json_str == "ORACLE_GATE")
								{
									type = GateType::ORACLE_GATE;
								}
								else if (json_str == "I_GATE")
								{
									type = GateType::I_GATE;
								}
								else if (json_str == "ECHO_GATE")
								{
									type = GateType::ECHO_GATE;
								}
								else if (json_str == "BARRIER_GATE")
								{
									type = GateType::BARRIER_GATE;
								}
								else
								{
									throw(std::runtime_error("error gate type:" + json_str));
								}
								break;
							}
							case 2:
							{
								prob = str->GetDouble();
								break;
							}
							default:
							{
								throw(std::runtime_error("invalid noise model"));
								break;
							}
							}
							i++;
						}
					}
					if (i != 3)
					{
						throw(std::runtime_error("invalid noise model"));
					}
					noise.add_noise_model(model, type, prob);
				}
			}
			else
			{
				throw(std::runtime_error("invalid noise model"));
			}
		}
		catch (const std::exception& e)
		{
			throw(std::runtime_error(e.what()));
		}
	}

	if (!json["readout_error"].Empty())
	{
		if (json["readout_error"].IsArray())
		{
			i = 0;
            double f0 = 0, f1 = 0;
			rapidjson::Value& iter = json["readout_error"];
			for (rapidjson::Value::ConstValueIterator str = iter.Begin(); str != iter.End(); str++)
			{
				switch (i)
				{
				case 0:
				{
					f0 = str->GetDouble();
					break;
				}
				case 1:
				{
					f1 = str->GetDouble();
					break;
				}
				default:
				{
					throw(std::runtime_error("invalid noise readout error"));
					break;
				}
				}
            }
            noise.set_readout_error({ {f0, 1 - f0}, {1 - f1, f1} });
		}
		else
		{
			throw(std::runtime_error("invalid noise readout error"));
		}
	}

	if (!json["rotation_error"].Empty())
	{
		noise.set_rotation_error(json["rotation_error"].GetDouble());
	}
}

QStat QPanda::HHL_solve_linear_equations(const QStat& A, const std::vector<double>& b,
	const uint32_t precision_cnt /* = 0*/)
{
	NoiseModel noise;
	std::vector<double> tmp_b = b;
	double norm_coffe_b = 0.0;
	for (const auto& item : tmp_b)
	{
		norm_coffe_b += (item * item);
	}

	if (abs(norm_coffe_b) < MAX_PRECISION)
	{
		QStat r;
		r.resize(b.size(), 0);
		return r;
	}

	norm_coffe_b = sqrt(norm_coffe_b);
	for (auto& item : tmp_b)
	{
		item = item / norm_coffe_b;
	}

	// build HHL quantum program
	auto machine = initQuantumMachine(CPU);
	machine->setConfigure({ 64, 64 });
	auto prog = QProg();
	HHLAlg hhl_alg(machine);
	QCircuit hhl_cir = hhl_alg.get_hhl_circuit(A, tmp_b, precision_cnt);
	prog << hhl_cir;
	// hhl_alg.m_hold->m_mutex.unlock();
	// PTraceCircuit(prog);
	//PTrace("HHL quantum circuit is running ...");
	auto start = chrono::system_clock::now();
	
	if (!noise_json.empty())
	{
		Get_Noise_Modle(noise);
	}
	directlyRun(prog, noise);
	auto stat = machine->getQState();
	auto end = chrono::system_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
	//PTrace("run HHL used: "
	//	<< double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den
	//	<< " s");

	machine->finalize();
	stat.erase(stat.begin(), stat.begin() + (stat.size() / 2));

	QStat stat_normed;
	for (auto& val : stat)
	{
		stat_normed.push_back(val * norm_coffe_b * hhl_alg.get_amplification_factor());
	}

	for (auto& val : stat_normed)
	{
		qcomplex_t tmp_val((abs(val.real()) < MAX_PRECISION ? 0.0 : val.real()), (abs(val.imag()) < MAX_PRECISION ? 0.0 : val.imag()));
		val = tmp_val;
	}

	// get solution
	QStat result;
	for (size_t i = 0; i < b.size(); ++i)
	{
		result.push_back(stat_normed.at(i));
	}
	return result;
}

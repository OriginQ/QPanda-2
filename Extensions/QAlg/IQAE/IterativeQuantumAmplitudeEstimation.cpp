#include "QAlg/IQAE/IterativeQuantumAmplitudeEstimation.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include "Core/Utilities/Tools/Utils.h"
#include <cmath>


USING_QPANDA
using namespace std;

IterativeAmplitudeEstimation::IterativeAmplitudeEstimation(
	const QCircuit& cir, // Quantum Circuit 
	const int qnumber, // number of qubits used by the cir.
	const double epsilon, // estimate the accuracy of 'a' (the amplitude of the ground state |1>).
	const double alpha,  // confidence is 1-alpha.
	const std::string confint_method, // statistical method for estimating confidence interval Chernoff-Hoeffding.
	const double min_ratio, // find the minimum magnification of the next K.
	const QMachineType QType // Quantum virtual machine type, currently only CPU is provided, other types of virtual machine types can be added later.
)
/*
* epsilon: (0, 0.5)
* ratio > 1
* the range of alpha : (0, 1)
* confint_method(Chernoff-Hoeffding) : {'CH', 'CP'}, only support 'CH' method in this vesion
*/
{
	m_qvm = QuantumMachineFactory::GetFactoryInstance().CreateByType(QType);
	m_qvm->init();
	m_qubits = m_qvm->qAllocMany(qnumber);
	m_cbits = m_qvm->cAllocMany(qnumber);

	m_cir = cir;
	m_qnumber = qnumber;
	m_epsilon = epsilon;
	m_alpha = alpha;
	m_confint_method = confint_method;
	m_min_ratio = min_ratio;
	m_QType = QType;
	m_N_max = 32 * std::log(2.0 * log(PI / 4 / epsilon) / log(m_min_ratio) / alpha) / std::pow(1 - 2.0 * std::sin(PI / 14), 2.0);
	m_round_max = ceil(log(PI / (8.0 * epsilon)) / std::log(min_ratio));
	m_L_max = std::pow(std::asin((2.0 / m_N_max) * std::log(2.0 * m_round_max / alpha)), 0.25);
	int alp = int(-log(alpha) / log(m_min_ratio));
	m_N_shots = /*50 + */int(-log(epsilon)) * int(-log(alpha) / log(m_min_ratio));
	if (m_N_shots >= m_N_max)
	{
		m_N_shots = m_N_max;
	}
};


QCircuit IterativeAmplitudeEstimation::grover_operator(QCircuit& cir, const QVec& qubits)
{
	int n = qubits.size();
	if (n > 1)
	{
		auto S0_qc = createEmptyCircuit();
		for (int i = 0; i < n; i++)
		{
			S0_qc << X(qubits[i]);
		}
		QVec control_qubit;
		for (int i = 0; i < n - 1; i++)
		{
			control_qubit.push_back(qubits[i]);
		}
		S0_qc << Z(qubits[n - 1]).control(control_qubit);
		for (int i = 0; i < n; i++)
		{
			S0_qc << X(qubits[i]);
		}
		auto S0_qc_m = getCircuitMatrix(S0_qc);
		auto S_psi0_qc = createEmptyCircuit();
		S_psi0_qc << Z(qubits[n - 1]);
		auto S_psi0_qc_m = getCircuitMatrix(S_psi0_qc);
		auto G_qc = createEmptyCircuit();
		G_qc << S_psi0_qc << cir.dagger() << S0_qc << cir;
		return G_qc;
	}
	else if (n == 1)
	{
		auto S0_qc = createEmptyCircuit();
		S0_qc << Z(qubits[0]);
		auto S_psi0_qc = createEmptyCircuit();
		S_psi0_qc << Z(qubits[0]);
		auto G_qc = createEmptyCircuit();
		G_qc << S_psi0_qc << cir.dagger() << S0_qc << cir;
		return G_qc;
	}
	else
	{
		return cir;
	}
}


QCircuit IterativeAmplitudeEstimation::_Gk_A_QC(const QCircuit& cir, const QCircuit& G, const QVec& qubits, int k)
{
	QCircuit Gk_A_qc = createEmptyCircuit();
	if (k == 0)
	{
		Gk_A_qc << cir;
	}
	else if (k > 0)
	{
		Gk_A_qc << cir;
		for (int k_id = 0; k_id < k; k_id++)
		{
			Gk_A_qc << G;
		}
	}
	return Gk_A_qc;
}


int IterativeAmplitudeEstimation::_QAE_in_QMachine(QCircuit& cir, const QVec& qubits, const int k, const int N)
{
	auto G_qc = grover_operator(cir, qubits);
	auto prog = createEmptyQProg();
	prog << _Gk_A_QC(cir, G_qc, m_qubits, k);
	prog << Measure({ m_qubits[m_qnumber - 1] }, { m_cbits[m_qnumber - 1] });

	auto result = m_qvm->runWithConfiguration(prog, m_cbits, N);
	int count_1 = 0;
	for (auto& aiter : result)
	{
		if (aiter.first.at(0) == '1')
		{
			count_1 = aiter.second;
		}
	}

	return count_1;
}


std::pair<double, double> IterativeAmplitudeEstimation::set_confidence_intervals_CH(double val, int round_max, int shots_num, double alpha)
{
	std::pair<double, double> CI(0.0, 0.0);
	double esp_val_ = std::sqrt(std::log(2.0 * round_max / alpha) / (2.0 * shots_num));
	double lower = 0.0;
	if (val - esp_val_ > 0.0)
	{
		lower = val - esp_val_;
	}
	double upper = 1.0;
	if (val + esp_val_ < 1.0)
	{
		upper = val + esp_val_;
	}
	CI = std::make_pair(lower, upper);

	return CI;
}


std::pair<double, int> IterativeAmplitudeEstimation::exec()
{
	int iter = 0;
	int k_i = 0;
	bool upper = true;
	m_theta_l = 0.0;
	m_theta_u = 1 / 4.0;
	int NumShotsAll = 0;
	int N = 0;
	int Count_1 = 0;
	int Num_shots = 0;
	std::pair<double, int> measureNum_P;
	std::vector<std::pair<int, double>> NumShots_a;

	while ((m_theta_u - m_theta_l) > m_epsilon / PI)
	{
		iter = iter + 1;
		std::pair<int, bool> next_k_up = find_nextK(k_i, m_theta_l, m_theta_u, upper);
		int K_i = 4 * next_k_up.first + 2;
		double esp = (m_theta_u - m_theta_l) * PI / 2.0;
		double scla = 3.0 / 4.0;
		m_N_shots = int(m_round_max * scla * log((log(PI / 4.0 / esp) / log(m_min_ratio)) * (2.0 / m_alpha)));
		if (m_N_shots > m_N_max)
		{
			m_N_shots = m_N_max;
		}

		if (K_i > std::ceil(m_L_max / m_epsilon))
		{
			N = std::ceil(m_N_shots * m_L_max / (m_epsilon * K_i * 10.0));
		}
		else
		{
			N = m_N_shots;
		}

		QCircuit cir = createEmptyCircuit();
		auto count_1 = _QAE_in_QMachine(m_cir, m_qubits, next_k_up.first, N);
		if (next_k_up.first == k_i)
		{
			Count_1 += count_1;
			Num_shots += N;
		}
		else
		{
			Count_1 = count_1;
			Num_shots = N;
		}
		double prob_1 = 1.0 * Count_1 / Num_shots;
		NumShotsAll += N;
		if (m_isWriteData == true)
		{
			NumShotsAll += N;
		}

		std::pair<double, double> a_CI(0.0, 0.0);
		if (m_confint_method == "CH")
		{
			int tem_shots = NumShotsAll;
			if (tem_shots > m_N_max)
			{
				tem_shots = m_N_max;
			}
			a_CI = set_confidence_intervals_CH(prob_1, m_round_max, tem_shots, m_alpha);
		}
		std::pair<double, double> theta_CI(0.0, 0.0);
		if (next_k_up.second)
		{
			theta_CI = std::make_pair(std::acos(1.0 - 2.0 * a_CI.first), std::acos(1.0 - 2.0 * a_CI.second));
		}
		else
		{
			theta_CI = std::make_pair(2 * PI - std::acos(1.0 - 2.0 * a_CI.second), 2 * PI - std::acos(1.0 - 2.0 * a_CI.first));
		}

		m_theta_l = (std::floor(K_i * m_theta_l) + theta_CI.first / 2.0 / PI) / (1.0 * K_i);
		m_theta_u = (std::floor(K_i * m_theta_u) + theta_CI.second / 2.0 / PI) / (1.0 * K_i);

		k_i = next_k_up.first;
		upper = next_k_up.second;

		double a_ = (std::pow(std::sin(m_theta_l * 2.0 * PI), 2.0) + std::pow(std::sin(m_theta_u * 2.0 * PI), 2.0)) / 2.0;
		if (m_isWriteData == true)
		{
			NumShots_a.push_back(std::make_pair(NumShotsAll, a_));
		}
	}
	if (m_isWriteData == true)
	{
		write_basedata(NumShots_a);
	}
	double P_1 = (std::pow(std::sin(m_theta_l * 2.0 * PI), 2.0) + std::pow(std::sin(m_theta_u * 2.0 * PI), 2.0)) / 2.0;
	measureNum_P = std::make_pair(P_1, NumShotsAll);
	m_result = P_1;
	return measureNum_P;
}


std::pair<int, bool> IterativeAmplitudeEstimation::find_nextK(int k_i, double theta_l, double theta_u, bool up_i)
{
	std::pair<int, bool> next_k(k_i, up_i);
	int K_i = 4 * k_i + 2;
	int K_max = std::floor(1 / 2.0 / (theta_u - theta_l));
	int K = K_max - (K_max - 2) % 4;

	while (K >= m_min_ratio * K_i)
	{
		double theta_i_min = K * theta_l;
		double theta_i_max = K * theta_u;
		double theta_i_min_ = theta_i_min - std::floor(theta_i_min);
		double theta_i_max_ = theta_i_max - std::floor(theta_i_max);

		if (theta_i_max_ <= 0.5 && theta_i_max_ >= theta_i_min_)
		{
			next_k = std::make_pair(std::round((K - 2) / 4), true);
			return next_k;
		}
		else if (theta_i_min_ >= 0.5 && theta_i_max_ >= theta_i_min_)
		{
			next_k = std::make_pair(std::round((K - 2) / 4), false);
			return next_k;
		}
		K = K - 4;
	}

	return next_k;
}


bool IterativeAmplitudeEstimation::write_basedata(const std::vector<std::pair<int, double>>& result)
{
	OriginCollection collection(m_filename_json, false);
	collection = { "Nsum", "a" };
	for (int i = 0; i < result.size(); i++)
	{
		//display the details
		//cout << result[i].first << ", " << result[i].second << endl;
		collection.insertValue(result[i].first, result[i].second);
	}

	if (!collection.write())
	{
		cerr << "write sort result failed!" << endl;
		return -1;
	}

	return collection.write();
}

void IterativeAmplitudeEstimation::save_Nsum_a(bool b)
{
	m_isWriteData = b;
}

IterativeAmplitudeEstimation::~IterativeAmplitudeEstimation()
{
	freeQVM();
}

double QPanda::iterative_amplitude_estimation(
	const QCircuit& cir,
	const QVec& qvec,
	const double epsilon,
	const double confidence
)
{
	int q_num = qvec.size();
	IterativeAmplitudeEstimation iqae(cir, q_num, epsilon, confidence);
	iqae.save_Nsum_a(true);
	double result = iqae.exec().first;
	return result;
}


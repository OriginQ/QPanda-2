
#include "Core/Utilities/QProgInfo/RandomizedBenchmarking.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include "Core/VirtualQuantumProcessor/MPSQVM/MPSTensor.h"
#include "Core/Utilities/Tools/QStatMatrix.h"
#include "Core/Utilities/QPandaNamespace.h"
#include <ThirdParty/mpfit/include/mpfit.h>
#include <set>
#include <vector>
#include <map>
#include <random>




USING_QPANDA
using namespace std;

class YGate : public RBGate
{
public:
	YGate() {}
	QGate qgate(Qubit* qbit) { return Y(qbit); }
	QStat unitary()
	{
		return{ 0, qcomplex_t(0, -1), qcomplex_t(0, 1), 0 };
	}
};

class XGate : public RBGate
{
public:
	XGate() {}
	QGate qgate(Qubit* qbit) { return X(qbit); }
	QStat unitary() 
	{
		return { 0, 1,1, 0 };
	}
};

class  XPowGate : public RBGate
{
public:
	XPowGate(double phi) { m_angle = PI * phi; }
	QGate qgate(Qubit* qbit) { return RX(qbit, m_angle); }

	QStat unitary()
	{
		QStat mat = { cos(m_angle / 2), qcomplex_t(0, -1 * sin(m_angle / 2)),
			qcomplex_t(0, -1 * sin(m_angle / 2)), cos(m_angle / 2) };
		return mat;
	}

private:
	double m_angle;
};

class YPowGate : public RBGate
{
public:
	YPowGate(double phi) { m_angle = PI * phi; }

	QGate qgate(Qubit* qbit) { return RY(qbit, m_angle); }
	QStat unitary()
	{
		QStat mat = { cos(m_angle / 2), -sin(m_angle / 2),
			sin(m_angle / 2), cos(m_angle / 2) };
		return mat;
	}
private:
	double m_angle;
};

RandomizedBenchmarking::RandomizedBenchmarking(MeasureQVMType type, QuantumMachine* qvm)
{
	m_qvm_type = type;
	if (m_qvm_type == MeasureQVMType::WU_YUAN)
		m_cloud_qvm = dynamic_cast<QCloudMachine*>(qvm);
	else
		m_mea_qvm = dynamic_cast<NoiseQVM*>(qvm);
}

RandomizedBenchmarking::~RandomizedBenchmarking()
{
}

std::map<int, double> RandomizedBenchmarking::single_qubit_rb(Qubit* qbit, const std::vector<int>& clifford_range, 
	int num_circuits, int shots, const std::vector<QGate> &interleaved_gates)
{
	auto cliffords = _single_qubit_cliffords();
	auto c1 = cliffords.c1_in_xy;
	std::vector<QStat> cfd_mats(c1.size());

	for (int i = 0; i < c1.size(); i++)
	{
		QStat tmp = { 1,0,0,1 };
		for (int j = 0; j < c1[i].size(); j++)
			tmp = c1[i][j]->unitary() * tmp;

		cfd_mats[i] = tmp;
	}

	std::vector<ClassicalCondition> cbits;
	if (m_qvm_type == MeasureQVMType::WU_YUAN)
	{
		cbits = m_cloud_qvm->cAllocMany(1);
	}
	else
	{
		cbits = m_mea_qvm->cAllocMany(1);
	}
	std::map<int, double> rb_result;
	for (const auto& num_cfds : clifford_range)
	{
		double total_probs = 0.0;
		for (int i = 0; i < num_circuits; i++)
		{
			QProg prog;
			prog << _random_single_q_clifford(qbit, num_cfds, c1, cfd_mats, interleaved_gates);
			prog << Measure(qbit, cbits[0]);
			if (m_qvm_type == MeasureQVMType::WU_YUAN)
			{
				auto res = m_cloud_qvm->real_chip_measure(prog, shots);
				if (res.find("0") != res.end())
					total_probs += res["0"];
			}
			else
			{
				std::map<string, size_t> res = m_mea_qvm->runWithConfiguration(prog, cbits, shots);
				if (res.find("0") != res.end())
					total_probs += res["0"] / (double)shots;
			}
		}
		rb_result[num_cfds] = total_probs / (double)num_circuits;
	}
	return rb_result;
}

QCircuit RandomizedBenchmarking::_random_single_q_clifford(Qubit* qbit, int num_cfds, const CliffordsSeq& cfds, 
	const std::vector<QStat>& cfd_matrices, const std::vector<QGate>& interleaved_gates)
{
	int clifford_group_size = cfds.size();
	std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
	std::uniform_int_distribution<> u(0, clifford_group_size - 1);
	std::vector<int> gate_ids(num_cfds);
	QCircuit gen_cir;

	for (int i = 0; i < num_cfds; i++)
	{
		int idx = u(gen);
		gate_ids[i] = idx;
		for (int j = 0; j < cfds[idx].size(); j++)
			gen_cir << cfds[idx][j]->qgate(qbit);

		for (auto gate : interleaved_gates)
		{
			gen_cir << gate;
		}
		gen_cir << BARRIER({ qbit });
	}

	QStat  cir_qstat = getCircuitMatrix(gen_cir);
	QMatrixXcd cir_mat = QMatrixXcd::Map(&cir_qstat[0], 2, 2);
	for (int i = 0; i < cfd_matrices.size(); i++)
	{
		QMatrixXcd tmp = QMatrixXcd::Map(&cfd_matrices[i][0], 2, 2);
		auto mat = tmp * cir_mat;
		if (abs(mat.trace()) / 2 > 0.999)
		{
			for (auto it : cfds[i])
				gen_cir << it->qgate(qbit);

			break;
		}
	}
	return gen_cir;
}


RandomizedBenchmarking::Cliffords RandomizedBenchmarking::_single_qubit_cliffords()
{
	Cliffords cfds;
	std::vector<double> phi0_vect = { 1.0, 0.5, -0.5 };
	std::vector<double> phi1_vect = { 0.0, 0.5, -0.5 };
	for (auto phi0 : phi0_vect)
	{
		for (auto phi1 : phi1_vect)
		{
			cfds.c1_in_xy.push_back({ make_shared<XPowGate>(phi0), make_shared< YPowGate>(phi1) });
			cfds.c1_in_xy.push_back({ make_shared<YPowGate>(phi0), make_shared< XPowGate>(phi1) });
		}
	}

	cfds.c1_in_xy.push_back({ make_shared < XPowGate>(0.0) });
	cfds.c1_in_xy.push_back({ make_shared < YGate>(), make_shared < XGate>() });


	std::vector<std::vector<double>>phi_xy = {
		{-0.5, 0.5, 0.5},
		{-0.5, -0.5, 0.5},
		{0.5, 0.5, 0.5},
		{-0.5, 0.5, -0.5} };

	for (auto phi : phi_xy)
	{
		cfds.c1_in_xy.push_back({ make_shared < XPowGate>(phi[0]), \
			make_shared < YPowGate>(phi[1]),make_shared < XPowGate>(phi[2]) });
	}


	cfds.s1 = {
		{make_shared < XPowGate>(0.0) },
		{make_shared<YPowGate>(0.5), make_shared<XPowGate>(0.5)},
		{make_shared<XPowGate>(-0.5), make_shared<YPowGate>(-0.5)}
	};

	cfds.s1_x = {
		{make_shared < XPowGate>(0.5) },
		{make_shared<XPowGate>(0.5), make_shared<YPowGate>(0.5), make_shared<XPowGate>(0.5)},
		{make_shared<YPowGate>(-0.5)}
	};

	cfds.s1_y = {
		{make_shared < YPowGate>(0.5) },
		{make_shared<XPowGate>(-0.5), make_shared<YPowGate>(-0.5), make_shared<XPowGate>(0.5)},
		{make_shared<YGate>(), make_shared<XPowGate>(0.5)}
	};
	return cfds;
}


std::vector<int >RandomizedBenchmarking::_split_two_q_clifford_idx(int idx)
{
	//Decompose the index for two-qubit Cliffords.
	int idx_0 = int(idx / 480);
	int idx_1 = int((idx % 480) * 0.05);
	int idx_2 = idx - idx_0 * 480 - idx_1 * 20;
	return { idx_0, idx_1, idx_2};
}

QCircuit RandomizedBenchmarking::_random_two_q_clifford(Qubit* q_0, Qubit* q_1, int num_cfds, const Cliffords& cfds, 
	const std::vector<QStat>& cfd_matrices, const std::vector<QGate>& interleaved_gates)
{
	auto _two_qubit_clifford = [&](int idx){
		auto split_idxs= _split_two_q_clifford_idx(idx);
		int idx_0 = split_idxs[0];
		int idx_1 = split_idxs[1];
		int idx_2 = split_idxs[2];
		QCircuit cir;
		cir << _two_qubit_clifford_starters(q_0, q_1, idx_0, idx_1, cfds)
			<< _two_qubit_clifford_mixers(q_0, q_1, idx_2, cfds);
		return cir;
	};

	int clifford_group_size = 11520;
	std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
	std::uniform_int_distribution<> u(0, clifford_group_size - 1);
	std::vector<int> gate_ids(num_cfds);

	QCircuit gen_cir;
	for (int i = 0; i < num_cfds; i++)
	{
		int idx = u(gen);
		gate_ids[i] = idx;
		gen_cir << _two_qubit_clifford(idx);

		for (auto gate : interleaved_gates)
		{
			gen_cir << gate;
		}
		QVec qv = { q_0, q_1 };
		gen_cir << BARRIER(qv);
	}

	auto mat_qstat = getCircuitMatrix(gen_cir);
	QMatrixXcd cir_mat = QMatrixXcd::Map(&mat_qstat[0], 4, 4);
	for (int i = 0; i < cfd_matrices.size(); i++)
	{
		QMatrixXcd tmp = QMatrixXcd::Map(&cfd_matrices[i][0], 4, 4);
		auto mat = tmp * cir_mat;
		if (abs(mat.trace()) / 4 > 0.999)
		{
			gen_cir << _two_qubit_clifford(i);
			break;
		}
	}
	return gen_cir;
}

QCircuit RandomizedBenchmarking::_two_qubit_clifford_starters(Qubit* q_0, Qubit* q_1, int idx_0, int idx_1, const Cliffords& cfds)
{
	auto c1 = cfds.c1_in_xy;
	QCircuit cir;
	for (auto it : c1[idx_0])
		cir << it->qgate(q_0);

	for (auto it : c1[idx_1])
		cir << it->qgate(q_1);

	return cir;
}

QCircuit RandomizedBenchmarking::_two_qubit_clifford_mixers(Qubit* q_0, Qubit* q_1, int idx_2, const Cliffords& cfds)
{
	QCircuit cir;
	auto s1 = cfds.s1;
	auto s1_x = cfds.s1_x;
	auto s1_y = cfds.s1_y;
	if (idx_2 == 1)
	{
		cir << CZ(q_0, q_1)
			<< RY(q_0, -0.5 * PI)
			<< RY(q_1, 0.5 * PI)
			<< CZ(q_0, q_1)
			<< RY(q_0, 0.5 * PI)
			<< RY(q_1, -0.5 * PI)
			<< CZ(q_0, q_1)
			<< RY(q_1, 0.5 * PI)
			;
	}
	else if (2 <= idx_2  && idx_2 <= 10)
	{
		cir << CZ(q_0, q_1);
		int idx_3 = int((idx_2 - 2) / 3);
		int idx_4 = (idx_2 - 2) % 3;
		for (auto it : s1[idx_3])
			cir << it->qgate(q_0);

		for (auto it : s1_y[idx_4])
			cir << it->qgate(q_1);
	}
	else if (idx_2 >= 11)
	{
		cir << CZ(q_0, q_1)
			<< RY(q_0, 0.5 * PI)
			<< RX(q_1, -0.5 * PI)
			<< CZ(q_0, q_1)
			;
		int idx_3 = int((idx_2 - 11) / 3);
		int idx_4 = (idx_2 - 11) % 3;
		for (auto it : s1_y[idx_3])
			cir << it->qgate(q_0);

		for (auto it : s1_x[idx_4])
			cir << it->qgate(q_1);
	}
	return cir;
}

std::vector <QStat>RandomizedBenchmarking::_two_qubit_clifford_matrices(Qubit* q_0, Qubit* q_1, const Cliffords& cfds)
{

	int clifford_group_size = 11520;
	vector<vector<QStat>> starters;

	auto c1 = cfds.c1_in_xy;
	for (int idx_0 = 0; idx_0 < 24; idx_0++)
	{
		vector<QStat> subset;
		for (int idx_1 = 0; idx_1 < 24; idx_1++)
		{
			QCircuit cir = _two_qubit_clifford_starters(q_0, q_1, idx_0, idx_1, cfds);
			subset.push_back(getCircuitMatrix(cir));
		}
		starters.push_back(subset);
	}
	vector<QStat> mixers;
	QStat eye(16, 0);
	eye[0] = eye[5] = eye[10] = eye[15] = 1;
	mixers.push_back(eye);

	for (int idx_2 = 1; idx_2 < 20; idx_2++)
	{
		QCircuit cir = _two_qubit_clifford_mixers(q_0, q_1, idx_2, cfds);
		mixers.push_back(getCircuitMatrix(cir));
	}

	vector < QStat >mats;
	for (int i = 0; i < clifford_group_size; i++)
	{
		auto idxs = _split_two_q_clifford_idx(i);
		int idx_0 = idxs[0];
		int idx_1 = idxs[1];
		int idx_2 = idxs[2];

		auto mat_qstat = mixers[idx_2] * starters[idx_0][idx_1];
		mats.push_back(mat_qstat);
	}
	return mats;
}


std::map<int, double> RandomizedBenchmarking::two_qubit_rb(Qubit* qbit0, Qubit* qbit1, const std::vector<int>& clifford_range, 
	int num_circuits, int shots, const std::vector<QGate>& interleaved_gates )
{
	std::map<int, double> rb_result;
	auto cfds = _single_qubit_cliffords();
	auto cfd_mats = _two_qubit_clifford_matrices(qbit0, qbit1, cfds);
	std::vector<ClassicalCondition> cbits;
	if (m_qvm_type == MeasureQVMType::WU_YUAN)
	{
		cbits = m_cloud_qvm->cAllocMany(2);
	}
	else
	{
		cbits = m_mea_qvm->cAllocMany(2);
	}
	for (const auto& num_cfds : clifford_range)
	{
		double total_probs = 0.0;
		for (int i = 0; i < num_circuits; i++)
		{
			QProg prog;
			prog<< _random_two_q_clifford(qbit0, qbit1, num_cfds, cfds, cfd_mats, interleaved_gates);
			prog << Measure(qbit0, cbits[0])
				<< Measure(qbit1, cbits[1]);
			if (m_qvm_type == MeasureQVMType::WU_YUAN)
			{
				auto res = m_cloud_qvm->real_chip_measure(prog, shots);
				if (res.find("00") != res.end())
					total_probs += res["00"];
			}
			else
			{
				std::map<string, size_t> res = m_mea_qvm->runWithConfiguration(prog, cbits, shots);
				if (res.find("00") != res.end())
					total_probs += (double)res["00"] / shots;
			}
	
		}
		rb_result[num_cfds] = total_probs / (double)num_circuits;
	}
	return rb_result;
}


std::map<int, double> QPanda::single_qubit_rb(NoiseQVM* qvm, Qubit* qbit, const std::vector<int>& clifford_range, int num_circuits, int shots, const std::vector<QGate>& interleaved_gates)
{
	RandomizedBenchmarking rb(MeasureQVMType::NOISE, qvm);
	return rb.single_qubit_rb(qbit, clifford_range, num_circuits, shots, interleaved_gates);
}

std::map<int, double> QPanda::double_qubit_rb(NoiseQVM* qvm, Qubit* qbit0, Qubit* qbit1, const std::vector<int>& clifford_range, int num_circuits, int shots, const std::vector<QGate>& interleaved_gates )
{
	RandomizedBenchmarking rb(MeasureQVMType::NOISE, qvm);
	return rb.two_qubit_rb(qbit0, qbit1, clifford_range, num_circuits, shots, interleaved_gates);
}

std::map<int, double> QPanda::single_qubit_rb(QCloudMachine* qvm, Qubit* qbit, const std::vector<int>& clifford_range, int num_circuits, int shots, const std::vector<QGate>& interleaved_gates )
{
	RandomizedBenchmarking rb(MeasureQVMType::WU_YUAN, qvm);
	return rb.single_qubit_rb(qbit, clifford_range, num_circuits, shots, interleaved_gates);
}

std::map<int, double> QPanda::double_qubit_rb(QCloudMachine* qvm, Qubit* qbit0, Qubit* qbit1, const std::vector<int>& clifford_range, int num_circuits, int shots, const std::vector<QGate>& interleaved_gates)
{
	RandomizedBenchmarking rb(MeasureQVMType::WU_YUAN, qvm);
	return rb.two_qubit_rb(qbit0, qbit1, clifford_range, num_circuits, shots, interleaved_gates);
}

struct vars_struct {
	double* x;
	double* y;
	double* ey;
};
int expfunc(int m, int n, double* p, double* dy, double** dvec, void* vars) {
	int i;
	struct vars_struct* v = (struct vars_struct*)vars;
	double* x, * y, * ey;

	x = v->x;
	y = v->y;
	ey = v->ey;

	for (i = 0; i < m; i++) {
		dy[i] = (y[i] - p[0] * pow(p[1], x[i]) - p[2]) / ey[i];
	}

	return 0;
}

double calc_single_qubit_fidelity(const std::map<int, double>& rb_result)
{
	const int NUM_PARAM = 3;
	double* x = new double[rb_result.size()];
	double* y = new double[rb_result.size()];
	double p[NUM_PARAM] = { 0.0, 0.5, 0.0 };       /* Initial conditions */
	//double pactual[NUM_PARAM] = {3.0, 0.70, 2.0};/* Actual values used to make data*/
	double perror[NUM_PARAM];                      /* Returned parameter errors */
	mp_par pars[NUM_PARAM];                        /* Parameter constraints */
	int i;
	struct vars_struct v;
	int status;
	mp_result result;

	memset(&result, 0, sizeof(result));  /* Zero results structure */
	result.xerror = perror;

	memset(pars, 0, sizeof(pars));    /* Initialize constraint structure */
	pars[1].limited[0] = 1;
	pars[1].limited[1] = 1;
	pars[1].limits[0] = 0.0;
	pars[1].limits[1] = 1.0;

	double* ey = new double[rb_result.size()];
	//double* ey = malloc(sizeof(double) * n);
	for (i = 0; i < rb_result.size(); ++i) ey[i] = 0.1;

	v.x = x;
	v.y = y;
	v.ey = ey;

	/* Call fitting function for 10 data points and 4 parameters (2
	   parameters fixed) */

	status = mpfit(expfunc, rb_result.size(), NUM_PARAM, p, pars, 0, (void*)&v, &result);

	double fidelity = 0.5 * (1 - p[1]);
	return fidelity;
}

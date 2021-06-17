#include "gtest/gtest.h"
#include "QPanda.h"


USING_QPANDA
using namespace std;

TEST(QProgInfo, rb)
{
	auto qvm = new NoiseQVM();
	qvm->init();
	auto qv = qvm->qAllocMany(4);
	std::vector<QVec> qvs = { {qv[0], qv[1]} };
	qvm->set_noise_model(DEPOLARIZING_KRAUS_OPERATOR, CZ_GATE, 0.05, qvs);

	std::vector<int > range = { 5,10,15 };
	std::map<int, double> res = single_qubit_rb(qvm,  qv[0], range, 10, 1000);
	//std::map<int, double> res = double_qubit_rb(noise_qvm,qv[0], qv[1], range, 10, 1000);

	for (auto it : res)
	{
		std::cout << it.first << "  :  " << it.second << std::endl;
	}

	qvm->finalize();
	delete qvm;
	
	getchar();
}

TEST(QProgInfo, xeb)
{
	auto noise_qvm = new NoiseQVM();
	noise_qvm->init();

	auto q = noise_qvm->qAllocMany(4);
	vector<QVec> qvs = { { q[0], q[1] } };
	noise_qvm->set_noise_model(DEPOLARIZING_KRAUS_OPERATOR, CZ_GATE, 0.1, qvs);
	vector<int> range = { 2,4,6,8,10 };
	auto r = double_gate_xeb(noise_qvm, q[0], q[1], range, 10, 1000, CZ_GATE);
	noise_qvm->finalize();
	delete noise_qvm;
	getchar();
}

TEST(QProgInfo, qv)
{
	auto noise_qvm = new NoiseQVM();
	noise_qvm->init();
	//set noise model
	noise_qvm->set_noise_model(NOISE_MODEL::DEPOLARIZING_KRAUS_OPERATOR, CZ_GATE, 0.005);
	std::vector <std::vector<int> >qubit_lists = { {1,2} };
	//std::vector <std::vector<int> >qubit_lists = { {1,2}, {1,2,3}, {1,2,3,4,5} };

	int ntrials = 50;
	int shots = 5000;
	auto res = calculate_quantum_volume(noise_qvm, qubit_lists, ntrials, shots);
	std::cout << "QV : " << res << std::endl;
	noise_qvm->finalize();
	delete noise_qvm;
	getchar();
}
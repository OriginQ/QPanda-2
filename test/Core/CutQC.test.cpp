#include <map>
#include <cstdlib>
#include <sstream>
#include <string>
#include <complex>
#include <algorithm>
#include "Core/Core.h"
#include "gtest/gtest.h"
#include "ThirdParty/Eigen/Dense"
#include "ThirdParty/Eigen/Sparse"
#include <EigenUnsupported/Eigen/KroneckerProduct>
#include "Extensions/Extensions.h"

#ifdef USE_EXTENSION

using namespace std;
USING_QPANDA

#define MINI_FIDELITY 0.97

static double fidelity(size_t qubit_num, size_t shots, std::map<std::string, size_t>& qvm_counts, std::map<std::string, double>& cut_result)
{
    std::map<std::string, double> qvm_result;

    for (auto val : qvm_counts)
    {
        auto qubits = val.first;
        auto counts = val.second;

        qvm_result[qubits] = (double)counts / shots;
    }

    double fidelity_result = 0.;

    for (uint64_t i = 0; i < (1ull << qubit_num); ++i)
    {
        auto bits = integerToBinary(i, qubit_num);

        auto qvm_iter_found = qvm_result.find(bits) != qvm_result.end();
        auto cut_iter_found = cut_result.find(bits) != cut_result.end();

        if (qvm_iter_found && cut_iter_found)
        {
            fidelity_result += (std::sqrt(qvm_result[bits] * cut_result[bits]));
        }
    }

    return fidelity_result * fidelity_result;
}

static double fidelity(std::map<std::string, double>actual_dist,
	std::map<std::string, double> mea_dist, int qnum)
{
	vector<string> bits_vect;
	for (int i = 0; i < (int)pow(2, qnum); i++)
	{
		bits_vect.push_back(dec2bin(i, qnum));
	}

	double fidelity = 0.0;
	for (auto bit : bits_vect)
	{
		if (actual_dist.find(bit) != actual_dist.end() &&
			mea_dist.find(bit) != mea_dist.end())
		{
			if (mea_dist[bit] > 0)
				fidelity += sqrt(actual_dist[bit] * mea_dist[bit]);
		}
	}
	return fidelity;
}

/*****************************************************************
*                   test cut
*/
static bool test_CutQC_1()
{
	auto qvm = initQuantumMachine();
	auto q = qvm->qAllocMany(5);
	auto c = qvm->cAllocMany(5);

	QCircuit cir1;
	//cir1 << H(q[0]) << X(q[1]) << Y(q[2]) << Z(q[3]) << S(q[4]) << CNOT(q[0], q[1]) << CNOT(q[1], q[2])
	//  	 << CNOT(q[2], q[3]) << CNOT(q[3], q[4]) << CNOT(q[4], q[0]) << CNOT(q[1], q[0]) << CNOT(q[2], q[0]) << CNOT(q[3], q[1]);

    cir1 << H(q[0])
        << X(q[1])
        << Y(q[2])
        << Z(q[3])
        << S(q[4])

        << CNOT(q[0], q[1])
        << U3(q[0], 1, 2, 3)

        << CNOT(q[1], q[2])
        << U3(q[1], 1, 2, 3)

        << CNOT(q[2], q[3])
        << U3(q[2], 1, 2, 3)

        << CNOT(q[3], q[4])
        << U3(q[3], 1, 2, 3)

        << CNOT(q[4], q[0])
        << U3(q[4], 1, 2, 3)

        << CNOT(q[1], q[0])
        << U3(q[3], 1, 2, 3)
        << CNOT(q[2], q[0])
        << U3(q[2], 1, 2, 3)
        << CNOT(q[3], q[1]);

    std::map<std::string, size_t> acutal_result;
    cout << "src cir" << cir1 << endl;
    {
        QProg test_prog;
        test_prog << cir1 << MeasureAll(q, c);  
        auto test_result = runWithConfiguration(test_prog, c, 10000);
        acutal_result = test_result;
        for (const auto& _i : test_result) {
            cout << _i.first << ": " << _i.second / 10000. << endl;
        }
    }
	
	std::map<std::string, double> result = exec_by_cutQC(cir1, qvm, "CPU", 3);
    for (const auto& val : result){
        cout << val.first << " : " << val.second << endl;
    }

	const auto f = fidelity(qvm->getAllocateQubitNum(), 10000, acutal_result, result);
    cout << "test_CutQC_1 fidelity : " << f << endl;

	destroyQuantumMachine(qvm);
	return f > MINI_FIDELITY;
}

static bool test_CutQC_2()
{
	auto qvm = initQuantumMachine();
	auto q = qvm->qAllocMany(6);
	auto c = qvm->cAllocMany(6);

	QCircuit cir1;
	cir1 << H(q[0]) << X(q[1]) << Y(q[2]) << Z(q[3]) << S(q[4]) << T(q[5]) << CNOT(q[1], q[2])
		<< CNOT(q[3], q[4]) << CNOT(q[0], q[1]) << CNOT(q[2], q[3]) << CNOT(q[4], q[5])
		<< CNOT(q[0], q[2]) << CNOT(q[3], q[4]) << CNOT(q[3], q[5]);

    std::map<std::string, size_t> acutal_result;
    cout << "src cir" << cir1 << endl;
    {
        QProg test_prog;
        test_prog << cir1 << MeasureAll(q, c);
        auto test_result = runWithConfiguration(test_prog, c, 4096);
        acutal_result = test_result;
        for (const auto& _i : test_result) {
            cout << _i.first << ": " << _i.second / 4096.0 << endl;
        }
    }

	std::map<std::string, double> result = exec_by_cutQC(cir1, qvm, "CPU", 3);
    for (auto val : result)
    {
        cout << val.first << " : " << val.second << endl;
    }

	const auto f = fidelity(qvm->getAllocateQubitNum(), 4096, acutal_result, result);
    cout << "fidelity : " << f << endl;

	destroyQuantumMachine(qvm);
	return f > MINI_FIDELITY;
}

static bool test_CutQC_3()
{
	auto qvm = initQuantumMachine();
	auto q = qvm->qAllocMany(6);
	auto c = qvm->cAllocMany(6);

	QCircuit cir1;
    cir1 << H(q[0]) << X(q[1])
        << Y(q[2]) << Z(q[3])
        << S(q[4]) << T(q[5])
        << CNOT(q[0], q[1])
        << CNOT(q[2], q[3])
        << CNOT(q[4], q[5]) << CNOT(q[1], q[2])
        << CNOT(q[3], q[4])
        << CNOT(q[0], q[1])
        << CNOT(q[2], q[3])
        << CNOT(q[4], q[5])
        << CNOT(q[1], q[3])
        //<< CNOT(q[4], q[2]) 
        << CNOT(q[0], q[5])

        << U3(q[0], 1, 2, 3)
        << U3(q[1], 1, 2, 3)
        << U3(q[2], 1, 2, 3);

    std::map<std::string, size_t> acutal_result;
    cout << "src cir" << cir1 << endl;
    {
        QProg test_prog;
        test_prog << cir1 << MeasureAll(q, c);
        auto test_result = runWithConfiguration(test_prog, c, 4096);
        acutal_result = test_result;
        for (const auto& _i : test_result) {
            cout << _i.first << ": " << _i.second / 4096.0 << endl;
        }
    }

	std::map<std::string, double> result = exec_by_cutQC(cir1, qvm, "CPU", 4);
    for (auto val : result)
    {
        cout << val.first << " : " << val.second << endl;
    }

	const auto f = fidelity(qvm->getAllocateQubitNum(), 4096, acutal_result, result);
    cout << "fidelity : " << f << endl;

	destroyQuantumMachine(qvm);
	return f > MINI_FIDELITY;
}

static bool test_CutQC_4()
{
	auto qvm = initQuantumMachine();
	auto q = qvm->qAllocMany(7);
	auto c = qvm->cAllocMany(7);

	QCircuit cir1;
	cir1 << H(q[0]) << X(q[1]) << Y(q[2]) << Z(q[3]) << S(q[4]) << T(q[5]) << T(q[6]).dagger() 
		<< CNOT(q[0], q[2]) << CNOT(q[3], q[6]) << CNOT(q[1], q[2]) << CNOT(q[3], q[5]) << CNOT(q[4], q[0])
		<< CNOT(q[1], q[3]) << CNOT(q[2], q[5]);

    std::map<std::string, size_t> acutal_result;
    cout << "src cir" << cir1 << endl;
    {
        QProg test_prog;
        test_prog << cir1 << MeasureAll(q, c);
        auto test_result = runWithConfiguration(test_prog, c, 4096);
        acutal_result = test_result;
        for (const auto& _i : test_result) {
            cout << _i.first << ": " << _i.second / 4096.0 << endl;
        }
    }

	std::map<std::string, double> result = exec_by_cutQC(cir1, qvm, "CPU", 3);
    for (auto val : result)
    {
        cout << val.first << " : " << val.second << endl;
    }

	const auto f = fidelity(qvm->getAllocateQubitNum(), 4096, acutal_result, result);
    cout << "fidelity : " << f << endl;

	destroyQuantumMachine(qvm);
	return f > MINI_FIDELITY;
}

static bool test_CutQC_5()
{
	auto qvm = initQuantumMachine();
	auto q = qvm->qAllocMany(7);
	auto c = qvm->cAllocMany(7);

	QCircuit cir1;
	cir1 << RZ(q[0], 0) << RZ(q[1], 1) << RZ(q[2], 2) << RZ(q[3], 3) << RZ(q[4], 4) << RZ(q[5], 5) << RZ(q[6], 6)
		<< H(q[0]) << X(q[1]) << Y(q[2]) << Z(q[3]) << S(q[4]) << T(q[5]) << T(q[6]).dagger()
		<< CNOT(q[0], q[2]) << CNOT(q[3], q[6]) << CNOT(q[1], q[2]) << CNOT(q[3], q[5]) 
		<< CNOT(q[4], q[0]) << CNOT(q[1], q[3]) << CNOT(q[2], q[5]) << CNOT(q[1], q[3])
		<< RZ(q[0], 0) << RZ(q[1], 1) << RZ(q[2], 2) << RZ(q[3], 3) << RZ(q[4], 4) << RZ(q[5], 5) << RZ(q[6], 6)
		<< RZ(q[5], 5) << RZ(q[6], 6);
	cout << "src cir" << cir1 << endl;

	QProg test_prog;
	test_prog << cir1;
	const auto test_result = probRunDict(test_prog, q, -1);
	for (const auto& _i : test_result) {
		cout << _i.first << ": " << _i.second << endl;
	}

	std::map<std::string, double> result = exec_by_cutQC(cir1, qvm, "CPU", 4);
	cout << "Recombiner result:\n";
    for (auto val : result){
        cout << val.first << " : " << val.second << endl;
    }

	const double f = fidelity(test_result, result, q.size());
	std::cout << "cutQC fidelity: " << f << std::endl;

	destroyQuantumMachine(qvm);
	return f > MINI_FIDELITY;
}

static bool test_CutQC_6()
{
	auto qvm = initQuantumMachine();
	auto q = qvm->qAllocMany(4);
	auto c = qvm->cAllocMany(4);

	QCircuit cir1;
	cir1 << RZ(q[0], 0) << RZ(q[1], 1) << RZ(q[2], 2) << RZ(q[3], 3)
		<< H(q[0]) << RX(q[1], PI/3.0) 
		<< CNOT(q[0], q[3]) << CNOT(q[1], q[2]) << CNOT(q[1], q[0]) << CNOT(q[2], q[3])
		<< H(q[2]) << RX(q[3], PI / 3.0)
		<< RZ(q[0], 0) << RZ(q[1], 1) << RZ(q[2], 2) << RZ(q[3], 3)
		<< U3(q[2], 2, 3, 5) << U3(q[3], 2, 3, 5);
	cout << "src cir" << cir1 << endl;

	std::vector<StitchesInfo> stitches;
	std::vector<uint32_t> qubit_permutation;
	std::vector<SubCircuit> sub_cir_info = cut_circuit(cir1, qvm, 3, stitches, qubit_permutation);

	uint32_t i = 0;
	for (auto& _sub_cir : sub_cir_info)
	{
		cout << "sub_cir_" << i << ":::::::::::::" << _sub_cir.m_cir << endl;
		++i;
	}

	/** 计算子线路数据
	*/
	RecombineFragment recombiner(sub_cir_info);
	std::vector<RecombineFragment::ResultDataMap> frag_data = recombiner.collect_fragment_data();

	/** 计算choi矩阵
	*/
	std::vector<RecombineFragment::ChoiMatrices> choi_states_vec;
	recombiner.direct_fragment_model(frag_data, choi_states_vec);

    //结果优化处理
    std::vector<RecombineFragment::ChoiMatrices> likely_choi_states_vec;
    recombiner.maximum_likelihood_model(choi_states_vec, likely_choi_states_vec);

	/** 重组子线路数据
	*/
	std::map<std::string, double> result = recombiner.recombine_using_insertions(likely_choi_states_vec, stitches, qubit_permutation);

	cout << "Recombiner result:\n";
    for (auto val : result)
    {
        cout << val.first << " : " << val.second << endl;
    }
	destroyQuantumMachine(qvm);
	return 0;
}

static bool test_CutQC_7()
{
	auto qvm = initQuantumMachine();
	auto q = qvm->qAllocMany(6);
	auto c = qvm->cAllocMany(6);

	QCircuit cir1;
	cir1 << H(q[0]) << X(q[1]) << Y(q[2]) << Z(q[3]) << S(q[4]) << T(q[5])
		<< CNOT(q[0], q[1]) << CNOT(q[2], q[3]) << CNOT(q[4], q[5]) << CNOT(q[1], q[2])
		<< CNOT(q[3], q[4]) << CNOT(q[0], q[1]) << CNOT(q[2], q[3]) << CNOT(q[4], q[5])
		<< CNOT(q[1], q[3]) << CNOT(q[4], q[2]) << CNOT(q[0], q[5]);
	cout << "src cir" << cir1 << endl;
	QProg prog;
	prog << cir1;
	auto probs_result = probRunDict(prog, q, -1);
	std::cout << "probRunDict : " << std::endl;
	for (auto iter : probs_result)
		std::cout << iter.first << " : " << iter.second << std::endl;

	std::map<std::string, double> result = exec_by_cutQC(cir1, qvm, "CPU", 4);
	for (auto val : result)
	{
		cout << val.first << " : " << val.second << endl;
	}

	const double f = fidelity(probs_result, result, q.size());
	std::cout << "cutQC fidelity : " << f << std::endl;

	destroyQuantumMachine(qvm);
	return f > MINI_FIDELITY;
}

static bool test_CutQC_7_2()
{
	auto qvm = initQuantumMachine();
	auto q = qvm->qAllocMany(12);
	auto c = qvm->cAllocMany(12);

	QCircuit cir1;
	cir1 << H(q[0]) << X(q[1]) << Y(q[2]) << Z(q[3]) << S(q[4]) << T(q[5])
		<< CNOT(q[0], q[1]) << CNOT(q[10], q[3]) << CNOT(q[4], q[8]) << CNOT(q[1], q[2])
		<< CNOT(q[11], q[8]) << CNOT(q[10], q[9]) << CNOT(q[7], q[3]) << CNOT(q[4], q[5])
		<< CNOT(q[4], q[1]) << CNOT(q[2], q[3]) << CNOT(q[6], q[8]) << CNOT(q[0], q[8])
		<< CNOT(q[10], q[5]) << CNOT(q[6], q[9]) << CNOT(q[8], q[6]) << CNOT(q[7], q[11])
		<< CNOT(q[11], q[2]) << CNOT(q[4], q[2]) << CNOT(q[4], q[7]);
	cout << "src cir" << cir1 << endl;
	QProg prog;
	prog << cir1;
	auto probs_result = probRunDict(prog, q, -1);
	std::cout << "probRunDict : " << std::endl;
	for (auto iter : probs_result)
		std::cout << iter.first << " : " << iter.second << std::endl;

	std::map<std::string, double> result = exec_by_cutQC(cir1, qvm, "CPU", 4);
	for (auto val : result)
	{
		cout << val.first << " : " << val.second << endl;
	}

	double f = fidelity(probs_result, result, q.size());
	std::cout << "cut QC  fidelity : " << f << std::endl;

	destroyQuantumMachine(qvm);
	return 1;
}

static bool test_CutQC_8()
{
	auto qvm = initQuantumMachine();
	/*auto q = qvm->qAllocMany(6);
	auto c = qvm->cAllocMany(6);*/

	QVec q;
	std::vector<ClassicalCondition> c;
	QProg prog = convert_originir_to_qprog("E://HHL_prog.ir", qvm, q, c);

	decompose_multiple_control_qgate(prog, qvm);
	//sub_cir_optimizer(prog, std::vector<std::pair<QCircuit, QCircuit>>(), QCircuitOPtimizerMode::Merge_U3);
	
	//test
	auto probs_result = probRunDict(prog, q, -1);
	std::cout << "probRunDict : " << std::endl;
	for (auto iter : probs_result)
		std::cout << iter.first << " : " << iter.second << std::endl;

	auto cir = QProgFlattening::prog_flatten_to_cir(prog);
	std::map<std::string, double> result = exec_by_cutQC(cir, qvm, "CPU", 4);
	for (auto val : result)
	{
		cout << val.first << " : " << val.second << endl;
	}

	double f = fidelity(probs_result, result, q.size());
	std::cout << "cut QC  fidelity : " << f << std::endl;

	destroyQuantumMachine(qvm);
	return 1;
}

TEST(CutQC, test1)
{
	bool test_val = true;
	try
	{
		test_val = test_val && test_CutQC_1();
		test_val = test_val && test_CutQC_2();
		test_val = test_val && test_CutQC_3();
		test_val = test_val && test_CutQC_4();
		test_val = test_val && test_CutQC_5();
		test_val = test_val && test_CutQC_7();
		//test_val = test_CutQC_7_2(); /**< very slow */
		//test_val = test_CutQC_8(); /**< very slow */
	}
	catch (const std::exception& e)
	{
		cout << "Got a exception: " << e.what() << endl; 
		test_val = false;
	}
	catch (...)
	{
		cout << "Got an unknow exception: " << endl;
		test_val = false;
	}

	cout << "CutQC test over." << endl;
	ASSERT_TRUE(test_val);
}

#endif
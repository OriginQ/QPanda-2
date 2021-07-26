#include <time.h>
#include <iostream>
#include <numeric>
#include "QPanda.h"
#include <functional>
#include "gtest/gtest.h"
#include "Core/Utilities/Tools/OriginCollection.h"
USING_QPANDA
using namespace std;

TEST(CPUQVMTest, testInit)
{
	return;
	CPUQVM qvm;
	ASSERT_THROW(auto qvec = qvm.allocateQubits(2), qvm_attributes_error);
	ASSERT_THROW(auto cvec = qvm.allocateCBits(2), qvm_attributes_error);

	qvm.init();
	ASSERT_NO_THROW(auto qvec = qvm.allocateQubits(2));
	ASSERT_NO_THROW(auto cvec = qvm.allocateCBits(2));

	ASSERT_THROW(auto qvec = qvm.allocateQubits(26), qalloc_fail);
	ASSERT_THROW(auto cvec = qvm.allocateCBits(257), calloc_fail);

	qvm.finalize();
	ASSERT_THROW(auto qvec = qvm.allocateQubits(2), qvm_attributes_error);
	ASSERT_THROW(auto cvec = qvm.allocateCBits(2), qvm_attributes_error);
	ASSERT_THROW(auto qvec = qvm.getAllocateQubit(), qvm_attributes_error);
	ASSERT_THROW(auto qvec = qvm.getAllocateCMem(), qvm_attributes_error);
	ASSERT_THROW(auto qvec = qvm.getResultMap(), qvm_attributes_error);
}

TEST(NoiseMachineTest, test)
{
	//return;
	rapidjson::Document doc;
	doc.Parse("{}");
	Value value(rapidjson::kObjectType);
	Value value_h(rapidjson::kArrayType);
	value_h.PushBack(DAMPING_KRAUS_OPERATOR, doc.GetAllocator());
	value_h.PushBack(0.5, doc.GetAllocator());
	value.AddMember("H", value_h, doc.GetAllocator());

	Value value_rz(rapidjson::kArrayType);
	value_rz.PushBack(DAMPING_KRAUS_OPERATOR, doc.GetAllocator());
	value_rz.PushBack(0.5, doc.GetAllocator());
	value.AddMember("RZ", value_rz, doc.GetAllocator());

	Value value_cnot(rapidjson::kArrayType);
	value_cnot.PushBack(DAMPING_KRAUS_OPERATOR, doc.GetAllocator());
	value_cnot.PushBack(0.5, doc.GetAllocator());
	value.AddMember("CPHASE", value_cnot, doc.GetAllocator());
	doc.AddMember("noisemodel", value, doc.GetAllocator());

	NoiseQVM qvm;
	qvm.init();
	auto qvec = qvm.allocateQubits(16);
	auto cvec = qvm.allocateCBits(16);
	auto prog = QProg();

	QCircuit  qft = CreateEmptyCircuit();
	for (auto i = 0; i < qvec.size(); i++)
	{
		qft << H(qvec[qvec.size() - 1 - i]);
		for (auto j = i + 1; j < qvec.size(); j++)
		{
			qft << CR(qvec[qvec.size() - 1 - j], qvec[qvec.size() - 1 - i], 2 * PI / (1 << (j - i + 1)));
		}
	}

	prog << qft << qft.dagger()
		<< MeasureAll(qvec, cvec);

	rapidjson::Document doc1;
	doc1.Parse("{}");
	auto& alloc = doc1.GetAllocator();
	doc1.AddMember("shots", 10, alloc);

	clock_t start = clock();
	auto result = qvm.runWithConfiguration(prog, cvec, doc1);
	clock_t end = clock();
	std::cout << end - start << endl;

	/*for (auto& aiter : result)
	{
		std::cout << aiter.first << " : " << aiter.second << endl;
	}*/

	ASSERT_EQ(result.begin()->second, 10);

	//auto state = qvm.getQState();
	//for (auto &aiter : state)
	//{
	//    std::cout << aiter << endl;
	//}
	qvm.finalize();

	//std::cout << "NoiseMachineTest.test  tests over!" << endl;
}

QStat run_one_qubit_circuit()
{
	CPUQVM cpu;
	cpu.init();

	auto q = cpu.qAllocMany(1);
	auto c = cpu.cAllocMany(1);

	QProg prog;
	prog << H(q[0])
		<< RX(q[0], PI / 4)
		<< RX(q[0], PI / 5)
		<< RX(q[0], PI / 3)
		<< RZ(q[0], PI / 4);

	cpu.directlyRun(prog);
	return cpu.getQState();
}

double getStateProb(complex<double> val)
{
	return val.real() * val.real() + val.imag() * val.imag();
}

TEST(QVM, PartialAmplitudeQVM)
{
	auto machine = new PartialAmplitudeQVM();
	machine->init();
	auto qlist = machine->allocateQubits(40);
	auto clist = machine->allocateCBits(40);
	auto Toffoli = X(qlist[20]);
	Toffoli.setControl({ qlist[18], qlist[19] });

	auto prog = QProg();
	prog << H(qlist[18])
		<< X(qlist[19]);
	//<< Toffoli;

	machine->run(prog);

	std::vector<string> subSet = { "0000000000000000000001000000000000000000" ,
								   "0000000000000000000010000000000000000000" ,
								   "0000000000000000000011000000000000000000" ,
								   "0000000000000000000100000000000000000000" ,
								   "0000000000000000000101000000000000000000" ,
								   "0000000000000000000110000000000000000000" ,
								   "0000000000000000000111000000000000000000" ,
								   "1000000000000000000000000000000000000000" };

	/*for (int i = 0; i < subSet.size(); ++i)
	{
		auto result = machine->PMeasure_bin_index(subSet[i]);
		std::cout << result << std::endl;
	}*/

	ASSERT_EQ(subSet.size(), 8);

	//std::cout << val.first << " : " << val.second << std::endl;


	//getchar();
}

TEST(QVM, SingleAmplitudeQVM)
{
	auto qvm = new SingleAmplitudeQVM();
	qvm->init();
	auto qv = qvm->qAllocMany(11);
	auto cv = qvm->cAllocMany(11);

	auto prog = QProg();
	for_each(qv.begin(), qv.end(), [&](Qubit* val) { prog << H(val); });
	prog << CZ(qv[1], qv[5])
		<< CZ(qv[3], qv[5])
		<< CZ(qv[2], qv[4])
		<< CZ(qv[3], qv[7])
		<< CZ(qv[0], qv[4])
		<< RY(qv[7], PI / 2)
		<< RX(qv[8], PI / 2)
		<< RX(qv[9], PI / 2)
		<< CR(qv[0], qv[1], PI)
		<< CR(qv[2], qv[3], PI)
		<< RY(qv[4], PI / 2)
		<< RZ(qv[5], PI / 4)
		<< RX(qv[6], PI / 2)
		<< RZ(qv[7], PI / 4)
		<< CR(qv[8], qv[9], PI)
		<< CR(qv[1], qv[2], PI)
		<< RY(qv[3], PI / 2)
		<< RX(qv[4], PI / 2)
		<< RX(qv[5], PI / 2)
		<< CR(qv[9], qv[1], PI)
		<< RY(qv[1], PI / 2)
		<< RY(qv[2], PI / 2)
		<< RZ(qv[3], PI / 4)
		<< CR(qv[7], qv[8], PI);

	// pMeasureBinindex : 获取对应（二进制）量子态概率
	// run 有三个参数，默认2个，
	//第一个执行的量子程序;
	//第二个为申请的量子比特
	//第三个为最大RANK，这里根据内存设置，默认30；
	//第四个就是quickBB优化的最大运行时间，默认5s
	qvm->run(prog, qv);
	//cout << qvm->pMeasureBinindex("00001100000") << endl;

	// pMeasureDecindex : 获取对应（10进制）量子态概率
	qvm->run(prog, qv);
	//cout << qvm->pMeasureDecindex("2") << endl;

	// getProbDict 获取对应量子比特所有量子态（如果申请比特数超过30， 该接口不提供使用）
	qvm->run(prog, qv);
	auto res_1 = qvm->getProbDict(qv);

	// probRunDict  上面两个接口的封装
	auto res = qvm->probRunDict(prog, qv);
	/*for (auto val : res)
	{
		std::cout << val.first << " : " << val.second << std::endl;
	}*/

	ASSERT_EQ(res.size(), 2048);
	qvm->finalize();
	delete(qvm);
	//getchar();
}

TEST(QubitAddr, test_0)
{
	// 量子比特可以和虚拟机 脱离关系
	auto qpool = OriginQubitPool::get_instance();
	auto cmem = OriginCMem::get_instance();

	//获取容器大小
	//std::cout << "set qubit pool capacity  before: "<< qpool->get_capacity() << std::endl;
	// 设置最大容器
	qpool->set_capacity(20);
	//std::cout << "set qubit pool capacity  after: " << qpool->get_capacity() << std::endl;

	// 构建虚拟机
	auto qvm = new CPUQVM();
	qvm->init();
	auto qv = qpool->qAllocMany(6);
	auto cv = cmem->cAllocMany(6);

	// 获取被申请的量子比特
	QVec used_qv;
	auto used_qv_size = qpool->get_allocate_qubits(used_qv);
	//std::cout << "allocate qubits number: " << used_qv_size << std::endl;


	auto prog = QProg();
	// 直接使用物理地址作为量子比特信息入参
	prog << H(0)
		<< H(1)
		<< H(2)
		<< H(4)
		<< X(5)
		<< X1(2)
		<< CZ(2, 3)
		<< RX(3, PI / 4)
		<< CR(4, 5, PI / 2)
		<< SWAP(3, 5)
		<< CU(1, 3, PI / 2, PI / 3, PI / 4, PI / 5)
		<< U4(4, 2.1, 2.2, 2.3, 2.4)
		<< BARRIER({ 0, 1,2,3,4,5 })
		<< BARRIER(0)
		;

	// 测量方法也可以使用比特物理地址 
	auto res_0 = qvm->probRunDict(prog, { 0,1,2,3,4,5 });
	// auto res_1 = qvm->probRunDict(prog, qv);  //同等上述方法

	// 同样经典比特地址也可以作为经典比特信息入参
	prog << Measure(0, 0)
		<< Measure(1, 1)
		<< Measure(2, 2)
		<< Measure(3, 3)
		<< Measure(4, 4)
		<< Measure(5, 5)
		;

	// 使用经典比特地址入参 
	vector<int> cbit_addrs = { 0,1,2,3,4,5 };
	auto res_2 = qvm->runWithConfiguration(prog, cbit_addrs, 5000);
	// auto res_3 = qvm->runWithConfiguration(prog, cv, 5000); //同等上述方法
	qvm->finalize();
	delete(qvm);

	auto qvm_noise = new NoiseQVM();
	qvm_noise->init();
	auto res_4 = qvm_noise->runWithConfiguration(prog, cbit_addrs, 5000);
	qvm_noise->finalize();
	delete(qvm_noise);

	ASSERT_EQ(res_2.size(), 48);

	//getchar();
}


void GHZ(int a)
{
	CPUQVM qvm;
	qvm.setConfigure({ 64,64 });
	qvm.init();

	auto q = qvm.qAllocMany(a);
	auto c = qvm.cAllocMany(a);

	auto prog = QProg();
	prog << H(q[0]);

	for (auto i = 0; i < a - 1; ++i)
	{
		prog << CNOT(q[i], q[i + 1]);
	}
	prog << MeasureAll(q, c);


	const string ss = "GHZ_" + to_string(a);
	write_to_originir_file(prog, &qvm, ss);
}

QStat run_origin_circuit()
{
	CPUQVM cpu;
	cpu.setConfigure({ 64,64 });
	cpu.init();

	auto q = cpu.qAllocMany(3);
	auto c = cpu.cAllocMany(3);

	QProg prog;
	prog << H(q[0])
		<< H(q[1])
		<< H(q[2])

		<< RX(q[0], PI / 6)
		<< RY(q[1], PI / 3)
		<< RX(q[2], PI / 6)

		<< CNOT(q[0], q[1])
		<< H(q[2])

		<< RY(q[0], PI / 4)
		<< RZ(q[1], PI / 4)

		<< RX(q[0], PI / 6)
		<< CR(q[2], q[1], PI / 6)

		<< RY(q[1], PI / 6)
		<< RY(q[2], PI / 3);

	QProg prog1;
	prog1 << H(q[0])
		<< CNOT(q[0], q[1])
		<< CNOT(q[0], q[2])
		<< CNOT(q[1], q[2])
		<< CZ(q[1], q[2]);

	cpu.directlyRun(prog);
	return cpu.getQState();
}

const double _SQ2 = 1 / 1.4142135623731;
const double _PI = 3.14159265358979;

const QStat mi{ 1., 0., 0., 1. };
const QStat mz{ 1., 0., 0., -1. };
const QStat mx{ 0., 1., 1., 0. };
const QStat my{ 0., qcomplex_t(0., -1.), qcomplex_t(0., 1.), 0. };
const QStat mh{ _SQ2, _SQ2, _SQ2, -_SQ2 };
const QStat mt{ 1, 0, 0, qcomplex_t(_SQ2, _SQ2) };

QStat m10 = mi * mz;
QStat m11 = mi;

//QStat m20 = mi * (-1) * mz;
QStat m20 = mi * mz;
QStat m21 = mx;

QStat m30 = mx;
QStat m31 = mh * mh * mx;

QStat m40 = my;
QStat m41 = mh * mt * mh * mt * mx;


QStat run_subset_circuit10()
{
	CPUQVM cpu;
	cpu.init();

	auto q = cpu.qAllocMany(2);
	auto c = cpu.cAllocMany(2);

	QProg prog;
	prog << H(q[0])
		<< H(q[1])

		<< RX(q[0], PI / 6)
		<< RY(q[1], PI / 3)

		<< CNOT(q[0], q[1])

		<< RY(q[0], PI / 4)
		<< U4(m10, q[1])

		<< RX(q[0], PI / 6);

	cpu.directlyRun(prog);
	return cpu.getQState();
}

QStat run_subset_circuit20()
{
	CPUQVM cpu;
	cpu.init();

	auto q = cpu.qAllocMany(2);
	auto c = cpu.cAllocMany(2);

	QProg prog;
	prog << H(q[0])
		<< H(q[1])

		<< RX(q[0], PI / 6)
		<< RY(q[1], PI / 3)

		<< CNOT(q[0], q[1])

		<< RY(q[0], PI / 4)
		<< U4(m20, q[1])

		<< RX(q[0], PI / 6);


	cpu.directlyRun(prog);
	return cpu.getQState();
}

QStat run_subset_circuit30()
{
	CPUQVM cpu;
	cpu.init();

	auto q = cpu.qAllocMany(2);
	auto c = cpu.cAllocMany(2);

	QProg prog;
	prog << H(q[0])
		<< H(q[1])

		<< RX(q[0], PI / 6)
		<< RY(q[1], PI / 3)

		<< CNOT(q[0], q[1])

		<< RY(q[0], PI / 4)
		<< U4(m30, q[1])

		<< RX(q[0], PI / 6);


	cpu.directlyRun(prog);
	return cpu.getQState();
}

QStat run_subset_circuit40()
{
	CPUQVM cpu;
	cpu.init();

	auto q = cpu.qAllocMany(2);
	auto c = cpu.cAllocMany(2);

	QProg prog;
	prog << H(q[0])
		<< H(q[1])

		<< RX(q[0], PI / 6)
		<< RY(q[1], PI / 3)

		<< CNOT(q[0], q[1])

		<< RY(q[0], PI / 4)
		<< U4(m40, q[1])

		<< RX(q[0], PI / 6);


	cpu.directlyRun(prog);
	return cpu.getQState();
}

QStat run_subset_circuit11()
{
	CPUQVM cpu;
	cpu.init();

	auto q = cpu.qAllocMany(2);
	auto c = cpu.cAllocMany(2);

	QProg prog;
	prog << U4(m11, q[0])
		<< H(q[1])

		<< RX(q[1], PI / 6)

		<< RZ(q[0], PI / 4)
		<< H(q[1])

		<< CR(q[1], q[0], PI / 6)

		<< RY(q[0], PI / 6)
		<< RY(q[1], PI / 3);

	cpu.directlyRun(prog);
	return cpu.getQState();
}

QStat run_subset_circuit21()
{
	CPUQVM cpu;
	cpu.init();

	auto q = cpu.qAllocMany(2);
	auto c = cpu.cAllocMany(2);

	QProg prog;
	prog << U4(m21, q[0])
		<< H(q[1])

		<< RX(q[1], PI / 6)

		<< RZ(q[0], PI / 4)
		<< H(q[1])

		<< CR(q[1], q[0], PI / 6)

		<< RY(q[0], PI / 6)
		<< RY(q[1], PI / 3);

	cpu.directlyRun(prog);
	return cpu.getQState();
}

QStat run_subset_circuit31()
{
	CPUQVM cpu;
	cpu.init();

	auto q = cpu.qAllocMany(2);
	auto c = cpu.cAllocMany(2);

	QProg prog;
	prog << U4(m31, q[0])
		<< H(q[1])

		<< RX(q[1], PI / 6)

		<< RZ(q[0], PI / 4)
		<< H(q[1])

		<< CR(q[1], q[0], PI / 6)

		<< RY(q[0], PI / 6)
		<< RY(q[1], PI / 3);

	cpu.directlyRun(prog);
	return cpu.getQState();
}

QStat run_subset_circuit41()
{
	CPUQVM cpu;
	cpu.init();

	auto q = cpu.qAllocMany(2);
	auto c = cpu.cAllocMany(2);

	QProg prog;
	prog << U4(m41, q[0])
		<< H(q[1])

		<< RX(q[1], PI / 6)

		<< RZ(q[0], PI / 4)
		<< H(q[1])

		<< CR(q[1], q[0], PI / 6)

		<< RY(q[0], PI / 6)
		<< RY(q[1], PI / 3);

	cpu.directlyRun(prog);
	return cpu.getQState();
}

void test_3bit()
{
	auto result = run_origin_circuit();

	auto result10 = run_subset_circuit10();
	auto result11 = run_subset_circuit11();

	auto result20 = run_subset_circuit20();
	auto result21 = run_subset_circuit21();

	auto result30 = run_subset_circuit30();
	auto result31 = run_subset_circuit31();

	auto result40 = run_subset_circuit40();
	auto result41 = run_subset_circuit41();

	//000
	auto c1 = result10[0] * result11[0] +
		result20[0] * result21[0] +
		result30[0] * result31[0] +
		result40[0] * result41[0];

	//001
	auto c2 = result10[0] * result11[1] +
		result20[0] * result21[1] +
		result30[0] * result31[1] +
		result40[0] * result41[1];

	//010
	auto c3 = result10[1] * result11[2] +
		result20[1] * result21[2] +
		result30[1] * result31[2] +
		result40[1] * result41[2];

	//011
	auto c4 = result10[1] * result11[3] +
		result20[1] * result21[3] +
		result30[1] * result31[3] +
		result40[1] * result41[3];

	//100
	auto c5 = result10[2] * result11[0] +
		result20[2] * result21[0] +
		result30[2] * result31[0] +
		result40[2] * result41[0];

	//101
	auto c6 = result10[2] * result11[1] +
		result20[2] * result21[1] +
		result30[2] * result31[1] +
		result40[2] * result41[1];

	//110
	auto c7 = result10[3] * result11[2] +
		result20[3] * result21[2] +
		result30[3] * result31[2] +
		result40[3] * result41[2];

	//111
	auto c8 = result10[3] * result11[3] +
		result20[3] * result21[3] +
		result30[3] * result31[3] +
		result40[3] * result41[3];

	QStat sub_result = { c1,c2,c3,c4,c5,c6,c7,c8 };

	prob_vec sub_probs;
	prob_vec ori_probs;
	for (auto val : sub_result)
	{
		sub_probs.emplace_back(std::norm(val));
	}

	for (auto val : result)
	{
		ori_probs.emplace_back(std::norm(val));
	}

	auto sum_sub_val = std::accumulate(sub_probs.begin(), sub_probs.end(), 0.);
	auto sum_ori_val = std::accumulate(ori_probs.begin(), ori_probs.end(), 0.);

	std::cout << "origin result | sub_result : " << endl;
	for (auto i = 0; i < result.size(); ++i)
	{
		std::cout << result[i] << " | " << sub_result[i] / 2. << endl;
	}
}

QStat run_cir(QProg& prog)
{
	CPUQVM cpu;
	cpu.init();

	auto q = cpu.qAllocMany(2);
	auto c = cpu.cAllocMany(2);

	cpu.directlyRun(prog);
	return cpu.getQState();
}


static double state_probs(QStat& state)
{
	double result = 0.;
	for (auto val : state)
	{
		result += std::norm(val);
	}

	return result;
}

static prob_vec state_to_probs(QStat& state)
{
	prob_vec probs;
	for (auto val : state)
	{
		probs.emplace_back(std::norm(val));
	}

	return probs;
}

prob_vec origin_probs_vec()
{
	//get origin quantum circuit result
	CPUQVM qvm;
	qvm.init();

	auto qv = qvm.qAllocMany(3);
	auto cv = qvm.cAllocMany(3);

	QProg prog;
	prog << H(qv[0])
		<< H(qv[1])
		<< RX(qv[0], PI / 6)
		<< RX(qv[1], PI / 6)
		<< CNOT(qv[0], qv[1])
		<< CNOT(qv[1], qv[2])
		<< RX(qv[1], PI / 6)
		<< RX(qv[2], PI / 6)
		<< RY(qv[1], PI / 6)
		<< RY(qv[2], PI / 6);

	std::cout << prog << std::endl;

	qvm.directlyRun(prog);
	auto origin_state = qvm.getQState();
	return state_to_probs(origin_state);
}

double get_expectation_O_000(prob_vec probs)
{
	//00 + 01 - 10 - 11
	return (probs[0] + probs[2]) * (probs[0] + probs[1] - probs[2] - probs[3]);
}

double get_expectation_P_000(prob_vec probs)
{
	//00 - 01 + 10 - 11
	return probs[0] * (probs[0] - probs[1] + probs[2] - probs[3]);
}

double get_expectation_001(prob_vec probs)
{
	return probs[0] * (probs[0] - probs[1] + probs[2] - probs[3]);
}


void cut_one_qubit_circuit()
{
	//get origin quantum circuit result
	auto origin_probs = origin_probs_vec();

	CPUQVM cpu;
	cpu.init();

	auto q = cpu.qAllocMany(3);
	auto c = cpu.cAllocMany(3);

	QProg up_origin_programs;
	up_origin_programs << H(q[0])
		<< H(q[1])
		<< RX(q[0], PI / 6)
		<< RX(q[1], PI / 6)
		<< CNOT(q[0], q[1]);

	QProg down_origin_programs;
	down_origin_programs << CNOT(q[0], q[1])
		<< RX(q[0], PI / 6)
		<< RX(q[1], PI / 6)
		<< RY(q[0], PI / 6)
		<< RY(q[1], PI / 6);

	std::vector<QProg> up_programs(8);

	up_programs[0] << up_origin_programs;

	up_programs[1] << up_origin_programs << H(q[1]);
	//up_programs[1] << up_origin_programs << RY(q[0], -PI / 2);

	up_programs[2] << up_origin_programs << S(q[1]).dagger() << H(q[1]);
	//up_programs[2] << up_origin_programs << RX(q[0], -PI / 2);

	up_programs[3] << up_origin_programs;
	//up_programs[3] << up_origin_programs << Z(q[0]);

	std::vector<QProg> down_programs(8);

	down_programs[0] << I(q[0]) << down_origin_programs;
	down_programs[1] << X(q[0]) << down_origin_programs;

	down_programs[2] << H(q[0]) << down_origin_programs;
	down_programs[3] << H(q[0]) << Z(q[0]) << down_origin_programs;

	down_programs[4] << H(q[0]) << S(q[0]) << down_origin_programs;
	down_programs[5] << H(q[0]) << S(q[0]) << Z(q[0]) << down_origin_programs;

	down_programs[6] << I(q[0]) << down_origin_programs;
	down_programs[7] << X(q[0]) << down_origin_programs;

	//std::vector<QStat> up_state;
	//std::vector<QStat> down_state;
	std::vector <prob_vec> up_probs;
	std::vector <prob_vec> down_probs;

	for (auto i = 0; i < 4; ++i)
	{
		auto up_result = run_cir(up_programs[i]);

		up_probs.emplace_back(state_to_probs(up_result));
	}

	for (auto i = 0; i < 8; ++i)
	{
		auto down_result = run_cir(down_programs[i]);

		down_probs.emplace_back(state_to_probs(down_result));
	}

	QStat result(2, .0);
	prob_vec probs_result(2, .0);

#if 0

	//probs_result[0] += up_probs[0][0] * down_probs[0][0] * (1. / 2);
	//probs_result[0] += up_probs[1][0] * down_probs[1][0] * (1. / 2);
	probs_result[0] += up_probs[2][0] * down_probs[2][0] * (up_probs[2][0] - up_probs[2][1]) * (down_probs[2][0] - down_probs[2][1]) * (1. / 2);
	probs_result[0] += up_probs[3][0] * down_probs[3][0] * (up_probs[3][0] - up_probs[3][1]) * (down_probs[3][0] - down_probs[3][1]) * (1. / 2) * (-1.);
	//probs_result[0] += up_probs[4][0] * down_probs[4][0] * (up_probs[4][0] - up_probs[4][1]) * (down_probs[4][0] - down_probs[4][1]) * (1. / 2);
	//probs_result[0] += up_probs[5][0] * down_probs[5][0] * (up_probs[5][0] - up_probs[5][1]) * (down_probs[5][0] - down_probs[5][1]) * (1. / 2) * (-1.);
	//probs_result[0] += up_probs[6][0] * down_probs[6][0] * (up_probs[6][0] - up_probs[6][1]) * (down_probs[6][0] - down_probs[6][1]) * (1. / 2);
	//probs_result[0] += up_probs[7][0] * down_probs[7][0] * (up_probs[7][0] - up_probs[7][1]) * (down_probs[7][0] - down_probs[7][1]) * (1. / 2) * (-1.);

	//probs_result[0] += up_probs[0][0] * down_probs[0][0] * (1. / 2);
	//probs_result[0] += up_probs[1][0] * down_probs[1][0] * (1. / 2);
	//probs_result[0] += (up_probs[2][0] - up_probs[2][1]) * (down_probs[2][0] - down_probs[2][1]) * (1. / 2);
	//probs_result[0] += (up_probs[3][0] - up_probs[3][1]) * (down_probs[3][0] - down_probs[3][1]) * (1. / 2) * (-1.);
	//probs_result[0] += (up_probs[4][0] - up_probs[4][1]) * (down_probs[4][0] - down_probs[4][1]) * (1. / 2);
	//probs_result[0] += (up_probs[5][0] - up_probs[5][1]) * (down_probs[5][0] - down_probs[5][1]) * (1. / 2) * (-1.);
	//probs_result[0] += (up_probs[6][0] - up_probs[6][1]) * (down_probs[6][0] - down_probs[6][1]) * (1. / 2);
	//probs_result[0] += (up_probs[7][0] - up_probs[7][1]) * (down_probs[7][0] - down_probs[7][1]) * (1. / 2) * (-1.);

	//probs_result[0] += up_probs[0][0] * down_probs[0][0] * (1. / 2);
	//probs_result[0] += up_probs[1][0] * down_probs[1][0] * (1. / 2);
	//probs_result[0] += up_probs[2][0] * down_probs[2][0] * (1. / 2);
	//probs_result[0] += up_probs[3][0] * down_probs[3][0] * (1. / 2) * (-1.);
	//probs_result[0] += up_probs[4][0] * down_probs[4][0] * (1. / 2);
	//probs_result[0] += up_probs[5][0] * down_probs[5][0] * (1. / 2) * (-1.);
	//probs_result[0] += up_probs[6][0] * down_probs[6][0] * (1. / 2);
	//probs_result[0] += up_probs[7][0] * down_probs[7][0] * (1. / 2) * (-1.);

	probs_result[1] += up_probs[0][1] * down_probs[0][1] * (1. / 2);
	probs_result[1] += up_probs[1][1] * down_probs[1][1] * (1. / 2);
	probs_result[1] += up_probs[2][1] * down_probs[2][1] * (1. / 2);
	probs_result[1] += up_probs[3][1] * down_probs[3][1] * (1. / 2) * (-1.);
	probs_result[1] += up_probs[4][1] * down_probs[4][1] * (1. / 2);
	probs_result[1] += up_probs[5][1] * down_probs[5][1] * (1. / 2) * (-1.);
	probs_result[1] += up_probs[6][1] * down_probs[6][1] * (1. / 2);
	probs_result[1] += up_probs[7][1] * down_probs[7][1] * (1. / 2) * (-1.);

#else

	if (1)
	{
		//probs_result[0] += up_probs[0][0] * (down_probs[0][0] + down_probs[1][1]);
		//probs_result[0] += get_expectation_O_000(up_probs[1]) * (get_expectation_P_000(down_probs[2]) - get_expectation_P_000(down_probs[3]));
		//probs_result[0] += get_expectation_O_000(up_probs[2]) * (get_expectation_P_000(down_probs[4]) - get_expectation_P_000(down_probs[5]));
		//probs_result[0] += get_expectation_O_000(up_probs[3]) * (get_expectation_P_000(down_probs[6]) - get_expectation_P_000(down_probs[7]));
		//probs_result[0] *= 1 / 2.;

		probs_result[0] += (up_probs[0][0] + up_probs[0][2]) * (down_probs[0][0] + down_probs[1][1]);
		probs_result[0] += get_expectation_O_000(up_probs[1]) * (get_expectation_P_000(down_probs[2]) - get_expectation_P_000(down_probs[3]));
		probs_result[0] += get_expectation_O_000(up_probs[2]) * (get_expectation_P_000(down_probs[4]) - get_expectation_P_000(down_probs[5]));
		probs_result[0] += get_expectation_O_000(up_probs[3]) * (get_expectation_P_000(down_probs[6]) - get_expectation_P_000(down_probs[7]));
		probs_result[0] *= 1 / 2.;
	}
	else
	{

	}

	std::cout << origin_probs[0] << " | " << probs_result[0] << endl;


#endif

	//auto origin_state = run_one_qubit_circuit();

	//auto p0 = state_probs(result);
	//auto p1 = state_probs(origin_state);

	//std::cout << "origin result | sub_result : " << endl;
	//for (auto i = 0; i < origin_probs.size(); ++i)
	//{
	//    std::cout << origin_probs[i] << " | " << probs_result[i] << endl;
	//}

	return;
}

void cut_one_qubit_circuit_2020()
{
	CPUQVM cpu;
	cpu.init();

	auto q = cpu.qAllocMany(1);
	auto c = cpu.cAllocMany(1);

	//origin circuit
	QProg prog;
	prog << H(q[0])
		<< RX(q[0], PI / 4)
		<< RX(q[0], PI / 5)
		<< RX(q[0], PI / 3)
		<< RZ(q[0], PI / 4);

	//P0,P1
	QStat matrix_p0 = { 1.,0.,0.,0. };
	QStat matrix_p1 = { 0.,0.,0.,1. };

	QProg up_origin_programs;
	up_origin_programs << H(q[0]) << RX(q[0], PI / 4) << RX(q[0], PI / 5);

	QProg down_origin_programs;
	down_origin_programs << RX(q[0], PI / 3) << RZ(q[0], PI / 4);

	std::vector<QProg> up_programs(4);

	up_programs[0] << up_origin_programs << I(q[0]);
	up_programs[1] << up_origin_programs << I(q[0]);

	//up_programs[2] << up_origin_programs << X(q[0]);
	//up_programs[3] << up_origin_programs << Y(q[0]);

	up_programs[2] << up_origin_programs << RY(q[0], PI / 2);
	up_programs[3] << up_origin_programs << RX(q[0], PI / 2);

	std::vector<QProg> down_programs(4);

	down_programs[0] << U4(q[0], matrix_p0) << down_origin_programs;
	down_programs[1] << U4(q[0], matrix_p1) << down_origin_programs;
	down_programs[2] << X(q[0]) << down_origin_programs;
	down_programs[3] << Y(q[0]) << down_origin_programs;

	std::vector<QStat> up_state;
	std::vector<QStat> down_state;
	for (auto i = 0; i < 4; ++i)
	{
		auto up_result = run_cir(up_programs[i]);
		auto down_result = run_cir(down_programs[i]);

		up_state.emplace_back(up_result);
		down_state.emplace_back(down_result);
	}

	QStat result(2, .0);

	result[0] += up_state[0][0] * down_state[0][0] * (1. / 2);
	result[0] += up_state[1][0] * down_state[1][0] * (1. / 2);
	result[0] += up_state[2][0] * down_state[2][0] * (1. / 2);
	result[0] += up_state[3][0] * down_state[3][0] * (1. / 2);

	result[1] += up_state[0][1] * down_state[0][1] * (1. / 2);
	result[1] += up_state[1][1] * down_state[1][1] * (1. / 2);
	result[1] += up_state[2][1] * down_state[2][1] * (1. / 2);
	result[1] += up_state[3][1] * down_state[3][1] * (1. / 2);

	auto origin_state = run_one_qubit_circuit();

	auto p0 = state_probs(result);
	auto p1 = state_probs(origin_state);

	std::cout << "origin result | sub_result : " << endl;
	for (auto i = 0; i < result.size(); ++i)
	{
		std::cout << origin_state[i] << " | " << result[i] << endl;
	}

	return;
}


void cut_one_qubit_circuit_2021()
{
	CPUQVM cpu;
	cpu.init();

	auto q = cpu.qAllocMany(1);
	auto c = cpu.cAllocMany(1);

	//origin circuit
	QProg prog;
	prog << H(q[0])
		<< RX(q[0], PI / 6)
		<< RX(q[0], PI / 6)
		<< RX(q[0], PI / 6)
		<< RZ(q[0], PI / 6);

	//P0,P1
	QStat matrix_p0 = { 1.,0.,0.,0. };
	QStat matrix_p1 = { 0.,0.,0.,1. };

	QProg up_origin_programs;
	up_origin_programs << H(q[0]) << RX(q[0], PI / 4) << RX(q[0], PI / 5);

	QProg down_origin_programs;
	down_origin_programs << RX(q[0], PI / 3) << RZ(q[0], PI / 4);

	std::vector<QProg> up_programs(4);

	up_programs[0] << up_origin_programs << I(q[0]);
	up_programs[1] << up_origin_programs << I(q[0]);

	up_programs[2] << up_origin_programs << X(q[0]);
	up_programs[3] << up_origin_programs << Y(q[0]);

	//up_programs[2] << up_origin_programs;// << RY(q[0], -PI / 2);
	//up_programs[3] << up_origin_programs;// << RX(q[0], -PI / 2);

	std::vector<QProg> down_programs(4);

	down_programs[0] << down_origin_programs;
	down_programs[1] << down_origin_programs;
	down_programs[2] << H(q[0]) << down_origin_programs;
	down_programs[3] << H(q[0]) << S(q[0]) << down_origin_programs;

	std::vector<QStat> up_state;
	std::vector<QStat> down_state;
	for (auto i = 0; i < 4; ++i)
	{
		auto up_result = run_cir(up_programs[i]);
		auto down_result = run_cir(down_programs[i]);

		up_state.emplace_back(up_result);
		down_state.emplace_back(down_result);
	}

	QStat result(2, .0);

	result[0] += up_state[0][0] * down_state[0][0] * (1. / 2);
	result[0] += up_state[1][0] * down_state[1][0] * (1. / 2);
	result[0] += up_state[2][0] * down_state[2][0] * (1. / 2);
	result[0] += up_state[3][0] * down_state[3][0] * (1. / 2);

	result[1] += up_state[0][1] * down_state[0][1] * (1. / 2);
	result[1] += up_state[1][1] * down_state[1][1] * (1. / 2);
	result[1] += up_state[2][1] * down_state[2][1] * (1. / 2);
	result[1] += up_state[3][1] * down_state[3][1] * (1. / 2);

	auto origin_state = run_one_qubit_circuit();

	auto p0 = state_probs(result);
	auto p1 = state_probs(origin_state);

	std::cout << "origin result | sub_result : " << endl;
	for (auto i = 0; i < result.size(); ++i)
	{
		std::cout << origin_state[i] << " | " << result[i] << endl;
	}

	return;
}



std::map<string, size_t> real_chip_result_split(Qnum merge_qubits, std::map<string, size_t> merge_result)
{
	QPANDA_ASSERT(merge_result.empty(), "merge_result is empty");

	int qubits_num = merge_result.begin()->first.size();

	std::map<string, size_t> result;
	for (auto val : merge_result)
	{
		std::string source_result;

		for (auto j = 0; j < merge_qubits.size(); ++j)
		{
			source_result.insert(source_result.begin(), val.first[qubits_num - 1 - j]);
		}

		result[source_result] += val.second;
	}

	return result;
}

QStat get_sub_set_result()
{
	MPSQVM mps;
	mps.setConfigure({ 64,64 });
	mps.init();

	auto q = mps.qAllocMany(3);
	auto c = mps.cAllocMany(3);

	QProg prog;
	prog << H(q[0])
		<< RX(q[0], PI / 3)
		<< RY(q[0], PI / 4)
		<< RZ(q[0], PI / 5)
		<< RX(q[1], PI / 3)
		<< RY(q[1], PI / 4)
		<< RZ(q[1], PI / 5)
		<< RX(q[2], PI / 3)
		<< RY(q[2], PI / 4)
		<< H(q[0])
		<< CNOT(q[0], q[2])
		<< CNOT(q[0], q[2])
		<< CR(q[0], q[1], PI / 3)
		<< CR(q[1], q[2], PI / 4)
		<< CR(q[0], q[2], PI / 6);

	//auto a = mps.pmeasure_bin_subset(prog, { "000000","000001" ,"000010" ,"000011" ,"000100" });
	//auto a = mps.pmeasure_bin_subset(prog, { "000","001" ,"010" ,"011" ,"100","101","110" ,"111" });
	auto a = mps.pmeasure_dec_subset(prog, { "0","1" ,"2" ,"3" ,"4","5","6" ,"7" });
	return a;
}

string test_jaon()
{
	string a = R"([{"key": ["0", "1"], "value": [0.5488410532814034, 0.45115894671859674]}])";



	std::map<string, double> merge_measure_result;
	merge_measure_result["000000"] = 0.21;
	merge_measure_result["000001"] = 0.31;


	rapidjson::Document result_doc;
	result_doc.SetObject();
	rapidjson::Document::AllocatorType& allocator = result_doc.GetAllocator();

	rapidjson::Value key_array(rapidjson::kArrayType);
	rapidjson::Value value_array(rapidjson::kArrayType);

	std::for_each(merge_measure_result.begin(), merge_measure_result.end(), [&](std::pair<std::string, double> val)
		{
			rapidjson::Value string_key(rapidjson::kStringType);
			string_key.SetString(val.first.c_str(), (rapidjson::SizeType)val.first.size(), allocator);

			key_array.PushBack(string_key, allocator);
			value_array.PushBack(val.second, allocator);
		});

	result_doc.AddMember("key", key_array, allocator);
	result_doc.AddMember("value", value_array, allocator);

	rapidjson::StringBuffer buffer;
	rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
	result_doc.Accept(writer);

	std::string result_json = buffer.GetString();

	rapidjson::Document result_array;
	result_array.SetArray();
	rapidjson::Document::AllocatorType& array_allocator = result_array.GetAllocator();

	rapidjson::Value final_string(rapidjson::kStringType);
	final_string.SetString(result_json.c_str(), (rapidjson::SizeType)result_json.size(), array_allocator);

	result_array.PushBack(final_string, array_allocator);

	rapidjson::StringBuffer final_buffer;
	rapidjson::Writer<rapidjson::StringBuffer> final_writer(final_buffer);
	result_array.Accept(final_writer);

	std::string final_json = final_buffer.GetString();

	return final_json;
}



using namespace Base64;
using namespace rapidjson;

std::string build_quantum_chip_config_data(std::string adj_json)
{
	//Qubit Count
	Document parse_doc;
	parse_doc.Parse(adj_json.c_str());
	auto qubit_count = parse_doc["adj"].Size();

	rapidjson::Document arch_doc;
	arch_doc.SetObject();

	rapidjson::Document::AllocatorType& arch_alloc = arch_doc.GetAllocator();

	arch_doc.AddMember("QubitCount", qubit_count, arch_alloc);
	arch_doc.AddMember("adj", parse_doc["adj"], arch_alloc);

	rapidjson::StringBuffer arch_buffer;
	rapidjson::Writer<rapidjson::StringBuffer> arch_writer(arch_buffer);
	arch_doc.Accept(arch_writer);

	//construct json
	rapidjson::Document doc;
	doc.SetObject();

	rapidjson::Document::AllocatorType& allocator = doc.GetAllocator();

	doc.AddMember("QuantumChipArch", arch_doc, arch_alloc);

	rapidjson::StringBuffer buffer;
	rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
	doc.Accept(writer);

	return buffer.GetString();
}

#if 0

int test()
{
	auto qubits_num = 6;
	auto originir_string = R"(QINIT 6 CREG 6 H q[0] H q[1] H q[2] H q[3] H q[4] H q[5] CR q[1],q[0],(1.570796) CR q[2],q[1],(1.570796) CR q[3],q[2],(1.570796) CR q[4],q[3],(1.570796) CR q[5],q[4],(1.570796) CZ q[0],q[1] CZ q[0],q[2] CZ q[0],q[3] CZ q[0],q[4] CZ q[0],q[5] CZ q[1],q[0] CR q[4],q[5],(1.570796) CR q[3],q[4],(1.570796) CR q[2],q[3],(1.570796) CR q[1],q[2],(1.570796) CR q[0],q[1],(1.570796) MEASURE q[0],c[0] MEASURE q[1],c[1] MEASURE q[2],c[2] MEASURE q[3],c[3] MEASURE q[4],c[4] MEASURE q[5],c[5])";
	std::string base64_adj = R"(ew0KICAgICJhZGoiOiBbDQogICAgICAgIFt7ICJ2IjogMSwgInciOiAwIH1dLA0KICAgICAgICBbeyAidiI6IDAsICJ3IjogMCB9LHsgInYiOiAyLCAidyI6IDAuOTkwOSB9XSwNCiAgICAgICAgW3sidiI6IDEsICJ3IjogMC45OTA5fSwgeyAidiI6IDMsICJ3IjogMC45NzA3IH1dLA0KICAgICAgICBbeyJ2IjogMiwgInciOiAwLjk3MDd9LCB7ICJ2IjogNCwgInciOiAwLjk5MDkgfV0sDQogICAgICAgIFt7InYiOiAzLCAidyI6IDAuOTkwOX0sIHsgInYiOiA1LCAidyI6IDAuOTkwOX1dLA0KICAgICAgICBbeyJ2IjogNCwgInciOiAwLjk5MDl9XQ0KICAgICAgXQ0KfQ==)";

	auto adj_code = decode((void*)base64_adj.c_str(), base64_adj.size());
	auto adj = string(adj_code.begin(), adj_code.end());

	std::cout << adj << std::endl;

	try
	{
		auto qvm = CPUQVM();

		Configuration config = { 256,256 };
		qvm.setConfig(config);
		qvm.init();

		QVec qubits;
		vector<ClassicalCondition> cbits;

		auto prog = convert_originir_string_to_qprog(originir_string, &qvm, qubits, cbits);
		std::cout << prog << std::endl;

		auto quantum_chip_arch_config_data = build_quantum_chip_config_data(adj);
		std::cout << quantum_chip_arch_config_data << std::endl;


		QVec mapping_qubits;
		auto bmt_mapped_prog = OBMT_mapping(prog, &qvm, mapping_qubits, 200, 1024, quantum_chip_arch_config_data);

		QPANDA_RETURN(qubits_num < qvm.getAllocateQubitNum(), -1);

		std::cout << "before mapping ----->" << std::endl;
		std::cout << prog << std::endl;

		std::cout << " after mapping ----->" << std::endl;
		std::cout << bmt_mapped_prog << std::endl;

	}
	catch (...)
	{
		return -1;
	}

	return 0;
}

#endif

TEST(QVM, MPSQVM)
{
	//test_real_chip_split();
	cut_one_qubit_circuit();

	return;
	MPSQVM mps;
	mps.setConfigure({ 64,64 });
	mps.init();

	auto q = mps.qAllocMany(3);
	auto c = mps.cAllocMany(3);

	QProg prog;
	prog << H(q[0])
		<< RX(q[0], PI / 3)
		<< RY(q[0], PI / 4)
		<< RZ(q[0], PI / 5)
		<< RX(q[1], PI / 3)
		<< RY(q[1], PI / 4)
		<< RZ(q[1], PI / 5)
		<< RX(q[2], PI / 3)
		<< RY(q[2], PI / 4)
		<< H(q[0])
		<< CNOT(q[0], q[2])
		<< CNOT(q[0], q[2])
		<< CR(q[0], q[1], PI / 3)
		<< CR(q[1], q[2], PI / 4)
		<< CR(q[0], q[2], PI / 6);

	auto q0 = { q[0] };
	auto q1 = { q[1] };
	std::vector<QVec> qs = { { q[0],q[1] } };

	//mps.set_noise_model(NOISE_MODEL::BITFLIP_KRAUS_OPERATOR, CNOT_GATE, 0.5, qs);
	//mps.set_noise_model(NOISE_MODEL::BITFLIP_KRAUS_OPERATOR, PAULI_X_GATE, 0.9999, q0);

	//mps.set_measure_error(NOISE_MODEL::BITFLIP_KRAUS_OPERATOR, 0.9999, q1);
	//mps.set_measure_error(NOISE_MODEL::BITFLIP_KRAUS_OPERATOR, 0.0001, q0);
	QStat id = { 1,0,0,0,
				0,1,0,0,
				0,0,1,0,
				0,0,0,1 };

	QStat _CNOT = { 1,0,0,0,
			0,1,0,0,
			0,0,0,1,
			0,0,1,0 };

	//mps.set_mixed_unitary_error(GateType::CNOT_GATE, { _CNOT,id }, { 0.5, 0.5 });

	//auto a = mps.runWithConfiguration(prog, c, 1000);
	mps.directlyRun(prog);
	auto a = mps.getQState();
	auto b = get_sub_set_result();

	for (auto i = 0; i < a.size(); ++i)
	{
		cout << a[i] << ": " << b[i] << endl;
		ASSERT_EQ(b[i], 0.21875);
	}
	
	mps.finalize();
	//getchar();
}

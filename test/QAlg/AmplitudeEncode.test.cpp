#include "gtest/gtest.h"
#include "QPanda.h"

USING_QPANDA
using namespace std;

static bool test_angle_encoding()
{
	auto qvm = new CPUQVM();
	qvm->init();
	//std::vector<double>data{ 0.4737709042140123,0.5287790950369405,0.2157088373614705,0.023769834145903144,0.4019987624599187,0.057654140832877024,0.5150212867886899,0.1366183026575457 };
	std::vector<double>data{PI,PI,PI};
	QProg prog;
	auto q = qvm->qAllocMany((int)data.size());
	Encode encode_b;
	encode_b.angle_encode(q,data);
	prog << encode_b.get_circuit();
	std:; cout << prog << std::endl;
	auto result = qvm->probRunDict(prog, q, -1);
	for (auto &val:result) {
		std::cout << val.first << ":" <<val.second << std::endl;
	}
	destroyQuantumMachine(qvm);
	return true;
}
static bool test_basic_encoding()
{
	auto qvm = new CPUQVM();
	qvm->init();
	//std::vector<double>data{ 0.4737709042140123,0.5287790950369405,0.2157088373614705,0.023769834145903144,0.4019987624599187,0.057654140832877024,0.5150212867886899,0.1366183026575457 };
	//std::vector<double>data{ 1,0,1,1 };
	string data = "1011";
	QProg prog;
	auto q = qvm->qAllocMany((int)data.size());
	Encode encode_b;
	encode_b.basic_encode(q, data);
	prog << encode_b.get_circuit();
	qvm->directlyRun(prog);
	auto result = qvm->probRunDict(prog,encode_b.get_out_qubits(),-1);
	for (auto val : result) {
		std::cout << val.first <<','<< val.second<< std::endl;
	}
	destroyQuantumMachine(qvm);
	return true;
}
static bool test_dense_encoding()
{
	auto qvm = new CPUQVM();
	qvm->init();
	//std::vector<double>data{ 0.4737709042140123,0.5287790950369405,0.2157088373614705,0.023769834145903144,0.4019987624599187,0.057654140832877024,0.5150212867886899,0.1366183026575457 };
	std::vector<double>data{ PI,PI,PI};
	QProg prog;
	auto q = qvm->qAllocMany(15);
	Encode encode_b;
	encode_b.dense_angle_encode(q, data);
	prog << encode_b.get_circuit();
	qvm->directlyRun(prog);
    std:: cout << prog << std::endl;
	auto result = qvm->probRunDict(prog,encode_b.get_out_qubits(),-1);
	for (auto &val : result)
	{
	    std:: cout << val.first << ':'<<val.second << std::endl;
	}
	destroyQuantumMachine(qvm);
	return true;
}
static bool test_ds_quantum_state_preparation_encoding()
{
	auto qvm = new CPUQVM();
	qvm->init();
	//std::vector<double>data{ -0.33164128327780906,-0.23433373435840027,-0.00022877626650846954,0.3466895582845477,0.36968371696590785,-0.21966111818142703,-0.28122454968118343,0.2954805505078439,0.2060812114310625,0.28816304973225565,0.13470785340813266,0.06207383012753573,-0.14157790059772607,0.15044744783305566,-0.28050173370498876,-0.3077795693383025 };
	//std::vector<double>data{ 0.15311858100051695,-0.0961350374871273,0.3859320687001368,-0.5634457467385428,0.1474901012487757,-0.45185782723129864,0.32284355187278985,-0.4132085412578166 };
	QProg prog;
	auto q = qvm->qAllocMany(15);
	Encode encode_b;
	std::map<std::string, double>mp;
	std::map<std::string, std::complex<double>>mp1;
	mp["000"] = -0.4012058758884066;
	mp["001"] = 0.9121413556170931;
	mp["111"] = 0.08385697660676902;
	mp1["010"].real(0.6793113376921358);
	mp1["010"].imag(0.1376859100584252);
	mp1["111"].real(0.720435424880283);
	mp1["111"].imag(0.02348393561289133);
	encode_b.ds_quantum_state_preparation(q,mp);
	prog << encode_b.get_circuit() << BARRIER(q);
	QVec out_qubits = encode_b.get_out_qubits();
	std::cout << prog << std::endl;
	auto result = qvm->probRunDict(prog, out_qubits, -1);
	auto normalization_constant = encode_b.get_normalization_constant();
	for (auto &val : result)
	{
		std::cout << val.first << ":" << val.second*normalization_constant*normalization_constant << std::endl;
	}
	destroyQuantumMachine(qvm);
	return true;
}
static bool test_sparse_isometry_encoding()
{
	auto qvm = new CPUQVM();
	qvm->init();
	//std::vector<double>data{ -0.33164128327780906,-0.23433373435840027,-0.00022877626650846954,0.3466895582845477,0.36968371696590785,-0.21966111818142703,-0.28122454968118343,0.2954805505078439,0.2060812114310625,0.28816304973225565,0.13470785340813266,0.06207383012753573,-0.14157790059772607,0.15044744783305566,-0.28050173370498876,-0.3077795693383025 };
	//std::vector<double>data{ 0.15311858100051695,-0.0961350374871273,0.3859320687001368,-0.5634457467385428,0.1474901012487757,-0.45185782723129864,0.32284355187278985,-0.4132085412578166 };
	std::vector<complex<double>>data{ qcomplex_t(-0.13792979841421985, 0.14135138210029866),qcomplex_t(-0.1390065521329917, 0.027905778367451294), qcomplex_t(0.1408303661581104, 0.1904935417178447), qcomplex_t(-0.413408661519566, 0.39732552760462225), qcomplex_t(-0.451018978400839, 0.05084981631268899), qcomplex_t(-0.28129960014056354, 0.28003182131256205), qcomplex_t(-0.15455196636034582, 0.0831192522241199), qcomplex_t(-0.18327917974198926, 0.35785589125336675) };
	QProg prog;
	auto q = qvm->qAllocMany(3);
	Encode encode_b;
	std::map<std::string, double>mp;
	std::map<std::string, complex<double>>mp1;
	mp["000"] = -0.4012058758884066;
	mp["001"] = 0.9121413556170931;
	mp["111"] = 0.08385697660676902;
	mp1["010"].real(0.6793113376921358);
	mp1["010"].imag(0.1376859100584252);
	mp1["111"].real(0.720435424880283);
	mp1["111"].imag(0.02348393561289133);
	//mp["000"] = 0.7071067811865476;
	//mp["111"] = 0.7071067811865476;
	//mp["100"] = 1;
	encode_b.sparse_isometry(q, mp);
	prog << encode_b.get_circuit() << BARRIER(q);
	QVec out_qubits = encode_b.get_out_qubits();
	std::cout << prog << std::endl;
	qvm->directlyRun(prog);
	auto result = qvm->getQState();
	for (auto &val : result) {
		std::cout << "Amplitude" << ":" << val << std::endl;
	}
	//std::cout << result << std::endl;
	/*auto result = qvm->probRunDict(prog, out_qubits, -1);
	double normalization_constant = encode_b.get_normalization_constant();
	for (auto &val : result)
	{
		std::cout << val.first << ":" << val.second*normalization_constant*normalization_constant << std::endl;
	}*/
	destroyQuantumMachine(qvm);
	return true;
}
static bool test_dc_Amplitude_encoding()
{
	auto qvm = new CPUQVM();
	qvm->init();
	//std::vector<double>data{ -0.33164128327780906,-0.23433373435840027,-0.00022877626650846954,0.3466895582845477,0.36968371696590785,-0.21966111818142703,-0.28122454968118343,0.2954805505078439,0.2060812114310625,0.28816304973225565,0.13470785340813266,0.06207383012753573,-0.14157790059772607,0.15044744783305566,-0.28050173370498876,-0.3077795693383025 };
	std::vector<double>data{ 0.15311858100051695,-0.0961350374871273,0.3859320687001368,-0.5634457467385428,0.1474901012487757,-0.45185782723129864,0.32284355187278985,-0.4132085412578166 };
	QProg prog;
	auto q = qvm->qAllocMany(15);
	Encode encode_b;
	encode_b.dc_amplitude_encode(q, data);
	prog << encode_b.get_circuit() << BARRIER(q);
	QVec out_qubits = encode_b.get_out_qubits();
	auto result = qvm->probRunDict(prog, out_qubits, -1);
	int k = 0;
	auto normalization_constant = encode_b.get_normalization_constant();
	for (auto &val : result)
	{
		double temp = k >= (int)data.size() ? 0 : data[k];
		std::cout << val.second << endl;
		std::cout << val.first << ":" <<  val.second*normalization_constant*normalization_constant << std::endl;
		++k;
	}
	destroyQuantumMachine(qvm);
	return true;
}
static bool test_schmidt_encoding()
{
	auto qvm = new CPUQVM();
	qvm->init();
	std::vector<double>data{ -0.33164128327780906,-0.23433373435840027,-0.00022877626650846954,0.3466895582845477,0.36968371696590785,-0.21966111818142703,-0.28122454968118343,0.2954805505078439,0.2060812114310625,0.28816304973225565,0.13470785340813266,0.06207383012753573,-0.14157790059772607,0.15044744783305566,-0.28050173370498876,-0.3077795693383025 };
	//std::vector<double>data{ 0.15311858100051695,-0.0961350374871273,0.3859320687001368,-0.5634457467385428,0.1474901012487757,-0.45185782723129864,0.32284355187278985,-0.4132085412578166 };
	//std::vector<double>data{ 0.6793113376921358, 0.1376859100584252, 0.720435424880283, 0.02348393561289133};
	
	QProg prog;
	auto q = qvm->qAllocMany(5);
	//auto circuit = QCircuit();
	//circuit << H(q[0]);
	//QStat q_mat(16);
	//q_mat[0] = (qcomplex_t)1.0;
	//q_mat[5]= (qcomplex_t)1.0;
	//q_mat[11] = (qcomplex_t)1.0;
	//q_mat[14] = (qcomplex_t)1.0;
	//circuit << QOracle(q, q_mat);
	Encode encode_b;
	encode_b.schmidt_encode(q, data);
	prog << encode_b.get_circuit() << BARRIER(q);
	std::cout << prog << std::endl;
	std::cout<< convert_qprog_to_originir(prog, qvm)<<std::endl;
	QVec out_qubits = encode_b.get_out_qubits();
	auto result = qvm->probRunDict(prog, q, -1);
	int k = 0;
	/*double normalization_constant = encode_b.get_normalization_constant();*/
	for (auto &val : result)
	{
		double temp = k >= (int)data.size() ? 0 : data[k];
		std::cout << "Amplitude:" << val.first << ":" << val.second << "," << "Originial value:" << temp * temp << std::endl;
		++k;
	}
	destroyQuantumMachine(qvm);
	return true;
}
static bool test_bid_Amplitude_encoding()
{
	auto qvm = new CPUQVM();
	qvm->init();
	std::vector<double>data{ -0.33164128327780906,-0.23433373435840027,-0.00022877626650846954,0.3466895582845477,0.36968371696590785,-0.21966111818142703,-0.28122454968118343,0.2954805505078439,0.2060812114310625,0.28816304973225565,0.13470785340813266,0.06207383012753573,-0.14157790059772607,0.15044744783305566,-0.28050173370498876,-0.3077795693383025 };
	//std::vector<double>data{ 0.15311858100051695,-0.0961350374871273,0.3859320687001368,-0.5634457467385428,0.1474901012487757,-0.45185782723129864,0.32284355187278985,-0.4132085412578166 };
	//std::vector<double>data{1.5,1.3,1.2,1.11,1.6,1.9,1.7};
	QProg prog;
	auto q = qvm->qAllocMany(15);
	Encode encode_b;
	encode_b.bid_amplitude_encode(q, data);
	prog << encode_b.get_circuit() << BARRIER(q);
	std::cout << prog << std::endl;
	QVec out_qubits = encode_b.get_out_qubits();
	auto result = qvm->probRunDict(prog, out_qubits, -1);
	int k = 0;
	auto normalization_constant = encode_b.get_normalization_constant();
	for (auto &val : result)
	{
		double temp=k >= (int)data.size() ? 0 : data[k];
		std::cout << val.first << ":" << val.second*normalization_constant*normalization_constant << std::endl;
		++k;
	}
	destroyQuantumMachine(qvm);
	return true;
}

static bool test_iqp_Amplitude_encoding()
{
	auto qvm = new CPUQVM();
	qvm->init();
	//std::vector<double>data{ -0.33164128327780906,-0.23433373435840027,-0.00022877626650846954,0.3466895582845477,0.36968371696590785,-0.21966111818142703,-0.28122454968118343,0.2954805505078439,0.2060812114310625,0.28816304973225565,0.13470785340813266,0.06207383012753573,-0.14157790059772607,0.15044744783305566,-0.28050173370498876,-0.3077795693383025 };
	//std::vector<double>data{ 0.15311858100051695,-0.0961350374871273,0.3859320687001368,-0.5634457467385428,0.1474901012487757,-0.45185782723129864,0.32284355187278985,-0.4132085412578166 };
	std::vector<double>data{ -1.45, 3, 2, -0.05 };
	QProg prog;
	auto q = qvm->qAllocMany(data.size());
	Encode encode_b;
	encode_b.iqp_encode(q, data);
	prog << encode_b.get_circuit() << BARRIER(q);
	std::cout << prog << std::endl;
	QVec out_qubits = encode_b.get_out_qubits();
	auto result = qvm->probRunDict(prog, out_qubits, -1);
	//qvm->directlyRun(prog);
	//auto result = qvm->getQState();
	int k = 0;
	//double normalization_constant = encode_b.get_normalization_constant();
	for (auto &val : result)
	{
		double temp = k >= (int)data.size() ? 0 : data[k];
		std::cout <<val.first<<':'<< val.second << std::endl;
		++k;
	}
	destroyQuantumMachine(qvm);
	return true;
}

static bool test_Amplitude_encoding1()
{
	auto qvm = new CPUQVM();
	qvm->init();
	//std::vector<double>data{ -0.33164128327780906,-0.23433373435840027,-0.00022877626650846954,0.3466895582845477,0.36968371696590785,-0.21966111818142703,-0.28122454968118343,0.2954805505078439,0.2060812114310625,0.28816304973225565,0.13470785340813266,0.06207383012753573,-0.14157790059772607,0.15044744783305566,-0.28050173370498876,-0.3077795693383025 };
	//std::vector<double>data{ 0.15311858100051695,-0.0961350374871273,0.3859320687001368,-0.5634457467385428,0.1474901012487757,-0.45185782723129864,0.32284355187278985,-0.4132085412578166 };
	//std::vector<complex<double>>data{qcomplex_t(0.33510517, 0.29640703), qcomplex_t(-0.11504671, -0.05932087), qcomplex_t(-0.11146904, 0.11053713), qcomplex_t(0.18010564, 0.15731823), qcomplex_t(-0.3284375, 0.40864014), qcomplex_t(-0.19391152, 0.2075545), qcomplex_t(-0.30391936, 0.36735511), qcomplex_t(0.32367291, 0.11680073)};
	//std::vector<complex<double>>data{ qcomplex_t(-0.13792979841421985, 0.14135138210029866),qcomplex_t(-0.1390065521329917, 0.027905778367451294), qcomplex_t(0.1408303661581104, 0.1904935417178447), qcomplex_t(-0.413408661519566, 0.39732552760462225), qcomplex_t(-0.451018978400839, 0.05084981631268899), qcomplex_t(-0.28129960014056354, 0.28003182131256205), qcomplex_t(-0.15455196636034582, 0.0831192522241199), qcomplex_t(-0.18327917974198926, 0.35785589125336675) };
	QStat test_qstat{ qcomplex_t(0.406270160898181, 0.0749820103002614),
	qcomplex_t(0.195605655043143, 0.486587726445213),
	qcomplex_t(0.241952125812726, 0.455619712343811),
	qcomplex_t(0.31466450344542, 0.434912822865699) };
	std::vector<double>data{ -11,-22,-3,4,5,6 };
	auto prog = QProg();
	auto q = qvm->qAllocMany(4);
	Encode encode_b;
	encode_b.amplitude_encode(q, data);
	prog << encode_b.get_circuit();
	std::cout << prog << std::endl;
	QVec out_qubits = encode_b.get_out_qubits();
	qvm->directlyRun(prog);
	auto result = qvm->getQState();
	//auto result = qvm->probRunDict(prog, q, -1);
	int k = 0;
	auto normalization_constant = encode_b.get_normalization_constant();
	for (auto &val : result) {
		std::cout << "Amplitude:" <<val*normalization_constant << std::endl;
	}
	/*for (auto &val : result)
	{
		double temp = k >= (int)data.size() ? 0 : data[k];
		std::cout << "Amplitude:" << val.first << ":" << val.second*normalization_constant*normalization_constant << "," << "Originial value:" << temp * temp << std::endl;
		++k;
	}*/
	destroyQuantumMachine(qvm);
	return true;
}
static bool test_Amplitude_encoding2()
{
	auto qvm = new CPUQVM();
	qvm->init();
	std::vector<double>data{ -0.33164128327780906,-0.23433373435840027,-0.00022877626650846954,0.3466895582845477,0.36968371696590785,-0.21966111818142703,-0.28122454968118343,0.2954805505078439,0.2060812114310625,0.28816304973225565,0.13470785340813266,0.06207383012753573,-0.14157790059772607,0.15044744783305566,-0.28050173370498876,-0.3077795693383025 };
	//std::vector<double>data{ 0.15311858100051695,-0.0961350374871273,0.3859320687001368,-0.5634457467385428,0.1474901012487757,-0.45185782723129864,0.32284355187278985,-0.4132085412578166 };
	//std::vector<double>data{ 1,2,3,4,5,6 };
	auto prog = QProg();
	auto q = qvm->qAllocMany(4);
	auto cir = amplitude_encode(q, data);
	std::cout << cir << std::endl;
	prog << cir;
	auto result = qvm->probRunList(prog, q, -1);
	int k = 0;
	for (auto &val : result)
	{
		std::cout << "Amplitude:" << val << "," << "Originial value:" << data[k] * data[k] << std::endl;
		++k;
	}
	destroyQuantumMachine(qvm);
	return true;
}
static bool test_Amplitude_encoding3()
{
	auto qvm = initQuantumMachine(QMachineType::CPU);
	auto q = qvm->allocateQubits(2);
	auto c = qvm->allocateCBits(2);


	QStat test_qstat{ qcomplex_t(0.406270160898181, 0.0749820103002614),
		qcomplex_t(0.195605655043143, 0.486587726445213),
		qcomplex_t(0.241952125812726, 0.455619712343811),
		qcomplex_t(0.31466450344542, 0.434912822865699) };

	QCircuit test_cir = amplitude_encode(q, test_qstat);

	auto prog = QProg();
	prog << test_cir;
	qvm->directlyRun(prog);
	auto stat = qvm->getQState();
	std::cout << "target_stat:\n" << stat << std::endl;

	destroyQuantumMachine(qvm);

	if (0 == mat_compare(stat, test_qstat, MAX_COMPARE_PRECISION)) {
		return true;
	}

	return false;
}
static bool test_Amplitude_encoding4()
{
	auto qvm = initQuantumMachine(QMachineType::CPU);
	auto q = qvm->allocateQubits(2);
	auto c = qvm->allocateCBits(2);


	QStat test_qstat{ qcomplex_t(0.406270160898181, 0.0749820103002614),
		qcomplex_t(0.195605655043143, 0.486587726445213),
		qcomplex_t(0.241952125812726, 0.455619712343811),
		qcomplex_t(0.31466450344542, 0.434912822865699) };
	vector<double>data{ 1,2,3,4,5,6 };
	auto prog = QProg();
	Encode encode_b;
	//encode_b.amplitude_encode_recursive(q, test_qstat);
	encode_b.amplitude_encode_recursive(q, test_qstat);
	prog << encode_b.get_circuit();
	auto k = encode_b.get_normalization_constant();
	std::cout << prog << std::endl;
	//QCircuit test_cir = amplitude_encode(q, test_qstat);

	//auto prog = QProg();
	//prog << test_cir;
	qvm->directlyRun(prog);
	auto stat = qvm->getQState();
	for (auto &i : stat) 
	{
		i = i * k;
	}
	std::cout << "target_stat:\n" << stat << std::endl;

	destroyQuantumMachine(qvm);

	if (0 == mat_compare(stat, test_qstat, MAX_COMPARE_PRECISION)) {
		return true;
	}

	return false;
}
TEST(AmplitudeEncode, test1)
{
	bool test_val = false;
	try
	{
		test_val = test_angle_encoding();
		test_val&= test_Amplitude_encoding1();
		test_val&= test_Amplitude_encoding2();
		test_val&= test_Amplitude_encoding3();
		test_val&= test_Amplitude_encoding4();
		test_val&= test_dc_Amplitude_encoding();
		test_val&= test_bid_Amplitude_encoding();
		test_val&= test_dense_encoding();
		test_val&= test_basic_encoding();
		test_val&= test_iqp_Amplitude_encoding();
		test_val&= test_ds_quantum_state_preparation_encoding();
		test_val&= test_sparse_isometry_encoding();
		test_val&= test_schmidt_encoding();
	}
	catch (const std::exception& e)
	{
		cout << "Got a exception: " << e.what() << endl;
	}
	catch (...)
	{
		cout << "Got an unknow exception: " << endl;
	}
	ASSERT_TRUE(test_val);
}
#include "gtest/gtest.h"
#include "QPanda.h"
#include "Core/Utilities/Encode/Encode.h"
USING_QPANDA
using namespace std;

static bool test_angle_encoding()
{
	auto qvm = new CPUQVM();
	qvm->init();
	//std::vector<double>data{ 0.4737709042140123,0.5287790950369405,0.2157088373614705,0.023769834145903144,0.4019987624599187,0.057654140832877024,0.5150212867886899,0.1366183026575457 };
	std::vector<double>data{ PI,PI,PI };
	QProg prog;
	auto q = qvm->qAllocMany((int)data.size());
	Encode encode_b;
	encode_b.angle_encode(q, data);
	prog << encode_b.get_circuit();
std:; cout << prog << std::endl;
	auto result = qvm->probRunDict(prog, q, -1);
	for (auto &val : result) {
		std::cout << val.first << ":" << val.second << std::endl;
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
	auto result = qvm->probRunDict(prog, encode_b.get_out_qubits(), -1);
	for (auto val : result) {
		std::cout << val.first << ',' << val.second << std::endl;
	}
	destroyQuantumMachine(qvm);
	return true;
}
static bool test_dense_encoding()
{
	auto qvm = new CPUQVM();
	qvm->init();
	//std::vector<double>data{ 0.4737709042140123,0.5287790950369405,0.2157088373614705,0.023769834145903144,0.4019987624599187,0.057654140832877024,0.5150212867886899,0.1366183026575457 };
	std::vector<double>data{ PI,PI,PI };
	QProg prog;
	auto q = qvm->qAllocMany(15);
	Encode encode_b;
	encode_b.dense_angle_encode(q, data);
	prog << encode_b.get_circuit();
	qvm->directlyRun(prog);
	std::cout << prog << std::endl;
	auto result = qvm->probRunDict(prog, encode_b.get_out_qubits(), -1);
	for (auto &val : result)
	{
		std::cout << val.first << ':' << val.second << std::endl;
	}
	destroyQuantumMachine(qvm);
	return true;
}
static bool test_ds_quantum_state_preparation_encoding()
{
	auto qvm = new CPUQVM();
	qvm->init();
	//std::vector<double>data{ -0.33164128327780906,-0.23433373435840027,-0.00022877626650846954,0.3466895582845477,0.36968371696590785,-0.21966111818142703,-0.28122454968118343,0.2954805505078439,0.2060812114310625,0.28816304973225565,0.13470785340813266,0.06207383012753573,-0.14157790059772607,0.15044744783305566,-0.28050173370498876,-0.3077795693383025 };
	std::vector<double>data{ 0.15311858100051695,-0.0961350374871273,0.3859320687001368,-0.5634457467385428,0.1474901012487757,-0.45185782723129864,0.32284355187278985,-0.4132085412578166 };
	QProg prog;
	auto q = qvm->qAllocMany(15);
	Encode encode_b;
	std::map<std::string, double>mp;
	std::map<std::string, std::complex<double>>mp1;
	mp["000"] = -0.4012058758884066;
	mp["001"] = 0.9121413556170931;
	mp["111"] = 0.08385697660676902;
	mp1["000"].real(0.37126468505062743);
	mp1["000"].imag(0.44434564717341857);
	mp1["001"].real(0.19987861562923684);
	mp1["001"].imag(0.3401208369570892);
	mp1["010"].real(0.5299242678696047);
	mp1["010"].imag(0.25498511099003784);
	mp1["100"].real(0.19536627105888826);
	mp1["100"].imag(0.3536675252024842);
	encode_b.ds_quantum_state_preparation(q, data);
	prog << encode_b.get_circuit() << BARRIER(q);
	QVec out_qubits = encode_b.get_out_qubits();
	std::cout << prog << std::endl;
	auto result = qvm->probRunDict(prog, out_qubits, -1);
	//double normalization_constant = encode_b.get_normalization_constant();
	for (auto &val : result)
	{
		std::cout << val.first << ":" << val.second << std::endl;
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
	mp1["000"].real(0.11340744356593455);
	mp1["000"].imag(0.060868874261929065);
	mp1["100"].real(0.20777139811704742);
	mp1["100"].imag(0.3642765242773252);
	mp1["011"].real(0.0018744875304074278);
	mp1["011"].imag(0.5721780883375533);
	mp1["010"].real(0.44608846562195503);
	mp1["010"].imag(0.5302652112262077);
	//mp["000"] = 0.7071067811865476;
	//mp["111"] = 0.7071067811865476;
	//mp["100"] = 1;
	encode_b.sparse_isometry(q, mp1);
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
static bool test_efficient_sparse_encoding()
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
	//std::vector<double>data{0, 0, 0, -0.5, 0.5, -0.5, 0.5, 0};
	mp["000"] = -0.4012058758884066;
	mp["001"] = 0.9121413556170931;
	mp["111"] = 0.08385697660676902;

	mp1["00000000"].real(0.24310913564079167);
	mp1["00000000"].imag(0.01159058947905379);
	mp1["00000010"].real(0.31904155866953476);
	mp1["00000010"].imag(0.09143678216616687);
	mp1["00100000"].real(0.0023527412511721974);
	mp1["00100000"].imag(0.06406955746132303);
	mp1["00000001"].real(0.03157574906272091);
	mp1["00000001"].imag(0.10105536367098665);
	mp1["10000000"].real(0.047173075703916816);
	mp1["10000000"].imag(0.16531181355873065);
	mp1["00010000"].real(0.2450898977164231);
	mp1["00010000"].imag(0.038670570507224736);
	mp1["00001100"].real(0.1109015505954131);
	mp1["00001100"].imag(0.07401422543633625);
	mp1["01100000"].real(0.013151642285606184);
	mp1["01100000"].imag(0.17502338909204943);

	mp1["00000011"].real(0.30331286890722187);
	mp1["00000011"].imag(0.05526812310938826);
	mp1["00001010"].real(0.3099021978658747);
	mp1["00001010"].imag(0.2289686686845677);
	mp1["00001011"].real(0.17504000170596018);
	mp1["00001011"].imag(0.12856765550595634);
	mp1["00110001"].real(0.15693304322067672);
	mp1["00110001"].imag(0.1910842222007265);
	mp1["01100001"].real(0.1234790138508381);
	mp1["01100001"].imag(0.29627849886345053);
	mp1["10010100"].real(0.09158112115755046);
	mp1["10010100"].imag(0.12798738473764856);
	mp1["01011100"].real(0.0183266884240271);
	mp1["01011100"].imag(0.31960954765528116);
	mp1["01111000"].real(0.2848150218881516);
	mp1["01111000"].imag(0.11119291552105852);
	//mp["000"] = 0.7071067811865476;
	//mp["111"] = 0.7071067811865476;
	//mp["100"] = 1;
	encode_b.efficient_sparse(q, data);
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
	//auto normalization_constant = encode_b.get_normalization_constant();
	for (auto &val : result)
	{
		double temp = k >= (int)data.size() ? 0 : data[k];
		std::cout << val.second << endl;
		std::cout << val.first << ":" << val.second << std::endl;
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
	auto q = qvm->qAllocMany(4);
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
	//std::cout<< convert_qprog_to_originir(prog, qvm)<<std::endl;
	QVec out_qubits = encode_b.get_out_qubits();
	//auto result = qvm->probRunDict(prog, q, -1);
	qvm->directlyRun(prog);
	auto result = qvm->getQState();
	int k = 0;
	/*double normalization_constant = encode_b.get_normalization_constant();*/
	for (auto &val : result)
	{
		//double temp = k >= (int)data.size() ? 0 : data[k];
		std::cout << "Amplitude:" << val << "," << "Originial value:" << data[k] << std::endl;
		++k;
	}
	destroyQuantumMachine(qvm);
	return true;
}
static bool test_mps_encoding()
{
	auto qvm = new CPUQVM();
	qvm->init();
	auto q = qvm->qAllocMany(2);
	//QVec q1, q2, q3;
	//for (int i = 0; i < 2; ++i) {
	//	q1.push_back(q[i]);
	//}
	//for (int i = 2; i < 4; ++i) {
	//	q2.push_back(q[i]);
	//}
	//for (int i = 4; i < 7; ++i) {
	//	q3.push_back(q[i]);
	//}
	std::vector<qcomplex_t>p = { qcomplex_t(0.4330127,0),qcomplex_t(0,0.5), qcomplex_t(0,0.55901699),qcomplex_t(0,0.5) };

	Encode encode_b = Encode();
	encode_b.approx_mps_encode(q, p, 1, 0);
	//int dim = 4;
	//Eigen::VectorXcd v2 = Eigen::VectorXcd::Random(1 << dim);
	//v2.normalize();
	//vector<std::complex<double>> data(v2.data(), v2.data() + v2.size());
	////double tmp = 0.0;
	////for (int i = 0; i < data.size(); ++i) {
	////	tmp += data[i].real() * data[i].real()+ data[i].imag() * data[i].imag();
	////}
	////std::cout << tmp << std::endl;
	////QStat data{ qcomplex_t(0.016149328485063213,0.28882784478720575), qcomplex_t(0.06798688402448427,0.08462169020140721), qcomplex_t(0.3604460814011521,0.3721195422668412), qcomplex_t(0.034335701555517434,0.2825307847328346),qcomplex_t(0.4136767437887583,0.31439966778277506),qcomplex_t(0.18393603540428655,0.30588919458289876), qcomplex_t(0.1234427644644139,0.07995535607117477), qcomplex_t(0.3653047043801535,0.05179716511548195) };
	////std::vector<double>data{ 0.00010455027646829108, 0.000112165951999467, 0.00012026976656399777, 0.00012888769278913994, 0.00013804668784637316, 0.00014777470219979341, 0.00015810068613349398, 0.0001690545938777599, 0.0001806673851507228, 0.00019297102392962486, 0.00020599847426405876, 0.00021978369294236823, 0.00023436161882216627, 0.00024976815863643215, 0.00026604016908805074, 0.00028321543504814667, 0.0003013326436767946, 0.00032043135428928994, 0.0003405519637965668, 0.00036173566755516817, 0.00038402441546985037, 0.0004074608632011414, 0.0004320883183403972, 0.00045795068142645415, 0.0004850923816909944, 0.0005135583074336693, 0.0005433937309437296, 0.0005746442279015633, 0.0006073555912116153, 0.000641573739237675, 0.0006773446184318551, 0.0007147141003707094, 0.0007537278732348019, 0.0007944313277921636, 0.0008368694379715253, 0.0008810866361370585, 0.0009271266832039047, 0.0009750325337614142, 0.0010248461964000976, 0.001076608589467171, 0.0011303593925058894, 0.0011861368936638033, 0.0012439778333856323, 0.0013039172447372308, 0.001365988290737279, 0.0014302220991040896, 0.0014966475948546922, 0.0015652913312227358, 0.0016361773193908685, 0.0017093268575604616, 0.001784758359908941, 0.0018624871860101265, 0.0019425254713170382, 0.0020248819593294146, 0.0021095618360881313, 0.0021965665676579272, 0.0022858937412759063, 0.002377536910857247, 0.0024714854475615033, 0.0025677243961309097, 0.002666234337718773, 0.0027669912599285603, 0.0028699664347840337, 0.0029751263053478445, 0.0030824323816982776, 0.003191841146964451, 0.0033033039741057185, 0.003416767054104374, 0.0035321713362186703, 0.0036494524809194107, 0.003768540826104412, 0.003889361367153092, 0.004011833751348344, 0.00413587228715262, 0.004261385968783476, 0.004388278516487203, 0.004516448432859853, 0.004645789075513201, 0.004776188746326735, 0.004907530797469909, 0.005039693754317577, 0.005172551455318753, 0.005305973208814569, 0.0054398239667334996, 0.005573964515025005, 0.005708251680622479, 0.005842538554657211, 0.005976674731573177, 0.006110506563722806, 0.006243877430952528, 0.006376628024616782, 0.0065085966453906005, 0.006639619514181747, 0.006769531095378266, 0.006898164431602282, 0.007025351489079015, 0.007150923512671384, 0.007274711389573353, 0.007396546020603659, 0.007516258697991907, 0.007633681488504293, 0.0077486476207162674, 0.007860991875202582, 0.007970550976385304, 0.008077163984753975, 0.00818067268815165, 0.00828092199080624, 0.008377760298776247, 0.008471039900477659, 0.008560617340960593, 0.008646353788612659, 0.008728115392981057, 0.00880577363242479, 0.008879205650335653, 0.00894829457869834, 0.009012929847798069, 0.009073007480928014, 0.009128430372997352, 0.009179108551995685, 0.009224959422328536, 0.009265907989103069, 0.009301887062511453, 0.009332837441532443, 0.00935870807624813, 0.0093794562081529, 0.009395047487914955, 0.009405456070136247, 0.009410664684745041, 0.009410664684745041, 0.009405456070136247, 0.009395047487914955, 0.0093794562081529, 0.00935870807624813, 0.009332837441532443, 0.009301887062511453, 0.009265907989103069, 0.009224959422328536, 0.009179108551995673, 0.009128430372997352, 0.009073007480928014, 0.009012929847798069, 0.00894829457869834, 0.008879205650335653, 0.00880577363242479, 0.008728115392981057, 0.008646353788612659, 0.008560617340960593, 0.008471039900477659, 0.008377760298776247, 0.00828092199080624, 0.00818067268815165, 0.008077163984753975, 0.007970550976385304, 0.007860991875202582, 0.0077486476207162674, 0.007633681488504293, 0.007516258697991907, 0.007396546020603659, 0.007274711389573353, 0.007150923512671384, 0.007025351489079015, 0.006898164431602282, 0.006769531095378266, 0.006639619514181747, 0.0065085966453906005, 0.006376628024616782, 0.006243877430952528, 0.006110506563722806, 0.005976674731573177, 0.005842538554657211, 0.005708251680622479, 0.005573964515024972, 0.0054398239667334996, 0.005305973208814569, 0.005172551455318753, 0.005039693754317577, 0.004907530797469909, 0.004776188746326735, 0.004645789075513201, 0.004516448432859853, 0.0043882785164871705, 0.004261385968783476, 0.00413587228715262, 0.004011833751348344, 0.003889361367153092, 0.0037685408261043814, 0.0036494524809194107, 0.0035321713362186703, 0.003416767054104374, 0.0033033039741057185, 0.003191841146964423, 0.0030824323816982776, 0.0029751263053478445, 0.0028699664347840337, 0.0027669912599285603, 0.002666234337718773, 0.0025677243961309097, 0.0024714854475615033, 0.002377536910857247, 0.0022858937412759063, 0.0021965665676579272, 0.0021095618360881313, 0.0020248819593294146, 0.0019425254713170382, 0.0018624871860101265, 0.001784758359908941, 0.0017093268575604616, 0.0016361773193908685, 0.0015652913312227358, 0.0014966475948546922, 0.0014302220991040896, 0.001365988290737279, 0.0013039172447372308, 0.0012439778333856323, 0.0011861368936638033, 0.0011303593925058894, 0.001076608589467171, 0.0010248461964000976, 0.0009750325337614142, 0.0009271266832039047, 0.0008810866361370585, 0.0008368694379715253, 0.0007944313277921636, 0.0007537278732348019, 0.0007147141003707094, 0.0006773446184318551, 0.000641573739237675, 0.0006073555912116153, 0.0005746442279015633, 0.0005433937309437296, 0.0005135583074336693, 0.0004850923816909944, 0.00045795068142645415, 0.00043208831834039106, 0.0004074608632011414, 0.00038402441546985037, 0.00036173566755516817, 0.0003405519637965668, 0.00032043135428928485, 0.0003013326436767946, 0.00028321543504814667, 0.00026604016908805074, 0.00024976815863643215, 0.00023436161882216627, 0.00021978369294236823, 0.00020599847426405876, 0.00019297102392962486, 0.0001806673851507228, 0.0001690545938777599, 0.00015810068613349398, 0.00014777470219979341, 0.00013804668784637316, 0.00012888769278913994, 0.00012026976656399777, 0.000112165951999467, 0.00010455027646829108 };
	////std::vector<double>data{ 0.00103373, 0.00406266, 0.01256227, 0.03102829, 0.06207927, 0.10191518
	////		 , 0.13893765, 0.15903557, 0.15442565, 0.12842124, 0.0922782, 0.05776894
	////		 , 0.03175181, 0.01543281, 0.00667795, 0.00258877 };
	////std::vector<double>data{ 0,0.5,0.5,0,0.5,0,0,0,0.5,0,0,0,0,0,0,0 };
	////std::vector<double>data{ 0.15311858100051695,-0.0961350374871273,0.3859320687001368,-0.5634457467385428,0.1474901012487757,-0.45185782723129864,0.32284355187278985,-0.4132085412578166 };
	////std::vector<double>data{ 0.6793113376921358, 0.1376859100584252, 0.720435424880283, 0.02348393561289133};
	////for (int i = 0; i < data.size(); ++i) {
	////	data[i] = sqrt(data[i]);
	////}
	//QProg prog;
	//auto q = qvm->qAllocMany(dim);
	////auto circuit = QCircuit();
	////circuit << H(q[0]);
	////QStat q_mat(16);
	////q_mat[0] = (qcomplex_t)1.0;
	////q_mat[5]= (qcomplex_t)1.0;
	////q_mat[11] = (qcomplex_t)1.0;
	////q_mat[14] = (qcomplex_t)1.0;
	////circuit << QOracle(q, q_mat);
	//
	//Encode encode_b;
	//auto start = chrono::system_clock::now();
	////std::cout << "start" << std::endl;
	//encode_b.approx_mps_encode(q, data, 2, 1000);
	//auto end = chrono::system_clock::now();
	//auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
	//std::cout << double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den
	//	<< " s" << std::endl;
	//v2.normalize();
	//vector<qcomplex_t> data1(v2.data(), v2.data() + v2.size());
	//cout << encode_b.get_fidelity(data1);
	QProg prog = QProg();
	prog << encode_b.get_circuit();
	//std::cout << prog << std::endl;
	//std::cout<< convert_qprog_to_originir(prog, qvm)<<std::endl;
	QVec out_qubits = encode_b.get_out_qubits();
	//auto result = qvm->probRunList(prog, q, -1);
	qvm->directlyRun(prog);
	auto result = qvm->getQState();
	int k = 0;
	/*double normalization_constant = encode_b.get_normalization_constant();*/
	//for (auto &val : result)
	//{
	//	//double temp = k >= (int)data.size() ? 0 : data[k];
	//	std::cout << "Prob:" << val << "," << "Originial value:" << data[k]*data[k] << std::endl;
	//	++k;
	//}
	for (auto &val : result)
	{
		//double temp = k >= (int)data.size() ? 0 : data[k];
		std::cout << "Amplitude:" << val << "," << "Originial value:" << p[k] << std::endl;
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

	for (auto &val : result)
	{
		double temp = k >= (int)data.size() ? 0 : data[k];
		std::cout << val.first << ":" << val.second << std::endl;
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
		std::cout << val.first << ':' << val.second << std::endl;
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
	//std::vector<double>data{ -11,-22,-3,4,5,6 };
	auto prog = QProg();
	auto q = qvm->qAllocMany(4);
	Encode encode_b;
	encode_b.amplitude_encode(q, test_qstat);
	prog << encode_b.get_circuit();
	std::cout << prog << std::endl;
	QVec out_qubits = encode_b.get_out_qubits();
	qvm->directlyRun(prog);
	auto result = qvm->getQState();
	//auto result = qvm->probRunDict(prog, q, -1);
	int k = 0;
	for (auto &val : result) {
		std::cout << "Amplitude:" << val << std::endl;
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
	//vector<double>data{ 1,2,3,4,5,6 };
	auto prog = QProg();
	Encode encode_b;
	//encode_b.amplitude_encode_recursive(q, test_qstat);
	encode_b.amplitude_encode_recursive(q, test_qstat);
	prog << encode_b.get_circuit();

	std::cout << prog << std::endl;
	//QCircuit test_cir = amplitude_encode(q, test_qstat);

	//auto prog = QProg();
	//prog << test_cir;
	qvm->directlyRun(prog);
	auto stat = qvm->getQState();

	std::cout << "target_stat:\n" << stat << std::endl;

	destroyQuantumMachine(qvm);

	if (0 == mat_compare(stat, test_qstat, MAX_COMPARE_PRECISION)) {
		return true;
	}

	return false;
}
TEST(Encode_, test1)
{
	bool test_val = false;
	try
	{
		test_val = test_angle_encoding();
		test_val &= test_efficient_sparse_encoding();
		test_val &= test_ds_quantum_state_preparation_encoding();
		test_val &= test_Amplitude_encoding1();
		test_val &= test_Amplitude_encoding2();
		test_val &= test_Amplitude_encoding3();
		test_val &= test_Amplitude_encoding4();
		test_val &= test_dc_Amplitude_encoding();
		test_val &= test_bid_Amplitude_encoding();
		test_val &= test_dense_encoding();
		test_val &= test_basic_encoding();
		test_val &= test_iqp_Amplitude_encoding();
		test_val &= test_sparse_isometry_encoding();
		test_val &= test_schmidt_encoding();
		test_val&test_mps_encoding();
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
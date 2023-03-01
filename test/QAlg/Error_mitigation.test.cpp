#include "gtest/gtest.h"

#include"include/QAlg/Error_mitigation/Correction.h"
#include"include/Components/Operator/PauliOperator.h"
#include "QAlg/Encode/Encode.h"
#include "Core/Utilities/QProgInfo/Visualization/QVisualization.h"
USING_QPANDA




bool test_gmres() {
	int n = 256;
	Eigen::SparseMatrix<double> S = Eigen::MatrixXd::Random(n, n).sparseView(0.5, 1);
	S = S.transpose() * S;
	MatrixReplacement A;
	A.attachMyMatrix(S);
	Eigen::VectorXd b(n), x;
	b.setRandom();
	Eigen::GMRES<MatrixReplacement, Eigen::IdentityPreconditioner> gmres;
	gmres.compute(A);
	x = gmres.solve(b);
	std::cout << "GMRES:    #iterations: " << gmres.iterations() << ", estimated error: " << gmres.error() << std::endl;
	return true;
}
bool test_miti() {
	auto machine = CPUQVM();
	machine.init();
	int N = 5;
	QVec q = machine.qAllocMany(N);
	auto c = machine.cAllocMany(N);
	NoiseModel noise;
	std::vector<std::vector<double>> prob_list(N);
	for (int i = 0; i < N; ++i) {
		prob_list[i] = { 0.9,0.87 };
	}
	noise.set_readout_error({ { 0.9,0.1 }, { 0.13,0.87 } });
	QProg prog = QProg();
	prog << H(q[0]);
	for (int i = 0; i < N - 1; ++i) {
		prog << CNOT(q[i], q[i + 1]);
	}
	prog << MeasureAll(q, c);
	auto res = machine.runWithConfiguration(prog, c, 8192, noise);
	std::vector<double> unmiti_prob(1 << N);
	for (int i = 0; i < res.size(); ++i) {
		unmiti_prob[i] = (res[ull2binary(i, N)] / 8192.0);
	}
	std::cout << "unmitigatin:" << std::endl;
	for (double i : unmiti_prob) {
		std::cout << i << std::endl;
	}

	Mitigation mitigation(q, &machine, noise, 8192);
	mitigation.readout_error_mitigation(Independ, unmiti_prob);
	auto result = mitigation.get_miti_result();
	std::cout << "mitigatin:" << std::endl;
	for (double i : result) {
		std::cout << i << std::endl;
	}
	return true;
}
bool test_zne()
{

	CPUQVM qvm;
	qvm.init();
	int N = 8;
	auto q = qvm.qAllocMany(N);
	auto c = qvm.cAllocMany(N);
	//Encode encode;
	//std::vector<double>data{ 0.00010455027646829108, 0.000112165951999467, 0.00012026976656399777, 0.00012888769278913994, 0.00013804668784637316, 0.00014777470219979341, 0.00015810068613349398, 0.0001690545938777599, 0.0001806673851507228, 0.00019297102392962486, 0.00020599847426405876, 0.00021978369294236823, 0.00023436161882216627, 0.00024976815863643215, 0.00026604016908805074, 0.00028321543504814667, 0.0003013326436767946, 0.00032043135428928994, 0.0003405519637965668, 0.00036173566755516817, 0.00038402441546985037, 0.0004074608632011414, 0.0004320883183403972, 0.00045795068142645415, 0.0004850923816909944, 0.0005135583074336693, 0.0005433937309437296, 0.0005746442279015633, 0.0006073555912116153, 0.000641573739237675, 0.0006773446184318551, 0.0007147141003707094, 0.0007537278732348019, 0.0007944313277921636, 0.0008368694379715253, 0.0008810866361370585, 0.0009271266832039047, 0.0009750325337614142, 0.0010248461964000976, 0.001076608589467171, 0.0011303593925058894, 0.0011861368936638033, 0.0012439778333856323, 0.0013039172447372308, 0.001365988290737279, 0.0014302220991040896, 0.0014966475948546922, 0.0015652913312227358, 0.0016361773193908685, 0.0017093268575604616, 0.001784758359908941, 0.0018624871860101265, 0.0019425254713170382, 0.0020248819593294146, 0.0021095618360881313, 0.0021965665676579272, 0.0022858937412759063, 0.002377536910857247, 0.0024714854475615033, 0.0025677243961309097, 0.002666234337718773, 0.0027669912599285603, 0.0028699664347840337, 0.0029751263053478445, 0.0030824323816982776, 0.003191841146964451, 0.0033033039741057185, 0.003416767054104374, 0.0035321713362186703, 0.0036494524809194107, 0.003768540826104412, 0.003889361367153092, 0.004011833751348344, 0.00413587228715262, 0.004261385968783476, 0.004388278516487203, 0.004516448432859853, 0.004645789075513201, 0.004776188746326735, 0.004907530797469909, 0.005039693754317577, 0.005172551455318753, 0.005305973208814569, 0.0054398239667334996, 0.005573964515025005, 0.005708251680622479, 0.005842538554657211, 0.005976674731573177, 0.006110506563722806, 0.006243877430952528, 0.006376628024616782, 0.0065085966453906005, 0.006639619514181747, 0.006769531095378266, 0.006898164431602282, 0.007025351489079015, 0.007150923512671384, 0.007274711389573353, 0.007396546020603659, 0.007516258697991907, 0.007633681488504293, 0.0077486476207162674, 0.007860991875202582, 0.007970550976385304, 0.008077163984753975, 0.00818067268815165, 0.00828092199080624, 0.008377760298776247, 0.008471039900477659, 0.008560617340960593, 0.008646353788612659, 0.008728115392981057, 0.00880577363242479, 0.008879205650335653, 0.00894829457869834, 0.009012929847798069, 0.009073007480928014, 0.009128430372997352, 0.009179108551995685, 0.009224959422328536, 0.009265907989103069, 0.009301887062511453, 0.009332837441532443, 0.00935870807624813, 0.0093794562081529, 0.009395047487914955, 0.009405456070136247, 0.009410664684745041, 0.009410664684745041, 0.009405456070136247, 0.009395047487914955, 0.0093794562081529, 0.00935870807624813, 0.009332837441532443, 0.009301887062511453, 0.009265907989103069, 0.009224959422328536, 0.009179108551995673, 0.009128430372997352, 0.009073007480928014, 0.009012929847798069, 0.00894829457869834, 0.008879205650335653, 0.00880577363242479, 0.008728115392981057, 0.008646353788612659, 0.008560617340960593, 0.008471039900477659, 0.008377760298776247, 0.00828092199080624, 0.00818067268815165, 0.008077163984753975, 0.007970550976385304, 0.007860991875202582, 0.0077486476207162674, 0.007633681488504293, 0.007516258697991907, 0.007396546020603659, 0.007274711389573353, 0.007150923512671384, 0.007025351489079015, 0.006898164431602282, 0.006769531095378266, 0.006639619514181747, 0.0065085966453906005, 0.006376628024616782, 0.006243877430952528, 0.006110506563722806, 0.005976674731573177, 0.005842538554657211, 0.005708251680622479, 0.005573964515024972, 0.0054398239667334996, 0.005305973208814569, 0.005172551455318753, 0.005039693754317577, 0.004907530797469909, 0.004776188746326735, 0.004645789075513201, 0.004516448432859853, 0.0043882785164871705, 0.004261385968783476, 0.00413587228715262, 0.004011833751348344, 0.003889361367153092, 0.0037685408261043814, 0.0036494524809194107, 0.0035321713362186703, 0.003416767054104374, 0.0033033039741057185, 0.003191841146964423, 0.0030824323816982776, 0.0029751263053478445, 0.0028699664347840337, 0.0027669912599285603, 0.002666234337718773, 0.0025677243961309097, 0.0024714854475615033, 0.002377536910857247, 0.0022858937412759063, 0.0021965665676579272, 0.0021095618360881313, 0.0020248819593294146, 0.0019425254713170382, 0.0018624871860101265, 0.001784758359908941, 0.0017093268575604616, 0.0016361773193908685, 0.0015652913312227358, 0.0014966475948546922, 0.0014302220991040896, 0.001365988290737279, 0.0013039172447372308, 0.0012439778333856323, 0.0011861368936638033, 0.0011303593925058894, 0.001076608589467171, 0.0010248461964000976, 0.0009750325337614142, 0.0009271266832039047, 0.0008810866361370585, 0.0008368694379715253, 0.0007944313277921636, 0.0007537278732348019, 0.0007147141003707094, 0.0006773446184318551, 0.000641573739237675, 0.0006073555912116153, 0.0005746442279015633, 0.0005433937309437296, 0.0005135583074336693, 0.0004850923816909944, 0.00045795068142645415, 0.00043208831834039106, 0.0004074608632011414, 0.00038402441546985037, 0.00036173566755516817, 0.0003405519637965668, 0.00032043135428928485, 0.0003013326436767946, 0.00028321543504814667, 0.00026604016908805074, 0.00024976815863643215, 0.00023436161882216627, 0.00021978369294236823, 0.00020599847426405876, 0.00019297102392962486, 0.0001806673851507228, 0.0001690545938777599, 0.00015810068613349398, 0.00014777470219979341, 0.00013804668784637316, 0.00012888769278913994, 0.00012026976656399777, 0.000112165951999467, 0.00010455027646829108 };
	//for (int i = 0; i < data.size(); ++i) {
	//	data[i] = sqrt(data[i]);
	//}
	//Encode encode_b;
	////std::cout << "start" << std::endl;
	//encode_b.approx_mps_encode(q, data, 2, 0);
	//QCircuit circuit = encode_b.get_circuit();
	QCircuit circuit = random_qcircuit(q, 10);
	//QCircuit circuit = QCircuit();
	//circuit << H(q[0]) <<CNOT(q[0],q[1]) << X(q[0]);
	QProg ori_prog = QProg();
	ori_prog << circuit << MeasureAll(q, c);
	//auto res0 = machine.get_expectation(ori_prog, ham.toHamiltonian(), qubits);
	auto result = qvm.runWithConfiguration(ori_prog, c, 8192);
	std::vector<double> unmiti_prob(1 << N);
	std::vector<double> ideal_prob(1 << N);
	for (int i = 0; i < result.size(); ++i) {
		ideal_prob[i] = (result[ull2binary(i, N)] / 8192.0);
	}


	//std::cout << "unmitigatin:" << std::endl;
	//for (double i : unmiti_prob) {
	//	std::cout << i << std::endl;
	//}
	//qvm.set_noise_model(NOISE_MODEL::BITFLIP_KRAUS_OPERATOR, GateType::PAULI_X_GATE, 0.1);

	//qvm.set_noise_model(NOISE_MODEL::DEPHASING_KRAUS_OPERATOR, GateType::HADAMARD_GATE, 0.1);

	//qvm.set_noise_model(NOISE_MODEL::DAMPING_KRAUS_OPERATOR, GateType::CNOT_GATE, 0.1);
	NoiseModel noise;
	noise.add_noise_model(NOISE_MODEL::BITFLIP_KRAUS_OPERATOR, GateType::PAULI_X_GATE, 0.1);
	noise.add_noise_model(NOISE_MODEL::BITFLIP_KRAUS_OPERATOR, GateType::PAULI_Y_GATE, 0.1);
	noise.add_noise_model(NOISE_MODEL::BITFLIP_KRAUS_OPERATOR, GateType::PAULI_Z_GATE, 0.1);
	noise.add_noise_model(NOISE_MODEL::DEPHASING_KRAUS_OPERATOR, GateType::PAULI_Z_GATE, 0.1);
	noise.add_noise_model(NOISE_MODEL::DEPHASING_KRAUS_OPERATOR, GateType::PAULI_X_GATE, 0.1);
	noise.add_noise_model(NOISE_MODEL::DEPHASING_KRAUS_OPERATOR, GateType::PAULI_Y_GATE, 0.1);
	noise.add_noise_model(NOISE_MODEL::DAMPING_KRAUS_OPERATOR, GateType::CNOT_GATE, 0.1);
	//noise.set_rotation_error(0.1);
	auto result1 = qvm.runWithConfiguration(ori_prog, c, 8192, noise);

	for (int i = 0; i < result1.size(); ++i) {
		unmiti_prob[i] = (result1[ull2binary(i, N)] / 8192.0);
	}
	std::cout << "unmitigatin:" << std::endl;
	std::cout << kl_divergence(ideal_prob, unmiti_prob) << std::endl;
	Mitigation mitigation(q, &qvm, noise, 10000);
	mitigation.zne_error_mitigation(circuit, { 1,2,3 });
	auto res = mitigation.get_miti_result();
	std::cout << "mitigatin:" << std::endl;
	std::cout << "kl_d: " << kl_divergence(ideal_prob, res) << std::endl;
	//for (double i : res) {
	//	std::cout << i << std::endl;
	//}
	//auto res0 = qvm.get_expectation(ori_prog, ham.toHamiltonian(), qubits,8192);

	//QProg prog = QProg();
	//QProg prog2 = QProg();
	//QProg prog3 = QProg();
	//QProg prog4 = QProg();
	//prog << circuit<<MeasureAll(q,c);
	//auto res1 = qvm.runWithConfiguration(prog, c,8192,noise);
	//prog2 << H(q[0]) << CNOT(q[0], q[1]) << X(q[0]) << X(q[0]) << X(q[0]) << I(q[0])<<MeasureAll(q,c);
	//auto res2 = qvm.runWithConfiguration(prog2, c, 8192,noise);
	//prog3 << circuit<<circuit<<circuit.dagger() <<MeasureAll(q,c);
	//auto res3 = qvm.runWithConfiguration(prog3, c, 8192,noise);
	////std::cout << "noise free:" << res0 << std::endl;
	//std::cout << "noise 1:" << (double)(res1["00"]/8192.0) << std::endl;
	//std::cout << "noise 2:" << (double)(res2["00"] / 8192.0) << std::endl;
	//std::cout << "noise 3:" << (double)(res3["00"] / 8192.0) << std::endl;
	return true;
}
bool test_quasi()
{


	CPUQVM qvm;
	qvm.init();

	int shots = 2048;
	int depth = 5;

	//std::cout << shots << std::endl;
	for (int i = 2; i < 11; ++i) {
		int N = i;
		auto q = qvm.qAllocMany(N);
		auto c = qvm.cAllocMany(N);
		for (int j = 5; j <= 20; j += 5) {
			QCircuit circuit = random_qcircuit(q, j, { "RX","RY","RZ","CNOT" });

			QProg ori_prog = QProg();
			ori_prog << circuit << MeasureAll(q, c);

			auto result = qvm.runWithConfiguration(ori_prog, c, shots);
			std::vector<double> unmiti_prob(1 << N);
			std::vector<double> ideal_prob(1 << N);

			for (int i = 0; i < result.size(); ++i) {
				ideal_prob[i] = (result[ull2binary(i, N)] / (double)shots);
			}

			NoiseModel noise;
			double p = 0.1;
			noise.add_noise_model(NOISE_MODEL::DEPOLARIZING_KRAUS_OPERATOR, GateType::CNOT_GATE, p);


			QProg prog = QProg();
			prog << circuit << MeasureAll(q, c);
			auto res = qvm.runWithConfiguration(prog, c, shots, noise);
			for (int i = 0; i < res.size(); ++i) {
				unmiti_prob[i] = (res[ull2binary(i, N)] / (double)shots);
			}
			std::cout << "Qubits:" << i << "," << "Depth:" << j << "," << "unmiti_kl_d: " << kl_divergence(ideal_prob, unmiti_prob) << std::endl;



			Mitigation mitigation(q, &qvm, noise, shots);
			std::vector<std::pair<double, GateType>> representation(4);
			double p2 = p / (16 + 14 * p);
			double p1 = 1 - 15 * p2;
			representation[0].first = p1;
			representation[0].second = GateType::I_GATE;
			representation[1].first = -p2;
			representation[1].second = GateType::PAULI_X_GATE;
			representation[2].first = -p2;
			representation[2].second = GateType::PAULI_Y_GATE;
			representation[3].first = -p2;
			representation[3].second = GateType::PAULI_Z_GATE;
			double norm = 1 + 30 * p2;

			//std::tuple<double, std::vector<std::pair<double, GateType>>> representation;
			auto tuple_tmp = std::make_tuple(norm, representation);
			mitigation.quasi_probability(circuit, tuple_tmp, 20);

			std::cout << "Qubits:" << i << "," << "Depth:" << j << "," << "miti_kl_d: " << kl_divergence(ideal_prob, mitigation.get_miti_result()) << std::endl;




		}
	}

	////QCircuit circuit = QCircuit();
	////circuit << H(q[0]);
	////for (int i = 0; i < N-1; ++i) {
	////	circuit << CNOT(q[i], q[i + 1]);
	////}
	//int cnot_nums = count_qgate_num(circuit, CNOT_GATE);
	//std::cout << cnot_nums << std::endl;	
	////QCircuit circuit = QCircuit();
	////circuit << X(q[0]) << H(q[1]) << CNOT(q[0], q[1]);
	//QProg ori_prog = QProg();
	//ori_prog << circuit << MeasureAll(q, c);
	///*auto pauli_opt = PauliOperator(PauliOperator::PauliMap{
	//	{"X1", 1} });*/
	////auto trace=qvm.get_expectation(ori_prog, pauli_opt.toHamiltonian(), q,shots);
	//auto result = qvm.runWithConfiguration(ori_prog, c, shots);
	//std::vector<double> unmiti_prob(1 << N);
	//std::vector<double> ideal_prob(1 << N);

	//for (int i = 0; i < result.size(); ++i) {
	//	ideal_prob[i] = (result[ull2binary(i, N)] / (double)shots);
	//}
	////std::cout << "ideal: " << std::endl;
	////for (int i = 0; i < (1 << N); ++i) {
	////	std::cout << ideal_prob[i] << std::endl;
	////}
	//NoiseModel noise;
	//double p = 0.1;
	//noise.add_noise_model(NOISE_MODEL::DEPOLARIZING_KRAUS_OPERATOR, GateType::CNOT_GATE, p);
	////noise.add_noise_model(NOISE_MODEL::DEPOLARIZING_KRAUS_OPERATOR, GateType::CNOT_GATE, p);
	////noise.add_noise_model(NOISE_MODEL::BITFLIP_KRAUS_OPERATOR, GateType::PAULI_Y_GATE, 0.1);
	////noise.add_noise_model(NOISE_MODEL::BITFLIP_KRAUS_OPERATOR, GateType::PA
	//// ULI_Z_GATE, 0.1);
	////noise.add_noise_model(NOISE_MODEL::DEPHASING_KRAUS_OPERATOR, GateType::PAULI_Z_GATE, 0.1);
	////noise.add_noise_model(NOISE_MODEL::DEPHASING_KRAUS_OPERATOR, GateType::PAULI_X_GATE, 0.1);
	////noise.add_noise_model(NOISE_MODEL::DEPHASING_KRAUS_OPERATOR, GateType::PAULI_Y_GATE, 0.1);
	////noise.add_noise_model(NOISE_MODEL::DAMPING_KRAUS_OPERATOR, GateType::CNOT_GATE, 0.1);
	//
	////QProg prog1 = QProg();
	////QProg prog2 = QProg();
	////QProg prog3 = QProg();
	////QProg prog4 = QProg();
	////prog << circuit << MeasureAll(q, c);
	////prog1 << circuit << I(q) << MeasureAll(q, c);
	////prog2 << circuit << X(q) << MeasureAll(q, c);
	////prog3 << circuit << Y(q) << MeasureAll(q, c);
	////prog4 << circuit << Z(q) << MeasureAll(q, c);

	//QProg prog = QProg();
	//prog << circuit << MeasureAll(q, c);
	//auto res = qvm.runWithConfiguration(prog, c, shots, noise);
	//for (int i = 0; i < res.size(); ++i) {
	//	unmiti_prob[i] = (res[ull2binary(i, N)] / (double)shots);
	//}
	//std::cout << "unmiti_kl_d: " << kl_divergence(ideal_prob, unmiti_prob) << std::endl;


	////std::cout << "unmitigation: " << std::endl;
	////for (int i = 0; i < (1 << N); ++i) {
	////	std::cout << unmiti_prob[i] << std::endl;
	////}
	//Mitigation mitigation(q, &qvm, noise, shots);
	//std::vector<std::pair<double, GateType>> representation(4);
	//double p2 = p / (16 + 14 * p);
	//double p1 = 1 - 15 * p2;
	//representation[0].first = p1;
	//representation[0].second = GateType::I_GATE;
	//representation[1].first =  -p2;
	//representation[1].second = GateType::PAULI_X_GATE;
	//representation[2].first = -p2;
	//representation[2].second = GateType::PAULI_Y_GATE;
	//representation[3].first = -p2;
	//representation[3].second = GateType::PAULI_Z_GATE;
	//double norm = 1 +30*p2;
	////std::cout << (p + 2) / (2 - 2 * p) << std::endl;
	//std::cout << norm << std::endl;
	//std::cout << p1 << std::endl;
	//int num_samples = ceil((double)(norm/0.03)* (double)(norm / 0.03));
	//std::cout << num_samples << std::endl;
	//mitigation.quasi_probability(circuit, { norm,representation }, 20);
	////auto result1 = qvm.runWithConfiguration(prog1, c, 8192,noise);
	////auto result2 = qvm.runWithConfiguration(prog2, c, 8192, noise);
	////auto result3 = qvm.runWithConfiguration(prog3, c, 8192, noise);
	////auto result4 = qvm.runWithConfiguration(prog4, c, 8192, noise);
	////std::vector<double> mitiprob(4);
	////for (int i = 0; i < (1<<N); ++i) {
	////	std::string s = ull2binary(i, N);
	////	mitiprob[i] = ((p + 2) / (2 - 2 * p)) * ((4 - p) / (4 + 2 * p)) * (result1[s]/8192.0) - ((p + 2) / (2 - 2 * p)) * ((p / (2 * p + 4))) * (result2[s] / 8192.0 + result3[s] / 8192.0 + result4[s] / 8192.0);
	////}
	////std::cout << "mitigation: " << std::endl;
	////for (int i = 0; i < (1 << N); ++i) {
	////	std::cout << mitigation.get_miti_result()[i] << std::endl;
	////}
	////std::cout << "unmiti_kl_d: " << kl_divergence(ideal_prob, unmiti_prob) << std::endl;
	//std::cout << "miti_kl_d: " << kl_divergence(ideal_prob, mitigation.get_miti_result()) << std::endl;
	return true;
}
TEST(Mitigation, test1)
{
	bool test_val = false;
	try
	{
		test_val = test_quasi();
	}
	catch (const std::exception& e)
	{
		std::cout << "Got a exception: " << e.what() << std::endl;
	}
	catch (...)
	{
		std::cout << "Got an unknow exception: " << std::endl;
	}

	ASSERT_TRUE(test_val);
}
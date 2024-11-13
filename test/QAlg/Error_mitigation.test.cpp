#include "gtest/gtest.h"

#include "QAlg/Error_mitigation/Correction.h"
#include "include/Components/Operator/PauliOperator.h"
#include "Core/Utilities/Encode/Encode.h"
#include "Core/Utilities/QProgInfo/Visualization/QVisualization.h"
#include "Extensions/PilotOSMachine/QPilotOSMachine.h"
USING_QPANDA


#ifdef USE_CURL

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
#if defined(USE_CURL)
    QPilotOSMachine qvm("Pilot");
    Configuration config;
    config.maxQubit = 72;
    config.maxCMem = 72;
    qvm.setConfig(config);
    qvm.init("https://10.9.12.9:10080", true, "F5BB86A31564481BBB14634FF4B1C26F");


    auto q = qvm.qAllocMany(72);
    auto c = qvm.cAllocMany(72);
    QVec used_q{ q[45],q[46] };
    std::vector<ClassicalCondition> used_c{ c[0],c[1] };
    Mitigation mit(q, &qvm, 1000);
    QProg prog = QProg();
    prog << H(q[45]) << CNOT(q[45], q[46]);
    prog << MeasureAll(used_q, { c[0],c[1] });
    std::vector < std::tuple<int, int, int>>amplify_factors{ {0,0,1},{0,1,1},{0,2,1} };
    std::vector<double>order{ 1.0,2.0,3.0 };
    auto res = mit.zne_circuit(prog, amplify_factors, true);
    mit.zne_error_mitigation(order, res);
    auto result = mit.get_miti_result();

#endif
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
        test_val = test_zne();
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
#endif
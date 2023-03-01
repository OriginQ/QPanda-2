#include <atomic>
#include <chrono>
#include <thread>
#include "QPanda.h"
#include "gtest/gtest.h"
#include "Extensions/Extensions.h"

#ifdef USE_CURL

#ifdef USE_EXTENSION
using namespace std;
using namespace PilotQVM;
using namespace QPanda;
using namespace rapidjson;

TEST(QPilotOSMachine, test)
{
    QPilotOSMachine QCM("Pilot");
    //QCM.init("https://10.9.12.210:10080", true);
	//QCM.init("https://10.9.12.154:10080", true);
    QCM.init("https://10.10.10.61:10080", false);
    auto q = QCM.allocateQubits(6);
    auto c = QCM.allocateCBits(6);

    auto measure_prog = QProg();
    measure_prog << HadamardQCircuit(q)
        << CZ(q[2], q[3])
        << Measure(q[0], c[0]);

    auto pmeasure_prog = QProg();
    pmeasure_prog << HadamardQCircuit(q)
        << CZ(q[1], q[5])
        << RX(q[2], PI / 4)
        << RX(q[1], PI / 4);

	QProg test_prog_1;
	test_prog_1 << H(q[0]) << H(q[1]) << H(q[2]) << H(q[3]) << H(q[4]) << X(q[5]) << X(q[0]) << X(q[4])
		<< H(q[5]) << CNOT(q[0], q[5]) << X(q[0]) << CNOT(q[1], q[5]) << H(q[0]) << H(q[1])
		<< CNOT(q[2], q[5]) << H(q[2]) << CNOT(q[3], q[5]) << H(q[3]) << CNOT(q[4], q[5]) << X(q[4]) << H(q[4]) 
		<< MeasureAll(q, c);

	const string test_ir_sunkf = "QINIT 12\nCREG 12\nH q[0]\nH q[1]\nH q[2]\nH q[3]\nH q[4]\nH q[5]\nH q[6]\nH q[7]\nH q[8]\nH q[9]\nH q[10]\nH q[11]\nCNOT q[0],q[1]\nCNOT q[1],q[2]\nCNOT q[2],q[3]\nCNOT q[3],q[4]\nCNOT q[4],q[5]\nCNOT q[5],q[6]\nCNOT q[6],q[7]\nCNOT q[7],q[8]\nCNOT q[8],q[9]\nCNOT q[9],q[10]\nCNOT q[10],q[11]\nMEASURE q[0],c[0]\nMEASURE q[1],c[1]\nMEASURE q[2],c[2]\nMEASURE q[3],c[3]\nMEASURE q[4],c[4]\nMEASURE q[5],c[5]\nMEASURE q[6],c [6]\nMEASURE q[7],c[7]\nMEASURE q[8],c[8]\nMEASURE q[9],c[9]\nMEASURE q[10],c[10]\nMEASURE q[11],c[11]";
	QProg test_prog_2 = convert_originir_string_to_qprog(test_ir_sunkf, &QCM);
	for (size_t i = 0; i < 2500; i++)
	{
		auto result0 = QCM.full_amplitude_measure(test_prog_2, 1000);
		std::cout << "-------------------full_amplitude_measure---------------------------" << std::endl;
		for (auto val : result0)
		{
			cout << val.first << " : " << val.second << endl;
		}
	}

    for (size_t i = 0; i < 1000; ++i)
    {
        auto result1 = QCM.full_amplitude_pmeasure(pmeasure_prog, { 0, 1, 2 });
        std::cout << "-------------------full_amplitude_pmeasure---------------------------" << std::endl;
        for (auto val : result1)
        {
            cout << val.first << " : " << val.second << endl;
        }

        auto result2 = QCM.partial_amplitude_pmeasure(pmeasure_prog, { "0", "1", "2" });
        std::cout << "-------------------partial_amplitude_pmeasure---------------------------" << std::endl;
        for (auto val : result2)
        {
            cout << val.first << " : " << val.second << endl;
        }

        auto result3 = QCM.single_amplitude_pmeasure(pmeasure_prog, "0");
        std::cout << "-------------------single_amplitude_pmeasure---------------------------" << std::endl;
        cout << "0" << " : " << result3 << endl;

        QCM.set_noise_model(NOISE_MODEL::BIT_PHASE_FLIP_OPRATOR, { 0.01 }, { 0.02 });
        auto result4 = QCM.noise_measure(measure_prog, 100);
        std::cout << "-------------------noise_measure---------------------------" << std::endl;
        for (auto val : result4)
        {
            cout << val.first << " : " << val.second << endl;
        }

        auto result7 = QCM.get_state_fidelity(measure_prog, 1000);
        std::cout << "-------------------get_state_fidelity---------------------------" << std::endl;
        cout << "result7:" << result7 << endl;
    }
    
	for (size_t test_cnt = 0; test_cnt < 4000; test_cnt++)
	{
		auto result_8 = QCM.real_chip_measure(test_prog_1);
		std::cout << "------------------- get_realchip result 8 ---------------------------" << std::endl;
		for (const auto& i : result_8) {
			cout << i.first << ":" << i.second << "\n";
		}
	}
	
    QCM.finalize();
}

#endif

#endif
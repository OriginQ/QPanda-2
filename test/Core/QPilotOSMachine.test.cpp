#include <atomic>
#include <chrono>
#include <thread>
#include "QPanda.h"
#include "gtest/gtest.h"
#include "Extensions/Extensions.h"


#if defined(USE_CURL)

#ifdef USE_EXTENSION
#include "Extensions/PilotOSMachine/QPilotMachine.h"
#define SUCCESS_STATE "3"
#define FAILED_STATE "4"
#define CANCELED_STATE "35"
using namespace std;
using namespace PilotQVM;
using namespace QPanda;
using namespace rapidjson;

namespace PilotQVM {
    struct PilotNoiseParams;
    struct PilotTaskQueryResult;
    class QPilotMachine;
}

static bool test_real_chip_expectation(QPilotOSMachine& qvm)
{
    auto q = qvm.allocateQubits(6);
    auto c = qvm.allocateCBits(6);

    QProg test_prog_1;
#if 0
    test_prog_1 << RY(0, -0.22606962488087734);
    std::string hamiltonian = "\"\": -1.0421749197656989,\"Z0\": -0.789688726711808,\"X0\": 0.18121046201519692";
    std::vector<uint32_t> qubits {0};
#else
    auto qv = {q[0],q[1]};
    //std::string originir =  "QINIT 6\nCREG 6\nX q[0]\nBARRIER q[0], q[1]\nH q[0]\nRX q[1], (-1.570796)\nCNOT q[1], q[0]\nBARRIER q[0], q[1]\nRZ q[0], (0.226019)\nBARRIER q[0], q[1]\nCNOT q[1], q[0]\nBARRIER q[0], q[1]\nH q[0]\nRX q[1], (1.570796)";
    test_prog_1 << X(q[0])
                << BARRIER(qv)
                << H(q[0])
                << RX(q[1], -1.570796)
                << CNOT(q[1], q[0])
                << BARRIER(qv)
                << RZ(q[0], 0.226019)
                << BARRIER(qv)
                << CNOT(q[1], q[0])
                << BARRIER(qv)
                << H(q[0])
                << RX(q[1], 1.570796);
    std::string hamiltonian = "\"\": -1.0534210769165204,\"Z0\": 0.394844363355904,\"Z1\": -0.39484436335590417,\"Z0 Z1\": -0.01124615715082114,\"X0 X1\": 0.18121046201519694";
    std::vector<uint32_t> qubits{0, 1};
#endif

    for (size_t test_cnt = 0; test_cnt < 3/*0000*/; test_cnt++)
    {
        auto result = qvm.real_chip_expectation(test_prog_1, hamiltonian, qubits, 1000, 72, true, true, true, {52, 53, 54});
        std::cout << "------------------- get result: " << result << " for real_chip_expectation ---------------------------" << std::endl;
    }

    return true;
}

static bool test_real_chip_expectation_async(QPilotOSMachine& qvm)
{
    auto q = qvm.allocateQubits(6);
    auto c = qvm.allocateCBits(6);

    QProg test_prog_1;
#if 0
    test_prog_1 << RY(0, -0.22606962488087734);
    std::string hamiltonian = "\"\": -1.0421749197656989,\"Z0\": -0.789688726711808,\"X0\": 0.18121046201519692";
    std::vector<uint32_t> qubits {0};
#else
    //std::string originir =  "QINIT 6\nCREG 6\nX q[0]\nBARRIER q[0], q[1]\nH q[0]\nRX q[1], (-1.570796)\nCNOT q[1], q[0]\nBARRIER q[0], q[1]\nRZ q[0], (0.226019)\nBARRIER q[0], q[1]\nCNOT q[1], q[0]\nBARRIER q[0], q[1]\nH q[0]\nRX q[1], (1.570796)";
    test_prog_1 << X(q[0])
                << BARRIER({0, 1})
                << H(q[0])
                << RX(q[1], -1.570796)
                << CNOT(q[1], q[0])
                << BARRIER({0, 1})
                << RZ(q[0], 0.226019)
                << BARRIER({0, 1})
                << CNOT(q[1], q[0])
                << BARRIER({0, 1})
                << H(q[0])
                << RX(q[1], 1.570796);
    std::string hamiltonian = "\"\": -1.0534210769165204,\"Z0\": 0.394844363355904,\"Z1\": -0.39484436335590417,\"Z0 Z1\": -0.01124615715082114,\"X0 X1\": 0.18121046201519694";
    std::vector<uint32_t> qubits{0, 1};
#endif
    PilotQVM::PilotTaskQueryResult res;

    for (size_t test_cnt = 0; test_cnt < 3/*0000*/; ++test_cnt)
    {
        const auto task_id = qvm.async_real_chip_expectation(test_prog_1, hamiltonian, qubits, 1000, 72);
        std::cout << "test_" << test_cnt << ": got task_id:" << task_id << " for async_real_chip_measure\n";
        std::cout << "On get result for async_real_chip_measure ...";
        do
        {
            qvm.query_task_state(task_id, res);
            std::this_thread::sleep_for(std::chrono::seconds(1));
            std::cout << ".";
        } while ((res.m_state != SUCCESS_STATE)|| res.m_state != FAILED_STATE && (res.m_state != CANCELED_STATE));
        std::cout << "Expectation: " << res.m_result << std::endl;
    }

    return true;
}

static bool test_real_chip_measure(QPilotOSMachine& qvm)
{
    auto q = qvm.allocateQubits(6);
    auto c = qvm.allocateCBits(6);

    QProg test_prog_1;
    test_prog_1 << H(q[0]) << H(q[1]) << H(q[2]) << H(q[3]) << H(q[4]) << X(q[5]) << X(q[0]) << X(q[4])
        << H(q[5]) << CNOT(q[0], q[5]) << X(q[0]) << CNOT(q[1], q[5]) << H(q[0]) << H(q[1])
        << CNOT(q[2], q[5]) << H(q[2]) << CNOT(q[3], q[5]) << H(q[3]) << CNOT(q[4], q[5]) << X(q[4]) << H(q[4])
        << MeasureAll(q, c);

    for (size_t test_cnt = 0; test_cnt < 3/*0000*/; test_cnt++)
    {
        auto result = qvm.real_chip_measure(test_prog_1, 1000);
        std::cout << "------------------- get result for real_chip_measure ---------------------------" << std::endl;
        for (const auto& i : result) {
            cout << i.first << ":" << i.second << "\n";
        }
    }

    return true;
}

static bool test_real_chip_measure_async(QPilotOSMachine& qvm)
{
    auto q = qvm.allocateQubits(6);
    auto c = qvm.allocateCBits(6);

    QProg test_prog_1;
    test_prog_1 << H(q[0]) << H(q[1]) << H(q[2]) << H(q[3]) << H(q[4]) << X(q[5]) << X(q[0]) << X(q[4])
        << H(q[5]) << CNOT(q[0], q[5]) << X(q[0]) << CNOT(q[1], q[5]) << H(q[0]) << H(q[1])
        << CNOT(q[2], q[5]) << H(q[2]) << CNOT(q[3], q[5]) << H(q[3]) << CNOT(q[4], q[5]) << X(q[4]) << H(q[4])
        << MeasureAll(q, c);

    PilotQVM::PilotTaskQueryResult res;

    for (size_t test_cnt = 0; test_cnt < 3/*0000*/; ++test_cnt)
    {
        const auto task_id = qvm.async_real_chip_measure(test_prog_1, 1000, 1);
        std::cout << "test_" << test_cnt << ": got task_id:" << task_id << " for async_real_chip_measure\n";
        std::cout << "On get result for async_real_chip_measure ...";
        do
        {
            qvm.query_task_state(task_id, res);
            std::this_thread::sleep_for(std::chrono::seconds(1));
            std::cout << std::endl << "state:" << res.m_state << std::endl;
        } while ((res.m_state != SUCCESS_STATE) && (res.m_state != FAILED_STATE) && (res.m_state != CANCELED_STATE));

        std::map<std::string, double> result;
        qvm.parse_task_result(res.m_result, result);
        std::cout << "\nGot result:\n";
        for (const auto& val : result){
            cout << val.first << ": " << val.second << "\n";
        }
    }

    return true;
}

static bool test_real_chip_query_task(QPilotOSMachine& qvm)
{
    PilotQVM::PilotTaskQueryResult res;
    const std::string task_id = "7A94F74AB17E49EA8A6B37A3AA960FF2";
    std::string compile_prog;
    bool with_compensate = true;
    for (size_t test_cnt = 0; test_cnt < 3/*0000*/; ++test_cnt)
    {
        do
        {
            qvm.query_task_state(task_id, res);
            std::this_thread::sleep_for(std::chrono::seconds(1));
            std::cout << ".";
        } while ((res.m_state != SUCCESS_STATE) && (res.m_state != FAILED_STATE) && (res.m_state != CANCELED_STATE));

        std::map<std::string, double> result;
        qvm.parse_task_result(res.m_result, result);
        std::cout << "\nGot result:\n";
        for (const auto& val : result) {
            cout << val.first << ": " << val.second << "\n";
        }
    }

    return true;
}

static bool test_real_chip_query_compile_info(QPilotOSMachine& qvm)
{
    PilotQVM::PilotTaskQueryResult res;
    const std::string task_id = "7A94F74AB17E49EA8A6B37A3AA960FF2";
    std::string compile_prog;
    bool with_compensate = true;
    for (size_t test_cnt = 0; test_cnt < 1/*0000*/; ++test_cnt)
    {
    if (qvm.query_compile_prog(task_id, compile_prog, with_compensate))
        {
            std::cout << "task compile prog: " << std::endl << compile_prog << std::endl;
        }
    }

    return true;
}


static bool test_full_amplitude_measure(QPilotOSMachine& qvm)
{
    const string test_ir_1 = "QINIT 12\nCREG 12\nH q[0]\nH q[1]\nH q[2]\nH q[3]\nH q[4]\nH q[5]\nH q[6]\nH q[7]\nH q[8]\nH q[9]\nH q[10]\nH q[11]\nCNOT q[0],q[1]\nCNOT q[1],q[2]\nCNOT q[2],q[3]\nCNOT q[3],q[4]\nCNOT q[4],q[5]\nCNOT q[5],q[6]\nCNOT q[6],q[7]\nCNOT q[7],q[8]\nCNOT q[8],q[9]\nCNOT q[9],q[10]\nCNOT q[10],q[11]\nMEASURE q[0],c[0]\nMEASURE q[1],c[1]\nMEASURE q[2],c[2]\nMEASURE q[3],c[3]\nMEASURE q[4],c[4]\nMEASURE q[5],c[5]\nMEASURE q[6],c [6]\nMEASURE q[7],c[7]\nMEASURE q[8],c[8]\nMEASURE q[9],c[9]\nMEASURE q[10],c[10]\nMEASURE q[11],c[11]";
    QProg test_prog_2 = convert_originir_string_to_qprog(test_ir_1, &qvm);

    for (size_t i = 0; i < 0; i++)
    {
        auto result0 = qvm.full_amplitude_measure(test_prog_2, 1000);
        std::cout << "-------------------full_amplitude_measure---------------------------" << std::endl;
        for (auto val : result0)
        {
            cout << val.first << " : " << val.second << endl;
        }
    }

    return true;
}

static bool test_full_amplitude_pmeasure(QPilotOSMachine& qvm)
{
    auto q = qvm.allocateQubits(6);
    auto c = qvm.allocateCBits(6);

    auto pmeasure_prog = QProg();
    pmeasure_prog << HadamardQCircuit(q)
        << CZ(q[1], q[5])
        << RX(q[2], PI / 4)
        << RX(q[1], PI / 4);

    for (size_t i = 0; i < 3; i++)
    {
        auto result1 = qvm.full_amplitude_pmeasure(pmeasure_prog, { 0, 1, 2 });
        std::cout << "-------------------full_amplitude_pmeasure---------------------------" << std::endl;
        for (auto val : result1)
        {
            cout << val.first << " : " << val.second << endl;
        }
    }

    return true;
}

static bool test_partial_amplitude_pmeasure(QPilotOSMachine& qvm)
{
    auto q = qvm.allocateQubits(6);
    auto c = qvm.allocateCBits(6);

    auto pmeasure_prog = QProg();
    pmeasure_prog << HadamardQCircuit(q)
        << CZ(q[1], q[5])
        << RX(q[2], PI / 4)
        << RX(q[1], PI / 4);

    for (size_t i = 0; i < 3; ++i)
    {
        auto result2 = qvm.partial_amplitude_pmeasure(pmeasure_prog, { "0", "1", "2" });
        std::cout << "-------------------partial_amplitude_pmeasure---------------------------" << std::endl;
        for (auto val : result2)
        {
            cout << val.first << " : " << val.second << endl;
        }
    }

    return true;
}

static bool test_single_amplitude_pmeasure(QPilotOSMachine& qvm)
{
    auto q = qvm.allocateQubits(6);
    auto c = qvm.allocateCBits(6);

    auto pmeasure_prog = QProg();
    pmeasure_prog << HadamardQCircuit(q)
        << CZ(q[1], q[5])
        << RX(q[2], PI / 4)
        << RX(q[1], PI / 4);

    for (size_t i = 0; i < 3; ++i)
    {
        auto result3 = qvm.single_amplitude_pmeasure(pmeasure_prog, "0");
        std::cout << "-------------------single_amplitude_pmeasure---------------------------" << std::endl;
        cout << "0" << " : " << result3 << endl;
    }

    return true;
}

static bool test_noise_measure(QPilotOSMachine& qvm)
{
    auto q = qvm.allocateQubits(6);
    auto c = qvm.allocateCBits(6);

    auto measure_prog = QProg();
    measure_prog << HadamardQCircuit(q)
        << CZ(q[2], q[3])
        << Measure(q[0], c[0]);

    for (size_t i = 0; i < 3; ++i)
    {
        qvm.set_noise_model(NOISE_MODEL::BIT_PHASE_FLIP_OPRATOR, { 0.01 }, { 0.02 });
        auto result4 = qvm.noise_measure(measure_prog, 100);
        std::cout << "-------------------noise_measure---------------------------" << std::endl;
        for (auto val : result4)
        {
            cout << val.first << " : " << val.second << endl;
        }
    }

    return true;
}

static bool test_get_state_fidelity(QPilotOSMachine& qvm)
{
    auto q = qvm.allocateQubits(6);
    auto c = qvm.allocateCBits(6);

    auto measure_prog = QProg();
    measure_prog << HadamardQCircuit(q)
        << CZ(q[2], q[3])
        << Measure(q[0], c[0]);

    for (size_t i = 0; i < 3; ++i)
    {
        auto result7 = qvm.get_state_fidelity(measure_prog, 1000);
        std::cout << "-------------------get_state_fidelity---------------------------" << std::endl;
        cout << "result7:" << result7 << endl;
    }

    return true;
}


TEST(QPilotOSMachine, test)
{
    bool test_val = true;
    try
    {
        QPilotOSMachine pilot_qvm("Pilot");
        //pilot_qvm.init("https://10.10.10.61:10080", true, "qcloud", "qcloud");
        pilot_qvm.init("https://10.9.12.9:10080", true, "qcloud", "qcloud");
        //pilot_qvm.init("https://10.10.8.36:10080", true, "qcloud", "qcloud");
        //pilot_qvm.init("https://10.9.11.172:10080", true, "qcloud", "qcloud");

        for (size_t i = 0; i < 1; ++i)
        {
            /*test_val = test_full_amplitude_measure(pilot_qvm);
            test_val = test_val && test_full_amplitude_pmeasure(pilot_qvm);
            test_val = test_val && test_partial_amplitude_pmeasure(pilot_qvm);
            test_val = test_val && test_single_amplitude_pmeasure(pilot_qvm);
            test_val = test_val && test_noise_measure(pilot_qvm);*/
            test_val = test_val && test_real_chip_measure(pilot_qvm);
            //test_val = test_val && test_real_chip_measure_async(pilot_qvm);
            //test_val = test_val && test_real_chip_query_task(pilot_qvm);
            //test_val = test_val && test_real_chip_query_compile_info(pilot_qvm);
            //test_val = test_val && test_real_chip_expectation_async(pilot_qvm);
            //test_val = test_val && test_real_chip_expectation(pilot_qvm);

            if (!test_val) {
                break;
            }
        }

        pilot_qvm.finalize();
    }
    catch (const std::exception& e)
    {
        std::cout << "Got a exception: " << e.what() << endl;
        test_val = false;
    }
    catch (...)
    {
        std::cout << "Got an unknow exception: " << endl;
        test_val = false;
    }

    ASSERT_TRUE(test_val);
}

#endif

#endif
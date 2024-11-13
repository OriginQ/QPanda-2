#include <atomic>
#include <chrono>
#include <thread>
#include <functional>
#include <ciso646>
#include "QPanda.h"
#include "gtest/gtest.h"

#ifdef USE_EXTENSION

#include "Extensions/Extensions.h"
#include "Extensions/PilotOSMachine/QPilotMachine.h"
#include "Extensions/PilotOSMachine/Def.h"
#include "Extensions/PilotOSMachine/JsonParser.h"
#include "Extensions/PilotOSMachine/JsonBuilder.h"

#if defined(USE_CURL)

#define SUCCESS_STATE "3"
#define FAILED_STATE "4"
#define CANCELED_STATE "35"
std::string Task_Id = "7B819848EF504D53B1E00FB1A1A51694"; /* 请输入需要查询的对应任务id */
using namespace std;
using namespace PilotQVM;
using namespace QPanda;
using namespace rapidjson;

namespace PilotQVM {
    struct PilotNoiseParams;
    struct PilotTaskQueryResult;
    class QPilotMachine;
}

struct EmComputeConf {
    uint16_t m_qem_base_method{ 0 };
    uint16_t m_qem_method{ 0 };
    uint32_t m_nl_shots{ 256 };
    uint32_t m_pattern{ 0 };
    uint32_t m_samples{ 10 };
    uint32_t m_qem_samples{ 300 };
    uint32_t m_qem_shots{ 256 };
    std::string m_file{ "common_generator.py" };
    std::string m_noise_model_file;

    std::vector<uint32_t> m_depths{ 2, 4, 8, 16, 32 };
    std::vector<double> m_noise_strength{ 0.0, 1.0, 2.0 };
    std::vector<std::string> m_expectations;
};

struct CaseConf {
	bool m_default{true};
	uint32_t m_loop{ 1 };
	uint32_t m_shots{1000};
	uint32_t m_backend_id{72};
	std::vector<std::string> m_originir;
	std::string m_hamiltonian;
	std::vector<uint32_t> m_qubits;
	std::vector<uint32_t> m_specified_block;
	EmComputeConf m_em_compute_conf;
};

struct GlobalConf {
    uint32_t m_server_port{ 10080 };
    uint32_t m_qubits{ 6 };                          /* for allocateQubits */
    bool m_output{ true };                           /* Output log! */
    std::string m_server_ip;
    std::string m_user{  };
    std::string m_pswd{  };
    std::string m_token;
    std::vector<std::string> m_cases;           /* 虽然如下map已经能够保证所有用例都能执行，但无序 */
    std::map<std::string, CaseConf> m_case_conf;     /* 每个测试用例一个originir,不能共用! */
};

GlobalConf global_conf;

static bool test_real_chip_expectation(QPilotOSMachine& qvm, const CaseConf &conf)
{
    size_t bit_num = conf.m_default ? 6 : global_conf.m_qubits;
    auto q = qvm.allocateQubits(bit_num);
    auto c = qvm.allocateCBits(bit_num);

    std::string originir = conf.m_default ? "QINIT 6\nCREG 6\nX q[0]\nBARRIER q[0], q[1]\nH q[0]\nRX q[1], (-1.570796)\nCNOT q[1], q[0]\nBARRIER q[0], q[1]\nRZ q[0], (0.226019)\nBARRIER q[0], q[1]\nCNOT q[1], q[0]\nBARRIER q[0], q[1]\nH q[0]\nRX q[1], (1.570796)" : conf.m_originir.at(0);
    QProg test_prog_1 = convert_originir_string_to_qprog(originir, &qvm);
    std::string hamiltonian = conf.m_default ? "\"\": -1.0534210769165204,\"Z0\": 0.394844363355904,\"Z1\": -0.39484436335590417,\"Z0 Z1\": -0.01124615715082114,\"X0 X1\": 0.18121046201519694" : conf.m_hamiltonian;
    int shots = conf.m_default ? 1000 : conf.m_shots;
    int backend_id = conf.m_default ? 72 : conf.m_backend_id;
    size_t loop{ 1 };
    std::vector<uint32_t> qubits{ 0, 1 };
    if (!conf.m_default) {
        qubits.clear();
        qubits = conf.m_qubits;
        loop = conf.m_loop;
    }
    std::vector<uint32_t> specified_block;
    for (auto& qubit : conf.m_specified_block) {
        specified_block.push_back(qubit);
    }

#if 0
    test_prog_1 << RY(0, -0.22606962488087734);
    std::string hamiltonian = "\"\": -1.0421749197656989,\"Z0\": -0.789688726711808,\"X0\": 0.18121046201519692";
    std::vector<uint32_t> qubits{ 0 };

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
#endif

    for (size_t test_cnt = 0; test_cnt < loop; test_cnt++)
    {
        //auto result = qvm.real_chip_expectation(test_prog_1, hamiltonian, qubits, shots, backend_id, true, true, true, specified_block);
        auto result = qvm.real_chip_expectation(test_prog_1, hamiltonian, qubits, shots, backend_id, true, true, true);
        std::cout << "------------------- get result for real_chip_exception ---------------------------" << std::endl;
        std::cout << "Expectation: " << result << std::endl;
    }

    qvm.Free_Qubits(q);
    qvm.Free_CBits(c);
    return true;
}

static bool test_real_chip_expectation_async(QPilotOSMachine& qvm, const CaseConf &conf)
{
    size_t bit_num = conf.m_default ? 6 : global_conf.m_qubits;
    auto q = qvm.allocateQubits(bit_num);
    auto c = qvm.allocateCBits(bit_num);

    std::string originir = conf.m_default ? "QINIT 6\nCREG 6\nX q[0]\nBARRIER q[0], q[1]\nH q[0]\nRX q[1], (-1.570796)\nCNOT q[1], q[0]\nBARRIER q[0], q[1]\nRZ q[0], (0.226019)\nBARRIER q[0], q[1]\nCNOT q[1], q[0]\nBARRIER q[0], q[1]\nH q[0]\nRX q[1], (1.570796)" : conf.m_originir.at(0);
    QProg test_prog_1 = convert_originir_string_to_qprog(originir, &qvm);
    std::string hamiltonian = conf.m_default ? "\"\": -1.0534210769165204,\"Z0\": 0.394844363355904,\"Z1\": -0.39484436335590417,\"Z0 Z1\": -0.01124615715082114,\"X0 X1\": 0.18121046201519694" : conf.m_hamiltonian;
    int shots = conf.m_default ? 1000 : conf.m_shots;
    int backend_id = conf.m_default ? 72 : conf.m_backend_id;
    size_t loop{ 1 };
    std::vector<uint32_t> qubits{ 0, 1 };
    if (!conf.m_default) {
        qubits.clear();
        qubits = conf.m_qubits;
        loop = conf.m_loop;
    }

#if 0
    test_prog_1 << RY(0, -0.22606962488087734);
    std::string hamiltonian = "\"\": -1.0421749197656989,\"Z0\": -0.789688726711808,\"X0\": 0.18121046201519692";
    std::vector<uint32_t> qubits{ 0 };
#else

#endif
    PilotQVM::PilotTaskQueryResult res;

	for (size_t test_cnt = 0; test_cnt < loop; ++test_cnt)
	{
		const auto task_id = qvm.async_real_chip_expectation(test_prog_1, hamiltonian, qubits, shots, backend_id);
		std::cout << "test_" << test_cnt << ": got task_id:" << task_id << " for async_real_chip_measure\n";
		std::cout << "On get result for async_real_chip_measure ...";
		/*do
		{
			qvm.query_task_state(task_id, res);
			std::this_thread::sleep_for(std::chrono::seconds(2));
			std::cout << ".";
		} while ((res.m_state != SUCCESS_STATE) && res.m_state != FAILED_STATE && (res.m_state != CANCELED_STATE));
		*/std::cout << std::endl << "Expectation: " << res.m_result_vec[0] << std::endl;
		double result;
		ErrorCode errCode;
		std::string errInfo;
		qvm.get_expectation_result(task_id, result, errCode, errInfo);
		std::cout << "Expectation: " << result << std::endl;
	}

    qvm.Free_Qubits(q);
    qvm.Free_CBits(c);
    return true;
}

static bool test_real_chip_measure_vec(QPilotOSMachine& qvm, const CaseConf &conf)
{
    size_t bit_num = conf.m_default ? 6 : global_conf.m_qubits;
    auto q = qvm.allocateQubits(bit_num);
    auto c = qvm.allocateCBits(bit_num);

    std::vector<QPanda::QProg> prog_vec;
    int shots = 1000;
    int backend_id = 72;
    size_t loop{ 1 };
    if (conf.m_default) {
        std::string originir = "QINIT 6\nCREG 6\nH q[0]\nH q[1]\nH q[2]\nH q[3]\nH q[4]\nX q[5]\nX q[0]\nX q[4]\nH q[5]\nCNOT q[0],q[5]\nX q[0]\nCNOT q[1],q[5]\nH q[0]\nH q[1]\nCNOT q[2],q[5]\nH q[2]\nCNOT q[3],q[5]\nH q[3]\nCNOT q[4],q[5]\nX q[4]\nH q[4]\nMEASURE q[0],c[0]\nMEASURE q[1],c[1]\nMEASURE q[2],c[2]\nMEASURE q[3],c[3]\nMEASURE q[4],c[4]\nMEASURE q[5],c[5]";
        auto test_prog_1 = convert_originir_string_to_qprog(originir, &qvm);
        prog_vec.push_back(test_prog_1);
        prog_vec.push_back(test_prog_1);
    }
    else {
        for (auto& originir : conf.m_originir) {
            prog_vec.push_back(convert_originir_string_to_qprog(originir, &qvm));
        }
        shots = conf.m_shots;
        backend_id = conf.m_backend_id;
        loop = conf.m_loop;
    }

	for (size_t test_cnt = 0; test_cnt < loop; test_cnt++)
	{
		auto result = qvm.real_chip_measure_vec(prog_vec, shots, backend_id);
		std::cout << "------------------- get result for real_chip_measure_vec ---------------------------" << std::endl;
		for (size_t i = 0; i < result.size(); i++) {
			std::cout << "result " << (i+1) << ":" << std::endl;
			for (const auto& i : result[i]) {
				cout << i.first << ":" << i.second << "\n";
			}
		}
		std::cout << std::endl;
	}

    qvm.Free_Qubits(q);
    qvm.Free_CBits(c);
    return true;
}

static bool test_real_chip_measure(QPilotOSMachine& qvm, const CaseConf& conf)
{
    cout << ("------------ test_real_chip_measure -------------\n");
    size_t bit_num = conf.m_default ? 6 : global_conf.m_qubits;
    auto q = qvm.allocateQubits(bit_num);
    auto c = qvm.allocateCBits(bit_num);

    std::string originir = conf.m_default ? "QINIT 6\nCREG 6\nH q[0]\nH q[1]\nH q[2]\nH q[3]\nH q[4]\nX q[5]\nX q[0]\nX q[4]\nH q[5]\nCNOT q[0],q[5]\nX q[0]\nCNOT q[1],q[5]\nH q[0]\nH q[1]\nCNOT q[2],q[5]\nH q[2]\nCNOT q[3],q[5]\nH q[3]\nCNOT q[4],q[5]\nX q[4]\nH q[4]\nMEASURE q[0],c[0]\nMEASURE q[1],c[1]\nMEASURE q[2],c[2]\nMEASURE q[3],c[3]\nMEASURE q[4],c[4]\nMEASURE q[5],c[5]" : conf.m_originir.at(0);
    auto prog = convert_originir_string_to_qprog(originir, &qvm);
    QVec qv;
    prog.get_used_qubits(qv);
    int shots = conf.m_default ? 1000 : conf.m_shots;
    int backend_id = conf.m_default ? 72 : conf.m_backend_id;
    size_t loop = conf.m_default ? 1: conf.m_loop;

    for (size_t test_cnt = 0; test_cnt < loop; test_cnt++)
    {
        auto result = qvm.real_chip_measure(prog, shots, backend_id, true, true, true, conf.m_specified_block);
        std::cout << "------------------- get result for real_chip_measure ---------------------------" << std::endl;
        for (auto j : result)
        {
            cout << j.first << ":" << j.second << endl;
        }
    }

    qvm.Free_Qubits(q);
    qvm.Free_CBits(c);
    return true;
}

static bool test_real_chip_measure_async(QPilotOSMachine& qvm, const CaseConf &conf)
{
    size_t bit_num = conf.m_default ? 6 : global_conf.m_qubits;
    auto q = qvm.allocateQubits(bit_num);
    auto c = qvm.allocateCBits(bit_num);

    std::string originir = conf.m_default ? "QINIT 6\nCREG 6\nH q[0]\nH q[1]\nH q[2]\nH q[3]\nH q[4]\nX q[5]\nX q[0]\nX q[4]\nH q[5]\nCNOT q[0],q[5]\nX q[0]\nCNOT q[1],q[5]\nH q[0]\nH q[1]\nCNOT q[2],q[5]\nH q[2]\nCNOT q[3],q[5]\nH q[3]\nCNOT q[4],q[5]\nX q[4]\nH q[4]\nMEASURE q[0],c[0]\nMEASURE q[1],c[1]\nMEASURE q[2],c[2]\nMEASURE q[3],c[3]\nMEASURE q[4],c[4]\nMEASURE q[5],c[5]" : conf.m_originir.at(0);
    auto test_prog_1 = convert_originir_string_to_qprog(originir, &qvm);
    int shots = conf.m_default ? 1000 : conf.m_shots;
    int backend_id = conf.m_default ? 72 : conf.m_backend_id;
    size_t loop = conf.m_default ? 1: conf.m_loop;
    PilotQVM::PilotTaskQueryResult res;

    for (size_t test_cnt = 0; test_cnt < loop; ++test_cnt)
    {
        const auto task_id = qvm.async_real_chip_measure(test_prog_1, shots, backend_id);
        std::cout << "get task_id :" << task_id << endl;
        std::cout << "On get result for async_real_chip_measure ...";
        do
        {
            qvm.query_task_state(task_id, res);
            std::this_thread::sleep_for(std::chrono::seconds(2));
            std::cout << std::endl << "state:" << res.m_state << std::endl;
        } while ((res.m_state != SUCCESS_STATE) && (res.m_state != FAILED_STATE) && (res.m_state != CANCELED_STATE));

        std::vector<std::map<std::string, double>> result;
        qvm.parse_task_result(res.m_result_vec, result);
        std::cout << "\nGot result:\n";
        for (const auto& res : result)
        {
            for (const auto& m : res)
            {
                std::cout << m.first << " : " << m.second << endl;
            }
        }
    }

    qvm.Free_Qubits(q);
    qvm.Free_CBits(c);
    return true;
}

static bool test_real_chip_measure_async_vec(QPilotOSMachine& qvm, const CaseConf &conf)
{
    size_t bit_num = conf.m_default ? 6 : global_conf.m_qubits;
    auto q = qvm.allocateQubits(bit_num);
    auto c = qvm.allocateCBits(bit_num);

	std::vector<std::string>ir_vec;
	int shots = 1000;
	int backend_id = 72;
    size_t loop{ 1 };
	if (conf.m_default) {
		std::string originir = conf.m_default ? "QINIT 6\nCREG 6\nH q[0]\nH q[1]\nH q[2]\nH q[3]\nH q[4]\nX q[5]\nX q[0]\nX q[4]\nH q[5]\nCNOT q[0],q[5]\nX q[0]\nCNOT q[1],q[5]\nH q[0]\nH q[1]\nCNOT q[2],q[5]\nH q[2]\nCNOT q[3],q[5]\nH q[3]\nCNOT q[4],q[5]\nX q[4]\nH q[4]\nMEASURE q[0],c[0]" : conf.m_originir.at(0);
        ir_vec.push_back(originir);
		ir_vec.push_back(originir);
	} 
    else {
		ir_vec = conf.m_originir;
		shots = conf.m_shots;
		backend_id = conf.m_backend_id;
        loop = conf.m_loop;
	}
	PilotQVM::PilotTaskQueryResult res;

	for (size_t test_cnt = 0; test_cnt < loop; ++test_cnt)
	{
		const auto task_id = qvm.async_real_chip_measure_vec(ir_vec, shots, backend_id);

		std::cout << "got task_id:" << task_id << " for async_real_chip_measure\nOn get result for async_real_chip_measure ...";
		do {
			qvm.query_task_state(task_id, res);
			std::this_thread::sleep_for(std::chrono::seconds(2));
			std::cout << std::endl << "state:" << res.m_state << std::endl;
		} while ((res.m_state != SUCCESS_STATE) && (res.m_state != FAILED_STATE) && (res.m_state != CANCELED_STATE));

        //std::vector<std::map<std::string, uint64_t>> result;
        std::vector<std::map<std::string, double>> result;
        qvm.parse_task_result(res.m_result_vec, result);
        for (const auto& res : result)
        {
            std::cout << "\nGot result:\n";
            for (const auto& m : res){
                std::cout << m.first << ": " << m.second << endl;
            }
        }
    }

    qvm.Free_Qubits(q);
    qvm.Free_CBits(c);
    return true;
}

static bool test_noise_learning_async(QPilotOSMachine& qvm, const CaseConf& conf)
{
    size_t bit_num = 6;
    auto q = qvm.allocateQubits(bit_num);
    auto c = qvm.allocateCBits(bit_num);

#if 0
    std::string originir = conf.m_default ? "QINIT 6\nCREG 6\nH q[0]\nH q[1]\nH q[2]\nH q[3]\nH q[4]\nX q[5]\nX q[0]\nX q[4]\nH q[5]\nCNOT q[0],q[5]\nX q[0]\nCNOT q[1],q[5]\nH q[0]\nH q[1]\nCNOT q[2],q[5]\nH q[2]\nCNOT q[3],q[5]\nH q[3]\nCNOT q[4],q[5]\nX q[4]\nH q[4]\nMEASURE q[0],c[0]\nMEASURE q[1],c[1]\nMEASURE q[2],c[2]\nMEASURE q[3],c[3]\nMEASURE q[4],c[4]\nMEASURE q[5],c[5]" : conf.m_originir.at(0);
    std::string script = "";
    int shots = conf.m_default ? 1000 : conf.m_shots;
    //int samples = conf.m_default ? 10 : conf.m_samples;
    std::vector<uint32_t> temp;
    //std::vector<uint32_t> circuitDepthList = conf.m_default ? temp : conf.m_circuit_list;
    for (size_t test_cnt = 0; test_cnt < 1/*0000*/; test_cnt++)
    {
        std::string task_id = qvm.async_noise_learning(true, isEMCompute, script, originir, shots, samples, circuitDepthList);
        std::cout << "get task_id for noise em_compute, task_id: " << task_id << endl;
    }
#endif

    qvm.Free_Qubits(q);
    qvm.Free_CBits(c);
    return true;
}

static bool test_noise_learning(QPilotOSMachine& qvm, const CaseConf& conf)
{
    size_t bit_num = 6;
    auto q = qvm.allocateQubits(bit_num);
    auto c = qvm.allocateCBits(bit_num);

    std::string param_json;
    size_t loop{ 1 };
    if (conf.m_default) {
        param_json = R"({ "ir": "QINIT 3\n CREG 3\n RY q[0],(1.43)\n RY q[1],(2.9)\n RY q[2],(1.47)\n CONTROL q[0]\n X q[1]\n ENDCONTROL\n RY q[0], (1.1)\n RY q[1],(-0.93)\n CONTROL q[0]\n X q[2]\n ENDCONTROL\n RY q[2],(-0.098)\n CONTROL q[1]\n X q[2]\n ENDCONTROL\n RY q[2],(-0.098)\n CONTROL q[0]\n X q[2]\n ENDCONTROL\n RY q[2], (-0.29)\n",
"pattern": 0,
"file": "common_generator.py",
"noise_model_file": "common_PER_experiment.pkl",
"samples": 10,
"nl_shots": 256,
"depths": [2, 4, 8],
"qem_base_method": 0 })";
    }
    else {
        JsonMsg::JsonBuilder json_builder;
        json_builder.add_member("ir", conf.m_originir.at(0));
        json_builder.add_member("pattern", conf.m_em_compute_conf.m_pattern);
        json_builder.add_member("file", conf.m_em_compute_conf.m_file);
        json_builder.add_member("noise_model_file", conf.m_em_compute_conf.m_noise_model_file);
        json_builder.add_member("samples", conf.m_em_compute_conf.m_samples);
        json_builder.add_member("nl_shots", conf.m_em_compute_conf.m_nl_shots);
        json_builder.add_array("depths", conf.m_em_compute_conf.m_depths);
        json_builder.add_member("qem_base_method", conf.m_em_compute_conf.m_qem_base_method);
        param_json = json_builder.get_json_str(true);
        loop = conf.m_loop;
    }

    for (size_t test_cnt = 0; test_cnt < loop; test_cnt++)
    {
        std::string task_id = qvm.noise_learning(param_json);
        std::cout << "get task_id for noise em_compute, task_id: " << task_id << endl;
    }

    qvm.Free_Qubits(q);
    qvm.Free_CBits(c);
    return true;
}

static bool test_em_compute(QPilotOSMachine& qvm, const CaseConf& conf)
{
    size_t bit_num = 6;
    auto q = qvm.allocateQubits(bit_num);
    auto c = qvm.allocateCBits(bit_num);

	std::string param_json;
    size_t loop{ 1 };
	if (conf.m_default) {
		param_json = R"({"ir":"QINIT 72\nCREG 72\nCZ q[45],q[46]\nCZ q[52],q[53]\nCZ q[46],q[52]\nCZ q[53],q[54]\nCZ q[54],q[48]",
"pattern": 0,
"file": "common_generator.py",
"samples": 40,
"nl_shots": 256,
"depths": [2, 4, 8, 16, 32],
"expectations": ["IZI", "IIZ"],
"qem_method": 0,
"qem_base_method": 0,
"noise_model_file": "",
"qem_samples": 300,
"qem_shots": 256,
"noise_strength": [0.0, 1.0, 2.0] })";
    }
    else {
        JsonMsg::JsonBuilder json_builder;
        json_builder.add_member("ir", conf.m_originir.at(0));
        json_builder.add_member("pattern", conf.m_em_compute_conf.m_pattern);
        json_builder.add_member("file", conf.m_em_compute_conf.m_file);
        json_builder.add_member("noise_model_file", conf.m_em_compute_conf.m_noise_model_file);
        json_builder.add_member("samples", conf.m_em_compute_conf.m_samples);
        json_builder.add_member("nl_shots", conf.m_em_compute_conf.m_nl_shots);
        json_builder.add_array("depths", conf.m_em_compute_conf.m_depths);
        json_builder.add_member("qem_base_method", conf.m_em_compute_conf.m_qem_base_method);

        json_builder.add_array("expectations", conf.m_em_compute_conf.m_expectations);
        json_builder.add_member("qem_method", conf.m_em_compute_conf.m_qem_method);
        json_builder.add_member("qem_samples", conf.m_em_compute_conf.m_qem_samples);
        json_builder.add_member("qem_shots", conf.m_em_compute_conf.m_qem_shots);
        json_builder.add_array("noise_strength", conf.m_em_compute_conf.m_noise_strength);
        param_json = json_builder.get_json_str(true);
        loop = conf.m_loop;
    }

    for (size_t test_cnt = 0; test_cnt < loop; test_cnt++)
    {
        auto result = qvm.em_compute(param_json);
        std::string output = "[";
        for (auto& ele : result) {
            output += std::to_string(ele) + " ";
        }
        if (output.at(output.size() - 1) == ' ') {
            output.erase(output.size() - 1);
        }
        cout << output + "]" << endl;
    }

    qvm.Free_Qubits(q);
    qvm.Free_CBits(c);
    return true;
}

static bool test_em_compute_async(QPilotOSMachine& qvm, const CaseConf& conf)
{
    size_t bit_num = 6;
    auto q = qvm.allocateQubits(bit_num);
    auto c = qvm.allocateCBits(bit_num);

	std::string param_json;
    size_t loop{ 1 };
	if (conf.m_default) {
		param_json = R"({"ir":"QINIT 72\nCREG 72\nCZ q[45],q[46]\nCZ q[52],q[53]\nCZ q[46],q[52]\nCZ q[53],q[54]\nCZ q[54],q[48]",
"pattern": 0,
"file": "common_generator.py",
"samples": 40,
"nl_shots": 256,
"depths": [2, 4, 8, 16, 32],
"expectations": ["IZI", "IIZ"],
"qem_method": 0,
"qem_base_method": 0,
"noise_model_file": "",
"qem_samples": 300,
"qem_shots": 256,
"noise_strength": [0.0, 1.0, 2.0] })";
    }
    else {
        JsonMsg::JsonBuilder json_builder;
        json_builder.add_member("ir", conf.m_originir.at(0));
        json_builder.add_member("pattern", conf.m_em_compute_conf.m_pattern);
        json_builder.add_member("file", conf.m_em_compute_conf.m_file);
        json_builder.add_member("noise_model_file", conf.m_em_compute_conf.m_noise_model_file);
        json_builder.add_member("samples", conf.m_em_compute_conf.m_samples);
        json_builder.add_member("nl_shots", conf.m_em_compute_conf.m_nl_shots);
        json_builder.add_array("depths", conf.m_em_compute_conf.m_depths);
        json_builder.add_member("qem_base_method", conf.m_em_compute_conf.m_qem_base_method);

        json_builder.add_array("expectations", conf.m_em_compute_conf.m_expectations);
        json_builder.add_member("qem_method", conf.m_em_compute_conf.m_qem_method);
        json_builder.add_member("qem_samples", conf.m_em_compute_conf.m_qem_samples);
        json_builder.add_member("qem_shots", conf.m_em_compute_conf.m_qem_shots);
        json_builder.add_array("noise_strength", conf.m_em_compute_conf.m_noise_strength);
        param_json = json_builder.get_json_str(true);
        loop = conf.m_loop;
    }

    for (size_t test_cnt = 0; test_cnt < loop; test_cnt++)
    {
        std::string task_id = qvm.async_em_compute(param_json);
        std::cout << "get task_id for noise em_compute, task_id: " << task_id << endl;

        PilotQVM::PilotTaskQueryResult res;
        do {
            qvm.query_task_state(task_id, res);
            std::this_thread::sleep_for(std::chrono::seconds(2));
            std::cout << "state:" << res.m_state << std::endl;
        } while ((res.m_state != SUCCESS_STATE) && (res.m_state != FAILED_STATE) && (res.m_state != CANCELED_STATE));
        cout << "[";
        for (auto& ele : res.m_result_vec) {
            cout << ele << " ";
        }
        cout << "]" << endl;
    }

    qvm.Free_Qubits(q);
    qvm.Free_CBits(c);
    return true;
}

static bool test_real_chip_query_task(QPilotOSMachine& qvm, const CaseConf &conf)
{
    PilotQVM::PilotTaskQueryResult res;
    std::string compile_prog;
    bool with_compensate = true;
    for (size_t test_cnt = 0; test_cnt < 1/*0000*/; ++test_cnt)
    {
        do
        {
            qvm.query_task_state(Task_Id, res);
            if (res.m_errCode != 0) {
                std::cout << "Query error!" << std::endl << "errCode: " << res.m_errCode << std::endl << "errInfo: " << res.m_errInfo << std::endl;
                return false;
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
            std::cout << ".";
        } while ((res.m_state != SUCCESS_STATE) && (res.m_state != FAILED_STATE) && (res.m_state != CANCELED_STATE));

        std::vector<std::map<std::string, double>> result;
        qvm.parse_task_result(res.m_result_vec, result);
        std::cout << "\nGot result:\n";
        for (auto res1 : result)
        {
            for (const auto& val : res1) {
                cout << val.first << ": " << val.second << "\n";
            }
        }
    }

    return true;
}

static bool test_real_chip_query_compile_info(QPilotOSMachine& qvm)
{
    PilotQVM::PilotTaskQueryResult res;
    std::string compile_prog;
    bool with_compensate = true;
    for (size_t test_cnt = 0; test_cnt < 1/*0000*/; ++test_cnt)
    {
        if (qvm.query_compile_prog(Task_Id, compile_prog, with_compensate))
        {
            std::cout << "task compile prog: " << std::endl << compile_prog << std::endl;
        }
    }

    return true;
}

static bool test_full_amplitude_measure(QPilotOSMachine& qvm, const CaseConf &conf)
{
    const string test_ir_1 = conf.m_default ? "QINIT 12\nCREG 12\nH q[0]\nH q[1]\nH q[2]\nH q[3]\nH q[4]\nH q[5]\nH q[6]\nH q[7]\nH q[8]\nH q[9]\nH q[10]\nH q[11]\nCNOT q[0],q[1]\nCNOT q[1],q[2]\nCNOT q[2],q[3]\nCNOT q[3],q[4]\nCNOT q[4],q[5]\nCNOT q[5],q[6]\nCNOT q[6],q[7]\nCNOT q[7],q[8]\nCNOT q[8],q[9]\nCNOT q[9],q[10]\nCNOT q[10],q[11]\nMEASURE q[0],c[0]\nMEASURE q[1],c[1]\nMEASURE q[2],c[2]\nMEASURE q[3],c[3]\nMEASURE q[4],c[4]\nMEASURE q[5],c[5]\nMEASURE q[6],c [6]\nMEASURE q[7],c[7]\nMEASURE q[8],c[8]\nMEASURE q[9],c[9]\nMEASURE q[10],c[10]\nMEASURE q[11],c[11]" : conf.m_originir.at(0);
    QProg test_prog_2 = convert_originir_string_to_qprog(test_ir_1, &qvm);

    for (size_t i = 0; i < 1; i++)
    {
        auto result0 = qvm.full_amplitude_measure(test_prog_2, conf.m_default ? 1000 : conf.m_shots);
        std::cout << "-------------------full_amplitude_measure---------------------------" << std::endl;
        for (auto val : result0)
        {
            cout << val.first << " : " << val.second << endl;
        }
    }

    return true;
}

static bool test_full_amplitude_pmeasure(QPilotOSMachine& qvm, const CaseConf &conf)
{
    size_t bit_num = conf.m_default ? 6 : global_conf.m_qubits;
    auto q = qvm.allocateQubits(bit_num);
    auto c = qvm.allocateCBits(bit_num);

    const std::string originir = conf.m_default ? "QINIT 6\nCREG 6\nH q[0]\nH q[1]\nH q[2]\nH q[3]\nH q[4]\nH q[5]\nCZ q[1],q[5]\nRX q[2],(0.78539816)\nRX q[1],(0.78539816)" : conf.m_originir.at(0);
    auto prog = convert_originir_string_to_qprog(originir, &qvm);
    Qnum qvec{ 0, 1, 2 };
    if (!conf.m_default) {
        qvec.clear();
        for (auto& ele : conf.m_qubits) {
            qvec.push_back(ele);
        }
    }

    for (size_t i = 0; i < 3; i++)
    {
        auto result1 = qvm.full_amplitude_pmeasure(prog, qvec);
        std::cout << "-------------------full_amplitude_pmeasure---------------------------" << std::endl;
        for (auto val : result1)
        {
            cout << val.first << " : " << val.second << endl;
        }
    }

    qvm.Free_Qubits(q);
    qvm.Free_CBits(c);
    return true;
}

static bool test_partial_amplitude_pmeasure(QPilotOSMachine& qvm, const CaseConf &conf)
{
    size_t bit_num = conf.m_default ? 6 : global_conf.m_qubits;
    auto q = qvm.allocateQubits(bit_num);
    auto c = qvm.allocateCBits(bit_num);

    const std::string originir = conf.m_default ? "QINIT 6\nCREG 6\nH q[0]\nH q[1]\nH q[2]\nH q[3]\nH q[4]\nH q[5]\nCZ q[1],q[5]\nRX q[2],(0.78539816)\nRX q[1],(0.78539816)" : conf.m_originir.at(0);
    auto prog = convert_originir_string_to_qprog(originir, &qvm);
    std::vector<std::string> qvec{ "0", "1", "2" };
    if (!conf.m_default) {
        qvec.clear();
        for (auto& ele : conf.m_qubits) {
            qvec.push_back(std::to_string(ele));
        }
    }

    for (size_t i = 0; i < 3; ++i)
    {
        auto result2 = qvm.partial_amplitude_pmeasure(prog, qvec);
        std::cout << "-------------------partial_amplitude_pmeasure---------------------------" << std::endl;
        for (auto val : result2)
        {
            cout << val.first << " : " << val.second << endl;
        }
    }

    qvm.Free_Qubits(q);
    qvm.Free_CBits(c);
    return true;
}

static bool test_single_amplitude_pmeasure(QPilotOSMachine& qvm, const CaseConf &conf)
{
    size_t bit_num = conf.m_default ? 6 : global_conf.m_qubits;
    auto q = qvm.allocateQubits(bit_num);
    auto c = qvm.allocateCBits(bit_num);

    const std::string originir = conf.m_default ? "QINIT 6\nCREG 6\nH q[0]\nH q[1]\nH q[2]\nH q[3]\nH q[4]\nH q[5]\nCZ q[1],q[5]\nRX q[2],(0.78539816)\nRX q[1],(0.78539816)" : conf.m_originir.at(0);
    auto prog = convert_originir_string_to_qprog(originir, &qvm);
    std::string qbit{ "0" };
    if (!conf.m_default && conf.m_qubits.size() > 0) {
        qbit = std::to_string(conf.m_qubits.at(0));
    }

    for (size_t i = 0; i < 3; ++i)
    {
        auto result3 = qvm.single_amplitude_pmeasure(prog, qbit);
        std::cout << "-------------------single_amplitude_pmeasure---------------------------" << std::endl;
        cout << "0" << " : " << result3 << endl;
    }

    qvm.Free_Qubits(q);
    qvm.Free_CBits(c);
    return true;
}

static bool test_noise_measure(QPilotOSMachine& qvm, const CaseConf &conf)
{
    size_t bit_num = conf.m_default ? 6 : global_conf.m_qubits;
    auto q = qvm.allocateQubits(bit_num);
    auto c = qvm.allocateCBits(bit_num);

    std::string originir = conf.m_default ? "QINIT 6\nCREG 6\nH q[0]\nH q[1]\nH q[2]\nH q[3]\nH q[4]\nH q[5]\nCZ q[2],q[3]\nMEASURE q[0],c[0]" : conf.m_originir.at(0);
    auto prog = convert_originir_string_to_qprog(originir, &qvm);
    int shots = conf.m_default ? 100 : conf.m_shots;
    for (size_t i = 0; i < 3; ++i)
    {
        qvm.set_noise_model(NOISE_MODEL::BIT_PHASE_FLIP_OPRATOR, { 0.01 }, { 0.02 });
        auto result4 = qvm.noise_measure(prog, shots);
        std::cout << "-------------------noise_measure---------------------------" << std::endl;
        for (auto val : result4)
        {
            cout << val.first << " : " << val.second << endl;
        }
    }

    qvm.Free_Qubits(q);
    qvm.Free_CBits(c);
    return true;
}

static bool test_get_state_fidelity(QPilotOSMachine& qvm, const CaseConf &conf)
{
    size_t bit_num = conf.m_default ? 2 : global_conf.m_qubits;
    auto q = qvm.allocateQubits(bit_num);
    auto c = qvm.allocateCBits(bit_num);

    std::string originir = conf.m_default ? "QINIT 2\nCREG 2\nH q[0]\nH q[1]\nCZ q[0],q[1]\nMEASURE q[0],c[0]" : conf.m_originir.at(0);
    auto prog = convert_originir_string_to_qprog(originir, &qvm);
    int shots = conf.m_default ? 1000 : conf.m_shots;
    PilotQVM::PilotTaskQueryResult res;
    for (size_t i = 0; i < 1; ++i)
    {
        //auto result7 = qvm.get_state_fidelity(prog, shots);
        //auto task_id = qvm.async_real_chip_QST_fidelity(prog, shots, 72);
        auto task_id = qvm.async_real_chip_QST_density(prog, shots, 72);
        //std::vector<std::map<std::string, double>> result;
        double result;
        ErrorCode errCode;
        std::string errInfo;
        qvm.get_qst_fidelity_result(task_id, result, errCode, errInfo);

        std::cout << result << std::endl;
        /*for (const auto& res : result)
        {
            std::cout << "\nGot result:\n";
            for (const auto& m : res)
            {
                std::cout << m.first << " : " << m.second << endl;
            }
        }*/

        /*std::cout << "-------------------get_state_fidelity---------------------------" << std::endl;
        cout << "taskId:" << task_id << endl;
        std::cout << "get task_id :" << task_id << endl;
        std::cout << "On get result for async_real_chip_measure ...";
        do
        {
            qvm.query_task_state(task_id, res);
            std::this_thread::sleep_for(std::chrono::seconds(2));
            std::cout << std::endl << "state:" << res.m_state << std::endl;
        } while ((res.m_state != SUCCESS_STATE) && (res.m_state != FAILED_STATE) && (res.m_state != CANCELED_STATE));

        std::cout << "------result------------\n" << res.m_resultJson;*/

        /*std::vector<std::map<std::string, double>> result;
        qvm.parse_task_result(res.m_result_vec, result);
        for (const auto& res : result)
        {
            std::cout << "\nGot result:\n";
            for (const auto& m : res)
            {
                std::cout << m.first << " : " << m.second << endl;
            }
        }*/
    }

    qvm.Free_Qubits(q);
    qvm.Free_CBits(c);
    return true;
}

static bool test_parse(QPilotOSMachine& qvm, const CaseConf &conf)
{
    //RandomCircuit rc;
    /*CPUQVM qm;
    auto q = qvm.allocateQubits(20);
    QProg random_prog = random_qprog(2, 10, 100, qm, q);
    std::cout << random_prog;*/
    //std::vector<std::map<std::string, double>> result;
    //std::string res;
    //qvm.parse_task_result(res, result);

    //for (size_t i = 0; i < result.size(); i++) {
    //    std::cout << "\nGot result" << i << ":\n";
    //    for (const auto& val : result[i]) {
    //        cout << val.first << ": " << val.second << "\n";
    //    }
    //}

    //std::string instruction = "[[{\"RPhi\":[52,12.6629986498404,540]},{\"RPhi\":[54,136.30560716587423,540]},{\"Measure\":[[65,66,59,60,48,54,53,52,45,46,40],570]}]]";
    //std::vector<uint32_t> measure_qubits;
    //JsonMsg::JsonParser jp;
    //jp.load_json(instruction);
    //auto &doc = jp.get_json_obj();
    //for (auto &v: doc.GetArray())
    //{
    //    for (auto &w: v.GetArray())
    //    {
    //        if (!w.HasMember("Measure")) {
    //            continue;
    //        }
    //        else
    //        {
    //            //std::vector<std::vector<std::uint32_t>> ary = w.GetArray();

    //            auto& ary = w["Measure"].GetArray();
    //            for (auto &it: ary)
    //            {
    //                for (auto &a : it.GetArray())
    //                {
    //                    measure_qubits.push_back(a.GetInt());

    //                }
    //            }
    //        }

    //    }
    //}

    return true;
}

void resolv_conf()
{
    ifstream infile("../../../test/Core/QPilotOSMachine.json");
    if (!infile.is_open())
    {
        std::cout << "Failed to open QPilotOSMachine.json" << std::endl;
        return;
    }
    std::stringstream ss;
    ss << infile.rdbuf();
    std::string context{ ss.str() };

    JsonMsg::JsonParser parser;
    if (!parser.load_json(context))
    {
        cout << "Failed to load_json, context: " << context << endl;
        return;
    }
    global_conf.m_server_ip = parser.get_string("server_ip");
    global_conf.m_server_port = parser.get_uint32("server_port");
    global_conf.m_qubits = parser.get_uint32("qubits");
    global_conf.m_output = parser.get_bool("output");
    global_conf.m_user = parser.get_string("user");
    global_conf.m_pswd = parser.get_string("pswd");
    global_conf.m_token = parser.get_string("token");

    if (global_conf.m_server_ip == "" or global_conf.m_server_port <= 0) {
        std::cerr << "'server_ip' or 'server_port' is absent!" << std::endl;
        std::this_thread::sleep_for(2ms);
        exit(0);
    }
    if (global_conf.m_user == "" or global_conf.m_token == "") {
        std::cerr << "'user' or 'token' is absent!" << std::endl;
        std::this_thread::sleep_for(2ms);
        exit(0);
    }

    parser.get_array("perform", global_conf.m_cases);

    auto &doc = parser.get_json_obj();
    for (auto &one_case : global_conf.m_cases)
    {
        auto &case_obj = doc[one_case.c_str()];
        CaseConf conf;

		if (case_obj.HasMember("default") and case_obj["default"].IsBool()) {
			conf.m_default = case_obj["default"].GetBool();
		}
		if (case_obj.HasMember("loop") and case_obj["loop"].IsUint()) {
			conf.m_loop = case_obj["loop"].GetUint();
		}
		if (case_obj.HasMember("shots") and case_obj["shots"].IsUint()) {
			conf.m_shots = case_obj["shots"].GetUint();
			conf.m_em_compute_conf.m_nl_shots = conf.m_shots;
		}
		if (case_obj.HasMember("originir")) {
			if (case_obj["originir"].IsString()) {
				conf.m_originir.push_back(case_obj["originir"].GetString());
			}
			else if (case_obj["originir"].IsArray()) {
				for (auto& ele : case_obj["originir"].GetArray()) {
					if (ele.IsString()) {
						conf.m_originir.push_back(ele.GetString());
					} else {
						std::cerr << "Field 'originir' type is error!" << std::endl;
					}
				}
			}
			else {
				std::cerr << "Field 'originir' type is error!" << std::endl;
			}
		}
		if (case_obj.HasMember("backend_id") and case_obj["backend_id"].IsUint()) {
			conf.m_backend_id = case_obj["backend_id"].GetUint();
		}
		if (case_obj.HasMember("specified_block") and case_obj["specified_block"].IsArray()) {
			for (auto& qubit : case_obj["specified_block"].GetArray()) {
				if (qubit.IsUint()) {
					conf.m_specified_block.push_back(qubit.GetUint());
				}
			}
		}
		if(case_obj.HasMember("qubits") and case_obj["qubits"].IsArray())
		{
			for(auto &qubit : case_obj["qubits"].GetArray())
			{
				if (qubit.IsUint()) {
					conf.m_qubits.push_back(qubit.GetUint());
				}
				else {
					std::cout << "Field 'qubits' type is error!" << std::endl;
				}
			}
		}
		if(case_obj.HasMember("hamiltonian") and case_obj["hamiltonian"].IsString())
		{
			conf.m_hamiltonian = case_obj["hamiltonian"].GetString();
		}
		if (case_obj.HasMember("samples") and case_obj["samples"].IsUint()) {
			conf.m_em_compute_conf.m_samples = case_obj["samples"].GetUint();
		}
		if (case_obj.HasMember("nl_shots") and case_obj["nl_shots"].IsUint()) {
			conf.m_em_compute_conf.m_nl_shots = case_obj["nl_shots"].GetUint();
		}
		if (case_obj.HasMember("pattern") and case_obj["pattern"].IsUint()) {
			conf.m_em_compute_conf.m_pattern = case_obj["pattern"].GetUint();
		}
		if (case_obj.HasMember("qem_base_method") and case_obj["qem_base_method"].IsUint()) {
			conf.m_em_compute_conf.m_qem_base_method = case_obj["qem_base_method"].GetUint();
		}
		if (case_obj.HasMember("file") and case_obj["file"].IsString()) {
			conf.m_em_compute_conf.m_file = case_obj["file"].GetString();
		}
		if (case_obj.HasMember("noise_model_file") and case_obj["noise_model_file"].IsString()) {
			conf.m_em_compute_conf.m_noise_model_file = case_obj["noise_model_file"].GetString();
		}
		if (case_obj.HasMember("depths") and case_obj["depths"].IsArray()) {
			conf.m_em_compute_conf.m_depths.clear();
			for (auto& depth : case_obj["depths"].GetArray()) {
				if (depth.IsUint()) {
					conf.m_em_compute_conf.m_depths.push_back(depth.GetUint());
				}
				else {
					std::cerr << "Field 'depths' type is error!" << std::endl;
				}
			}
		}

        if (case_obj.HasMember("qem_method") and case_obj["qem_method"].IsUint()) {
            conf.m_em_compute_conf.m_qem_method = case_obj["qem_method"].GetUint();
        }
        if (case_obj.HasMember("qem_samples") and case_obj["qem_samples"].IsUint()) {
            conf.m_em_compute_conf.m_qem_samples = case_obj["qem_samples"].GetUint();
        }
        if (case_obj.HasMember("qem_shots") and case_obj["qem_shots"].IsUint()) {
            conf.m_em_compute_conf.m_qem_shots = case_obj["qem_shots"].GetUint();
        }
        if (case_obj.HasMember("expectations") and case_obj["expectations"].IsArray()) {
            conf.m_em_compute_conf.m_expectations.clear();
            for (auto& expectation : case_obj["expectations"].GetArray()) {
                if (expectation.IsString()) {
                    conf.m_em_compute_conf.m_expectations.push_back(expectation.GetString());
                }
                else {
                    std::cerr << "Field 'expectations' type is error!" << std::endl;
                }
            }
        }
        if (case_obj.HasMember("noise_strength") and case_obj["noise_strength"].IsArray()) {
            conf.m_em_compute_conf.m_noise_strength.clear();
            for (auto& noise_strength_ele : case_obj["noise_strength"].GetArray()) {
                if (noise_strength_ele.IsDouble()) {
                    conf.m_em_compute_conf.m_noise_strength.push_back(noise_strength_ele.GetDouble());
                }
                else {
                    std::cerr << "Field 'noise_strength' type is error!" << std::endl;
                }
            }
        }
        global_conf.m_case_conf.emplace(one_case, conf);
    }
}

TEST(QPilotOSMachine, test)
{
    bool test_val = true;
    /*{
        MPSQVM machine;
        machine.setConfigure({ 64,64 });
        machine.init();

        auto q = machine.qAllocMany(63);
        auto c = machine.cAllocMany(5);
        QProg prog;
        prog << H(q[0]) << RPhi(q[45], 0.3, 0.5) << Measure(q[45], c[0]);
        auto result2 = machine.runWithConfiguration(prog, 1000);
        cout << "ssssssssss" << endl;
    }*/

	try
	{
		resolv_conf();
		QPilotOSMachine pilot_qvm("Pilot");
		/* init接口后两个参数为司南页面服务的账号和密码，示例中为测试账号，为方便任务管理建议使用个人账号 */
		//pilot_qvm.init("https://10.9.12.17:10080", true, "9B89981FF7244B3D83C9CBF0A65E956C");
		pilot_qvm.init(std::string("https://") + global_conf.m_server_ip + ":" + std::to_string(global_conf.m_server_port), global_conf.m_output, global_conf.m_token);

        std::map<std::string, std::function<bool(QPilotOSMachine &, const CaseConf &)>> name2func;
        name2func.emplace("full_amplitude_measure", test_full_amplitude_measure);
        name2func.emplace("full_amplitude_pmeasure", test_full_amplitude_pmeasure);
        name2func.emplace("partial_amplitude_pmeasure", test_partial_amplitude_pmeasure);
        name2func.emplace("single_amplitude_pmeasure", test_single_amplitude_pmeasure);
        name2func.emplace("noise_measure", test_noise_measure);
        name2func.emplace("real_chip_measure", test_real_chip_measure);
        name2func.emplace("real_chip_measure_vec", test_real_chip_measure_vec);
        name2func.emplace("real_chip_measure_async", test_real_chip_measure_async);
        name2func.emplace("real_chip_measure_async_vec", test_real_chip_measure_async_vec);
        name2func.emplace("noise_learning", test_noise_learning);
        name2func.emplace("noise_learning_async", test_noise_learning_async);
        name2func.emplace("em_compute", test_em_compute);
        name2func.emplace("em_compute_async", test_em_compute_async);
        name2func.emplace("real_chip_expectation", test_real_chip_expectation);
        name2func.emplace("real_chip_expectation_async", test_real_chip_expectation_async);
        name2func.emplace("get_state_fidelity", test_get_state_fidelity);
        name2func.emplace("test_parse", test_parse);

        for (auto& one_case : global_conf.m_cases) {
            if (name2func.find(one_case) == name2func.end()) {
                test_val = false;
                break;
            }
            test_val = name2func[one_case](pilot_qvm, global_conf.m_case_conf[one_case]);
        }
#if 0
        for (size_t i = 0; i < 1; ++i)
        {
            test_val = test_full_amplitude_measure(pilot_qvm);
            test_val = test_full_amplitude_pmeasure(pilot_qvm);
            test_val = test_partial_amplitude_pmeasure(pilot_qvm);
            test_val = test_single_amplitude_pmeasure(pilot_qvm);
            test_val = test_noise_measure(pilot_qvm); */
                /* 以上五个接口为集群计算，仅适用于61环境测试 */

                test_val = test_real_chip_measure(pilot_qvm);
            test_val = test_real_chip_measure_async(pilot_qvm);
            test_val = test_real_chip_expectation_async(pilot_qvm);
            test_val = test_real_chip_expectation(pilot_qvm);
            /* 以上四个接口为真实芯片计算接口，可在61和12.9环境测试 */

            /*test_val = test_val && test_real_chip_query_task(pilot_qvm);
            test_val = test_val && test_real_chip_query_compile_info(pilot_qvm);*/
            /* 以上两个接口为查询任务结果和编译信息接口 */

            if (!test_val) {
                break;
            }
        }
#endif
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

#endif    //use_curl

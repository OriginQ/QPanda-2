#include <atomic>
#include <chrono>
#include <string>
#include "QPanda.h"
#include <algorithm>
#include "gtest/gtest.h"

#include "ThirdParty/rapidjson/writer.h"
#include "ThirdParty/rapidjson/document.h"
#include "ThirdParty/rapidjson/stringbuffer.h"

#include "ThirdParty/Eigen/Dense"
#include "ThirdParty/Eigen/Sparse"
#include <EigenUnsupported/Eigen/KroneckerProduct>

#define CHECK_RUN_TIME_BEGIN                  auto start_time = chrono::system_clock::now()\

#define CHECK_RUN_TIME_END_AND_COUT_SECONDS(argv)   auto final_time = chrono::system_clock::now();\
                                                    auto duration = chrono::duration_cast<chrono::microseconds>(final_time - start_time);\
                                                    std::cout << argv << " -> run_time counts :" << double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den << std::endl\

using namespace std;
using namespace rapidjson;

constexpr size_t SHOTS = 100;

QCircuit qft_prog(const QVec& qvec)
{
    QCircuit qft = CreateEmptyCircuit();
    for (auto i = 0; i < qvec.size(); i++)
    {
        qft << H(qvec[qvec.size() - 1 - i]);
        for (auto j = i + 1; j < qvec.size(); j++)
        {
            qft << CR(qvec[qvec.size() - 1 - j],
                qvec[qvec.size() - 1 - i], 2 * PI / (1 << (j - i + 1)));
        }
    }
    return qft;
}

QProg ghz_prog(const QVec& q)
{
    auto prog = QProg();
    prog << H(q[0]);

    for (auto i = 0; i < q.size() - 1; ++i)
    {
        prog << CNOT(q[i], q[i + 1]);
    }

    return prog;
}

static std::string build_chain_typed_quantum_chip_config_data(size_t qubit_num)
{
    rapidjson::Document arch_doc;
    arch_doc.SetObject();

    rapidjson::Document::AllocatorType &arch_alloc = arch_doc.GetAllocator();

    rapidjson::Value adj_object(rapidjson::kObjectType);
    for (SizeType qubit_addr = 0; qubit_addr < qubit_num; qubit_addr++)
    {
        rapidjson::Value qubit_array(rapidjson::kArrayType);

        if (qubit_addr != 0)
        {
            rapidjson::Value qubit_value(rapidjson::kObjectType);

            qubit_value.AddMember("v", qubit_addr - 1, arch_alloc);
            qubit_value.AddMember("w", 1.0, arch_alloc);

            qubit_array.PushBack(qubit_value, arch_alloc);
        }

        if (qubit_addr != (qubit_num - 1))
        {
            rapidjson::Value qubit_value(rapidjson::kObjectType);

            qubit_value.AddMember("v", qubit_addr + 1, arch_alloc);
            qubit_value.AddMember("w", 1.0, arch_alloc);

            qubit_array.PushBack(qubit_value, arch_alloc);
        }

        std::string qubit_addr_str = to_string(qubit_addr);

        rapidjson::Value qubit_value(kStringType);
        qubit_value.SetString(qubit_addr_str.c_str(), (rapidjson::SizeType)qubit_addr_str.size(), arch_alloc);

        adj_object.AddMember(qubit_value, qubit_array, arch_alloc);
    }

    arch_doc.AddMember("adj", adj_object, arch_alloc);
    arch_doc.AddMember("QubitCount", (SizeType)qubit_num, arch_alloc);

    //construct final json
    rapidjson::Document doc;
    doc.SetObject();

    rapidjson::Document::AllocatorType &alloc = doc.GetAllocator();

    doc.AddMember("QuantumChipArch", arch_doc, alloc);

    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);

    return buffer.GetString();
}

bool test_matrix_decompose_mps()
{
    auto qvm = initQuantumMachine(QMachineType::CPU);
    auto q = qvm->allocateQubits(2);
    auto c = qvm->allocateCBits(2);

    QCircuit test_cir;
    test_cir << H(q[0]) << H(q[1]) << CNOT(q[0], q[1]) << T(q[1]);
    //test_cir << CNOT(q[0], q[1]);
    QStat stat_0;
    /*{
        auto prog = QProg();
        prog << test_cir;
        qvm->directlyRun(prog);
        stat_0 = qvm->getQState();
        std::cout << "stat_0:\n" << stat << endl;
    }*/
    const QStat target_matrix = getCircuitMatrix(test_cir);
    std::cout << "target_matrix:\n" << target_matrix << endl;

    auto cir = matrix_decompose_qr({ q[0], q[1] }, target_matrix);
    //auto cir = matrix_decompose_householder({ q[0], q[1] }, target_matrix);
    std::cout << "decomposed circuit:" << cir << endl;
    const auto mat_2 = getCircuitMatrix(cir);
    std::cout << "mat_2:\n" << mat_2 << endl;

    auto prog = QProg();
    prog << cir;
    qvm->directlyRun(prog);
    auto stat = qvm->getQState();

    destroyQuantumMachine(qvm);

    if (0 == mat_compare(target_matrix, mat_2, MAX_COMPARE_PRECISION)) {
        return true;
    }

    return false;
}


TEST(MPS, test)
{
    auto machine = initQuantumMachine(CPU);
    machine->setConfigure({ 64,64 });

    auto qv = machine->allocateQubits(36);
    auto cv = machine->allocateCBits(36);


    auto _grover_prog = ghz_prog(qv);
    _grover_prog << MeasureAll(qv, cv);

    cout << convert_qprog_to_originir(_grover_prog, machine);
    getchar();

	test_matrix_decompose_mps();
    return;
#if 1
    try
    {
        QCloudMachine QCM;
        QCM.init("C60FBD87EF084DBA820945D052218AA8", true);
        //QCM.init();

        //QVec qv;
        //std::vector<ClassicalCondition> cv;
        //QProg prog1 = convert_originir_to_qprog("E:\\N4.txt", &QCM, qv, cv);

        QCM.set_qcloud_api("http://10.10.10.197:8060");
        QCM.set_real_chip_api("http://10.10.10.197:8060");

        QVec qv;
        vector<ClassicalCondition> cv;
        auto ir = convert_originir_to_qprog("E:\\N4.txt", &QCM, qv, cv);

        TASK_STATUS status;
        auto result = QCM.full_amplitude_measure(ir,1000);
        cout << 123 << endl;
        /* TASK_STATUS status;
         auto taskid = QCM.get_expectation_commit(prog, hamiltonian, q, status);
         if (status == TASK_STATUS::FAILED)
         {
             cout << QCM.get_last_error() << endl;
         }
         else
         {
             auto result = QCM.get_expectation_exec(taskid, status);
             if (status == TASK_STATUS::FAILED)
             {
                 cout << QCM.get_last_error() << endl;
             }
             else
             {
                 auto result = QCM.get_expectation_exec(taskid, status);
                 std::cout << result << std::endl;
             }
         }*/

        QCM.finalize();
    }
    catch (const std::exception e)
    {
        cout << e.what() << endl;
    }
    catch (...)
    {
        cout << 123 << endl;
    }

    //QProg measure_prog;
    //measure_prog << HadamardQCircuit(q) << MeasureAll(q, c);
    //auto result2 = QCM.get_state_tomography_density(measure_prog, 1000, REAL_CHIP_TYPE::ORIGIN_WUYUAN_D4);
    //for (auto val : result2)
    //{
    //    cout << val << endl;
    //}

    //auto result3 = QCM.get_state_fidelity(measure_prog, 1000, REAL_CHIP_TYPE::ORIGIN_WUYUAN_D4);
    //cout << result3 << endl;


    return;
#endif
}
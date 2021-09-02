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

QStat cpu_backend(size_t qubit_num, QProg& prog)
{
    CPUQVM qvm;
    qvm.setConfigure({ 64,64 });
    qvm.init();

    auto q = qvm.qAllocMany(qubit_num);
    auto c = qvm.cAllocMany(qubit_num);

    qvm.directlyRun(prog);
    return qvm.getQState();
}

TEST(MPS, test)
{
#if 0
    PartialAmplitudeQVM mps;

    mps.init();
    auto qa = mps.qAllocMany(4);
    auto ca = mps.cAllocMany(4);

    QProg pp;
    //pp << H(qa[0]) << CNOT(qa[0], qa[1]) << Measure(qa[0], ca[0]);
    pp << H(qa[0]) << S(qa[0]);// << Measure(qa[0], ca[0]);

    //QVec qv;
    //std::vector<ClassicalCondition> cv;
    //auto prog1 = convert_originir_to_qprog("D:\\123.txt", &mps, qv, cv);
    //auto result = mps.runWithConfiguration(pp, ca, 1000);
    mps.run(pp);
    auto result = mps.pmeasure_dec_index("0");
    auto result1 = mps.pmeasure_dec_index("1");

#endif 

#if 0
    //通过QCloudMachine创建量子云虚拟机
    QCloudMachine QCM;;

    //通过传入当前用户的token来初始化
    QCM.init("5075D2CF755640C99B586A3E10C73437",true);
    auto q = QCM.allocateQubits(6);
    auto c = QCM.allocateCBits(6);

    //构建量子程序
    auto measure_prog = QProg();
    measure_prog << HadamardQCircuit(q)
        << RX(q[1], PI / 4)
        << Measure(q[0], c[0])
        << Measure(q[1], c[1]);

    //调用真实芯片计算接口，需要量子程序和测量次数两个参数
    auto result = QCM.real_chip_measure(measure_prog, 1000, REAL_CHIP_TYPE::ORIGIN_WUYUAN_D4);
    for (auto val : result)
    {
        cout << val.first << " : " << val.second << endl;
    }

    //auto result1 = QCM.real_chip_task(measure_prog, 1000, true, true, REAL_CHIP_TYPE::ORIGIN_WUYUAN_D4);
    //for (auto val : result1)
    //{
    //    cout << val.first << " : " << val.second << endl;
    //}

    auto result2 = QCM.get_state_tomography_density(measure_prog, 1000, REAL_CHIP_TYPE::ORIGIN_WUYUAN_D4);
    for (auto val : result2)
    {
        cout << val << endl;
    }

    auto result3 = QCM.get_state_fidelity(measure_prog, 1000, REAL_CHIP_TYPE::ORIGIN_WUYUAN_D4);
    cout << result3 << endl;


    QCM.finalize();
    return;

#endif


    auto machine = new PartialAmplitudeQVM();
    //auto machine = new SingleAmplitudeQVM();
     machine->init();
    auto qlist = machine->qAllocMany(10);

    // 构建量子程序
    QProg prog;
    for_each(qlist.begin(), qlist.end(), [&](Qubit *val) { prog << H(val); });
    prog << CZ(qlist[1], qlist[5])
        << CZ(qlist[3], qlist[7])
        << RZ(qlist[7], PI / 4)
        << U2(qlist[5], PI / 4, 1.)
        << U3(qlist[5], PI / 4, 1., 2.)
        << U4(qlist[5], PI / 4, 1., 2., 3.)
        << RX(qlist[4], PI / 4)
        << RY(qlist[3], PI / 4)
        << CZ(qlist[2], qlist[6])
        << RZ(qlist[3], PI / 4)
        << RZ(qlist[8], PI / 4)
        << CZ(qlist[9], qlist[5])
        << RZ(qlist[9], PI / 4)
        << SWAP(qlist[0], qlist[1])
        << CR(qlist[2], qlist[7], PI / 4)
        << X(qlist[5]).control({ qlist[1], qlist[2] });

    // 获取量子态所有分量的振幅
    //machine->run(prog, qlist);
    machine->run(prog);

    // 打印特定量子态分量的振幅
    //cout << machine->pMeasureDecindex("0") << endl;
    cout << machine->pmeasure_dec_index("0") << endl;
    cout << machine->pmeasure_dec_index("1") << endl;
    cout << machine->pmeasure_dec_index("2") << endl;
    cout << machine->pmeasure_dec_index("3") << endl;
    cout << machine->pmeasure_dec_index("4") << endl;
    cout << machine->pmeasure_dec_index("5") << endl;
    cout << machine->pmeasure_dec_index("6") << endl;
    cout << machine->pmeasure_dec_index("7") << endl;

    cout << "=========================" << endl;

    auto re = cpu_backend(10,prog);

    cout << re[0] << endl;
    cout << re[1] << endl;
    cout << re[2] << endl;
    cout << re[3] << endl;
    cout << re[4] << endl;
    cout << re[5] << endl;
    cout << re[6] << endl;
    cout << re[7] << endl;

    getchar();
}
#include <thread>
#include <string>
#include <sstream>
#include <iostream>
#include "gtest/gtest.h"

#include "QPandaConfig.h"
#include "Core/Utilities/Tools/MultiControlGateDecomposition.h"
#include "Core/Utilities/UnitaryDecomposer/UniformlyControlledGates.h"

#include <Eigen/Eigen>
#include <Eigen/Dense>

#include "Core/Core.h"
#include "Core/QuantumCloud/QCloudMachine.h"

#include "Core/Utilities/CommunicationProtocol/CommunicationProtocolEncode.h"
#include "Core/Utilities/CommunicationProtocol/CommunicationProtocolDecode.h"

using namespace std;
USING_QPANDA

string online_api_key = "302e020100301006072a8648ce3d020106052b8104001c041730150201010410b6d33ad8772eb9705e844394453a3c8a/6327";
string test_api_key = "302e020100301006072a8648ce3d020106052b8104001c0417301502010104104615aad5ebb390b75b8cac9311b97cac/10139";

static bool matrix_compare(const QStat& mat1, const QStat& mat2, const double precision /*= 0.000001*/)
{
    if (mat1.size() != mat2.size())
        return false;

    qcomplex_t ratio; // constant value
    for (size_t i = 0; i < mat1.size(); ++i)
    {
        if ((abs(mat2.at(i).real() - 0.0) > precision) || (abs(mat2.at(i).imag() - 0.0) > precision))
        {
            ratio = mat1.at(i) / mat2.at(i);
            if (!isfinite(ratio.real()) || !isfinite(ratio.imag()))
                return false;
            
            if (precision < abs(sqrt(ratio.real()*ratio.real() + ratio.imag()*ratio.imag()) - 1.0))
                return false;
            
            break;
        }
    }

    qcomplex_t tmp_val;
    for (size_t i = 0; i < mat1.size(); ++i)
    {
        tmp_val = ratio * mat2.at(i);
        if ((abs(mat1.at(i).real() - tmp_val.real()) > precision) ||
            (abs(mat1.at(i).imag() - tmp_val.imag()) > precision))
        {
            std::cout << abs(mat1.at(i).real() - tmp_val.real()) << std::endl;
            std::cout << abs(mat1.at(i).imag() - tmp_val.imag()) << std::endl;

            return false;
        }
    }

    return true;
}

#if defined(USE_CURL)

void test_qcloud_big_data()
{
    auto machine = QCloudMachine();
    machine.setConfigure({ 72,72 });

    //test : http://qcloud4test.originqc.com/zh
    machine.init(test_api_key, true);
    machine.set_qcloud_url("http://oqcs.originqc.com");

    //online
    //machine.init(online_api_key, true);

    auto q = machine.qAllocMany(6);
    auto c = machine.cAllocMany(6);

    auto measure_prog = QProg();

    for (size_t i = 0; i < 1024 * 128; i++)
        measure_prog << H(q[0]) << CNOT(q[0], q[1]);

    measure_prog << Measure(q[0], c[0]) << Measure(q[1], c[1]);

    std::vector<QProg> big_prog_array;
    for (size_t i = 0; i < 9; i++)
        big_prog_array.emplace_back(measure_prog);

    auto prog = QProg();
    prog << H(q[1]) << Measure(q[1], c[0]);

    auto big_batch_result = machine.batch_real_chip_measure(big_prog_array, 1000, RealChipType::ORIGIN_72);
    for (auto val : big_batch_result)
    {
        for (auto single_item : val)
            cout << single_item.first << " : " << single_item.second << endl;
    }

    auto batch_id = machine.async_batch_real_chip_measure(big_prog_array, 1000, RealChipType::ORIGIN_72);

    std::vector<QProg> prog_array;
    for (size_t i = 0; i < 8; i++)
        prog_array.emplace_back(prog);

    auto batch_result = machine.batch_real_chip_measure(prog_array, 100, RealChipType::ORIGIN_72);
    for (auto val : batch_result)
    {
        for (auto single_item : val)
            cout << single_item.first << " : " << single_item.second << endl;
    }

    std::cout << "test_qcloud_big_data passed. " << std::endl;
}

void test_comm_protocol_encode_data()
{
    auto machine = CPUQVM();
    machine.setConfigure({ 72,72 });

    auto q = machine.qAllocMany(8);
    auto c = machine.cAllocMany(8);

    auto circuit = QCircuit();
    circuit << RXX(q[0], q[1], 2);
    circuit << RYY(q[0], q[1], -3);
    circuit << RZZ(q[0], q[1], 3);
    circuit << RZX(q[0], q[1], 4);

    auto prog = QProg();
    //prog << random_qcircuit(q, 15);
    prog << H(q[0]);
    //prog << circuit;
    //prog << circuit.dagger();
    //prog << CR(q[0], q[1], -4);
    //prog << U1(q[0], 2);
    //prog << U2(q[0], 2, 3);
    //prog << U3(q[0], 2, -3, 4);
    //prog << U4(q[0], -2, 3, 4, 5);
    //prog << RPhi(q[0], 1 , -2);
    //prog << BARRIER(q);
    //prog << Toffoli(q[0], q[1], q[2]);
    //prog << MeasureAll(q,c);

    CommProtocolConfig config;

    config.open_mapping = true;
    config.open_error_mitigation = false;
    config.optimization_level = 2;
    config.circuits_num = 5;
    config.shots = 1000;

    std::cout << prog << std::endl;

    auto encode_data = comm_protocol_encode(prog, config);

    cout << "========================" << endl;

    CommProtocolConfig decode_config;
    auto decode_progs = comm_protocol_decode(decode_config, encode_data, &machine);

    for (size_t i = 0; i < decode_progs.size(); i++)
        std::cout << decode_progs[i] << std::endl;

    auto encode_matrix = get_unitary(prog);
    auto decode_matrix = get_unitary(decode_progs[0]);

    if(matrix_compare(encode_matrix, decode_matrix, 1e-3))
        std::cout << "test_comm_protocol_encode_data passed. " << std::endl;
    else
        std::cout << "test_comm_protocol_encode_data failed. " << std::endl;
}


#include <chrono>
void test_comm_protocol_encode(int qubits_num, int depth, int prog_length)
{
    auto machine = CPUQVM();
    machine.setConfigure({ 72,72 });

    auto q = machine.qAllocMany(qubits_num);
    auto c = machine.cAllocMany(qubits_num);

    auto prog_array = std::vector<QProg>(prog_length);
    for (size_t i = 0; i < prog_length; i++)
        prog_array[i] << random_qcircuit(q, depth);

    CommProtocolConfig config;

    config.open_mapping = true;
    config.open_error_mitigation = false;
    config.optimization_level = 2;
    config.circuits_num = 5;
    config.shots = 1000;

    //std::cout << prog_array[0] << std::endl;
    //std::cout << "-----------" << std::endl;
    //std::cout << prog_array[1] << std::endl;

    auto encode_start_time = std::chrono::high_resolution_clock::now();

    auto encode_data = comm_protocol_encode(prog_array, config);
    size_t encode_data_bytes = encode_data.size() * sizeof(char);
    std::cout << " Size of encode_data: " << encode_data_bytes << " bytes" << std::endl;
    auto encode_end_time = std::chrono::high_resolution_clock::now();
    auto encode_duration = std::chrono::duration_cast<std::chrono::milliseconds>(encode_end_time - encode_start_time);
    std::cout << " Encode execution time: " << encode_duration.count() << " milliseconds" << std::endl;

    cout << "========================" << endl;

    CommProtocolConfig decode_config;

    auto decode_start_time = std::chrono::high_resolution_clock::now();
    auto decode_progs = comm_protocol_decode(decode_config, encode_data, &machine);
    auto decode_end_time = std::chrono::high_resolution_clock::now();
    auto decode_duration = std::chrono::duration_cast<std::chrono::milliseconds>(decode_end_time - decode_start_time);
    std::cout << " Decode execution time: " << decode_duration.count() << " milliseconds" << std::endl;

    //std::cout << decode_progs[0] << std::endl;
    //std::cout << "-------------" << std::endl;
    //std::cout << decode_progs[1] << std::endl;

    auto encode_matrix = get_unitary(prog_array[0]);
    auto decode_matrix = get_unitary(decode_progs[0]);

    if (matrix_compare(encode_matrix, decode_matrix, 1e-3))
        std::cout << "test_comm_protocol_encode_data passed. " << std::endl;
    else
        std::cout << "test_comm_protocol_encode_data failed. " << std::endl;
}

void test_qcloud_real_chip()
{
    auto machine = QCloudMachine();
    machine.setConfigure({ 72,72 });

    //test : http://qcloud4test.originqc.com/zh
    machine.init(test_api_key, true);
    machine.set_qcloud_url("http://oqcs.originqc.com");

    //online
    //machine.init(online_api_key, true);

    auto q = machine.qAllocMany(6);
    auto c = machine.cAllocMany(6);

    auto measure_prog = QProg();

    for (size_t i = 0; i < 1024 * 256; i++)
        measure_prog << H(q[0]) << CNOT(q[0], q[1]);

    //measure_prog << H(q[0]) << H(q[1]) << Measure(q[0], c[0]);

    std::vector<QProg> prog_array;
    for (size_t i = 0; i < 8; i++)
        prog_array.emplace_back(measure_prog);

    auto prog = QProg();
    prog << H(q[1]) << Measure(q[1], c[0]);

    auto batch_result = machine.batch_real_chip_measure(prog_array, 100, RealChipType::ORIGIN_72);
    for (auto val : batch_result)
    {
        for (auto single_item : val)
            cout << single_item.first << " : " << single_item.second << endl;
    }

    auto real_chip_result = machine.real_chip_measure(measure_prog, 
        1000, 
        RealChipType::ORIGIN_72,
        true,
        true,
        true);

    for (auto val : real_chip_result)
        cout << val.first << " : " << val.second << endl;

    std::cout << "test_qcloud_real_chip passed. " << std::endl;
}

void test_qcloud_async()
{
    auto machine = QCloudMachine();
    machine.setConfigure({ 72,72 });

    //test : http://qcloud4test.originqc.com/zh
    machine.init(test_api_key, 
        true,
        true,
        false);

    //online
    //machine.init(online_api_key, true, true, false);

    machine.set_qcloud_url("http://oqcs.originqc.com");

    auto q = machine.qAllocMany(6);
    auto c = machine.cAllocMany(6);

    std::vector<QProg> prog_array;

    for (size_t i = 0; i < 2; i++)
    {
        auto measure_prog = QProg();
        measure_prog 
            << H(q[0]) 
            << CNOT(q[0], q[1]) 
            << CNOT(q[1], q[2])
            << Measure(q[0], c[0]) 
            << Measure(q[1], c[1]);

        prog_array.emplace_back(measure_prog);
    }

    auto prog = QProg();
    prog << H(q[1]) << Measure(q[1], c[0]);

    //auto price_result = machine.estimate_price(10, 1000);

    auto batch_result = machine.batch_real_chip_measure(prog_array, 1000, RealChipType::ORIGIN_72);
    for (auto val : batch_result)
    {
        for (auto single_item : val)
            cout << single_item.first << " : " << single_item.second << endl;
    }

    auto batch_id = machine.async_batch_real_chip_measure(prog_array, 1000, RealChipType::ORIGIN_72);
    std::cout << "batch_id : " << batch_id << std::endl;

    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    auto batch_query_result = machine.query_batch_state_result(batch_id, 10);
    for (auto val : batch_query_result)
    {
        for (auto single_item : val)
            cout << single_item.first << " : " << single_item.second << endl;
    }

    auto real_chip_measure_id = machine.async_real_chip_measure(prog, 1000, RealChipType::ORIGIN_72);
    std::cout << "real_chip_measure_id : " << real_chip_measure_id << std::endl;

    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    auto real_chip_measure_result = machine.query_state_result(real_chip_measure_id);

    for (auto val : real_chip_measure_result)
        std::cout << val.first << " : " << val.second << std::endl;
    
    auto real_chip_result = machine.real_chip_measure(prog,
        1000,
        RealChipType::ORIGIN_72,
        true,
        true,
        true);

    for (auto val : real_chip_result)
        cout << val.first << " : " << val.second << endl;

    machine.finalize();

    std::cout << "test_qcloud_async passed. " << std::endl;
}

void test_benchmark()
{
    //QV
    auto qubit_lists = std::vector<std::vector<int>>{ {3, 4},{2, 3, 5} };

    auto ntrials = 10;

    auto qv_config = QCloudTaskConfig();
    qv_config.cloud_token = "302e020100301006072a8648ce3d020106052b8104001c041730150201010410b6d33ad8772eb9705e844394453a3c8a/6327";
    qv_config.shots = 1000;

    auto qv_result = calculate_quantum_volume(qv_config, qubit_lists, ntrials);

    //RB
    auto rb_range = { 5, 10, 15 };

    auto rb_config = QCloudTaskConfig();
    rb_config.cloud_token = "302e020100301006072a8648ce3d020106052b8104001c041730150201010410b6d33ad8772eb9705e844394453a3c8a/6327";
    rb_config.shots = 1000;

    auto single_rb_result = single_qubit_rb(rb_config, 0, rb_range, 10);

    auto double_rb_result = double_qubit_rb(rb_config, 0, 1, rb_range, 10);

    //XEB
    auto range = std::vector<int>{ 2, 4, 6, 8, 10 };

    auto xeb_config = QCloudTaskConfig();
    xeb_config.cloud_token = "302e020100301006072a8648ce3d020106052b8104001c041730150201010410b6d33ad8772eb9705e844394453a3c8a/6327";

    auto res = double_gate_xeb(xeb_config, 0, 1, range, 10);
    return;
}

void test_qcloud()
{
    auto machine = QCloudMachine();
    machine.setConfigure({ 72,72 });

    //test : http://qcloud4test.originqc.com/zh
    //machine.init(test_api_key, true);

    //online
    machine.init(online_api_key, true);

    //machine.set_qcloud_url("http://oqcs.originqc.com");

    auto q = machine.qAllocMany(6);
    auto c = machine.cAllocMany(6);

    auto measure_prog = QProg();
    measure_prog << H(q[0]) << H(q[1]) << Measure(q[0], c[0]);

    std::vector<QProg> prog_array;

    for (size_t i = 0; i < 10; i++)
        prog_array.emplace_back(measure_prog);

    auto prog = QProg();
    prog << H(q[1]) << Measure(q[1], c[0]);

    auto result0 = machine.full_amplitude_measure(prog, 1000);
    cout << "full_amplitude_measure result : " << endl;
    for (auto val : result0)
        cout << val.first << " : " << val.second << endl;

    auto pmeasure_prog = QProg();
    pmeasure_prog << H(q[0]) << H(q[1]);

    auto result1 = machine.full_amplitude_pmeasure(pmeasure_prog, { 0, 1, 2 });
    cout << "full_amplitude_pmeasure result : " << endl;
    for (auto val : result1)
        cout << val.first << " : " << val.second << endl;

    auto result2 = machine.partial_amplitude_pmeasure(pmeasure_prog, { "0", "1", "2" });
    cout << "partial_amplitude_pmeasure result : " << endl;
    for (auto val : result2)
        cout << val.first << " : " << val.second << endl;

    auto result3 = machine.single_amplitude_pmeasure(pmeasure_prog, "0");
    cout << "single_amplitude_pmeasure result : " << endl;
    cout << "0 : " << result3 << endl;

    machine.set_noise_model(NOISE_MODEL::BIT_PHASE_FLIP_OPRATOR, { 0.01 }, { 0.02 });
    auto result41 = machine.noise_measure(measure_prog, 100);
    cout << "noise_measure result : " << endl;
    for (auto val : result41)
        cout << val.first << " : " << val.second << endl;

    auto result4 = machine.real_chip_measure(measure_prog, 1000);
    cout << "real_chip_measure result : " << endl;
    for (auto val : result4)
        cout << val.first << " : " << val.second << endl;

    auto result5 = machine.get_state_tomography_density(measure_prog, 1000);
    cout << "get_state_tomography_density result : " << endl;
    for (auto val : result5)
    {
        for (auto val1 : val)
            cout << val1 << endl;
    }

    auto result6 = machine.get_state_fidelity(measure_prog, 1000);
    cout << "fidelity : " << result6 << endl;

    machine.finalize();

    cout << "test qcloud passed. " << endl;
}

bool static state_compare(const QStat& state1, const QStat& state2)
{
    QPANDA_RETURN(state1.size() != state2.size(), false);

    for (auto i = 0; i < state1.size(); ++i)
    {
        if (std::fabs(state1[i].real() - state2[i].real()) > 1e-6)
            return false;
        if (std::fabs(state1[i].imag() - state2[i].imag()) > 1e-6)
            return false;
    }

    return true;
}

#include "Core/Utilities/Tools/RandomCircuit.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
void test_ldd_oracle()
{
    CPUQVM machine;
    machine.init();

    auto q = machine.qAllocMany(5);
    auto c = machine.cAllocMany(5);

    auto tar_qubits = QVec{ q[0], q[1], q[2] };
    auto ctr_qubits = QVec{ q[3], q[4] };

    auto rand_circuit = random_qcircuit(tar_qubits, 10);
    auto rand_prog = QProg(rand_circuit);

    auto rand_matrix = get_unitary(rand_prog);

    QProg prog;
    prog << X(q[3]) << X(q[4])
        << RX(q[0], 1)
        << MS(q[0], q[1]).control(ctr_qubits)
        << RXX(q[0], q[1], 1).control(ctr_qubits)
        << RYY(q[0], q[1], 2).control(ctr_qubits)
        << RZZ(q[0], q[1], 3).control(ctr_qubits)
        << RZX(q[0], q[1], 4).control(ctr_qubits)
        << QOracle(tar_qubits, rand_matrix).control(ctr_qubits);

    machine.directlyRun(prog);
    auto origin_state = machine.getQState();

    auto ldd_prog = ldd_decompose(prog);
    machine.initState();
    machine.directlyRun(ldd_prog);
    auto ldd_state = machine.getQState();

    for (auto val : origin_state)
        std::cout << val << endl;

    std::cout << "================" << endl;

    for (auto val : ldd_state)
        std::cout << val << endl;

    std::cout << prog << endl;
    std::cout << ldd_prog << endl;

    if (state_compare(origin_state, ldd_state))
        cout << "Test Ldd Decompose Multi Control Oracle Passed." << endl;
    else
        cout << "Test Ldd Decompose Multi Control Oracle Failed." << endl;

    return;
}



TEST(CloudHttp, QCloud)
{
    //test_benchmark();
    //test_qcloud();
    //test_ldd_oracle();
    test_qcloud_async();
    //test_qcloud_big_data();
    //test_qcloud_real_chip();
    //test_comm_protocol_encode_data();
    //test_comm_protocol_encode(8, 8, 5);
    cout << "Test CloudHttp.QCloud Passed." << endl;
}

#endif // USE_CURL


#include "Core/Utilities/Tools/QCloudConfig.h"
#include "ThirdParty/rapidjson/document.h"
#include "ThirdParty/rapidjson/writer.h"
#include "ThirdParty/rapidjson/stringbuffer.h"
#include "Core/Utilities/Tools/Uinteger.h"
#include "Core/Utilities/Compiler/QProgToOriginIR.h"

USING_QPANDA
using namespace std;
using namespace rapidjson;

void QPanda::json_string_transfer_encoding(std::string& str)
{
    //delete "\r\n" from recv json, Transfer-Encoding: chunked
    int pos = 0;
    while ((pos = str.find("\n")) != -1)
    {
        str.erase(pos, 1);
    }

    return;
}

void QPanda::params_verification(std::string dec_amplitude, size_t qubits_num)
{
    uint128_t amplitude(dec_amplitude.c_str());

    uint128_t max_amplitude = (uint128_t("1") << qubits_num) - 1;

    if (max_amplitude < amplitude)
        QCERR_AND_THROW(run_fail, "amplitude params > max_amplitude");
}

void QPanda::params_verification(std::vector<std::string> dec_amplitudes, size_t qubits_num)
{
    for (size_t i = 0; i < dec_amplitudes.size(); i++)
    {
        uint128_t amplitude(dec_amplitudes[i].c_str());

        uint128_t max_amplitude = (uint128_t("1") << qubits_num) - 1;

        if (max_amplitude < amplitude)
            QCERR_AND_THROW(run_fail, "amplitude params > max_amplitude");
    }
}


std::string QPanda::to_string_array(const Qnum values)
{
    std::string string_array;
    for (auto val : values)
    {
        string_array.append(to_string(val));
        if (val != values.back())
        {
            string_array.append(",");
        }
    }

    return string_array;
}

std::string QPanda::to_string_array(const std::vector<string> values)
{
    std::string string_array;
    for (auto val : values)
    {
        string_array.append(val);
        if (val != values.back())
        {
            string_array.append(",");
        }
    }

    return string_array;
}

void QPanda::construct_cluster_task_json(
    rabbit::document& doc,
    std::string prog_str,
    std::string token,
    size_t qvm_type,
    size_t qubit_num,
    size_t cbit_num,
    size_t measure_type,
    std::string task_name)
{
    doc.insert("code", prog_str);
    doc.insert("apiKey", token);
    doc.insert("QMachineType", qvm_type);
    doc.insert("codeLen", prog_str.size());
    doc.insert("qubitNum", qubit_num);
    doc.insert("taskFrom", 4);
    doc.insert("measureType", measure_type);
    doc.insert("classicalbitNum", cbit_num);
    doc.insert("taskName", task_name);

    return;
}

void QPanda::construct_real_chip_task_json(
    rabbit::document& doc,
    std::string prog_str,
    std::string token,
    bool is_amend,
    bool is_mapping,
    bool is_optimization,
    size_t qvm_type,
    size_t qubit_num,
    size_t cbit_num,
    size_t measure_type,
    size_t shots,
    size_t chip_id,
    std::string task_name)
{
    doc.insert("code", prog_str);
    doc.insert("apiKey", token);
    doc.insert("isAmend", (int)!is_amend);
    doc.insert("mappingFlag", (int)!is_mapping);
    doc.insert("circuitOptimization", (int)!is_optimization);
    doc.insert("QMachineType", qvm_type);
    doc.insert("codeLen", prog_str.size());
    doc.insert("qubitNum", qubit_num);
    doc.insert("taskFrom", 4);
    doc.insert("measureType", measure_type);
    doc.insert("classicalbitNum", cbit_num);
    doc.insert("shot", (size_t)shots);
    doc.insert("chipId", (size_t)chip_id);
    doc.insert("taskName", task_name);
}

void QPanda::real_chip_task_validation(int shots, QProg &prog)
{
    QVec qubits;
    std::vector<int> cbits;

    auto qbit_num = get_all_used_qubits(prog, cbits);
    auto cbit_num = get_all_used_class_bits(prog, cbits);

    QPANDA_ASSERT(qbit_num > 6 || cbit_num > 6, "real chip qubit num or cbit num are not less or equal to 6");
    QPANDA_ASSERT(shots > 10000 || shots < 1000, "real chip shots must be in range [1000,10000]");

    TraversalConfig traver_param;
    QProgCheck prog_check;
    prog_check.execute(prog.getImplementationPtr(), nullptr, traver_param);

    if (!traver_param.m_can_optimize_measure)
    {
        QCERR("measure must be last");
        throw run_fail("measure must be last");
    }

    return;
}


size_t QPanda::recv_json_data(void *ptr, size_t size, size_t nmemb, void *stream)
{
    std::string data((const char*)ptr, 0, (size_t)(size * nmemb));
    *((std::stringstream*)stream) << data << std::endl;
    return size * nmemb;
}

string QPanda::hamiltonian_to_json(const QHamiltonian& hamiltonian)
{
    Document doc;
    doc.SetObject();
    Document::AllocatorType &alloc = doc.GetAllocator();

    Value hamilton_arr(kArrayType);

    for (auto i = 0; i < hamiltonian.size(); ++i)
    {
        const auto& item = hamiltonian[i];

        Value hamilton_item(kObjectType);
        Value temp_pauli_param_array(rapidjson::kArrayType);
        Value temp_pauli_type_array(rapidjson::kArrayType);

        for (auto val : item.first)
        {
            temp_pauli_param_array.PushBack(val.first, alloc);

            rapidjson::Value string_key(kStringType);
            string_key.SetString(std::string(1, val.second).c_str(), 1, alloc);
            temp_pauli_type_array.PushBack(string_key, alloc);
        }

        hamilton_item.AddMember("pauli_type", temp_pauli_type_array, alloc);
        hamilton_item.AddMember("pauli_param", temp_pauli_param_array, alloc);
        hamilton_item.AddMember("hamiltonian_param", item.second, alloc);
        hamilton_arr.PushBack(hamilton_item, alloc);
    }

    doc.AddMember("hamiltonian", hamilton_arr, alloc);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    doc.Accept(writer);

    std::string hamiltonian_str = buffer.GetString();
    return hamiltonian_str;
}
QHamiltonian QPanda::json_to_hamiltonian(const std::string& hamiltonian_json)
{
    try
    {
        rabbit::document cfg_doc;
        cfg_doc.parse(hamiltonian_json);
        const rabbit::array hamiltonion_arr = cfg_doc["hamiltonian"];
        QHamiltonian result;
        for (auto &ele : hamiltonion_arr)
        {
            QTerm qterm;
            auto &pauli_type_arr = ele["pauli_type"];
            auto &pauli_param_arr = ele["pauli_param"];
            size_t type_size = pauli_type_arr.size();
            size_t param_size = pauli_param_arr.size();
            if (type_size != param_size)
            {
                QCERR_AND_THROW(run_fail, "parse json error");
            }
            for (auto i = 0; i < type_size; ++i)
            {
                qterm.insert(std::make_pair(pauli_param_arr[i].as_uint(), pauli_type_arr[i].as_string().at(0)));
            }
            result.emplace_back(qterm, ele["hamiltonian_param"].as_double());
        }

        return result;
    }
    catch (std::exception& e)
    {
        QCERR(e.what());
        throw run_fail("hamiltonian json error");
    }
}
void QPanda::construct_multi_prog_json(QuantumMachine* qvm, rabbit::array& code_array, size_t& code_len, std::vector<QProg>& prog_array)
{
    //convert prog to originir 
    std::vector<string> originir_array;

    for (auto& val : prog_array)
    {
        auto prog_str = convert_qprog_to_originir(val, qvm);
        code_len += prog_str.size();
        originir_array.emplace_back(prog_str);
    }

    for (auto i = 0; i < originir_array.size(); ++i)
    {
        rabbit::object code_value;
        code_value.insert("code", originir_array[i]);
        code_value.insert("id", (size_t)i);
        code_value.insert("step", (size_t)i);
        code_value.insert("breakPoint", "0");
        code_value.insert("isNow", (size_t)!i);
        code_array.push_back(code_value);
    }
}
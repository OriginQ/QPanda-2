#include "Core/Utilities/Tools/QCloudConfig.h"
#include "ThirdParty/rapidjson/document.h"
#include "ThirdParty/rapidjson/writer.h"
#include "ThirdParty/rapidjson/stringbuffer.h"
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
    //construct json
    Document doc;
    doc.SetObject();
    Document::AllocatorType &alloc = doc.GetAllocator();

    Value hamiltonian_value_array(rapidjson::kObjectType);
    Value hamiltonian_param_array(rapidjson::kArrayType);

    Value pauli_parm_array(rapidjson::kArrayType);
    Value pauli_type_array(rapidjson::kArrayType);

    for (auto i = 0; i < hamiltonian.size(); ++i)
    {
        const auto& item = hamiltonian[i];

        Value temp_pauli_parm_array(rapidjson::kArrayType);
        Value temp_pauli_type_array(rapidjson::kArrayType);

        for (auto val : item.first)
        {
            temp_pauli_parm_array.PushBack(val.first, alloc);

            rapidjson::Value string_key(kStringType);
            string_key.SetString(std::string(1, val.second).c_str(), 1, alloc);

            temp_pauli_type_array.PushBack(string_key, alloc);
        }

        pauli_parm_array.PushBack(temp_pauli_parm_array, alloc);
        pauli_type_array.PushBack(temp_pauli_type_array, alloc);

        hamiltonian_param_array.PushBack(item.second, alloc);
    }

    hamiltonian_value_array.AddMember("pauli_type", pauli_type_array, alloc);
    hamiltonian_value_array.AddMember("pauli_parm", pauli_parm_array, alloc);

    doc.AddMember("hamiltonian_value", hamiltonian_value_array, alloc);
    doc.AddMember("hamiltonian_param", hamiltonian_param_array, alloc);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    doc.Accept(writer);

    std::string hamiltonian_str = buffer.GetString();
    return hamiltonian_str;
}

QHamiltonian QPanda::json_to_hamiltonian(const std::string& hamiltonian_json)
{
    Document doc;
    if (doc.Parse(hamiltonian_json.c_str()).HasParseError())
    {
        QCERR(hamiltonian_json.c_str());
        throw run_fail("result json parse error");
    }

    try
    {
        QHamiltonian result;

        const rapidjson::Value& hamiltonian_value_array = doc["hamiltonian_value"];
        const rapidjson::Value& hamiltonian_param_array = doc["hamiltonian_param"];

        const rapidjson::Value& pauli_type_array = hamiltonian_value_array["pauli_type"];
        const rapidjson::Value& pauli_parm_array = hamiltonian_value_array["pauli_parm"];

        QPANDA_ASSERT(pauli_type_array.Size() != pauli_parm_array.Size(), "hamiltonian json error");

        for (SizeType i = 0; i < pauli_type_array.Size(); ++i)
        {
            QTerm qterm;

            const rapidjson::Value &pauli_type_value = pauli_type_array[i];
            const rapidjson::Value &pauli_parm_value = pauli_parm_array[i];

            const rapidjson::Value &hamiltonian_parm = hamiltonian_param_array[i];

            QPANDA_ASSERT(pauli_type_value.Size() != pauli_parm_value.Size(), "hamiltonian json error");

            for (SizeType j = 0; j < pauli_type_value.Size(); ++j)
            {
                size_t pauli_parm = pauli_parm_value[j].GetInt();
                string pauli_type = pauli_type_value[j].GetString();
                qterm.insert(std::make_pair(pauli_parm, pauli_type[0]));
            }

            result.emplace_back(std::make_pair(qterm, hamiltonian_parm.GetDouble()));
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
#include <map>
#include "QPanda.h"
#include "Core/Utilities/Tools/JsonConfigParam.h"
#include <algorithm>  
#include "gtest/gtest.h"
using namespace std;
USING_QPANDA

TEST(ReadConfig, Instructions)
{
    std::cout << "======================================" << std::endl;
	JsonConfigParam config;
    config.load_config(CONFIG_PATH);

    std::map<std::string, std::map<std::string, uint32_t>> ins_config;
    bool is_success = config.getInstructionConfig(ins_config);

    for (auto val : ins_config)
    {
        cout << val.first << "======"<<endl;
        for (auto val1 : val.second)
        {
            cout << val1.first << " : " << val1.second << endl;
        }
    }
}


TEST(ReadConfig, ClassName)
{
    std::cout << "======================================" << std::endl;
	JsonConfigParam config;
    config.load_config(CONFIG_PATH);
    map<string, string> class_names;
    bool is_success = config.getClassNameConfig(class_names);

    for (auto &val : class_names)
    {
        std::cout << val.second << std::endl;
    }
}

TEST(ReadConfig, Metadata)
{
    std::cout << "======================================" << std::endl;
	JsonConfigParam config;
    config.load_config(CONFIG_PATH);
    int qubit_number;
    vector<vector<double>> matrix;
    bool is_success = config.getMetadataConfig(qubit_number, matrix);

    std::cout << "qubits count: " << qubit_number << std::endl;
    std::cout << "matrix: " << std::endl;
    for (int i = 0; i < qubit_number; i++)
    {
        for (int j = 0; j < qubit_number; j++)
        {
            std::cout << matrix[i][j] << "  ";
        }
        std::cout << std::endl;
    }
}


TEST(ReadConfig, QGateConfig)
{
    std::cout << "======================================" << std::endl;
	JsonConfigParam config;
    config.load_config(CONFIG_PATH);
    vector<string> single_gates;
    vector<string> double_gates;
    bool is_success = config.getQGateConfig(single_gates, double_gates);

    std::cout << "Single Gates: " << std::endl;
    for (auto &val : single_gates)
    {
        std::cout << val << std::endl;
    }

    std::cout << "Double Gates: " << std::endl;
    for (auto &val : double_gates)
    {
        std::cout << val << std::endl;
    }
}

TEST(ReadConfig, QGateTimeConfig)
{
    std::cout << "======================================" << std::endl;
	JsonConfigParam config;
    config.load_config(CONFIG_PATH);
    map<GateType, size_t> gate_time;
    bool is_success = config.getQGateTimeConfig(gate_time);

    std::cout << "gate time: " << std::endl;
    for (auto &val : gate_time)
    {
        std::cout << val.first << ", " << val.second << std::endl;
    }
}


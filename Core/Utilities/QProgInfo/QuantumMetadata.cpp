#include "Core/Utilities/QProgInfo/QuantumMetadata.h"
#include <algorithm>
#include <string>
#include "Core/Utilities/Tools/TranformQGateTypeStringAndEnum.h"
#include "Core/Utilities/Tools/JsonConfigParam.h"

using namespace std;

USING_QPANDA


QuantumMetadata::QuantumMetadata(const string &filename)
{
    if (m_config.load_config(filename.c_str()))
    {
        m_is_config_exist = true;
    }
    else
    {
        m_is_config_exist = false;
    }
}

bool QuantumMetadata::getMetadata(int &qubit_num, std::vector<std::vector<double> > &matrix)
{
    if (!m_is_config_exist)
    {
        qubit_num = 4;
        matrix = { {0,1,1,0},
                    {1,0,0,1},
                    {1,0,0,1},
                    {0,1,1,0} };
    }
    else
    {
        return m_config.getMetadataConfig(qubit_num, matrix);
    }
    return true;
}

bool QuantumMetadata::getQGate(std::vector<string> &single_gates, std::vector<string> &double_gates)
{
    if (!m_is_config_exist)
    {
        single_gates.push_back("RX");
        single_gates.push_back("RY");
        single_gates.push_back("RZ");
        single_gates.push_back("X1");
        single_gates.push_back("H");
        single_gates.push_back("S");

        double_gates.push_back("CNOT");
        double_gates.push_back("CZ");
        double_gates.push_back("ISWAP");
    }
    else
    {
        return m_config.getQGateConfig(single_gates, double_gates);
    }

    return true;
}

bool QuantumMetadata::getGateTime(std::map<GateType, size_t> &gate_time_map)
{
    if (!m_is_config_exist)
    {
        insertGateTimeMap({"RX", 1}, gate_time_map);
        insertGateTimeMap({"RY", 1}, gate_time_map);
        insertGateTimeMap({"RZ", 1}, gate_time_map);
        insertGateTimeMap({"X1", 1}, gate_time_map);
        insertGateTimeMap({"H", 1}, gate_time_map);
        insertGateTimeMap({"S", 1 }, gate_time_map);
        insertGateTimeMap({"U3", 1}, gate_time_map);

        insertGateTimeMap({"CNOT", 2}, gate_time_map);
        insertGateTimeMap({"CZ", 2}, gate_time_map);
        insertGateTimeMap({"ISWAP", 2}, gate_time_map);
    }
    else
    {
        return m_config.getQGateTimeConfig(gate_time_map);
    }
    return true;
}

void QuantumMetadata::insertGateTimeMap(const pair<string, size_t> &gate_time, map<GateType, size_t> &gate_time_map)
{

    pair<GateType, size_t> gate_type_time(TransformQGateType::getInstance()[gate_time.first],                                            gate_time.second);
    gate_time_map.insert(gate_type_time);

    return ;
}


QuantumMetadata::~QuantumMetadata()
{
}


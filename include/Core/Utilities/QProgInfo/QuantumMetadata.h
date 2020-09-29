/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

QuantumMetadata.h
Author: Wangjing
Created in 2018-8-31

Classes for get the shortes path of graph

*/

#ifndef QUBITCONFIG_H
#define QUBITCONFIG_H

#include "Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"
#include "Core/Utilities/Tools/TranformQGateTypeStringAndEnum.h"
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include "Core/Utilities/Tools/JsonConfigParam.h"

QPANDA_BEGIN

/**
* @brief  Parse xml config and get metadata
* @ingroup Utilities
*/
class QuantumMetadata
{   
public:
    QuantumMetadata(const std::string & filename = CONFIG_PATH);
    QuantumMetadata & operator =(const QuantumMetadata &) = delete;

    bool getMetadata(int &qubit_num, std::vector<std::vector<double>> &matrix);
    bool getQGate(std::vector<std::string> &single_gates, std::vector<std::string> &double_gates);
    bool getGateTime(std::map<GateType, size_t> &gate_time_map);

    ~QuantumMetadata();
private:
    void insertGateTimeMap(const std::pair<std::string, size_t> &gate_time,
                           std::map<GateType, size_t> &gate_time_map);
	JsonConfigParam m_config;
    bool m_is_config_exist;
};

QPANDA_END
#endif // QubitConfig_H

/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

JsonConfigParam.h
Author: Wangjing
Created in 2018-8-31

Classes for get the shortes path of graph

*/
#ifndef JSON_CONFIG_PARAM_H
#define JSON_CONFIG_PARAM_H
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"
#include <iostream>
#include <string>
#include <fstream>
#include <map>
#include <vector>
#include "Core/Utilities/Tools/QPandaException.h"
#include "ThirdParty/rapidjson/rapidjson.h"
#include "ThirdParty/rapidjson/rapidjson.h"
#include "ThirdParty/rapidjson/document.h"
#include "ThirdParty/rapidjson/writer.h"
#include "ThirdParty/rapidjson/prettywriter.h"
#include "ThirdParty/rapidjson/stringbuffer.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"

QPANDA_BEGIN


#define CONFIG_PATH  "QPandaConfig.json"
#define QCIRCUIT_OPTIMIZER ("QCircuitOptimizer")
#define VIRTUAL_Z_CONFIG ("Metadata")
#define QUBIT_ADJACENT_MATRIX ("QubitAdjacentMatrix")
#define HIGH_FREQUENCY_QUBIT ("HighFrequencyQubit")
#define COMPENSATE_ANGLE ("CompensateAngle")

/**
* @brief json  configuration parameter
* @ingroup Utilities
*/
class JsonConfigParam
{
public:
	JsonConfigParam() {}
	virtual ~JsonConfigParam() {};

	/**
    * @brief Load config data
    * @ingroup Utilities
    * @param[in] const std::string It can be configuration file or configuration data, which can be distinguished by file suffix,
			 so the configuration file must be end with ".json", default is CONFIG_PATH
    * @return Return false if any error occurs, otherwise return true
    */
	bool load_config(const std::string config_data = CONFIG_PATH);

	rapidjson::Document& get_root_element() { return m_doc; }

	bool getMetadataConfig(int &qubit_num, std::vector<std::vector<double>> &qubit_matrix);
	bool getClassNameConfig(std::map<std::string, std::string> &class_names);

	bool getQuantumCloudConfig(std::map<std::string, std::string> &cloud_config);
	bool getQGateConfig(std::vector<std::string> &single_gates, std::vector<std::string> &double_gates);
	bool getQGateTimeConfig(std::map<GateType, size_t> &gate_time);
	bool getInstructionConfig(std::map<std::string, std::map<std::string, uint32_t>> &);
	
	/**
	* @brief  read topological structure from json config file
	* @ingroup Utilities
	* @param[in]  const rapidjson::Value&  json value
	* @param[out]  int& qubit number
	* @param[out]  std::vector<std::vector<int>>  qubit matrix
	* @return     bool
	*/
	static bool readAdjacentMatrix(const rapidjson::Value& AdjacentMatrixElement, int &qubit_num, std::vector<std::vector<double>> &qubit_matrix);

	/** @brief load quantum circuit topological structure*/
	static bool loadQuantumTopoStructure(const std::string &xmlStr, const std::string& dataElementStr, int &qubitsCnt, std::vector< std::vector<double>> &vec, const std::string configFile = "");

private:
	rapidjson::Document m_doc;
	std::string m_json_content;
};

/**
* @brief Time Sequence Config
* @ingroup Utilities
*/
class TimeSequenceConfig
{
public:
	TimeSequenceConfig() 
		:m_load_config(false)
	{}
	~TimeSequenceConfig() {}

	void load_config(const std::string config_data = CONFIG_PATH);
	int get_measure_time_sequence();
	int get_ctrl_node_time_sequence();
	int get_swap_gate_time_sequence();
	int get_single_gate_time_sequence();
	int get_reset_time_sequence();

	int read_config(const char* config_type_str, int val);

private:
	JsonConfigParam m_config_file;
	bool m_load_config;
};

class QCircuitOptimizerConfig
{
#define ANGLE_VAR_BASE 1024

public:
	QCircuitOptimizerConfig(const std::string config_data = CONFIG_PATH);
	~QCircuitOptimizerConfig();

	bool get_replace_cir(std::vector<std::pair<QCircuit, QCircuit>>& replace_cir_vec,std::string key_name = QCIRCUIT_OPTIMIZER);
	bool get_u3_replace_cir(std::vector<std::pair<QCircuit, QCircuit>>& replace_cir_vec);

private:
	QCircuit read_cir(const rapidjson::Value& gates);
	QGate build_sing_ratation_gate(std::string gate_name, Qubit* target_qubit, double angle);
	QGate build_double_ratation_gate(std::string gate_name, Qubit* target_qubit, double angle,double phi);
	QGate build_three_ratation_gate(std::string gate_name, Qubit* target_qubit, double theta,double phi,double lamda);
	QGate build_sing_gate(std::string gate_name, Qubit* target_qubit);
	QGate build_double_gate(std::string gate_name, QVec qubits);
	QGate build_double_ratation_gate(std::string gate_name, QVec qubits, double angle);
	double angle_str_to_double(const std::string angle_str);

private:
	JsonConfigParam m_config_file;
	CPUQVM m_qvm;
	QVec m_qubits;
};

class QuantumChipConfig
{
public:
	QuantumChipConfig() {}
	~QuantumChipConfig() {}

	/**
	* @brief Load config data
	* @ingroup Utilities
	* @param[in] const std::string It can be configuration file or configuration data, which can be distinguished by file suffix,
			 so the configuration file must be end with ".json", default is CONFIG_PATH
	* @return Return false if any error occurs, otherwise return true
	*/
	bool load_config(const std::string config_data = CONFIG_PATH) {
		return m_config_reader.load_config(config_data);
	}

	bool read_adjacent_matrix(size_t &qubit_num, std::vector<std::vector<int>> &qubit_matrix);
	std::vector<int> read_high_frequency_qubit();
	std::vector<double> read_compensate_angle();
	void read_compensate_angle(std::map<std::pair<int, int>, std::vector<double>>&);
	size_t get_double_gate_clock(const size_t default_val = 3);
	size_t get_single_gate_clock(const size_t default_val = 1);

protected:
	const rapidjson::Value& get_virtual_z_config();

private:
	JsonConfigParam m_config_reader;
	std::string m_json_content;
};

QPANDA_END
#endif // JSON_CONFIG_PARAM_H

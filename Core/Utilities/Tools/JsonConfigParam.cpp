#include "Core/Utilities/Tools/JsonConfigParam.h"
#include <algorithm>
#include "Core/Utilities/Tools/TranformQGateTypeStringAndEnum.h"
#include "Core/Utilities/Tools/PraseExpressionStr.h"
#include <string.h>
#include "Core/Utilities/Tools/ArchGraph.h"

using namespace std;
USING_QPANDA

#define Q_GATE_TIME_SEQUENCE_CONFIG ("QGateTimeSequence")
#define Q_MEASURE_TIME_SEQUENCE ("QMeasureTimeSequence")
#define Q_SWAP_TIME_SEQUENCE ("QSwapTimeSequence")
#define Q_CONTROL_GATE_TIME_SEQUENCE ("QGateControlTimeSequence")
#define Q_SINGLE_GATE_TIME_SEQUENCE ("QGateSingleTimeSequence")
#define Q_RESET_TIME_SEQUENCE ("QResetNodeTimeSequence")

#define META_DATA ("Metadata")
#define CLASS_NAME_CONFIG ("ClassNameConfig")
#define QUBIT_COUNT ("QubitCount")
#define QUBIT_ADJACENT_MATRIX ("QubitAdjacentMatrix")
#define Q_CLOUD_CONFIG ("QuantumCloudConfig")
#define BACK_ENDS ("backends")
#define Q_GATE ("QGate")
#define SINGLE_GATE ("SingleGate")
#define DOUBLE_GATE ("DoubleGate")
#define MICRO_ARCHITECTURE ("Micro_Architecture")

#define DOUBLE_GATE_CLOCK "DoubleGateClock"
#define SINGLE_GATE_CLOCK "SingleGateClock"

#define QCIRCUIT_REPLACE ("replace")

/*******************************************************************
*                 class JsonConfigParam
********************************************************************/
bool JsonConfigParam::load_config(const std::string config_data/* = CONFIG_PATH*/)
{
	if (config_data.length() < 6)
	{
		return false;
	}

	string suffix = config_data.substr(config_data.length() - 5);
	transform(suffix.begin(), suffix.end(), suffix.begin(), ::tolower);
	if (0 == suffix.compare(".json"))
	{
		std::ifstream reader(config_data);
		if (!reader.is_open())
		{
			return false;
		}

		m_json_content = std::string((std::istreambuf_iterator<char>(reader)), std::istreambuf_iterator<char>());
		reader.close();
	}
	else
	{
		m_json_content = config_data;
	}
	
	if (m_doc.Parse(m_json_content.c_str()).HasParseError())
	{
		QCERR_AND_THROW(run_fail, "Error: failed to parse the config file.");
	}

	return true;
}

bool JsonConfigParam::getMetadataConfig(int &qubit_num, std::vector<std::vector<double>> &qubit_matrix) 
{
	std::unique_ptr<ArchGraph> p_arch_graph = JsonBackendParser<ArchGraph>::Parse(m_doc);
	qubit_num = p_arch_graph->get_vertex_count();
	qubit_matrix = p_arch_graph->get_adj_weight_matrix();

	return true;
}

bool JsonConfigParam::getClassNameConfig(std::map<std::string, std::string> &class_names)
{
	if (m_doc.HasMember(CLASS_NAME_CONFIG))
	{
		auto& class_name_array = m_doc[CLASS_NAME_CONFIG];
		for (rapidjson::Value::ConstMemberIterator iter = class_name_array.MemberBegin(); iter != class_name_array.MemberEnd(); ++iter)
		{
			std::string str_key = iter->name.GetString();
			std::string str_val = class_name_array[str_key.c_str()].GetString();
			class_names.insert(std::pair<std::string, std::string>(str_key, str_val));
		}

		return true;
	}

	return false;
}

bool JsonConfigParam::readAdjacentMatrix(const rapidjson::Value& AdjacentMatrixElement, int &qubit_num, std::vector<std::vector<double>> &qubit_matrix)
{
	qubit_matrix.clear();

	if (!(AdjacentMatrixElement.HasMember(QUBIT_COUNT)))
	{
		return false;
	}
	qubit_num = AdjacentMatrixElement[QUBIT_COUNT].GetInt();

	if ((AdjacentMatrixElement.HasMember(QUBIT_ADJACENT_MATRIX)) && AdjacentMatrixElement[QUBIT_ADJACENT_MATRIX].IsArray())
	{
		const rapidjson::Value& arr_row = AdjacentMatrixElement[QUBIT_ADJACENT_MATRIX];
		for (int i = 0; i < arr_row.Size(); ++i) 
		{
			std::vector<double> tmp_vec;
			const rapidjson::Value& arr_col = arr_row[i];
			for (int j = 0; j < arr_col.Size(); ++j)
			{
				tmp_vec.push_back(arr_col[j].GetDouble());
			}
			qubit_matrix.push_back(tmp_vec);
		}

		return true;
	}

	return false;
}

bool JsonConfigParam::getQuantumCloudConfig(std::map<std::string, std::string> &cloud_config)
{
	if (!(m_doc.HasMember(Q_CLOUD_CONFIG)))
	{
		return false;
	}

	auto& tmp_config = m_doc[Q_CLOUD_CONFIG];
	for (rapidjson::Value::ConstMemberIterator iter = tmp_config.MemberBegin(); iter != tmp_config.MemberEnd(); ++iter)
	{
		std::string str_key = iter->name.GetString();
		if (tmp_config[str_key.c_str()].IsString())
		{
			std::string str_val = tmp_config[str_key.c_str()].GetString();
			cloud_config.insert(std::pair<std::string, std::string>(str_key, str_val));
		}
	}

	return true;
}

bool JsonConfigParam::loadQuantumTopoStructure(const std::string &xmlStr, const std::string& dataElementStr, 
	int &qubitsCnt, std::vector< std::vector<double>> &vec, const std::string configFile/* = ""*/)
{
	rapidjson::Document doc;
	if (configFile.empty())
	{
		doc.Parse(xmlStr.c_str());
	}
	else
	{
		std::ifstream reader(configFile);
		if (!reader.is_open())
		{
			QCERR_AND_THROW(run_fail, "Error: failed to open the config file.");
		}

		std::string json_content = std::string((std::istreambuf_iterator<char>(reader)), std::istreambuf_iterator<char>());
		reader.close();

		if (doc.Parse(json_content.c_str()).HasParseError())
		{
			QCERR_AND_THROW(run_fail, "Error: failed to parse the config file.");
		}
	}

	if (!(doc.HasMember(BACK_ENDS)))
	{
		return false;
	}

	const auto& back_end_config = doc[BACK_ENDS];
	if (!(back_end_config.HasMember(dataElementStr.c_str())))
	{
		return false;
	}

	const auto& metadata_element = back_end_config[dataElementStr.c_str()];
	return readAdjacentMatrix(metadata_element, qubitsCnt, vec);
}

bool JsonConfigParam::getQGateConfig(std::vector<std::string> &single_gates, std::vector<std::string> &double_gates)
{
	if (!(m_doc.HasMember(Q_GATE)))
	{
		return false;
	}

	const auto& gate_config = m_doc[Q_GATE];
	if (!(gate_config.HasMember(SINGLE_GATE)))
	{
		return false;
	}

	const auto& single_gate_config = gate_config[SINGLE_GATE];
	for (rapidjson::Value::ConstMemberIterator iter = single_gate_config.MemberBegin(); iter != single_gate_config.MemberEnd(); ++iter)
	{
		std::string str_gate = iter->name.GetString();
		const auto& gate_attribute = single_gate_config[str_gate.c_str()];

		std::transform(str_gate.begin(), str_gate.end(), str_gate.begin(), ::toupper);
		single_gates.emplace_back(str_gate);
		for (rapidjson::Value::ConstMemberIterator attribute_iter = gate_attribute.MemberBegin();
			attribute_iter != gate_attribute.MemberEnd(); ++attribute_iter)
		{
			std::string str_attribute = attribute_iter->name.GetString();
			int val = gate_attribute[str_attribute.c_str()].GetInt();
		}
	}

	const auto& double_gate_config = gate_config[DOUBLE_GATE];
	for (rapidjson::Value::ConstMemberIterator iter = double_gate_config.MemberBegin(); iter != double_gate_config.MemberEnd(); ++iter)
	{
		std::string str_gate = iter->name.GetString();
		const auto& gate_attribute = double_gate_config[str_gate.c_str()];

		std::transform(str_gate.begin(), str_gate.end(), str_gate.begin(), ::toupper);
		double_gates.emplace_back(str_gate);
		
		for (rapidjson::Value::ConstMemberIterator attribute_iter = gate_attribute.MemberBegin();
			attribute_iter != gate_attribute.MemberEnd(); ++attribute_iter)
		{
			std::string str_attribute = attribute_iter->name.GetString();
			int val = gate_attribute[str_attribute.c_str()].GetInt();
		}
	}

	return true;
}

bool JsonConfigParam::getQGateTimeConfig(std::map<GateType, size_t> &gate_time)
{
	gate_time.clear();

	if (!(m_doc.HasMember(Q_GATE)))
	{
		return false;
	}

	const auto& gate_config = m_doc[Q_GATE];
	if (!(gate_config.HasMember(SINGLE_GATE)))
	{
		return false;
	}

	TransformQGateType &tranform_gate_type = TransformQGateType::getInstance();
	const auto& single_gate_config = gate_config[SINGLE_GATE];
	for (rapidjson::Value::ConstMemberIterator iter = single_gate_config.MemberBegin(); iter != single_gate_config.MemberEnd(); ++iter)
	{
		std::string str_gate = iter->name.GetString();
		GateType gate_type = tranform_gate_type[str_gate];
		const auto& gate_attribute = single_gate_config[str_gate.c_str()];
		for (rapidjson::Value::ConstMemberIterator attribute_iter = gate_attribute.MemberBegin();
			attribute_iter != gate_attribute.MemberEnd(); ++attribute_iter)
		{
			std::string str_attribute = attribute_iter->name.GetString();
			int val = gate_attribute[str_attribute.c_str()].GetInt();
			gate_time.insert({ gate_type, val });
		}
	}

	const auto& double_gate_config = gate_config[DOUBLE_GATE];
	for (rapidjson::Value::ConstMemberIterator iter = double_gate_config.MemberBegin(); iter != double_gate_config.MemberEnd(); ++iter)
	{
		std::string str_gate = iter->name.GetString();
		GateType gate_type = tranform_gate_type[str_gate];
		const auto& gate_attribute = double_gate_config[str_gate.c_str()];
		for (rapidjson::Value::ConstMemberIterator attribute_iter = gate_attribute.MemberBegin();
			attribute_iter != gate_attribute.MemberEnd(); ++attribute_iter)
		{
			std::string str_attribute = attribute_iter->name.GetString();
			int val = gate_attribute[str_attribute.c_str()].GetInt();
			gate_time.insert({ gate_type, val });
		}
	}

	return true;
}

bool JsonConfigParam::getInstructionConfig(std::map<std::string, std::map<std::string, uint32_t>> &ins_config)
{
	if (!(m_doc.HasMember(MICRO_ARCHITECTURE)))
	{
		return false;
	}

	auto& instruction_element = m_doc[MICRO_ARCHITECTURE];
	for (rapidjson::Value::ConstMemberIterator group_iter = instruction_element.MemberBegin(); 
		group_iter != instruction_element.MemberEnd(); ++group_iter)
	{
		std::string str_group = group_iter->name.GetString();
		map<string, uint32_t> group_config;
		auto& tmp_group = instruction_element[str_group.c_str()];
		for (rapidjson::Value::ConstMemberIterator item_iter = tmp_group.MemberBegin();
			item_iter != tmp_group.MemberEnd(); ++item_iter)
		{
			std::string str_key = item_iter->name.GetString();
			int str_val = tmp_group[str_key.c_str()].GetInt();
			group_config.insert(make_pair(str_key, str_val));
		}

		ins_config.insert(make_pair(str_group, group_config));
	}

	return true;
}

/*******************************************************************
*                 class TimeSequenceConfig
********************************************************************/
void TimeSequenceConfig::load_config(const std::string config_data /*= CONFIG_PATH*/)
{
	m_load_config = m_config_file.load_config(config_data);
}

int TimeSequenceConfig::read_config(const char* config_type_str, int val)
{
	if (!m_load_config)
	{
		return val;
	}

	auto& config_elem = (m_config_file.get_root_element())[Q_GATE_TIME_SEQUENCE_CONFIG];
	int ret = val; //default val
	if (config_elem.HasMember(config_type_str) && config_elem[config_type_str].IsInt())
	{
		ret = config_elem[config_type_str].GetInt();
	}

	return ret;
}

int TimeSequenceConfig::get_measure_time_sequence()
{
	return read_config(Q_MEASURE_TIME_SEQUENCE, 2);
}

int TimeSequenceConfig::get_ctrl_node_time_sequence()
{
	return read_config(Q_CONTROL_GATE_TIME_SEQUENCE, 2);
}

int TimeSequenceConfig::get_swap_gate_time_sequence()
{
	return read_config(Q_SWAP_TIME_SEQUENCE, 2);
}

int TimeSequenceConfig::get_single_gate_time_sequence()
{
	return read_config(Q_SINGLE_GATE_TIME_SEQUENCE, 1);
}

int TimeSequenceConfig::get_reset_time_sequence()
{
	return read_config(Q_RESET_TIME_SEQUENCE, 1);
}

/*******************************************************************
*                 class QCircuitOptimizerConfig
********************************************************************/
QCircuitOptimizerConfig::QCircuitOptimizerConfig(const std::string config_data /*= CONFIG_PATH*/)
{ 
	m_config_file.load_config(config_data);
	m_qvm.init();
}

QCircuitOptimizerConfig::~QCircuitOptimizerConfig()
{
	m_qvm.finalize();
}

bool QCircuitOptimizerConfig::get_replace_cir(std::vector<std::pair<QCircuit, QCircuit>>& replace_cir_vec, string key_name)
{
	auto& doc = m_config_file.get_root_element();
	if (!(doc.HasMember(key_name.c_str())))
	{
		return false;
	}

	auto& optimizer_config = doc[key_name.c_str()];
	if (!(optimizer_config.HasMember(QCIRCUIT_REPLACE)) || (!(optimizer_config[QCIRCUIT_REPLACE].IsArray())))
	{
		return false;
	}
	auto& replace_cir_config = optimizer_config[QCIRCUIT_REPLACE];
	for (size_t i = 0; i < replace_cir_config.Size(); ++i)
	{
		auto& replace_item = replace_cir_config[i];
		{
			int qubit_num = replace_item["qubits"].GetInt();
			if (m_qubits.size() < qubit_num)
			{
				auto q = m_qvm.allocateQubits(qubit_num - m_qubits.size());
				m_qubits += q;
			}
			auto& src_cir_config = replace_item["src"];
			QCircuit src_cir = read_cir(src_cir_config);
			auto& dst_cir_config = replace_item["dst"];
			QCircuit dst_cir = read_cir(dst_cir_config);
			replace_cir_vec.push_back(std::make_pair(src_cir, dst_cir));
		}
	}

	return true;
}

bool QCircuitOptimizerConfig::get_u3_replace_cir(std::vector<std::pair<QCircuit, QCircuit>>& replace_cir_vec)
{
	return get_replace_cir(replace_cir_vec,"U3Optimizer");
}

QCircuit QCircuitOptimizerConfig::read_cir(const rapidjson::Value& gates)
{
	QCircuit ret_cir;
	for (rapidjson::Value::ConstMemberIterator gate_iter = gates.MemberBegin(); gate_iter != gates.MemberEnd(); ++gate_iter)
	{
		std::string gate_name = gate_iter->name.GetString();
		transform(gate_name.begin(), gate_name.end(), gate_name.begin(), ::toupper);
		auto& gate_para = gate_iter->value;
		if ((0 == strcmp(gate_name.c_str(), "H")) ||
			(0 == strcmp(gate_name.c_str(), "X")) ||
			(0 == strcmp(gate_name.c_str(), "Y")) ||
			(0 == strcmp(gate_name.c_str(), "Z")) ||
			(0 == strcmp(gate_name.c_str(), "T")) ||
			(0 == strcmp(gate_name.c_str(), "S")))
		{
			ret_cir << build_sing_gate(gate_name, { m_qubits[gate_para[0].GetInt()] });
		}
		else if ((0 == strcmp(gate_name.c_str(), "CNOT")) ||
			(0 == strcmp(gate_name.c_str(), "CZ")) ||
			(0 == strcmp(gate_name.c_str(), "SWAP")) ||
			(0 == strcmp(gate_name.c_str(), "SQISWAP")))
		{
			ret_cir << build_double_gate(gate_name, { m_qubits[gate_para[0].GetInt()], m_qubits[gate_para[1].GetInt()] });
		}
		else if ((0 == strcmp(gate_name.c_str(), "RX")) ||
			(0 == strcmp(gate_name.c_str(), "RY")) ||
			(0 == strcmp(gate_name.c_str(), "RZ")) ||
			(0 == strcmp(gate_name.c_str(), "U1")))
		{
			double angle = 0;
			if(gate_para[1].IsDouble())
			{
				angle = gate_para[1].GetDouble();
			}
			else if (gate_para[1].IsString())
			{
				string angle_str =  gate_para[1].GetString();
				angle = angle_str_to_double(angle_str);
			}
			else
			{
				QCERR_AND_THROW(run_fail, "Error: angle config error.");
			}
			
			ret_cir << build_sing_ratation_gate(gate_name, m_qubits[gate_para[0].GetInt()], angle);
		}
		else if(0 == strcmp(gate_name.c_str(), "RPhi") ||
				0 == strcmp(gate_name.c_str(), "RPHI"))
		{
			double angle = gate_para[1].GetDouble();
			double phi =gate_para[2].GetDouble();

			ret_cir << build_double_ratation_gate("RPhi", m_qubits[gate_para[0].GetInt()], angle,phi);
		}
		else if(0 == strcmp(gate_name.c_str(), "U3"))
		{
			double theta = 0;
			if (gate_para[1].IsString())
			{
				string theta_str =  gate_para[1].GetString();
				theta = angle_str_to_double(theta_str);
			}
			else
			{
				theta = gate_para[1].GetDouble();
			}

			double phi=0;
			if(gate_para[2].IsString())
			{
				string phi_str =  gate_para[2].GetString();
				phi = angle_str_to_double(phi_str);
			}
			else 
			{
				phi = gate_para[2].GetDouble();
			}

			double lamda=0;
			if (gate_para[3].IsString())
			{
				string lamda_str =  gate_para[3].GetString();
				lamda = angle_str_to_double(lamda_str);
			}
			else
			{
				lamda = gate_para[3].GetDouble();
			}
			
			ret_cir << build_three_ratation_gate(gate_name, m_qubits[gate_para[0].GetInt()], theta,phi,lamda);
		}
		else if ((0 == strcmp(gate_name.c_str(), "ISWAP")) ||
			(0 == strcmp(gate_name.c_str(), "CR")))
		{
			string angle_str = gate_para[1].GetString();
			//string to double 
			double angle = angle_str_to_double(angle_str);
			ret_cir << build_double_ratation_gate(gate_name, { m_qubits[gate_para[0].GetInt()], m_qubits[gate_para[1].GetInt()] }, angle);
		}
		else
		{
			QCERR_AND_THROW(run_fail, "Error: unknow error on read_cir form config file.");
		}
	}

	return ret_cir;
}

double QCircuitOptimizerConfig::angle_str_to_double(const string angle_str)
{
	double ret = 0.0;
	if (0 == strncmp(angle_str.c_str(), "theta_", 6))
	{
		ret = ANGLE_VAR_BASE * atoi(angle_str.c_str() + 6);
	}
	else
	{
		ret = ParseExpressionStr().parse(angle_str);
	}	

	return ret;
}

inline QGate QCircuitOptimizerConfig::build_sing_ratation_gate(std::string gate_name, Qubit* target_qubit, double angle)
{
	const auto p_fac = QGateNodeFactory::getInstance();
	return QGateNodeFactory::getInstance()->getGateNode(gate_name, { target_qubit }, angle);
}

inline QGate QCircuitOptimizerConfig::build_double_ratation_gate(std::string gate_name, Qubit* target_qubit, double angle,double phi)
{
	const auto p_fac = QGateNodeFactory::getInstance();
	return QGateNodeFactory::getInstance()->getGateNode(gate_name, { target_qubit }, angle, phi);
}

inline QGate QCircuitOptimizerConfig::build_three_ratation_gate(std::string gate_name, Qubit* target_qubit, double theta,double phi,double lamda)
{
	const auto p_fac = QGateNodeFactory::getInstance();
	return QGateNodeFactory::getInstance()->getGateNode(gate_name, { target_qubit }, theta,phi,lamda);
}

inline QGate QCircuitOptimizerConfig::build_sing_gate(std::string gate_name, Qubit* target_qubit)
{
	return QGateNodeFactory::getInstance()->getGateNode(gate_name, { target_qubit });
}

inline QGate QCircuitOptimizerConfig::build_double_gate(std::string gate_name, QVec qubits)
{
	return QGateNodeFactory::getInstance()->getGateNode(gate_name, qubits);
}

inline QGate QCircuitOptimizerConfig::build_double_ratation_gate(std::string gate_name, QVec qubits, double angle)
{
	return QGateNodeFactory::getInstance()->getGateNode(gate_name, qubits, angle);
}

/*******************************************************************
*                 class QuantumChipConfig
********************************************************************/
bool QuantumChipConfig::read_adjacent_matrix(size_t &qubit_num, std::vector<std::vector<int>> &qubit_matrix)
{
	std::unique_ptr<ArchGraph> p_arch_graph = JsonBackendParser<ArchGraph>::Parse(m_config_reader.get_root_element());
	qubit_num = p_arch_graph->get_vertex_count();
	qubit_matrix = p_arch_graph->get_adjacent_matrix();
	return true;
}

std::vector<int> QuantumChipConfig::read_high_frequency_qubit() 
{
	std::vector<int> high_frequency_qubits;

	auto& virtual_z_config = get_virtual_z_config();
	if (!(virtual_z_config.HasMember(HIGH_FREQUENCY_QUBIT)))
	{
		return high_frequency_qubits;
	}

	auto& high_frequency_qubit_conf = virtual_z_config[HIGH_FREQUENCY_QUBIT];
	for (int i = 0; i < high_frequency_qubit_conf.Size(); ++i)
	{
		high_frequency_qubits.push_back(high_frequency_qubit_conf[i].GetInt());
	}

	return high_frequency_qubits;
}

size_t QuantumChipConfig::get_double_gate_clock(const size_t default_val/* = 3*/)
{
	auto& virtual_z_config = get_virtual_z_config();
	if ((virtual_z_config.HasMember(DOUBLE_GATE_CLOCK)) && (virtual_z_config[DOUBLE_GATE_CLOCK].IsInt()))
	{
		return virtual_z_config[DOUBLE_GATE_CLOCK].GetInt();
	}

	return default_val;
}

size_t QuantumChipConfig::get_single_gate_clock(const size_t default_val/* = 1*/)
{
	auto& virtual_z_config = get_virtual_z_config();
	if ((virtual_z_config.HasMember(SINGLE_GATE_CLOCK)) && (virtual_z_config[SINGLE_GATE_CLOCK].IsInt()))
	{
		return virtual_z_config[SINGLE_GATE_CLOCK].GetInt();
	}

	return default_val;
}

std::vector<double> QuantumChipConfig::read_compensate_angle()
{
	std::vector<double> compensate_angle;
	auto& virtual_z_config = get_virtual_z_config();
	if (!(virtual_z_config.HasMember(COMPENSATE_ANGLE)))
	{
		return compensate_angle;
	}

	auto& compensate_angle_conf = virtual_z_config[COMPENSATE_ANGLE];
	for (int i = 0; i < compensate_angle_conf.Size(); ++i)
	{
		compensate_angle.push_back(compensate_angle_conf[i].GetDouble());
	}

	return compensate_angle;
}

void QuantumChipConfig::read_compensate_angle(std::map<std::pair<int, int>, std::vector<double>>& compensate_angle_map)
{
	compensate_angle_map.clear();
	auto& virtual_z_config = get_virtual_z_config();
	if (!(virtual_z_config.HasMember(COMPENSATE_ANGLE)))
	{
		QCERR("Failed to read_compensate_angle.");
		return;
	}

	auto& compensate_angle_conf = virtual_z_config[COMPENSATE_ANGLE];
	std::string qubit_str;
	string qubit_1;
	string qubit_2;
	for (rapidjson::Value::ConstMemberIterator iter = compensate_angle_conf.MemberBegin(); iter != compensate_angle_conf.MemberEnd(); ++iter)
	{
		std::string str_key = iter->name.GetString();
		size_t start_pos = str_key.find_first_of('(') + 1;
		qubit_str = str_key.substr(start_pos);
		qubit_1 = qubit_str.substr(0, qubit_str.find_first_of(','));
		start_pos = qubit_str.find_first_of(',') + 1;
		qubit_2 = qubit_str.substr(start_pos, qubit_str.find_first_of(')') - start_pos);
		auto qubis_pair = std::make_pair(stoi(qubit_1), stoi(qubit_2));
		
		std::vector<double> angle_vec;
		const auto& angle_val = compensate_angle_conf[str_key.c_str()];
		for (int i = 0; i < angle_val.Size(); ++i)
		{
			if (angle_val[i].IsDouble())
			{
				angle_vec.push_back(angle_val[i].GetDouble());
			}
			else if (angle_val[i].IsInt())
			{
				angle_vec.push_back((double)(angle_val[i].GetInt()));
			}
			else
			{
				QCERR_AND_THROW(run_fail, "Error: compensate_angle_conf error.");
			}
		}

		compensate_angle_map.insert(std::make_pair(qubis_pair, angle_vec));
	}

	return ;
}

const rapidjson::Value& QuantumChipConfig::get_virtual_z_config()
{
	auto& doc = m_config_reader.get_root_element();
	if (!(doc.HasMember(VIRTUAL_Z_CONFIG)))
	{
		QCERR_AND_THROW(init_fail, "Error: virtual_Z_config error.");
	}

	return doc[VIRTUAL_Z_CONFIG];
}

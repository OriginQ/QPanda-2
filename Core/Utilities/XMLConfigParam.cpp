#include "XMLConfigParam.h"
#include <algorithm>
#include "Core/Utilities/TranformQGateTypeStringAndEnum.h"

using namespace std;
USING_QPANDA
XmlConfigParam::XmlConfigParam() :
    m_root_element(nullptr)
{

}

bool XmlConfigParam::loadFile(const std::string & filename)
{
    if (!m_doc.LoadFile(filename.c_str()))
    {
        return false;
    }

    m_root_element = m_doc.RootElement();
    return true;
}

bool XmlConfigParam::getMetadataConfig(int &qubit_num, std::vector<std::vector<int>> &qubit_matrix)
{
    if (!m_root_element)
    {
        return false;
    }

    TiXmlElement *metadata_element = m_root_element->FirstChildElement("Metadata");
    if (!metadata_element)
    {
        return false;
    }

    TiXmlElement *qubit_num_element = metadata_element->FirstChildElement("QubitCount");
    qubit_num = std::stoi(qubit_num_element->GetText());
    vector<int> arr(qubit_num, 0);
    qubit_matrix.resize(qubit_num, arr);

    TiXmlElement *matrix_element = metadata_element->FirstChildElement("QubitMatrix");
    if (!matrix_element)
    {
        return false;
    }

    for (TiXmlElement *qubit_element = matrix_element->FirstChildElement("Qubit");
        qubit_element;
        qubit_element = qubit_element->NextSiblingElement("Qubit"))
    {
        string attr = qubit_element->Attribute("QubitNum");
        if (attr.empty())
        {
            return false;
        }

        int i  = std::stoi(attr);
        if (!i || i > qubit_num)
        {
            return false;
        }

        for (TiXmlElement *adjacent_qubit_element = qubit_element->FirstChildElement("AdjacentQubit");
            adjacent_qubit_element;
            adjacent_qubit_element = adjacent_qubit_element->NextSiblingElement("AdjacentQubit"))
        {
            string attr = adjacent_qubit_element->Attribute("QubitNum");
            if (attr.empty())
            {
                return false;
            }

            int j = std::stoi(attr);
            if (!j || j > qubit_num)
            {
                return false;
            }
            string item_text = adjacent_qubit_element->GetText();
            qubit_matrix[i - 1][j - 1] = std::stoi(item_text);
        }
    }

    return true;
}

bool XmlConfigParam::getClassNameConfig(map<string, string> &class_names)
{
    if (!m_root_element)
    {
        return false;
    }

    TiXmlElement *class_name_config_element = m_root_element->FirstChildElement("ClassNameConfig");
    if (!class_name_config_element)
    {
        return false;
    }

    for(TiXmlElement *class_msg_element = class_name_config_element->FirstChildElement();
        class_msg_element;
        class_msg_element = class_msg_element->NextSiblingElement())
    {
        if (!class_msg_element->GetText())
            continue;
        class_names.insert(pair<string, string>(class_msg_element->Value(), class_msg_element->GetText()));
    }

    return true;
}

bool XmlConfigParam::getQuantumCloudConfig(std::map<std::string, std::string>& cloud_config)
{
    if (!m_root_element)
    {
        return false;
    }

    TiXmlElement *cloud_config_element = m_root_element->FirstChildElement("QuantumCloudConfig");
    if (!cloud_config_element)
    {
        return false;
    }

    for (TiXmlElement *class_msg_element = cloud_config_element->FirstChildElement();
        class_msg_element;
        class_msg_element = class_msg_element->NextSiblingElement())
    {
        if (!class_msg_element->GetText())
            continue;
        cloud_config.insert(pair<string, string>(class_msg_element->Value(), class_msg_element->GetText()));
    }

    return true;
}

bool XmlConfigParam::getQGateConfig(std::vector<std::string> &single_gates, std::vector<std::string>& double_gates)
{
    TiXmlElement *qgate_element = m_root_element->FirstChildElement("QGate");
    if (!qgate_element)
    {
        return false;
    }

    TiXmlElement *single_gate_element = qgate_element->FirstChildElement("SingleGate");
    if (!single_gate_element)
    {
        return false;
    }

    for (TiXmlElement *gate_element = single_gate_element->FirstChildElement("Gate");
        gate_element;
        gate_element = gate_element->NextSiblingElement("Gate"))
    {
        if (gate_element)
        {
            string gate_str = gate_element->GetText();
            std::transform(gate_str.begin(), gate_str.end(), gate_str.begin(), ::toupper);
            single_gates.emplace_back(gate_str);
        }
    }

    TiXmlElement *double_gate_element = qgate_element->FirstChildElement("DoubleGate");
    if (!double_gate_element)
    {
        return false;
    }

    for (TiXmlElement *gate_element = double_gate_element->FirstChildElement("Gate");
        gate_element;
        gate_element = gate_element->NextSiblingElement("Gate"))
    {
        if (gate_element)
        {
            string gate_str = gate_element->GetText();
            std::transform(gate_str.begin(), gate_str.end(), gate_str.begin(), ::toupper);
            double_gates.emplace_back(gate_str);
        }
    }

    return true;
}

bool XmlConfigParam::getQGateTimeConfig(std::map<GateType, size_t>& gate_time)
{
    TiXmlElement *qgate_element = m_root_element->FirstChildElement("QGate");
    if (!qgate_element)
    {
        return false;
    }
    TransformQGateType &tranform_gate_type = TransformQGateType::getInstance();

    TiXmlElement *single_gate_element = qgate_element->FirstChildElement("SingleGate");
    if (!single_gate_element)
    {
        return false;
    }

    for (TiXmlElement *gate_element = single_gate_element->FirstChildElement("Gate");
        gate_element;
        gate_element = gate_element->NextSiblingElement("Gate"))
    {
        if (gate_element)
        {
            string gate_str = gate_element->GetText();
            std::transform(gate_str.begin(), gate_str.end(), gate_str.begin(), ::toupper);
            GateType gate_type = tranform_gate_type[gate_str];
            size_t time = std::stoll(gate_element->Attribute("time"));
            gate_time.insert({gate_type, time});
        }
    }

    TiXmlElement *double_gate_element = qgate_element->FirstChildElement("DoubleGate");
    if (!double_gate_element)
    {
        return false;
    }

    for (TiXmlElement *gate_element = double_gate_element->FirstChildElement("Gate");
        gate_element;
        gate_element = gate_element->NextSiblingElement("Gate"))
    {
        if (gate_element)
        {
            string gate_str = gate_element->GetText();
            std::transform(gate_str.begin(), gate_str.end(), gate_str.begin(), ::toupper);
            GateType gate_type = tranform_gate_type[gate_str];
            size_t time = std::stoll(gate_element->Attribute("time"));
            gate_time.insert({ gate_type, time });
        }
    }

    return true;
}

bool XmlConfigParam::getInstructionConfig(std::map<std::string, std::map<std::string, uint32_t>>& ins_config)
{
    if (!m_root_element)
    {
        return false;
    }

    TiXmlElement *instruction_element = 
        m_root_element->FirstChildElement("Micro-Architecture");
    if (!instruction_element)
    {
        return false;
    }

    for (TiXmlElement *group_element = instruction_element->FirstChildElement(); 
        group_element; group_element = group_element->NextSiblingElement())
    {
        map<string, uint32_t> group_config;
        for (TiXmlElement *element = group_element->FirstChildElement();
            element;element = element->NextSiblingElement())
        {
            group_config.insert(make_pair(element->Value(), stoul(element->GetText())));
        }
        ins_config.insert(make_pair(group_element->Value(), group_config));
    }
    return true;
}


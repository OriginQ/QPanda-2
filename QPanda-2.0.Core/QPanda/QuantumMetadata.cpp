#include "QuantumMetadata.h"
#include <algorithm>
#include <string>
#include "QPandaException.h"
const size_t _G_qubit_count = 4;
vector<vector<int>> _G_qubitMatrix = { {0,1,1,0},
                                       {1,0,0,1},
                                       {1,0,0,1},
                                       {0,1,1,0} };

QuantumMetadata::QuantumMetadata():m_root_element(nullptr)
{

}

QuantumMetadata::QuantumMetadata(const string & filename):
    m_doc(filename.c_str()), m_root_element(nullptr)
{
    if (!m_doc.LoadFile())
    {
        throw param_error_exception("load file failure", false);
    }
    m_root_element = m_doc.RootElement();
}

size_t QuantumMetadata::getQubitCount()
{
    if (!m_root_element)
    {
        return _G_qubit_count;
    }

    TiXmlElement *first_element = m_root_element->FirstChildElement("QubitCount");
    if (!first_element)
    {
        return false;
    }

    const char *c_str = first_element->GetText();
    size_t qubit_count = strtoul(c_str, nullptr, 0);
    return qubit_count;
}

bool QuantumMetadata::getQubiteMatrix(vector<vector<int>> & qubit_matrix)
{
    if (!m_root_element)
    {
        for (auto first_aiter : _G_qubitMatrix)
        {
            vector<int> temp;
            for (auto second_aiter : first_aiter)
            {
                temp.push_back(second_aiter);
            }
            qubit_matrix.push_back(temp);
        }
        return true;
    }

    TiXmlElement *first_element = m_root_element->FirstChildElement("QubitCount");
    if (!first_element)
    {
        return false;
    }

    const char *c_str = first_element->GetText();
    size_t qubit_count = strtoul(c_str, nullptr, 0);
    vector<vector<int>> tmp_vec(qubit_count, vector<int>(qubit_count, 0));
    qubit_matrix = tmp_vec;

    TiXmlElement *matrix_element = m_root_element->FirstChildElement("QubitMatrix");
    if (!matrix_element)
    {
        return false;
    }

    for (TiXmlElement *qubit_element = matrix_element->FirstChildElement("Qubit");
         qubit_element;
         qubit_element = qubit_element->NextSiblingElement("Qubit"))
    {
        const char *attr = qubit_element->Attribute("QubitNum");
        if (!attr)
        {
            return false;
        }

        size_t i = strtoul(attr, nullptr, 0);
        if (!i || i > qubit_count)
        {
            return false;
        }

        for (TiXmlElement *adjacent_qubit_element = qubit_element->FirstChildElement("AdjacentQubit");
             adjacent_qubit_element;
             adjacent_qubit_element = adjacent_qubit_element->NextSiblingElement("AdjacentQubit"))
        {
            const char* attr = adjacent_qubit_element->Attribute("QubitNum");
            if (!attr)
            {
                return false;
            }

            size_t j = strtoul(attr, nullptr, 0);
            if (!j || j > qubit_count)
            {
                return false;
            }
            const char *item_text = adjacent_qubit_element->GetText();
            qubit_matrix[i-1][j-1] = atoi(item_text);
        }
    }

    return true;
}

bool QuantumMetadata::getSingleGate(vector<string> &single_gate)
{
    if (!m_root_element)
    {
        single_gate.push_back("RX");
        single_gate.push_back("RY");
        single_gate.push_back("RZ");
        single_gate.push_back("X1");
        single_gate.push_back("H");
        single_gate.push_back("S");
        return true;
    }

    TiXmlElement *single_gate_element = m_root_element->FirstChildElement("SingleGate");
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
            transform(gate_str.begin(), gate_str.end(), gate_str.begin(), ::toupper);
            single_gate.emplace_back(gate_str);
        }
    }

    return true;
}

bool QuantumMetadata::getDoubleGate(vector<string> &double_gate)
{
    if (!m_root_element)
    {
        double_gate.push_back("CNOT");
        double_gate.push_back("CZ");
        double_gate.push_back("ISWAP");
        return true;
    }

    TiXmlElement *double_gate_element = m_root_element->FirstChildElement("DoubleGate");
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
            transform(gate_str.begin(), gate_str.end(), gate_str.begin(), ::toupper);
            double_gate.emplace_back(gate_str);
        }
    }

    return true;
}

bool QuantumMetadata::getGateTime(map<int, size_t> &gate_time_map)
{    
    if (!m_root_element)
    {   
        insertGateTimeMap({"RX", 2}, gate_time_map);
        insertGateTimeMap({"RY", 2}, gate_time_map);
        insertGateTimeMap({"RZ", 2}, gate_time_map);
        insertGateTimeMap({"X1", 2}, gate_time_map);
        insertGateTimeMap({"H", 2}, gate_time_map);
        insertGateTimeMap({"S", 2}, gate_time_map);

        insertGateTimeMap({"CNOT", 2}, gate_time_map);
        insertGateTimeMap({"CZ", 2}, gate_time_map);
        insertGateTimeMap({"ISWAP", 2}, gate_time_map);
        
        return true;
    }

    TiXmlElement *single_gate_element = m_root_element->FirstChildElement("SingleGate");
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
            transform(gate_str.begin(), gate_str.end(), gate_str.begin(), ::toupper);
            size_t time = std::stoi(gate_element->Attribute("time"));
            pair<string, size_t> single_gate = {gate_str, time};
            insertGateTimeMap(single_gate, gate_time_map);
        }
    }

    TiXmlElement *double_gate_element = m_root_element->FirstChildElement("DoubleGate");
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
            transform(gate_str.begin(), gate_str.end(), gate_str.begin(), ::toupper);
            size_t time = std::stoi(gate_element->Attribute("time"));
            pair<string, size_t> double_gate = {gate_str, time};
            insertGateTimeMap(double_gate, gate_time_map);
        }
    }

    return true;
}

void QuantumMetadata::insertGateTimeMap(const pair<string, size_t> &gate_time, map<int, size_t> &gate_time_map)
{

    pair<int, size_t> gate_type_time(QGateTypeStringToEnum::getInstance()[gate_time.first],
                                            gate_time.second);
    gate_time_map.insert(gate_type_time);

    return ;
}


QuantumMetadata::~QuantumMetadata()
{
}


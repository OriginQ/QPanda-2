/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

XMLConfigParam.h
Author: Wangjing
Created in 2018-8-31

Classes for get the shortes path of graph

*/
#ifndef XMLCONFIGPARAM_H
#define XMLCONFIGPARAM_H
#include "Core/Utilities/QPandaNamespace.h"
#include "ThirdParty/TinyXML/tinyxml.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"
#include <iostream>
#include <string>
#include <map>
#include <vector>

#define CONFIG_PATH  "./QPandaConfig.xml"

QPANDA_BEGIN


class XmlConfigParam
{
public:
    XmlConfigParam() ;
    bool loadFile(const std::string &filename);
    bool getMetadataConfig(int &qubit_num, std::vector<std::vector<int>> &qubit_matrix);
    bool getClassNameConfig(std::map<std::string, std::string> &class_names);

    bool getQuantumCloudConfig(std::map<std::string, std::string> &cloud_config);
    bool getQGateConfig(std::vector<std::string> &single_gates, std::vector<std::string> &double_gates);
    bool getQGateTimeConfig(std::map<GateType, size_t> &gate_time);
    bool getInstructionConfig(std::map<std::string, std::map<std::string, uint32_t>> &);
    virtual ~XmlConfigParam() {};
private:
    TiXmlDocument m_doc;
    TiXmlElement *m_root_element;
    std::string m_filename;
};
QPANDA_END
#endif // XMLCONFIGPARAM_H

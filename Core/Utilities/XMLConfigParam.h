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

#include "TinyXML/tinyxml.h"
#include <iostream>
#include <string>
#include <map>
#include "QPandaNamespace.h"

QPANDA_BEGIN



class XmlConfigParam
{
public:
    XmlConfigParam() = delete;
    XmlConfigParam(const std::string &filename);

    /*
    Parse xml config file and metadata config path
    param:
        path: output metadata config file path
    return:
        sucess or not

    Note:
    */
    bool getMetadataPath(std::string &path);

    /*
    Parse xml config file and get class name
    param:
        class_name_map: output all class name
    return:
        sucess or not

    Note:
    */
    bool getClassNameConfig(std::map<std::string, std::string> &class_name_map);

    virtual ~XmlConfigParam();
private:
    TiXmlDocument m_doc;
    TiXmlElement *m_root_element;
};
QPANDA_END
#endif // XMLCONFIGPARAM_H

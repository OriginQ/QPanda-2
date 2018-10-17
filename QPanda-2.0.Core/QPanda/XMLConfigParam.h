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

#pragma once

#include "TinyXML/tinyxml.h"
#include <iostream>
#include <string>
#include <map>

using namespace std;


class XmlConfigParam
{
public:
    XmlConfigParam() = delete;
    XmlConfigParam(const string &filename);

    /*
    Parse xml config file and metadata config path
    param:
        path: output metadata config file path
    return:
        sucess or not

    Note:
    */
    bool getMetadataPath(string &path);

    /*
    Parse xml config file and get class name
    param:
        class_name_map: output all class name
    return:
        sucess or not

    Note:
    */
    bool getClassNameConfig(map<string, string> &class_name_map);

    virtual ~XmlConfigParam();
private:
    TiXmlDocument m_doc;
    TiXmlElement *m_root_element;
};

#endif // XMLCONFIGPARAM_H

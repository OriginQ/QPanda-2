#ifndef XMLCONFIGPARAM_H
#define XMLCONFIGPARAM_H

#pragma once

#include "tinyxml.h"
#include <iostream>
#include <string>
#include <map>

using namespace std;


class XmlConfigParam
{
public:
    XmlConfigParam() = delete;
    XmlConfigParam(const string &xmlFile);

    bool getClassNameConfig(map<string, string> &classNameMap);

    virtual ~XmlConfigParam();
private:
    TiXmlDocument m_doc;
    TiXmlElement *m_rootElement;
};

#endif // XMLCONFIGPARAM_H

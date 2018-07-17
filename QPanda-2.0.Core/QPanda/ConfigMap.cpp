#include "ConfigMap.h"
#include "QPandaException.h"
#include "XMLConfigParam.h"
ConfigMap::ConfigMap()
{
#if defined(__linux__)
    m_sConfigFilePath.assign("./ConfigFile/Config.xml");
#elif defined(_WIN32)
    m_sConfigFilePath.assign("../Config.xml");
#endif

    XmlConfigParam xml(m_sConfigFilePath);
    xml.getClassNameConfig(m_configMap);
    string metadataPath;
}


ConfigMap::~ConfigMap()
{
}

void ConfigMap::insert(CONFIGPAIR & configPair)
{
    auto aiter = m_configMap.find(configPair.first);
    if (aiter != m_configMap.end())
    {
        aiter->second.assign(configPair.second);
    }
    else
    {
        m_configMap.insert(configPair);
    }
}

string ConfigMap::operator[](const char * name)
{
    string sName = name;
    auto aiter = m_configMap.find(sName);
    if (aiter == m_configMap.end())
        throw param_error_exception("param error",false);
    return aiter->second;
}

ConfigMap _G_configMap;
#include "ConfigMap.h"
#include "XMLConfigParam.h"
#include <cstdlib>
using namespace std;
USING_QPANDA
#ifdef _WIN32
#pragma warning(disable : 4996)
#endif

ConfigMap::ConfigMap()
{
    char * config_path = nullptr;
    string metadata_path = "";
    config_path = getenv("QPANDA_CONFIG_PATH");
    if (nullptr != config_path)
    {
#ifdef _WIN32
        metadata_path.append(config_path);
        metadata_path.append("\\MetadataConfig.xml");
        m_sConfigFilePath.append(config_path);
        m_sConfigFilePath.append("\\Config.xml");
#else
        metadata_path.append(config_path);
        metadata_path.append("/MetadataConfig.xml");
        m_sConfigFilePath.append(config_path);
        m_sConfigFilePath.append("/Config.xml");
#endif
        XmlConfigParam xml(m_sConfigFilePath);
        xml.getClassNameConfig(m_configMap);
        string metadataPath(metadata_path);
        CONFIGPAIR metadataPathPair = { "MetadataPath",metadataPath };
        insert(metadataPathPair);
    }
    else
    {
        insert(CONFIGPAIR("QProg", "OriginProgram"));
        insert(CONFIGPAIR("QCircuit", "OriginCircuit"));
        insert(CONFIGPAIR("QIfProg", "OriginQIf"));
        insert(CONFIGPAIR("QWhileProg", "OriginQWhile"));
        insert(CONFIGPAIR("QMeasure", "OriginMeasure"));
        insert(CONFIGPAIR("QuantumMachine", "CPUQVM"));
        insert(CONFIGPAIR("QubitPool", "OriginQubitPool"));
        insert(CONFIGPAIR("Qubit", "OriginQubit"));
        insert(CONFIGPAIR("PhysicalQubit", "OriginPhysicalQubit"));
        insert(CONFIGPAIR("CBit", "OriginCBit"));
        insert(CONFIGPAIR("CMem", "OriginCMem"));
        insert(CONFIGPAIR("QResult", "OriginQResult"));
        insert(CONFIGPAIR("CExpr", "OriginCExpr"));
        insert(CONFIGPAIR("ClassicalProg", "OriginClassicalProg"));
        CONFIGPAIR metadataPathPair = { "MetadataPath","" };
        insert(metadataPathPair);
    }
}


ConfigMap & ConfigMap::getInstance()
{
    static ConfigMap config;
    return config;
}

ConfigMap::~ConfigMap()
{
}

void ConfigMap::insert(CONFIGPAIR  configPair)
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
    {
        QCERR("param error");
        throw invalid_argument("param error");
    }

    return aiter->second;
}

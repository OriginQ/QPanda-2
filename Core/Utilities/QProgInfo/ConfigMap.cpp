#include "Core/Utilities/QProgInfo/ConfigMap.h"
#include <cstdlib>

using namespace std;
USING_QPANDA
#ifdef _WIN32
#pragma warning(disable : 4996)
#endif


ConfigMap::ConfigMap(const string &filename)
{

	JsonConfigParam config_file;
    if (config_file.load_config(filename))
    {
        if (config_file.getClassNameConfig(m_configMap))
        {
            return ;
        }
    }

    insert(CONFIGPAIR("QProg", "OriginProgram"));
    insert(CONFIGPAIR("QCircuit", "OriginCircuit"));
    insert(CONFIGPAIR("QIfProg", "OriginQIf"));
    insert(CONFIGPAIR("QWhileProg", "OriginQWhile"));
    insert(CONFIGPAIR("QMeasure", "OriginMeasure"));
	insert(CONFIGPAIR("QReset", "OriginReset"));
    insert(CONFIGPAIR("QuantumMachine", "CPUQVM"));
    insert(CONFIGPAIR("QubitPool", "OriginQubitPool"));
    insert(CONFIGPAIR("Qubit", "OriginQubit"));
    insert(CONFIGPAIR("PhysicalQubit", "OriginPhysicalQubit"));
    insert(CONFIGPAIR("CBit", "OriginCBit"));
    insert(CONFIGPAIR("CMem", "OriginCMem"));
    insert(CONFIGPAIR("QResult", "OriginQResult"));
    insert(CONFIGPAIR("CExpr", "OriginCExpr"));
    insert(CONFIGPAIR("ClassicalProg", "OriginClassicalProg"));
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


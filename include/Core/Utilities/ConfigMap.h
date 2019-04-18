#ifndef _CONFIG_MAP_H
#define _CONFIG_MAP_H
#include <map>
#include <string>
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/Utilities/XMLConfigParam.h"

QPANDA_BEGIN


typedef std::pair<std::string, std::string> CONFIGPAIR;
class ConfigMap
{
public:
    static ConfigMap &getInstance();
    ~ConfigMap();
    std::string operator [](const char *);
protected:
    ConfigMap(const std::string &filename = CONFIG_PATH);
    ConfigMap(const ConfigMap &);
    ConfigMap &operator=(const ConfigMap &);
private:
    void insert(CONFIGPAIR);
    std::map<std::string, std::string> m_configMap;
};
QPANDA_END

#endif // !_CONFIG_MAP_H



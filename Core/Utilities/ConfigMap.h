#ifndef _CONFIG_MAP_H
#define _CONFIG_MAP_H
#include <map>
#include <string>
#include "QPandaNamespace.h"
QPANDA_BEGIN
typedef std::pair<std::string, std::string> CONFIGPAIR;
class ConfigMap
{
public:
    static ConfigMap &getInstance();
    ~ConfigMap();
    std::string operator [](const char *);
protected:
    ConfigMap();
    ConfigMap(const ConfigMap &);
    ConfigMap &operator=(const ConfigMap &);
private:
    void insert(CONFIGPAIR);
    std::map<std::string, std::string> m_configMap;
    std::string m_sConfigFilePath;
};
QPANDA_END

#endif // !_CONFIG_MAP_H



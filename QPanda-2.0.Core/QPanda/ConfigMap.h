#ifndef _CONFIG_MAP_H
#define _CONFIG_MAP_H
#include <map>
#include <string>

using std::pair;
using std::map;
using std::string;
typedef pair<string, string> CONFIGPAIR;
class ConfigMap
{
public:
    static ConfigMap &getInstance();
    ~ConfigMap();
    string operator [](const char *);
protected:
    ConfigMap();
    ConfigMap(const ConfigMap &);
    ConfigMap &operator=(const ConfigMap &);
private:
    void insert(CONFIGPAIR);
    map<string, string> m_configMap;
    string m_sConfigFilePath;
};

#endif // !_CONFIG_MAP_H



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
    ConfigMap();
    ~ConfigMap();
    
    string operator [](const char *);
private:
    void insert(CONFIGPAIR &);
    map<string, string> m_configMap;
    string m_sConfigFilePath;
};

extern ConfigMap _G_configMap;

#endif // !_CONFIG_MAP_H



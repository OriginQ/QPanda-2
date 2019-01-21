#include "XMLConfigParam.h"
using namespace std;
USING_QPANDA
XmlConfigParam::XmlConfigParam(const string &filename) :
    m_doc(filename.c_str()), m_root_element(0)
{
    if (!m_doc.LoadFile())
    {
        QCERR("load file failure");
        throw invalid_argument("load file failure");
    }
    m_root_element = m_doc.RootElement();
}

bool XmlConfigParam::getMetadataPath(string &path)
{
    if (!m_root_element)
    {
        return false;
    }
    TiXmlElement *metadata_path_element = m_root_element->FirstChildElement("MetadataPath");

    if (!metadata_path_element)
    {
        return false;
    }

    path = metadata_path_element->GetText();
    return true;
}

bool XmlConfigParam::getClassNameConfig(map<string, string> &class_name_map)
{
    if (!m_root_element)
    {
        return false;
    }

    TiXmlElement *class_name_config_element = m_root_element->FirstChildElement("ClassNameConfig");
    if (!class_name_config_element)
    {
        return false;
    }

    for(TiXmlElement *class_msg_element = class_name_config_element->FirstChildElement();
        class_msg_element;
        class_msg_element = class_msg_element->NextSiblingElement())
    {
        if (!class_msg_element->GetText())
            continue;
        class_name_map.insert(pair<string, string>(class_msg_element->Value(), class_msg_element->GetText()));
    }

    return true;
}

XmlConfigParam::~XmlConfigParam()
{

}

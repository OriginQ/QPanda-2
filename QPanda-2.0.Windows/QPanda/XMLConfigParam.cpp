#include "XMLConfigParam.h"

XmlConfigParam::XmlConfigParam(const string &xmlFile) :
    m_doc(xmlFile.c_str()), m_rootElement(0)
{
    if (!m_doc.LoadFile())
    {
        throw exception();
    }
    m_rootElement = m_doc.RootElement();
}

bool XmlConfigParam::getClassNameConfig(map<string, string> &classNameMap)
{
    if (!m_rootElement)
    {
        return false;
    }

    TiXmlElement *classNameConfigEle = m_rootElement->FirstChildElement("ClassNameConfig");
    if (!classNameConfigEle)
    {
        return false;
    }

    for(TiXmlElement *classMsgEle = classNameConfigEle->FirstChildElement();
        classMsgEle;
        classMsgEle = classMsgEle->NextSiblingElement())
    {
        if (!classMsgEle->GetText())
            continue;
        classNameMap.insert(pair<string, string>(classMsgEle->Value(), classMsgEle->GetText()));
    }

    return true;
}

XmlConfigParam::~XmlConfigParam()
{

}

#include "QuantumMetadata.h"
#include <algorithm>
#include <string>


QuantumMetadata::QuantumMetadata(const string & xmlFile):
    m_doc(xmlFile.c_str()), m_rootElement(nullptr)
{
    if (!m_doc.LoadFile())
    {
        throw exception();
    }
    m_rootElement = m_doc.RootElement();
}

size_t QuantumMetadata::getQubitCount()
{
    if (!m_rootElement)
    {
        return false;
    }

    TiXmlElement *firstEle = m_rootElement->FirstChildElement("QubitCount");
    if (!firstEle)
    {
        return false;
    }

    const char *c_str = firstEle->GetText();
    size_t qubitCount = strtoul(c_str, nullptr, 0);
    return qubitCount;
}

bool QuantumMetadata::getQubiteMatrix(vector<vector<int>> & qubitMatrix)
{
    if (!m_rootElement)
    {
        return false;
    }

    TiXmlElement *firstEle = m_rootElement->FirstChildElement("QubitCount");
    if (!firstEle)
    {
        return false;
    }

    const char *c_str = firstEle->GetText();
    size_t qubitCount = strtoul(c_str, nullptr, 0);
    vector<vector<int>> tmpVec(qubitCount, vector<int>(qubitCount, 0));
    qubitMatrix = tmpVec;

    TiXmlElement *matrixEle = m_rootElement->FirstChildElement("QubitMatrix");
    if (!matrixEle)
    {
        return false;
    }

    for (TiXmlElement *qubitEle = matrixEle->FirstChildElement("Qubit");
         qubitEle;
         qubitEle = qubitEle->NextSiblingElement("Qubit"))
    {
        const char *attr = qubitEle->Attribute("QubitNum");
        if (!attr)
        {
            return false;
        }

        size_t i = strtoul(attr, nullptr, 0);
        if (!i || i > qubitCount)
        {
            return false;
        }

        for (TiXmlElement *adjacentQubitEle = qubitEle->FirstChildElement("AdjacentQubit");
             adjacentQubitEle;
             adjacentQubitEle = adjacentQubitEle->NextSiblingElement("AdjacentQubit"))
        {
            const char* attr = adjacentQubitEle->Attribute("QubitNum");
            if (!attr)
            {
                return false;
            }

            size_t j = strtoul(attr, nullptr, 0);
            if (!j || j > qubitCount)
            {
                return false;
            }
            const char *itemText = adjacentQubitEle->GetText();
            qubitMatrix[i-1][j-1] = atoi(itemText);
        }
    }

    return true;
}

bool QuantumMetadata::getSingleGate(vector<string> &singleGate)
{
    if (!m_rootElement)
    {
        return false;
    }

    TiXmlElement *singleGateEle = m_rootElement->FirstChildElement("SingleGate");
    if (!singleGateEle)
    {
        return false;
    }

    for (TiXmlElement *gateEle = singleGateEle->FirstChildElement("Gate");
         gateEle;
         gateEle = gateEle->NextSiblingElement("Gate"))
    {
        if (gateEle)
        {
            string gateStr = gateEle->GetText();
            transform(gateStr.begin(), gateStr.end(), gateStr.begin(), ::toupper);
            singleGate.emplace_back(gateStr);
        }
    }

    return true;
}

bool QuantumMetadata::getDoubleGate(vector<string> &doubleGate)
{
    if (!m_rootElement)
    {
        return false;
    }

    TiXmlElement *doubleGateEle = m_rootElement->FirstChildElement("DoubleGate");
    if (!doubleGateEle)
    {
        return false;
    }

    for (TiXmlElement *gateEle = doubleGateEle->FirstChildElement("Gate");
         gateEle;
         gateEle = gateEle->NextSiblingElement("Gate"))
    {
        if (gateEle)
        {
            string gateStr = gateEle->GetText();
            transform(gateStr.begin(), gateStr.end(), gateStr.begin(), ::toupper);
            doubleGate.emplace_back(gateStr);
        }
    }

    return true;
}


QuantumMetadata::~QuantumMetadata()
{
}

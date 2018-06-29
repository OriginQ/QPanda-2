#ifndef QUBITCONFIG_H
#define QUBITCONFIG_H

#include "tinyxml.h"
#include <iostream>
#include <vector>

using namespace std;



class QuantumMetadata
{
public:
    QuantumMetadata() = delete;
    QuantumMetadata & operator =(const QuantumMetadata &) = delete;
    QuantumMetadata(const string & xmlFile);

    bool getQubiteMatrix(vector<vector<int> > &qubitMatrix);
    bool getSingleGate(vector<string> &singleGate);
    bool getDoubleGate(vector<string> &doubleGate);

    ~QuantumMetadata();

private:
    TiXmlDocument m_doc;
    TiXmlElement *m_rootElement;
};


#endif // QubitConfig_H

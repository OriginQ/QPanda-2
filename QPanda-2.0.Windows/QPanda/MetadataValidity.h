#ifndef METADATAVALIDITYFUNCTIONVECTOR_H
#define METADATAVALIDITYFUNCTIONVECTOR_H

#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <functional>


using namespace std;

enum SingleGateTransferType
{
    SingleGateInvalid = -1,
    ArbitraryRotation,
    DoubleContinuous,
    SingleContinuousAndDiscrete,
    DoubleDiscrete
};

enum DoubleGateTransferType
{
    DoubleGateInvalid = -1,
    DoubleBitGate
};


int arbitraryRotationMetadataValidity(vector<string>&sQGateVector, vector<string>&sValidQGateVector);
int doubleContinuousMetadataValidity(vector<string>&sQGateVector, vector<string>&sValidQGateVector);
int singleContinuousAndDiscreteMetadataValidity(vector<string>&sQGateVector, vector<string>&sValidQGateVector);
int doubleDiscreteMetadataValidity(vector<string>&sQGateVector, vector<string>&sValidQGateVector);

int doubleGateMetadataValidity(vector<string>&sQGateVector, vector<string>&sValidQGateVector);


typedef  function<int(vector<string>&, vector<string>&)> MetadataValidityFunction;
class MetadataValidity
{
public:
    void push_back(MetadataValidityFunction func);
    MetadataValidityFunction operator[](int i);
    size_t size();

    virtual ~MetadataValidity();

private:
    vector<MetadataValidityFunction> m_MetadataValidityFunctionVector;
};

/****************************************************************************************************
*
*SingleGateTypeValidator 类GateType（）得到 SingleGateType
*
*****************************************************************************************************/

class SingleGateTypeValidator
{
public:
    SingleGateTypeValidator();
    static int GateType(vector<string>&sQGateVector, vector<string>&sValidQGateVector);
    virtual ~SingleGateTypeValidator();

    MetadataValidity m_MetadataValidityFunctionVector;
private:

};


/****************************************************************************************************
*
*DoubleGateTypeValidator 类GateType（）得到 DoubleGateType
*
*****************************************************************************************************/

class DoubleGateTypeValidator
{
public:
    DoubleGateTypeValidator();
    static int GateType(vector<string>&sQGateVector, vector<string>&sValidQGateVector);
    virtual ~DoubleGateTypeValidator();

    MetadataValidity m_MetadataValidityFunctionVector;
private:

};


#endif // METADATAVALIDITYFUNCTIONVECTOR_H

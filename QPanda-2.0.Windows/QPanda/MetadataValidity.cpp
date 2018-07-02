#include "MetadataValidity.h"


void MetadataValidity::push_back(MetadataValidityFunction func)
{
    m_MetadataValidityFunctionVector.push_back(func);
}

MetadataValidityFunction MetadataValidity::operator[](int i)
{
    if ((size_t)i >= m_MetadataValidityFunctionVector.size())
    {
        throw exception();
    }

    return m_MetadataValidityFunctionVector[i];
}

size_t MetadataValidity::size()
{
    return m_MetadataValidityFunctionVector.size();
}

MetadataValidity::~MetadataValidity()
{

}

int arbitraryRotationMetadataValidity(vector<string> &sQGateVector, vector<string> &sValidQGateVector)
{
    sValidQGateVector.clear();
    for (auto &val : sQGateVector)
    {
        if ("U3" == val || "U2" == val)
        {
            sValidQGateVector.emplace_back(val);
            return SingleGateTransferType::ArbitraryRotation;
        }
    }

    return SingleGateTransferType::SingleGateInvalid;
}

int doubleContinuousMetadataValidity(vector<string> &sQGateVector, vector<string> &sValidQGateVector)
{
    sValidQGateVector.clear();
    string tmp("");

    for (auto &val : sQGateVector)
    {
        if ("RX" == val || "RY" == val || "RZ" == val || "U1" == val)
        {
            if (!tmp.empty() && val != tmp)
            {
                if (("RZ" == val && "U1" == tmp) || ("U1" == val && "RZ" == tmp))
                    continue;

                sValidQGateVector.emplace_back(tmp);
                sValidQGateVector.emplace_back(val);
                return SingleGateTransferType::DoubleContinuous;
            }

            tmp = val;
        }
    }

    return SingleGateTransferType::SingleGateInvalid;
}

int singleContinuousAndDiscreteMetadataValidity(vector<string> &sQGateVector, vector<string> &sValidQGateVector)
{
    sValidQGateVector.clear();
    string tmp("");

    for (auto &val : sQGateVector)
    {
        if (!tmp.empty() && ("X" == val || "Y" == val || "H" == val))
        {
            string compare = tmp;
            if ("U1" == tmp)
            {
                compare = "RZ";
            }

            if (compare != "R" + val)
            {
                sValidQGateVector.emplace_back(tmp);
                sValidQGateVector.emplace_back(val);
                return SingleGateTransferType::SingleContinuousAndDiscrete;
            }
        }

        if ("RX" == val || "RY" == val || "RZ" == val || "U1" == val)
        {
            tmp = val;
        }
    }

    return SingleGateTransferType::SingleGateInvalid;
}

int doubleDiscreteMetadataValidity(vector<string> &sQGateVector, vector<string> &sValidQGateVector)
{
    sValidQGateVector.clear();
    string tmp("");

    for (auto &val : sQGateVector)
    {
        if ("RX" == val || "RY" == val || "RZ" == val || "U1" == val)
        {
            return SingleGateTransferType::SingleGateInvalid;
        }

        if ("H" == val || "X" == val || "Y" == val || "T" == val)
        {
            if (!tmp.empty() && tmp != val && ("T" == tmp || "T" == val))
            {
                sValidQGateVector.emplace_back(tmp);
                sValidQGateVector.emplace_back(val);
                return SingleGateTransferType::DoubleDiscrete;
            }

            tmp = val;
        }
    }

    return SingleGateTransferType::SingleGateInvalid;
}

int doubleGateMetadataValidity(vector<string> &sQGateVector, vector<string> &sValidQGateVector)
{
    sValidQGateVector.clear();
    string tmp("");

    for (auto &val : sQGateVector)
    {
        if ("CNOT" == val || "SWAP" == val)
        {
            if (!tmp.empty() && val != tmp)
            {
                sValidQGateVector.emplace_back(tmp);
                sValidQGateVector.emplace_back(val);
                return DoubleGateTransferType::DoubleBitGate;
            }

            tmp = val;
        }
    }

    return DoubleGateTransferType::DoubleGateInvalid;
}



SingleGateTypeValidator::SingleGateTypeValidator()
{
    m_MetadataValidityFunctionVector.push_back(arbitraryRotationMetadataValidity);
    m_MetadataValidityFunctionVector.push_back(doubleContinuousMetadataValidity);
    m_MetadataValidityFunctionVector.push_back(singleContinuousAndDiscreteMetadataValidity);
    m_MetadataValidityFunctionVector.push_back(doubleDiscreteMetadataValidity);
}

int SingleGateTypeValidator::GateType(vector<string> &sQGateVector, vector<string> &sValidQGateVector)
{
    static SingleGateTypeValidator validator;

    int ret = SingleGateTransferType::SingleGateInvalid;
    for (size_t i = 0; i < validator.m_MetadataValidityFunctionVector.size(); i++)
    {
        ret = validator.m_MetadataValidityFunctionVector[i](sQGateVector, sValidQGateVector);
        if (SingleGateTransferType::SingleGateInvalid != ret)
        {
            return ret;
        }
    }

    return ret;
}

SingleGateTypeValidator::~SingleGateTypeValidator()
{

}


DoubleGateTypeValidator::DoubleGateTypeValidator()
{
    m_MetadataValidityFunctionVector.push_back(doubleGateMetadataValidity);
}

int DoubleGateTypeValidator::GateType(vector<string> &sQGateVector, vector<string> &sValidQGateVector)
{
    static DoubleGateTypeValidator validator;

    int ret = DoubleGateTransferType::DoubleGateInvalid;
	for (size_t i = 0; i < validator.m_MetadataValidityFunctionVector.size(); i++)
	{
		ret = validator.m_MetadataValidityFunctionVector[i](sQGateVector, sValidQGateVector);
		if (DoubleGateTransferType::DoubleGateInvalid != ret)
		{
			return ret;
		}
	}

    return ret;
}

DoubleGateTypeValidator::~DoubleGateTypeValidator()
{

}

#include "MetadataValidity.h"
using namespace std;
USING_QPANDA

void MetadataValidity::push_back(MetadataValidity_cb func)
{
    m_metadata_validity_functions.push_back(func);
}

MetadataValidity_cb MetadataValidity::operator[](int i)
{
    if ((size_t)i >= m_metadata_validity_functions.size())
    {
        QCERR("size is out of range");
        throw invalid_argument("size is out of range");
    }

    return m_metadata_validity_functions[i];
}

size_t MetadataValidity::size()
{
    return m_metadata_validity_functions.size();
}

MetadataValidity::~MetadataValidity()
{

}

int QPanda::arbitraryRotationMetadataValidity(vector<string> &gates, vector<string> &valid_gates)
{
    valid_gates.clear();
    for (auto &val : gates)
    {
        if ("U3" == val || "U2" == val)
        {
            valid_gates.emplace_back(val);
            return SingleGateTransferType::ARBITRARY_ROTATION;
        }
    }

    return SingleGateTransferType::SINGLE_GATE_INVALID;
}

int QPanda::doubleContinuousMetadataValidity(vector<string> &gate_vector, vector<string> &valid_gates)
{
    valid_gates.clear();
    string tmp("");

    for (auto &val : gate_vector)
    {
        if ("RX" == val || "RY" == val || "RZ" == val || "U1" == val)
        {
            if (!tmp.empty() && val != tmp)
            {
                if (("RZ" == val && "U1" == tmp) || ("U1" == val && "RZ" == tmp))
                    continue;

                valid_gates.emplace_back(tmp);
                valid_gates.emplace_back(val);
                return SingleGateTransferType::DOUBLE_CONTINUOUS;
            }

            tmp = val;
        }
    }

    return SingleGateTransferType::SINGLE_GATE_INVALID;
}

int QPanda::singleContinuousAndDiscreteMetadataValidity(vector<string> &gates, vector<string> &valid_gates)
{
    valid_gates.clear();
    string tmp("");

    for (auto &val : gates)
    {
        if (("RX" == tmp && ("Y1" == val || "Z1" == val || "H" == val || "S" == val || "T" == val))
            || ("RY" == tmp && ("X1" == val || "Z1" == val || "S" == val || "T" == val))
            || ("RZ" == tmp && ("X1" == val || "Y1" == val || "H" == val))
            || ("U1" == tmp && ("X1" == val || "Y1" == val || "H" == val)))
        {
            valid_gates.emplace_back(tmp);
            valid_gates.emplace_back(val);
            return SingleGateTransferType::SINGLE_CONTINUOUS_DISCRETE;
        }

        if (tmp.empty() && ("RX" == val || "RY" == val || "RZ" == val || "U1" == val))
        {
            tmp = val;
        }
    }

    return SingleGateTransferType::SINGLE_GATE_INVALID;
}

int QPanda::doubleDiscreteMetadataValidity(vector<string> &gates, vector<string> &valid_gates)
{
    valid_gates.clear();
    string tmp("");

    for (auto &val : gates)
    {
        if ("RX" == val || "RY" == val || "RZ" == val || "U1" == val)
        {
            return SingleGateTransferType::SINGLE_GATE_INVALID;
        }

        if ("H" == val || "X1" == val || "Y1" == val || "T" == val)
        {
            if (!tmp.empty() && tmp != val && ("T" == tmp || "T" == val))
            {
                valid_gates.emplace_back(tmp);
                valid_gates.emplace_back(val);
                return SingleGateTransferType::DOUBLE_DISCRETE;
            }

            tmp = val;
        }
    }

    return SingleGateTransferType::SINGLE_GATE_INVALID;
}

int QPanda::doubleGateMetadataValidity(vector<string> &gates, vector<string> &valid_gates)
{
    valid_gates.clear();

    for (auto &val : gates)
    {
        if ("CNOT" == val || "ISWAP" == val)
        {
            valid_gates.emplace_back(val);
            return DoubleGateTransferType::DOUBLE_BIT_GATE;
        }
    }

    return DoubleGateTransferType::DOUBLE_GATE_INVALID;
}



SingleGateTypeValidator::SingleGateTypeValidator()
{
    m_metadata_validity_functions.push_back(arbitraryRotationMetadataValidity);
    m_metadata_validity_functions.push_back(doubleContinuousMetadataValidity);
    m_metadata_validity_functions.push_back(singleContinuousAndDiscreteMetadataValidity);
    m_metadata_validity_functions.push_back(doubleDiscreteMetadataValidity);
}

int SingleGateTypeValidator::GateType(vector<string> &gates, vector<string> &valid_gates)
{
    static SingleGateTypeValidator validator;

    int ret = SingleGateTransferType::SINGLE_GATE_INVALID;
    for (size_t i = 0; i < validator.m_metadata_validity_functions.size(); i++)
    {
        ret = (int)validator.m_metadata_validity_functions[(int)i](gates, valid_gates);
        if (SingleGateTransferType::SINGLE_GATE_INVALID != ret)
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
    m_metadata_validity_functions.push_back(doubleGateMetadataValidity);
}

int DoubleGateTypeValidator::GateType(vector<string> &gates, vector<string> &valid_gates)
{
    static DoubleGateTypeValidator validator;

    int ret = DoubleGateTransferType::DOUBLE_GATE_INVALID;
    for (size_t i = 0; i < validator.m_metadata_validity_functions.size(); i++)
    {
        ret = (int)validator.m_metadata_validity_functions[(int)i](gates, valid_gates);
        if (DoubleGateTransferType::DOUBLE_GATE_INVALID != ret)
        {
            return ret;
        }
    }

    return ret;
}

DoubleGateTypeValidator::~DoubleGateTypeValidator()
{

}

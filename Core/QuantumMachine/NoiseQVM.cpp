#include "OriginQuantumMachine.h"
#include "NoiseQPU/NoiseCPUImplQPU.h"
#include "Utilities/MetadataValidity.h"
#include "TranformQGateTypeStringAndEnum.h"
#include "Transform/TransformDecomposition.h"
#include "Core/Utilities/QPandaException.h"
USING_QPANDA
using namespace std;

NoiseQVM::NoiseQVM()
{
    m_gates_matrix = { {"X","Y","Z",
                        "T","S","H",
                        "RX","RY","RZ",
                        "U1" },
                       { "CNOT" } };
    m_valid_gates_matrix.resize(m_gates_matrix.size());
}


void NoiseQVM::_getValidGatesMatrix()
{
    if (SingleGateTransferType::SINGLE_GATE_INVALID == 
        SingleGateTypeValidator::GateType(m_gates_matrix[MetadataGateType::METADATA_SINGLE_GATE],
        m_valid_gates_matrix[MetadataGateType::METADATA_SINGLE_GATE]))   /* single gate data MetadataValidity */
    {
        finalize();
        QCERR("gates valid error");
        throw runtime_error("gates valid error");
    }
    if (m_gates_matrix.size() >= 2)
    {
        if (DoubleGateTransferType::DOUBLE_GATE_INVALID ==
            DoubleGateTypeValidator::GateType(m_gates_matrix[MetadataGateType::METADATA_DOUBLE_GATE],
            m_valid_gates_matrix[MetadataGateType::METADATA_DOUBLE_GATE]))   /* double gate data MetadataValidity */
        {
            finalize();
            QCERR("gates valid error");
            throw runtime_error("gates valid error");
        }
    }

}

void NoiseQVM::init()
{
    try
    {
        _start();
        _getValidGatesMatrix();
        rapidjson::Document doc;
        doc.Parse("{}");
        auto & alloc = doc.GetAllocator();
        Value noise_model_value(rapidjson::kObjectType);
        for (auto a : m_gates_matrix[MetadataGateType::METADATA_SINGLE_GATE])
        {
            Value value(rapidjson::kArrayType);
            value.PushBack(NOISE_MODEL::DAMPING_KRAUS_OPERATOR, alloc);
            value.PushBack(0.5, alloc);
            noise_model_value.AddMember(Value().SetString(a.c_str(), alloc).Move(), value, alloc);
        }

        doc.AddMember("noisemodel", noise_model_value, alloc);
        _pGates = new NoisyCPUImplQPU(doc);
        _ptrIsNull(_pGates, "NoisyCPUImplQPU");
    }
    catch (const std::exception&e)
    {
        finalize();
        QCERR(e.what());
        throw init_fail(e.what());
    }
}
void NoiseQVM::initGates(rapidjson::Document & doc)
{
    if (!doc.HasMember("gates"))
    {
        QCERR("doc do not include gates");
        throw invalid_argument("doc do not include gates");
    }

    auto doc_iter =doc.FindMember("gates");
    auto & gates = (*doc_iter).value;
    if (!gates.IsArray())
    {
        QCERR("gates is not array");
        throw invalid_argument("gates is not array");
    }
    m_gates_matrix.resize(0);

    for (auto first_layer_iter = gates.MemberBegin();
        first_layer_iter != gates.MemberEnd();
        first_layer_iter++)
    {
        auto &second_layer = (*first_layer_iter).value;
        vector<string> temp;
        if (!(*first_layer_iter).value.IsArray())
        {

            if (!(*first_layer_iter).value.IsString())
            {
                QCERR("first_layer_iter is not string or array");
                throw invalid_argument("first_layer_iter is not  string or array");
            }
            temp.push_back((*first_layer_iter).value.GetString());
        }
        else
        {
            for (auto second_layer_iter = second_layer.MemberBegin();
                second_layer_iter != second_layer.MemberEnd();
                second_layer_iter++)
            {
                if (!(*second_layer_iter).value.IsString())
                {
                    QCERR("second_layer_iter is not string");
                    throw invalid_argument("second_layer_iter is not string");
                }
                temp.push_back((*second_layer_iter).value.GetString());

            }
        }
        m_gates_matrix.push_back(temp);
    }
    m_valid_gates_matrix.reserve(m_gates_matrix.size());
}

void NoiseQVM::init(rapidjson::Document & doc)
{
    if (!doc.HasMember("noisemodel"))
    {
        init();
    }
    try
    {
        if (doc.HasMember("gates"))
        {
            initGates(doc);
        }
        else
        {
            _start();
            _getValidGatesMatrix();
            _pGates = new NoisyCPUImplQPU(doc);
            _ptrIsNull(_pGates, "NoisyCPUImplQPU");
        }
    }
    catch (const std::exception&e)
    {
        finalize();
        QCERR(e.what());
        throw init_fail(e.what());
    }
}

void NoiseQVM::run(QProg & prog)
{
    try
    {
        vector<vector<int>> adjacent_matrixes;
        TransformDecomposition traversal_vec(m_valid_gates_matrix, m_gates_matrix, adjacent_matrixes);
        traversal_vec.TraversalOptimizationMerge(dynamic_cast<QNode *>(&prog));

        auto _pParam = new QuantumGateParam();
        _ptrIsNull(_pParam, "_pParam");

        _pParam->m_qubit_number = _Qubit_Pool->getMaxQubit() - _Qubit_Pool->getIdleQubit();
        _pGates->initState(_pParam);

        prog.getImplementationPtr()->execute(_pGates, _pParam);

        /* aiter has been used in line 120 */
        for (auto aiter : _pParam->m_return_value)
        {
            _QResult->append(aiter);
        }

        delete _pParam;
        _pParam = nullptr;
    }
    catch (const std::exception&e)
    {
        QCERR(e.what());
        throw run_fail(e.what());
    }
   
}

REGISTER_QUANTUM_MACHINE(NoiseQVM);

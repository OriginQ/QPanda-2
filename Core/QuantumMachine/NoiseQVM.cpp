#include "OriginQuantumMachine.h"
#include "NoiseQPU/NoisyCPUImplQPU.h"
#include "Utilities/MetadataValidity.h"
#include "TranformQGateTypeStringAndEnum.h"
#include "Transform/TransformDecomposition.h"
USING_QPANDA
using namespace std;

NoiseQVM::NoiseQVM()
{
    m_gates_matrix = { {"X","Y","Z",
                        "T","S","H",
                        "RX","RY","RZ",
                        "U1" },
                       { "CNOT" } };
    m_valid_gates_matrix.resize(2);

}

void NoiseQVM::start()
{
    _Qubit_Pool =
        QubitPoolFactory::GetFactoryInstance().
        GetPoolWithoutTopology(_Config.maxQubit);
    _CMem =
        CMemFactory::GetFactoryInstance().
        GetInstanceFromSize(_Config.maxCMem);
    _QResult =
        QResultFactory::GetFactoryInstance().
        GetEmptyQResult();
    _QMachineStatus =
        QMachineStatusFactory::
        GetQMachineStatus();

    if ((nullptr == _Qubit_Pool) ||
        (nullptr == _CMem) ||
        (nullptr == _QResult) ||
        (nullptr == _QMachineStatus))
    {
        QCERR("new fail");
        throw std::runtime_error("new fail");
    }

    SingleGateTypeValidator::GateType(m_gates_matrix[MetadataGateType::METADATA_SINGLE_GATE],
        m_valid_gates_matrix[MetadataGateType::METADATA_SINGLE_GATE]);   /* single gate data MetadataValidity */
    DoubleGateTypeValidator::GateType(m_gates_matrix[MetadataGateType::METADATA_DOUBLE_GATE],
        m_valid_gates_matrix[MetadataGateType::METADATA_DOUBLE_GATE]);   /* double gate data MetadataValidity */
}

void NoiseQVM::init()
{
    start();
    rapidjson::Document doc;
    doc.Parse("{}");
    auto & alloc = doc.GetAllocator();

    for (auto a : m_gates_matrix[MetadataGateType::METADATA_SINGLE_GATE])
    {
        Value value(rapidjson::kArrayType);
        value.PushBack(NOISE_MODEL::DAMPING_KRAUS_OPERATOR, alloc);
        value.PushBack(0.5, alloc);
        doc.AddMember(Value().SetString(a.c_str(), alloc).Move(), value, alloc);
    }

    for (auto a : m_gates_matrix[MetadataGateType::METADATA_DOUBLE_GATE])
    {
        Value value(rapidjson::kArrayType);
        value.PushBack(NOISE_MODEL::DEPHASING_KRAUS_OPERATOR, alloc);
        value.PushBack(0.5, alloc);
        doc.AddMember(Value().SetString(a.c_str(), alloc).Move(), value, alloc);
    }

    _pGates = new NoisyCPUImplQPU(doc);

    if (nullptr == _pGates)
    {
        QCERR("new NoisyCPUImplQPU fail");
        throw std::runtime_error("new NoisyCPUImplQPU fail");
    }
}

bool NoiseQVM::init(rapidjson::Document & doc)
{
    start();
    _pGates = new NoisyCPUImplQPU(doc);

    if (nullptr == _pGates)
    {
        QCERR("new NoisyCPUImplQPU fail");
        throw std::runtime_error("new NoisyCPUImplQPU fail");
    }
    return true;
}

void NoiseQVM::run(QProg & prog)
{
    vector<vector<int>> adjacent_matrixes;
    TransformDecomposition traversal_vec(m_valid_gates_matrix, m_gates_matrix, adjacent_matrixes);
    traversal_vec.TraversalOptimizationMerge(dynamic_cast<QNode *>(&prog));

    _pParam = new QuantumGateParam();

    _pParam->m_qbit_number = _Qubit_Pool->getMaxQubit() - _Qubit_Pool->getIdleQubit();

    _pGates->initState(_pParam);

    prog.getImplementationPtr()->execute(_pGates, _pParam);

    /* aiter has been used in line 120 */
    for (auto aiter : _pParam->m_return_value)
    {
        _QResult->append(aiter);
    }

    delete _pParam;
    _pParam = nullptr;
    return;
}

REGISTER_QUANTUM_MACHINE(NoiseQVM);
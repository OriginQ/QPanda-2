#include "OriginQMachine.h"
#include "Factory.h"

#include "QPanda.h"
#include "QPanda/QuantumMetadata.h"
#include "QPanda/ConfigMap.h"
#include "../QPanda/MetadataValidity.h"
#include "Transform/QCircuitParse.h"
#include "Transform/TransformDecomposition.h"
OriginQMachine::OriginQMachine()
{
}


OriginQMachine::~OriginQMachine()
{
}

void OriginQMachine::init()
{
    auto metadata_path = ConfigMap::getInstance()["MetadataPath"];
    QuantumMetadata * metadata;
    if(metadata_path.size()<= 0)
        metadata = new QuantumMetadata();
    else
    {
        metadata =  new QuantumMetadata(metadata_path);
    }
    size_t stQubitCount = metadata->getQubitCount();
    m_Config.maxQubit = stQubitCount;
    m_Config.maxCMem = stQubitCount;
    
    metadata->getSingleGate(m_sSingleGateVector);
    metadata->getDoubleGate(m_sDoubleGateVector);

    metadata->getQubiteMatrix(m_qubitMatrix);

    m_pQubitPool =
        QubitPoolFactory::GetFactoryInstance().
        GetPoolWithoutTopology(m_Config.maxQubit);
    m_pCMem =
        CMemFactory::GetFactoryInstance().
        GetInstanceFromSize(m_Config.maxCMem);
    QProg  temp = CreateEmptyQProg();
    m_iQProgram = temp.getPosition();
    QNodeMap::getInstance().addNodeRefer(m_iQProgram);
    m_pQResult =
        QResultFactory::GetFactoryInstance().
        GetEmptyQResult();
    m_pQMachineStatus =
        QMachineStatusFactory::
        GetQMachineStatus();

    if (SingleGateTypeValidator::GateType(m_sSingleGateVector, m_sValidSingleGateVector) < 0)
    {
        throw metadate_error_exception();
    }
    if (DoubleGateTypeValidator::GateType(m_sDoubleGateVector, m_sValidDoubleGateVector) < 0)
    {
        throw metadate_error_exception();
    }


}

Qubit * OriginQMachine::Allocate_Qubit()
{
    if (m_pQubitPool == nullptr)
    {
        // check if the pointer is nullptr
        // Before init
        // After finalize
        throw(invalid_pool());
    }
    else
    {
        return m_pQubitPool->Allocate_Qubit();
    }
}

Qubit * OriginQMachine::Allocate_Qubit(size_t stQubitAddr)
{
    if (m_pQubitPool == nullptr)
    {
        // check if the pointer is nullptr
        // Before init
        // After finalize
        throw(invalid_pool());
    }
    else
    {
        return m_pQubitPool->Allocate_Qubit(stQubitAddr);
    }
}

CBit * OriginQMachine::Allocate_CBit()
{
    if (m_pCMem == nullptr)
    {
        // check if the pointer is nullptr
        // Before init
        // After finalize
        throw(invalid_cmem());
    }
    else
    {
        return m_pCMem->Allocate_CBit();
    }
}

CBit * OriginQMachine::Allocate_CBit(size_t stCBitaddr)
{
    if (m_pCMem == nullptr)
    {
        // check if the pointer is nullptr
        // Before init
        // After finalize
        throw(invalid_cmem());
    }
    else
    {
        return m_pCMem->Allocate_CBit(stCBitaddr);
    }
}

void OriginQMachine::Free_Qubit(Qubit * pQubit)
{
    this->m_pQubitPool->Free_Qubit(pQubit);
}

void OriginQMachine::Free_CBit(CBit * pCBit)
{
    this->m_pCMem->Free_CBit(pCBit);
}

void OriginQMachine::load(QProg & loadProgram)
{
    QNodeAgency temp(&loadProgram, nullptr, nullptr);
    if (!temp.verify())
    {
        throw load_exception();
    }
    QNodeMap::getInstance().deleteNode(m_iQProgram);
    m_iQProgram = loadProgram.getPosition();
    if (!QNodeMap::getInstance().addNodeRefer(m_iQProgram))
        throw exception();
}

void OriginQMachine::append(QProg &prog)
{
    QNodeAgency tempAgency(&prog, nullptr, nullptr);
    if (!tempAgency.verify())
    {
        throw load_exception();
    }
    auto aiter = QNodeMap::getInstance().getNode(m_iQProgram);
    if (nullptr == aiter)
        throw circuit_not_found_exception("cant found this QProgam", false);
    AbstractQuantumProgram * temp = dynamic_cast<AbstractQuantumProgram *>(aiter);
    temp->pushBackNode(&prog);


}

void OriginQMachine::run()
{
    vector<vector<string>> sValidQGateMatrix = { m_sValidSingleGateVector,m_sValidDoubleGateVector };
    vector<vector<string>> validMatrix = { m_sSingleGateVector,m_sDoubleGateVector };
    vector<vector<int> > qubitMatrix;
    TransformDecomposition td(sValidQGateMatrix, validMatrix,qubitMatrix);
    
    auto pProg = QNodeMap::getInstance().getNode(m_iQProgram);
    
    size_t iQGateCount = countQGateUnderQProg(dynamic_cast<AbstractQuantumProgram *>(pProg));
    cout <<"Before optimization : "<< iQGateCount << endl;

    td.TraversalOptimizationMerge(pProg);

    size_t iPostoptimalityQGateCount = countQGateUnderQProg(dynamic_cast<AbstractQuantumProgram *>(pProg));
    cout <<"postoptimality : "<< iPostoptimalityQGateCount << endl;
    exit(0);
}

QMachineStatus * OriginQMachine::getStatus() const
{
    return nullptr;
}

QResult * OriginQMachine::getResult()
{
    return nullptr;
}

void OriginQMachine::finalize()
{
    QNodeMap::getInstance().deleteNode(m_iQProgram);
    delete m_pQubitPool;
    delete m_pCMem;
    delete m_pQResult;
    delete m_pQMachineStatus;
}




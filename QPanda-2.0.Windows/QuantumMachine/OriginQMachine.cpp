#include "OriginQMachine.h"
#include "Factory.h"
#include "QPanda.h"
#include "QPanda/QuantumMetadata.h"
#include "QPanda/ConfigMap.h"
#include "../QPanda/MetadataValidity.h"
#include "QuantumInstructionHandle/QCircuitParse.h"
OriginQMachine::OriginQMachine()
{
}


OriginQMachine::~OriginQMachine()
{
}

void OriginQMachine::init()
{
    QuantumMetadata metadata(_G_configMap["MetadataPath"]);
    size_t stQubitCount = metadata.getQubitCount();
    m_Config.maxQubit = stQubitCount;
    m_Config.maxCMem = stQubitCount;
    
    metadata.getSingleGate(m_sSingleGateVector);
    metadata.getDoubleGate(m_sDoubleGateVector);

    metadata.getQubiteMatrix(m_qubitMatrix);

    m_pQubitPool =
        Factory::
        QubitPoolFactory::GetFactoryInstance().
        GetPoolWithoutTopology(m_Config.maxQubit);
    m_pCMem =
        Factory::
        CMemFactory::GetFactoryInstance().
        GetInstanceFromSize(m_Config.maxCMem);
    QProg  temp = CreateEmptyQProg();
    m_iQProgram = temp.getPosition();
    _G_QNodeMap.addNodeRefer(m_iQProgram);
    m_pQResult =
        Factory::
        QResultFactory::GetFactoryInstance().
        GetEmptyQResult();
    m_pQMachineStatus =
        Factory::
        QMachineStatusFactory::
        GetQMachineStatus();

    if (SingleGateTypeValidator::GateType(m_sSingleGateVector, m_sVildSingleGateVector) <= 0)
    {
        throw metadate_error_exception();
    }
    if (DoubleGateTypeValidator::GateType(m_sDoubleGateVector, m_sVildDoubleGateVector) <= 0)
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
    _G_QNodeMap.deleteNode(m_iQProgram);
    m_iQProgram = loadProgram.getPosition();
    if (!_G_QNodeMap.addNodeRefer(m_iQProgram))
        throw exception();
}

void OriginQMachine::append(QProg &prog)
{
    QNodeAgency tempAgency(&prog, nullptr, nullptr);
    if (!tempAgency.verify())
    {
        throw load_exception();
    }
    auto aiter = _G_QNodeMap.getNode(m_iQProgram);
    if (nullptr == aiter)
        throw circuit_not_found_exception("cant found this QProgam", false);
    AbstractQuantumProgram * temp = dynamic_cast<AbstractQuantumProgram *>(aiter);
    temp->pushBackNode(&prog);
}

void OriginQMachine::run()
{
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
    _G_QNodeMap.deleteNode(m_iQProgram);
    delete m_pQubitPool;
    delete m_pCMem;
    delete m_pQResult;
    delete m_pQMachineStatus;
}

REGISTER_QUANTUM_MACHINE(OriginQMachine);


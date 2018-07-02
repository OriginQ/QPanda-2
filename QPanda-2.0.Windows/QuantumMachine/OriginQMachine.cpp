#include "OriginQMachine.h"
#include "Factory.h"
#include "QPanda.h"
#include "QPanda/QuantumMetadata.h"
#include "QPanda/ConfigMap.h"

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
}

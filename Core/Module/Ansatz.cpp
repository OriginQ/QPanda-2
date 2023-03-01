#include "Core/Module/Ansatz.h"
USING_QPANDA
using AngleParameter = QGATE_SPACE::AbstractSingleAngleParameter;

static std::map<GateType, AnsatzGateType> ansatz_mapping =
{
  {GateType::PAULI_X_GATE, AnsatzGateType::AGT_X},
  {GateType::HADAMARD_GATE, AnsatzGateType::AGT_H},
  {GateType::CNOT_GATE, AnsatzGateType::AGT_X},
  {GateType::RX_GATE, AnsatzGateType::AGT_RX},
  {GateType::RY_GATE, AnsatzGateType::AGT_RY},
  {GateType::RZ_GATE, AnsatzGateType::AGT_RZ},
  {GateType::X_HALF_PI, AnsatzGateType::AGT_X1},
  {GateType::Z_HALF_PI, AnsatzGateType::AGT_Z1}
};

AnsatzCircuit::AnsatzCircuit() {}

AnsatzCircuit::AnsatzCircuit(QGate& gate) : AnsatzCircuit()
{
    execute(gate.getImplementationPtr(), nullptr);
}

AnsatzCircuit::AnsatzCircuit(AnsatzGate& gate) : AnsatzCircuit()
{
    m_ansatz.emplace_back(gate);
    m_thetas.emplace_back(gate.theta);
}

AnsatzCircuit::AnsatzCircuit(QCircuit& circuit, const Thetas& theta_list) : AnsatzCircuit()
{
    TraversalInterface::execute(circuit.getImplementationPtr(), nullptr);
    
    if (!theta_list.empty())
        m_thetas = theta_list;
}

AnsatzCircuit::AnsatzCircuit(Ansatz& ansatz, const Thetas& theta_list) : AnsatzCircuit()
{
    m_ansatz = ansatz;

    if (!theta_list.empty())
    {
        m_thetas = theta_list;
    }
    else
    {
        for (auto val : ansatz)
            m_thetas.emplace_back(val.theta);
    }
}

AnsatzCircuit::AnsatzCircuit(const AnsatzCircuit& ansatz_circuit, const Thetas& theta_list) : AnsatzCircuit()
{
    m_ansatz = ansatz_circuit.get_ansatz_list();
    m_thetas = ansatz_circuit.get_thetas_list();
    
    if (!theta_list.empty())
        m_thetas = theta_list;
}

void AnsatzCircuit::set_thetas(const Thetas& theta_list)
{
    if (m_thetas.size() != theta_list.size())
        QCERR_AND_THROW(run_fail, "theta list error");

    m_thetas = theta_list;
}

QCircuit AnsatzCircuit::qcircuit()
{
    auto qubit_pool = OriginQubitPool::get_instance();
    qubit_pool->set_capacity(64);

    QCircuit circuit;

    for (size_t i = 0; i < m_ansatz.size(); ++i)
    {
        QCircuit sub_cir;

        auto tar_qubit_addr = m_ansatz[i].target;
        auto ctr_qubit_addr = m_ansatz[i].control;

        auto tar_qubit = qubit_pool->allocateQubitThroughPhyAddress(tar_qubit_addr);

        switch (m_ansatz[i].type)
        {
            case AnsatzGateType::AGT_X:
                sub_cir << X(tar_qubit);
                break;

            case AnsatzGateType::AGT_H:
                sub_cir << H(tar_qubit);
                break;

            case AnsatzGateType::AGT_RX:
                sub_cir << RX(tar_qubit, m_thetas[i]);
                break;

            case AnsatzGateType::AGT_RY:
                sub_cir << RY(tar_qubit, m_thetas[i]);
                break;

            case AnsatzGateType::AGT_RZ:
                sub_cir << RZ(tar_qubit, m_thetas[i]);
                break;

            default:
                break;
        }

        if (ctr_qubit_addr != -1)
        {
            auto ctr_qubit = qubit_pool->allocateQubitThroughPhyAddress(ctr_qubit_addr);
            sub_cir.setControl({ ctr_qubit });
        }

        circuit << sub_cir;
    }

    return circuit;
}

void AnsatzCircuit::insert(QGate& gate)
{
    execute(gate.getImplementationPtr(), nullptr);
    return;
}


void AnsatzCircuit::insert(AnsatzGate& ansatz)
{
    m_ansatz.emplace_back(ansatz);
    m_thetas.emplace_back(ansatz.theta);

    return;
}

void AnsatzCircuit::insert(Ansatz& ansatz)
{
    for (auto val : ansatz)
    {
        m_ansatz.emplace_back(val);
        m_thetas.emplace_back(val.theta);
    }

    return;
}

void AnsatzCircuit::insert(QCircuit& circuit)
{
    TraversalInterface::execute(circuit.getImplementationPtr(), nullptr);
    return;
}

void AnsatzCircuit::insert(AnsatzCircuit& ansatz_circuit, const Thetas& thetas)
{
    auto temp_ansatz = ansatz_circuit.get_ansatz_list();
    auto temp_thetas = ansatz_circuit.get_thetas_list();

    if (!thetas.empty())
        temp_thetas = thetas;

    m_ansatz.insert(m_ansatz.end(), temp_ansatz.begin(), temp_ansatz.end());
    m_thetas.insert(m_thetas.end(), temp_thetas.begin(), temp_thetas.end());
    return;
}

void AnsatzCircuit::execute(std::shared_ptr<AbstractQGateNode> node, std::shared_ptr<QNode> parent_node)
{
    GateType type = (GateType)node->getQGate()->getGateType();
    if (ansatz_mapping.find(type) == ansatz_mapping.end())
        QCERR_AND_THROW(run_fail,"unsupported ansatz gate")

    QVec tar_qubits;
    node->getQuBitVector(tar_qubits);

    QVec ctr_qubits;
    node->getControlVector(ctr_qubits);

    AnsatzGateType ansa_type = ansatz_mapping[type];
    switch (type)
    {
    case GateType::X_HALF_PI:
    case GateType::Z_HALF_PI:
    case GateType::PAULI_X_GATE:
    case GateType::HADAMARD_GATE:
    {
        auto ctr_qubit_addr = ctr_qubits.empty() ? -1 : ctr_qubits[0]->get_phy_addr();
        AnsatzGate ansatz(ansa_type, tar_qubits[0]->get_phy_addr(), -1, ctr_qubit_addr);
        insert(ansatz);
        break;
    }
    case GateType::RX_GATE:
    case GateType::RY_GATE:
    case GateType::RZ_GATE:
    {
        auto param = dynamic_cast<AngleParameter*>(node->getQGate())->getParameter();
        auto ctr_qubit_addr = ctr_qubits.empty() ? -1 : ctr_qubits[0]->get_phy_addr();
        AnsatzGate ansatz(ansa_type, tar_qubits[0]->get_phy_addr(), param, ctr_qubit_addr);
        insert(ansatz);
        break;
    }
    case GateType::CNOT_GATE:
    {
        auto tar_qubit_addr = tar_qubits[1]->get_phy_addr();
        auto ctr_qubit_addr = tar_qubits[0]->get_phy_addr();
        AnsatzGate ansatz(AnsatzGateType::AGT_X, tar_qubit_addr, -1, ctr_qubit_addr);
        insert(ansatz);
        break;
    }

    default:
        break;
    }

    return;
}
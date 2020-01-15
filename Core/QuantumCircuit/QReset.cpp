#include "Core/QuantumCircuit/QReset.h"
#include "Core/Utilities/QProgInfo/ConfigMap.h"

USING_QPANDA
using namespace std;

QReset  QPanda::Reset(Qubit * target_qubit)
{
	QReset reset(target_qubit);
	return reset;
}

QReset::QReset(const QReset & old_reset)
{
	m_reset = old_reset.m_reset;
}

QReset::QReset(std::shared_ptr<AbstractQuantumReset> node)
{
	if (!node)
	{
		QCERR("this shared_ptr is null");
		throw invalid_argument("this shared_ptr is null");
	}
	m_reset = node;
}

QReset::QReset(Qubit * qubit)
{
	auto class_name = ConfigMap::getInstance()["QReset"];
	auto reset = QResetFactory::getInstance().getQuantumReset(class_name, qubit);
	m_reset.reset(reset);
}

std::shared_ptr<AbstractQuantumReset> QReset::getImplementationPtr()
{
	if (!m_reset)
	{
		QCERR("Unknown internal error");
		throw runtime_error("Unknown internal error");
	}
	return m_reset;
}

QReset::~QReset()
{
	m_reset.reset();
}

Qubit * QReset::getQuBit() const
{
	if (nullptr == m_reset)
	{
		QCERR("Unknown internal error");
		throw runtime_error("Unknown internal error");
	}
	return m_reset->getQuBit();
}

NodeType QReset::getNodeType() const
{
	if (!m_reset)
	{
		QCERR("Unknown internal error");
		throw runtime_error("Unknown internal error");
	}
	return (dynamic_pointer_cast<QNode>(m_reset))->getNodeType();
}

void QResetFactory::registClass(string name, CreateReset method)
{
	m_reset_map.insert(pair<string, CreateReset>(name, method));
}

AbstractQuantumReset * QResetFactory::getQuantumReset(std::string & classname, Qubit * pQubit)
{
	auto aiter = m_reset_map.find(classname);
	if (aiter != m_reset_map.end())
	{
		return aiter->second(pQubit);
	}
	else
	{
		QCERR("can not find targit reset class");
		throw runtime_error("can not find targit reset class");
	}
}

NodeType OriginReset::getNodeType() const
{
	return m_node_type;
}

OriginReset::OriginReset(Qubit * qubit) :
	m_target_qubit(qubit),
	m_node_type(RESET_NODE)
{
}

Qubit * OriginReset::getQuBit() const
{
	return m_target_qubit;
}

REGISTER_RESET(OriginReset);
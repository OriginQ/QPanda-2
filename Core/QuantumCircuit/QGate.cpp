#include "QGate.h"
#include "Core/Utilities/QProgInfo/ConfigMap.h"
#include "Core/Utilities/Tools/QPandaException.h"
#include <type_traits>
#include "Core/Utilities/Tools/QStatMatrix.h"

using namespace QGATE_SPACE;
using namespace std;
USING_QPANDA
static QGateNodeFactory* _gs_pGateNodeFactory = QGateNodeFactory::getInstance();
QGATE_FUN_MAP QGateParseMap::m_qgate_function_map = {};
QGate::~QGate()
{
	m_qgate_node.reset();
}

QGate::QGate(std::shared_ptr<AbstractQGateNode> node)
{
	if (!node)
	{
		QCERR("this shared_ptr is null");
		throw invalid_argument("this shared_ptr is null");
	}

	m_qgate_node = node;
}


QGate::QGate(const QGate& old_Gate)
{
	m_qgate_node = old_Gate.m_qgate_node;
}


QGate::QGate(QVec& qs, QuantumGate* QGate)
{
	if (nullptr == QGate)
	{
		QCERR("qgate param err");
		throw invalid_argument("qgate param err");
	}
	m_qgate_node.reset(new OriginQGate(qs, QGate));
}

NodeType QGate::getNodeType() const
{
	if (!m_qgate_node)
	{
		QCERR("Unknown internal error");
		throw runtime_error("Unknown internal error");
	}

	auto temp = dynamic_pointer_cast<QNode>(m_qgate_node);
	return temp->getNodeType();
}

size_t QGate::getQuBitVector(QVec& vector) const
{
	if (!m_qgate_node)
	{
		QCERR("Unknown internal error");
		throw runtime_error("Unknown internal error");
	}

	return m_qgate_node->getQuBitVector(vector);
}

size_t QGate::getTargetQubitNum() const
{
	if (!m_qgate_node)
	{
		QCERR("Unknown internal error");
		throw runtime_error("Unknown internal error");
	}
	return m_qgate_node->getTargetQubitNum();
}

size_t QGate::getControlQubitNum() const
{
	if (!m_qgate_node)
	{
		QCERR("Unknown internal error");
		throw runtime_error("Unknown internal error");
	}
	return m_qgate_node->getControlQubitNum();
}


QuantumGate* QGate::getQGate() const
{
	if (!m_qgate_node)
	{
		QCERR("Unknown internal error");
		throw runtime_error("Unknown internal error");
	}
	return m_qgate_node->getQGate();
}

bool QGate::setDagger(bool is_dagger)
{
	if (!m_qgate_node)
	{
		QCERR("Unknown internal error");
		throw runtime_error("Unknown internal error");
	}
	return m_qgate_node->setDagger(is_dagger);
}

bool QGate::setControl(QVec qubit_vector)
{
	if (!m_qgate_node)
	{
		QCERR("Unknown internal error");
		throw runtime_error("Unknown internal error");
	}
	return m_qgate_node->setControl(qubit_vector);
}

void QGate::clear_control()
{
	return m_qgate_node->clear_control();
}

std::shared_ptr<AbstractQGateNode> QGate::getImplementationPtr()
{
	return m_qgate_node;
}

/*****************************************************************
Name        : dagger
Description : dagger the QGate
argin       :
argout      :
Return      : new QGate
*****************************************************************/
QGate QGate::dagger()
{
	QVec qubit_vector;
	this->getQuBitVector(qubit_vector);
	QVec control_qubit_vector;
	this->getControlVector(control_qubit_vector);

	auto qgate = getQGate();
	auto temp_gate = copy_qgate(std::move(qgate), qubit_vector);
	temp_gate.setControl(control_qubit_vector);
	temp_gate.setDagger(this->isDagger() ^ true);
	return temp_gate;

}

/*****************************************************************
Name        : dagger
Description : set controlQubit to QGate
argin       :
argout      :
Return      : new QGate
*****************************************************************/
QGate QGate::control(QVec control_qubit_vector)
{
	QVec qubit_vector;
	this->getQuBitVector(qubit_vector);
	this->getControlVector(control_qubit_vector);

	auto qgate = getQGate();
	auto temp_gate = copy_qgate(std::move(qgate), qubit_vector);
	temp_gate.setControl(control_qubit_vector);
	temp_gate.setDagger(this->isDagger());
	return temp_gate;
}

bool QGate::isDagger() const
{
	if (!m_qgate_node)
	{
		QCERR("Unknown internal error");
		throw runtime_error("Unknown internal error");
	}

	return m_qgate_node->isDagger();
}

void QGate::remap(QVec qubit_vector)
{
	if (!m_qgate_node)
	{
		QCERR("Unknown internal error");
		throw runtime_error("Unknown internal error");
	}

	return m_qgate_node->remap(qubit_vector);
}

size_t QGate::getControlVector(QVec& qubit_vector) const
{
	if (!m_qgate_node)
	{
		QCERR("Unknown internal error");
		throw runtime_error("Unknown internal error");
	}

	return m_qgate_node->getControlVector(qubit_vector);
}



OriginQGate::~OriginQGate()
{
	if (nullptr != m_qgate)
	{
		delete m_qgate;
	}
}

OriginQGate::OriginQGate(QVec& qubit_vector, QuantumGate* qgate) :m_Is_dagger(false)
{
	if (nullptr == qgate)
	{
		QCERR("qgate param err");
		throw invalid_argument("qgate param err");
	}
	if (0 == qubit_vector.size())
	{
		QCERR("qubit_vector err");
		throw invalid_argument("qubit_vector err");
	}
	m_qgate = qgate;
	for (auto aiter = qubit_vector.begin(); aiter != qubit_vector.end(); ++aiter)
	{
		m_qubit_vector.push_back(*aiter);
	}
	m_node_type = GATE_NODE;
}

NodeType OriginQGate::getNodeType() const
{
	return m_node_type;
}

size_t OriginQGate::getQuBitVector(QVec& vector) const
{
	for (auto aiter : m_qubit_vector)
	{
		vector.push_back(aiter);
	}
	return m_qubit_vector.size();
}

size_t OriginQGate::getTargetQubitNum() const
{
	return m_qubit_vector.size();
}

size_t OriginQGate::getControlQubitNum() const
{
	return m_control_qubit_vector.size();
}

Qubit* OriginQGate::popBackQuBit()
{
	auto temp = m_qubit_vector.back();
	m_qubit_vector.pop_back();
	return temp;
}

QuantumGate* OriginQGate::getQGate() const
{
	if (nullptr == m_qgate)
	{
		QCERR("m_qgate is null");
		throw runtime_error("m_qgate is null");
	}
	return m_qgate;
}

void OriginQGate::setQGate(QuantumGate* qgate)
{
	if ((nullptr != m_qgate) && (qgate != m_qgate))
	{
		delete m_qgate;
	}
	m_qgate = qgate;
}

bool OriginQGate::setDagger(bool is_dagger)
{
	m_Is_dagger = is_dagger;
	return m_Is_dagger;
}

bool OriginQGate::setControl(QVec qubit_vector)
{
	for (auto aiter : qubit_vector)
	{
		m_control_qubit_vector.push_back(aiter);
	}
	return true;
}

bool OriginQGate::isDagger() const
{
	return m_Is_dagger;
}

size_t OriginQGate::getControlVector(QVec& qubit_vector) const
{
	for (auto aiter : m_control_qubit_vector)
	{
		qubit_vector.push_back(aiter);
	}
	return m_control_qubit_vector.size();
}

void OriginQGate::PushBackQuBit(Qubit* qubit)
{

	if (nullptr == qubit)
	{
		QCERR("param error");
		throw invalid_argument("param error");
	}

	m_qubit_vector.push_back(qubit);

}

void OriginQGate::remap(QVec qubit_vector)
{
	if (m_qubit_vector.size() != qubit_vector.size())
	{
		QCERR_AND_THROW(run_fail, "Error: failed to remap qubit, the size of new qubit_vec is error.");
	}
	m_qubit_vector.swap(qubit_vector);
}

void QGateParseSingleBit(QuantumGate* qgate,
	QVec& qubit_vector,
	QPUImpl* qgates,
	bool is_dagger,
	QVec& control_qubit_vector,
	GateType type)
{
	if (nullptr == qgate)
	{
		QCERR("param error");
		throw invalid_argument("param error");
	}

	QStat matrix;
	qgate->getMatrix(matrix);
	Qubit* qubit = *(qubit_vector.begin());
	size_t bit = qubit->getPhysicalQubitPtr()->getQubitAddr();
	if (control_qubit_vector.size() == 0)
	{
		qgates->unitarySingleQubitGate(bit, matrix, is_dagger, type);
	}
	else
	{
		size_t temp;
		vector<size_t> bit_num_vector;
		for (auto aiter : control_qubit_vector)
		{
			temp = aiter->getPhysicalQubitPtr()->getQubitAddr();
			bit_num_vector.push_back(temp);
		}
		bit_num_vector.push_back(bit);
		qgates->controlunitarySingleQubitGate(bit, bit_num_vector, matrix, is_dagger, type);
	}

}

void QGateParseDoubleBit(QuantumGate* qgate,
	QVec& qubit_vector,
	QPUImpl* qgates,
	bool is_dagger,
	QVec& control_qubit_vector,
	GateType type)
{
	QStat matrix;
	qgate->getMatrix(matrix);
	auto aiter = qubit_vector.begin();
	Qubit* qubit = *aiter;
	aiter++;
	Qubit* qubit2 = *aiter;
	size_t bit = qubit->getPhysicalQubitPtr()->getQubitAddr();
	size_t bit2 = qubit2->getPhysicalQubitPtr()->getQubitAddr();

	if (control_qubit_vector.size() == 0)
	{
		qgates->unitaryDoubleQubitGate(bit, bit2, matrix, is_dagger, type);
	}
	else
	{
		size_t temp;
		vector<size_t> bit_num_vector;
		for (auto aiter : control_qubit_vector)
		{
			temp = aiter->getPhysicalQubitPtr()->getQubitAddr();
			bit_num_vector.push_back(temp);
		}
		bit_num_vector.push_back(bit2);
		bit_num_vector.push_back(bit);
		qgates->controlunitaryDoubleQubitGate(bit, bit2, bit_num_vector, matrix, is_dagger, type);
	}
}

void QGateParseOracleBit(QuantumGate* qgate,
    QVec& qubit_vector,
    QPUImpl* qgates,
    bool is_dagger,
    QVec& control_qubit_vector,
    GateType type)
{
    if (nullptr == qgate)
    {
        QCERR("param error");
        throw invalid_argument("param error");
    }

    QStat matrix;
    qgate->getMatrix(matrix);

    Qnum targets(qubit_vector.size());
    for (size_t i = 0; i < qubit_vector.size(); i++)
    {
        targets[i] = qubit_vector[i]->get_phy_addr();
    }

    if (control_qubit_vector.size() == 0)
    {
        qgates->OracleGate(targets, matrix, is_dagger);
    }
    else
    {
        vector<size_t> controls(control_qubit_vector.size());
        for (size_t i = 0; i < control_qubit_vector.size(); i++)
        {
            controls[i] = control_qubit_vector[i]->get_phy_addr();
        }

        controls.insert(controls.end(), targets.begin(), targets.end());
        qgates->controlOracleGate(targets, controls, matrix, is_dagger);
    }

}

#define REGISTER_QGATE_PARSE(BitCount,FunctionName) \
class insertQGateMapHelper_##FunctionName \
{ \
public: \
     inline insertQGateMapHelper_##FunctionName(int bitCount,QGATE_FUN pFunction) \
    { \
        QGateParseMap::insertMap(bitCount, pFunction); \
    } \
};\
insertQGateMapHelper_##FunctionName _G_insertQGateHelper##FunctionName(BitCount, FunctionName)

REGISTER_QGATE_PARSE(1, QGateParseSingleBit);
REGISTER_QGATE_PARSE(2, QGateParseDoubleBit);
REGISTER_QGATE_PARSE(-1, QGateParseOracleBit);

/*Singl Gate*/
QGate QPanda::I(Qubit* qubit)
{
	string name = "I";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit });
}

QCircuit QPanda::I(const QVec& qubits)
{
	string name = "I";
	QCircuit cir = QCircuit();
	for (auto &qubit : qubits)
	{
		cir << _gs_pGateNodeFactory->getGateNode(name, { qubit });		
	}
	return cir;	
}

QGate QPanda::X(Qubit* qubit)
{
	string name = "X";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit });
}

QCircuit QPanda::X(const QVec& qubits)
{
	string name = "X";
	QCircuit cir = QCircuit();
	for (auto &qubit : qubits)
	{
		cir << _gs_pGateNodeFactory->getGateNode(name, { qubit });
	}
	return cir;
}

QGate QPanda::X1(Qubit* qubit)
{
	string name = "X1";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit });
}

QCircuit QPanda::X1(const QVec& qubits)
{
	string name = "X1";
	QCircuit cir = QCircuit();
	for (auto &qubit : qubits)
	{
		cir << _gs_pGateNodeFactory->getGateNode(name, { qubit });
	}
	return cir;
}

QGate QPanda::RX(Qubit* qubit, double angle)
{
	string name = "RX";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit }, angle);
}

QCircuit QPanda::RX(const QVec& qubits, double angle)
{
	string name = "RX";
	QCircuit cir = QCircuit();
	for (auto &qubit : qubits)
	{
		cir << _gs_pGateNodeFactory->getGateNode(name, { qubit }, angle);
	}
	return cir;
}

QGate QPanda::U1(Qubit* qubit, double angle)
{
	string name = "U1";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit }, angle);
}

QCircuit QPanda::U1(const QVec& qubits, double angle)
{
	string name = "U1";
	QCircuit cir = QCircuit();
	for (auto &qubit : qubits)
	{
		cir << _gs_pGateNodeFactory->getGateNode(name, { qubit }, angle);
	}

	return cir;
}

QGate QPanda::P(Qubit* qubit, double angle)
{
	string name = "P";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit }, angle);
}

QCircuit QPanda::P(const QVec& qubits, double angle)
{
	string name = "P";
	QCircuit cir = QCircuit();
	for (auto &qubit : qubits)
	{
		cir << _gs_pGateNodeFactory->getGateNode(name, { qubit }, angle);
	}

	return cir;
}

QGate QPanda::P(int qaddr, double angle)
{
	return P(get_qubit_by_phyaddr(qaddr), angle);
}

QCircuit QPanda::P(const std::vector<int>& qaddrs, double angle)
{
	QCircuit cir = QCircuit();
	for (auto &qaddr : qaddrs)
	{
		cir << P(get_qubit_by_phyaddr(qaddr), angle);
	}

	return cir;
}


QGate QPanda::U2(Qubit* qubit, double phi, double lambda)
{
	string name = "U2";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit }, phi, lambda);
}

QCircuit QPanda::U2(const QVec& qubits, double phi, double lambda)
{
	string name = "U2";
	QCircuit cir = QCircuit();
	for (auto &qubit : qubits)
	{
		cir << _gs_pGateNodeFactory->getGateNode(name, { qubit }, phi, lambda);
	}

	return cir;
}

QGate QPanda::U3(Qubit* qubit, double theta, double phi, double lambda)
{
	string name = "U3";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit }, theta, phi, lambda);
}

QCircuit QPanda::U3(const QVec& qubits, double theta, double phi, double lambda)
{
	string name = "U3";
	QCircuit cir = QCircuit();
	for (auto &qubit : qubits)
	{
		cir << _gs_pGateNodeFactory->getGateNode(name, { qubit }, theta, phi, lambda);
	}
	return cir;
}

QGate QPanda::U3(Qubit* qubit, QStat& matrix)
{
	string name = "U3";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit }, matrix);
}

QCircuit QPanda::U3(const QVec& qubits, QStat& matrix)
{
	string name = "U3";
	QCircuit cir = QCircuit();
	for (auto &qubit : qubits)
	{
		cir << _gs_pGateNodeFactory->getGateNode(name, { qubit }, matrix);
	}
	return cir;
}

QGate QPanda::Y(Qubit* qubit)
{
	string name = "Y";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit });
}

QCircuit QPanda::Y(const QVec& qubits)
{
	string name = "Y";
	QCircuit cir = QCircuit();
	for (auto &qubit : qubits)
	{
		cir << _gs_pGateNodeFactory->getGateNode(name, { qubit });
	}

	return cir;
}

QGate QPanda::Y1(Qubit* qubit)
{
	string name = "Y1";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit });
}

QCircuit QPanda::Y1(const QVec& qubits)
{
	string name = "Y1";
	QCircuit cir = QCircuit();
	for (auto &qubit : qubits)
	{
		cir << _gs_pGateNodeFactory->getGateNode(name, { qubit });
	}

	return cir;
}

QGate QPanda::RY(Qubit* qubit, double angle)
{
	string name = "RY";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit }, angle);
}

QCircuit QPanda::RY(const QVec& qubits, double angle)
{
	string name = "RY";
	QCircuit cir = QCircuit();
	for (auto &qubit : qubits)
	{
		cir << _gs_pGateNodeFactory->getGateNode(name, { qubit },angle);
	}

	return cir;
}

QGate QPanda::Z(Qubit* qubit)
{
	string name = "Z";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit });
}

QCircuit QPanda::Z(const QVec& qubits)
{
	string name = "Z";
	QCircuit cir = QCircuit();
	for (auto &qubit : qubits)
	{
		cir << _gs_pGateNodeFactory->getGateNode(name, { qubit });
	}

	return cir;
}

QGate QPanda::Z1(Qubit* qubit)
{
	string name = "Z1";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit });
}

QCircuit QPanda::Z1(const QVec& qubits)
{
	string name = "Z1";
	QCircuit cir = QCircuit();
	for (auto &qubit : qubits)
	{
		cir << _gs_pGateNodeFactory->getGateNode(name, { qubit });
	}

	return cir;
}

QGate QPanda::RZ(Qubit* qubit, double angle)
{
	string name = "RZ";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit }, angle);
}

QCircuit QPanda::RZ(const QVec& qubits, double angle)
{
	string name = "RZ";
	QCircuit cir = QCircuit();
	for (auto &qubit : qubits)
	{
		cir << _gs_pGateNodeFactory->getGateNode(name, { qubit }, angle);
	}

	return cir;
}

QGate QPanda::RPhi(Qubit* qubit, double angle, double phi)
{
	string name = "RPhi";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit }, angle, phi);
}

QCircuit QPanda::RPhi(const QVec& qubits, double angle, double phi)
{
	string name = "RPhi";
	QCircuit cir = QCircuit();
	for (auto &qubit : qubits)
	{
		cir << _gs_pGateNodeFactory->getGateNode(name, { qubit },angle, phi);
	}

	return cir;
}

QGate QPanda::S(Qubit* qubit)
{
	string name = "S";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit });
}

QCircuit QPanda::S(const QVec& qubits)
{
	string name = "S";
	QCircuit cir = QCircuit();
	for (auto &qubit : qubits)
	{
		cir << _gs_pGateNodeFactory->getGateNode(name, { qubit });
	}

	return cir;
}

QGate QPanda::T(Qubit* qubit)
{
	string name = "T";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit });
}

QCircuit QPanda::T(const QVec& qubits)
{
	string name = "T";
	QCircuit cir = QCircuit();
	for (auto &qubit : qubits)
	{
		cir << _gs_pGateNodeFactory->getGateNode(name, { qubit });
	}

	return cir;
}

QGate  QPanda::H(Qubit* qubit)
{
	string name = "H";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit });
}

QCircuit QPanda::H(const QVec& qubits)
{
	string name = "H";
	QCircuit cir = QCircuit();
	for (auto &qubit : qubits)
	{
		cir << _gs_pGateNodeFactory->getGateNode(name, { qubit });
	}

	return cir;
}

QGate  QPanda::ECHO(Qubit* qubit)
{
	string name = "ECHO";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit });
}

QCircuit QPanda::ECHO(const QVec& qubits)
{
	string name = "ECHO";
	QCircuit cir = QCircuit();
	for (auto &qubit : qubits)
	{
		cir << _gs_pGateNodeFactory->getGateNode(name, { qubit });
	}

	return cir;
}

QGate  QPanda::BARRIER(Qubit* qubit)
{
	string name = "BARRIER";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit });
}

QGate  QPanda::BARRIER(QVec qubits)
{
	if (qubits.size() < 1)
	{
		throw std::runtime_error("Error: BARRIER Create");
	}

	string name = "BARRIER";
	auto gate = _gs_pGateNodeFactory->getGateNode(name, qubits[0]);

	if (qubits.size() > 1)
	{
		gate.setControl(QVec(qubits.begin() + 1, qubits.end()));
	}

	return gate;
}

/** Construct QGate by Qubit physics addr */
QGate QPanda::I(int qaddr)
{
	return I(get_qubit_by_phyaddr(qaddr));
}

QCircuit QPanda::I(const std::vector<int>& qaddrs)
{
	QCircuit cir = QCircuit();
	for (auto &qaddr : qaddrs)
	{
		cir << I(get_qubit_by_phyaddr(qaddr));
	}

	return cir;
}

QGate QPanda::X(int qaddr)
{
	return X(get_qubit_by_phyaddr(qaddr));
}

QCircuit QPanda::X(const std::vector<int>& qaddrs)
{
	QCircuit cir = QCircuit();
	for (auto &qaddr : qaddrs)
	{
		cir << X(get_qubit_by_phyaddr(qaddr));
	}

	return cir;
}

QGate QPanda::X1(int qaddr)
{
	return X1(get_qubit_by_phyaddr(qaddr));
}

QCircuit QPanda::X1(const std::vector<int>& qaddrs)
{
	QCircuit cir = QCircuit();
	for (auto &qaddr : qaddrs)
	{
		cir << X1(get_qubit_by_phyaddr(qaddr));
	}

	return cir;
}

QGate QPanda::RX(int  qaddr, double angle)
{
	return RX(get_qubit_by_phyaddr(qaddr), angle);
}

QCircuit QPanda::RX(const std::vector<int>& qaddrs, double angle)
{
	QCircuit cir = QCircuit();
	for (auto &qaddr : qaddrs)
	{
		cir << RX(get_qubit_by_phyaddr(qaddr), angle);
	}

	return cir;
}

QGate QPanda::U1(int  qaddr, double angle)
{
	return U1(get_qubit_by_phyaddr(qaddr), angle);
}

QCircuit QPanda::U1(const std::vector<int>& qaddrs, double angle)
{
	QCircuit cir = QCircuit();
	for (auto &qaddr : qaddrs)
	{
		cir << U1(get_qubit_by_phyaddr(qaddr), angle);
	}

	return cir;
}

QGate QPanda::U2(int  qaddr, double phi, double lambda)
{
	return U2(get_qubit_by_phyaddr(qaddr), phi, lambda);
}

QCircuit QPanda::U2(const std::vector<int>& qaddrs, double angle, double lambda)
{
	QCircuit cir = QCircuit();
	for (auto &qaddr : qaddrs)
	{
		cir << U2(get_qubit_by_phyaddr(qaddr), angle, lambda);
	}

	return cir;
}

QGate QPanda::U3(int  qaddr, double theta, double phi, double lambda)
{
	return U3(get_qubit_by_phyaddr(qaddr), theta, phi, lambda);
}

QCircuit QPanda::U3(const std::vector<int>& qaddrs, double theta, double phi, double lambda)
{
	QCircuit cir = QCircuit();
	for (auto &qaddr : qaddrs)
	{
		cir << U3(get_qubit_by_phyaddr(qaddr), theta, phi, lambda);
	}

	return cir;
}

QGate QPanda::Y(int  qaddr)
{
	return Y(get_qubit_by_phyaddr(qaddr));
}

QCircuit QPanda::Y(const std::vector<int>& qaddrs)
{
	QCircuit cir = QCircuit();
	for (auto &qaddr : qaddrs)
	{
		cir << Y(get_qubit_by_phyaddr(qaddr));
	}

	return cir;
}

QGate QPanda::Y1(int  qaddr)
{
	return Y1(get_qubit_by_phyaddr(qaddr));
}

QCircuit QPanda::Y1(const std::vector<int>& qaddrs)
{
	QCircuit cir = QCircuit();
	for (auto &qaddr : qaddrs)
	{
		cir << Y1(get_qubit_by_phyaddr(qaddr));
	}

	return cir;
}

QGate QPanda::RY(int  qaddr, double angle)
{
	return RY(get_qubit_by_phyaddr(qaddr), angle);
}

QCircuit QPanda::RY(const std::vector<int>& qaddrs, double angle)
{
	QCircuit cir = QCircuit();
	for (auto &qaddr : qaddrs)
	{
		cir << RY(get_qubit_by_phyaddr(qaddr), angle);
	}

	return cir;
}

QGate QPanda::Z(int  qaddr)
{
	return Z(get_qubit_by_phyaddr(qaddr));
}

QCircuit QPanda::Z(const std::vector<int>& qaddrs)
{
	QCircuit cir = QCircuit();
	for (auto &qaddr : qaddrs)
	{
		cir << Z(get_qubit_by_phyaddr(qaddr));
	}

	return cir;
}

QGate QPanda::Z1(int  qaddr)
{
	return Z1(get_qubit_by_phyaddr(qaddr));
}

QCircuit QPanda::Z1(const std::vector<int>& qaddrs)
{
	QCircuit cir = QCircuit();
	for (auto &qaddr : qaddrs)
	{
		cir << Z1(get_qubit_by_phyaddr(qaddr));
	}

	return cir;
}

QGate QPanda::RZ(int  qaddr, double angle)
{
	return RZ(get_qubit_by_phyaddr(qaddr), angle);
}

QCircuit QPanda::RZ(const std::vector<int>& qaddrs, double angle)
{
	QCircuit cir = QCircuit();
	for (auto &qaddr : qaddrs)
	{
		cir << RZ(get_qubit_by_phyaddr(qaddr), angle);
	}

	return cir;
}

QGate QPanda::RPhi(int  qaddr, double angle, double phi)
{
	return RPhi(get_qubit_by_phyaddr(qaddr), angle, phi);
}

QCircuit QPanda::RPhi(const std::vector<int>& qaddrs, double angle, double phi)
{
	QCircuit cir = QCircuit();
	for (auto &qaddr : qaddrs)
	{
		cir << RPhi(get_qubit_by_phyaddr(qaddr), angle, phi);
	}

	return cir;
}

QGate QPanda::S(int  qaddr)
{
	return S(get_qubit_by_phyaddr(qaddr));
}

QCircuit QPanda::S(const std::vector<int>& qaddrs)
{
	QCircuit cir = QCircuit();
	for (auto &qaddr : qaddrs)
	{
		cir << S(get_qubit_by_phyaddr(qaddr));
	}

	return cir;
}

QGate QPanda::T(int  qaddr)
{
	return T(get_qubit_by_phyaddr(qaddr));
}

QCircuit QPanda::T(const std::vector<int>& qaddrs)
{
	QCircuit cir = QCircuit();
	for (auto &qaddr : qaddrs)
	{
		cir << T(get_qubit_by_phyaddr(qaddr));
	}

	return cir;
}

QGate QPanda::H(int  qaddr)
{
	return H(get_qubit_by_phyaddr(qaddr));
}

QCircuit QPanda::H(const std::vector<int>& qaddrs)
{
	QCircuit cir = QCircuit();
	for (auto &qaddr : qaddrs)
	{
		cir << H(get_qubit_by_phyaddr(qaddr));
	}

	return cir;
}

QGate QPanda::ECHO(int  qaddr)
{
	return ECHO(get_qubit_by_phyaddr(qaddr));
}

QCircuit QPanda::ECHO(const std::vector<int>& qaddrs)
{
	QCircuit cir = QCircuit();
	for (auto &qaddr : qaddrs)
	{
		cir << ECHO(get_qubit_by_phyaddr(qaddr));
	}

	return cir;
}

QGate QPanda::BARRIER(int  qaddr)
{
	return BARRIER(get_qubit_by_phyaddr(qaddr));
}

QGate QPanda::BARRIER(std::vector<int> qaddrs)
{
	return BARRIER(get_qubits_by_phyaddrs(qaddrs));
}

/*Double Gate*/
QGate QPanda::iSWAP(Qubit* targitBit_fisrt, Qubit* targitBit_second)
{
	string name = "ISWAP";
	return _gs_pGateNodeFactory->getGateNode(name, { targitBit_fisrt, targitBit_second });
}

QCircuit QPanda::iSWAP(const QVec& targitBits_first, const  QVec& targitBits_second)
{
	if (targitBits_first.size() == 0 || targitBits_second.size() == 0)
	{
		QCERR("qubit_vector err");
		throw invalid_argument("qubit_vector err");
	}

	string name = "ISWAP";
	auto cir = QCircuit();
	if (targitBits_first.size() == targitBits_second.size())
	{
		for (int i = 0; i < targitBits_first.size(); ++i)
		{
			if (targitBits_first[i] != targitBits_second[i])
			{
				cir << _gs_pGateNodeFactory->getGateNode(name, { targitBits_first[i], targitBits_second[i] });
			}
			else
			{
				QCERR("double_gate qubit err");
				throw invalid_argument("double_gate qubit");
			}
		}
	}
	else
	{
		QCERR("qubit_vector size err");
		throw invalid_argument("qubit_vector size");
	}

	return cir;
}

QGate QPanda::iSWAP(Qubit* targitBit_fisrt, Qubit* targitBit_second, double theta)
{
	string name = "ISWAPTheta";
	return _gs_pGateNodeFactory->getGateNode(name,
		{ targitBit_fisrt,targitBit_second },
		theta);
}

QCircuit QPanda::iSWAP(const QVec& targitBits_first, const QVec& targitBits_second, double theta)
{
	if (targitBits_first.size() == 0 || targitBits_second.size() == 0)
	{
		QCERR("qubit_vector err");
		throw invalid_argument("qubit_vector err");
	}

	string name = "ISWAPTheta";
	auto cir = QCircuit();
	if (targitBits_first.size() == targitBits_second.size())
	{
		for (int i = 0; i < targitBits_first.size(); ++i)
		{
			if (targitBits_first[i] != targitBits_second[i])
			{
				cir << _gs_pGateNodeFactory->getGateNode(name, { targitBits_first[i], targitBits_second[i] },theta);
			}
			else
			{
				QCERR("double_gate qubit err");
				throw invalid_argument("double_gate qubit");
			}
		}
	}
	else
	{
		QCERR("qubit_vector size err");
		throw invalid_argument("qubit_vector size");
	}

	return cir;
}

QGate QPanda::CR(Qubit* control_qubit, Qubit* targit_qubit, double theta)
{
	string name = "CPHASE";
	return _gs_pGateNodeFactory->getGateNode(name, { control_qubit, targit_qubit }, theta);
}

QCircuit QPanda::CR(const QVec& targitBits_first, const QVec& targitBits_second, double theta)
{
	if (targitBits_first.size() == 0 || targitBits_second.size() == 0)
	{
		QCERR("qubit_vector err");
		throw invalid_argument("qubit_vector err");
	}

	string name = "CPHASE";
	auto cir = QCircuit();
	if (targitBits_first.size() == targitBits_second.size())
	{
		for (int i = 0; i < targitBits_first.size(); ++i)
		{
			if (targitBits_first[i] != targitBits_second[i])
			{
				cir << _gs_pGateNodeFactory->getGateNode(name, { targitBits_first[i], targitBits_second[i] }, theta);
			}
			else
			{
				QCERR("double_gate qubit err");
				throw invalid_argument("double_gate qubit");
			}
		}
	}
	else
	{
		QCERR("qubit_vector size err");
		throw invalid_argument("qubit_vector size");
	}

	return cir;
}

QGate QPanda::CR(int control_qaddr, int target_qaddr, double theta)
{
	return CR(get_qubit_by_phyaddr(control_qaddr), get_qubit_by_phyaddr(target_qaddr), theta);
}

QCircuit QPanda::CR(const std::vector<int>& control_qaddrs, const std::vector<int>& target_qaddrs, double theta)
{
	if (control_qaddrs.size() == 0 || target_qaddrs.size() == 0)
	{
		QCERR("qubit_vector err");
		throw invalid_argument("qubit_vector err");
	}

	QCircuit cir = QCircuit();
	if (control_qaddrs.size() == target_qaddrs.size())
	{
		for (int i = 0; i < control_qaddrs.size(); ++i)
		{
			if (control_qaddrs[i] != target_qaddrs[i])
			{
				cir << CR(get_qubit_by_phyaddr(control_qaddrs[i]), get_qubit_by_phyaddr(target_qaddrs[i]), theta);
			}
			else
			{
				QCERR("double_gate qubit err");
				throw invalid_argument("double_gate qubit");
			}
		}
	}
	else
	{
		QCERR("qubit_vector size err");
		throw invalid_argument("qubit_vector size");
	}

	return cir;
}

QGate QPanda::SqiSWAP(Qubit* targitBit_fisrt, Qubit* targitBit_second)
{
	string name = "SQISWAP";
	return _gs_pGateNodeFactory->getGateNode(name,
		{ targitBit_fisrt,targitBit_second });
}

QCircuit QPanda::SqiSWAP(const QVec& targitBits_first, const QVec& targitBits_second)
{
	if (targitBits_first.size() == 0 || targitBits_second.size() == 0)
	{
		QCERR("qubit_vector err");
		throw invalid_argument("qubit_vector err");
	}

	string name = "SQISWAP";
	auto cir = QCircuit();
	if (targitBits_first.size() == targitBits_second.size())
	{
		for (int i = 0; i < targitBits_first.size(); ++i)
		{
			if (targitBits_first[i] != targitBits_second[i])
			{
				cir << _gs_pGateNodeFactory->getGateNode(name, { targitBits_first[i], targitBits_second[i] });
			}
			else
			{
				QCERR("double_gate qubit err");
				throw invalid_argument("double_gate qubit");
			}
		}
	}
	else
	{
		QCERR("qubit_vector size err");
		throw invalid_argument("qubit_vector size");
	}

	return cir;
}

QGate QPanda::SqiSWAP(int control_qaddr, int target_qaddr)
{
	return SqiSWAP(get_qubit_by_phyaddr(control_qaddr), get_qubit_by_phyaddr(target_qaddr));
}

QCircuit QPanda::SqiSWAP(const std::vector<int>& control_qaddrs, const std::vector<int>&  target_qaddrs)
{
	if (control_qaddrs.size() == 0 || target_qaddrs.size() == 0)
	{
		QCERR("qubit_vector err");
		throw invalid_argument("qubit_vector err");
	}

	QCircuit cir = QCircuit();
	if (control_qaddrs.size() == target_qaddrs.size())
	{
		for (int i = 0; i < control_qaddrs.size(); ++i)
		{
			if (control_qaddrs[i] != target_qaddrs[i])
			{
				cir << SqiSWAP(get_qubit_by_phyaddr(control_qaddrs[i]), get_qubit_by_phyaddr(target_qaddrs[i]));
			}
			else
			{
				QCERR("double_gate qubit err");
				throw invalid_argument("double_gate qubit");
			}
		}
	}
	else
	{
		QCERR("qubit_vector size err");
		throw invalid_argument("qubit_vector size");
	}

	return cir;
}

QGate QPanda::SWAP(Qubit* targitBit_fisrt, Qubit* targitBit_second)
{
	string name = "SWAP";
	return _gs_pGateNodeFactory->getGateNode(name,
		{ targitBit_fisrt,targitBit_second });
}

QCircuit QPanda::SWAP(const QVec& targitBits_first, const QVec& targitBits_second)
{
	if (targitBits_first.size() == 0 || targitBits_second.size() == 0)
	{
		QCERR("qubit_vector err");
		throw invalid_argument("qubit_vector err");
	}

	string name = "SWAP";
	auto cir = QCircuit();
	if (targitBits_first.size() == targitBits_second.size())
	{
		for (int i = 0; i < targitBits_first.size(); ++i)
		{
			if (targitBits_first[i] != targitBits_second[i])
			{
				cir << _gs_pGateNodeFactory->getGateNode(name, { targitBits_first[i], targitBits_second[i] });
			}
			else
			{
				QCERR("double_gate qubit err");
				throw invalid_argument("double_gate qubit");
			}
		}
	}
	else
	{
		QCERR("qubit_vector size err");
		throw invalid_argument("qubit_vector size");
	}

	return cir;
}

QGate QPanda::SWAP(int control_qaddr, int target_qaddr)
{
	return SWAP(get_qubit_by_phyaddr(control_qaddr), get_qubit_by_phyaddr(target_qaddr));
}

QCircuit QPanda::SWAP(const std::vector<int>& control_qaddrs, const std::vector<int>& target_qaddrs)
{
	if (control_qaddrs.size() == 0 || target_qaddrs.size() == 0)
	{
		QCERR("qubit_vector err");
		throw invalid_argument("qubit_vector err");
	}

	QCircuit cir = QCircuit();
	if (control_qaddrs.size() == target_qaddrs.size())
	{
		for (int i = 0; i < control_qaddrs.size(); ++i)
		{
			if (control_qaddrs[i] != target_qaddrs[i])
			{
				cir << SWAP(get_qubit_by_phyaddr(control_qaddrs[i]), get_qubit_by_phyaddr(target_qaddrs[i]));
			}
			else
			{
				QCERR("double_gate qubit err");
				throw invalid_argument("double_gate qubit");
			}
		}
	}
	else
	{
		QCERR("qubit_vector size err");
		throw invalid_argument("qubit_vector size");
	}

	return cir;
}

QGate QPanda::iSWAP(int control_qaddr, int target_qaddr)
{
	return iSWAP(get_qubit_by_phyaddr(control_qaddr), get_qubit_by_phyaddr(target_qaddr));
}

QCircuit QPanda::iSWAP(const std::vector<int>& control_qaddrs, const std::vector<int>& target_qaddrs)
{
	if (control_qaddrs.size() == 0 || target_qaddrs.size() == 0)
	{
		QCERR("qubit_vector err");
		throw invalid_argument("qubit_vector err");
	}

	QCircuit cir = QCircuit();
	if (control_qaddrs.size() == target_qaddrs.size())
	{
		for (int i = 0; i < control_qaddrs.size(); ++i)
		{
			if (control_qaddrs[i] != target_qaddrs[i])
			{
				cir << iSWAP(get_qubit_by_phyaddr(control_qaddrs[i]), get_qubit_by_phyaddr(target_qaddrs[i]));
			}
			else
			{
				QCERR("double_gate qubit err");
				throw invalid_argument("double_gate qubit");
			}
		}
	}
	else
	{
		QCERR("qubit_vector size err");
		throw invalid_argument("qubit_vector size");
	}

	return cir;
}

QGate QPanda::iSWAP(int control_qaddr, int target_qaddr, double theta)
{
	return iSWAP(get_qubit_by_phyaddr(control_qaddr), get_qubit_by_phyaddr(target_qaddr), theta);
}

QCircuit QPanda::iSWAP(const std::vector<int>& control_qaddrs, const std::vector<int>& target_qaddrs, double theta)
{
	if (control_qaddrs.size() == 0 || target_qaddrs.size() == 0)
	{
		QCERR("qubit_vector err");
		throw invalid_argument("qubit_vector err");
	}

	QCircuit cir = QCircuit();
	if (control_qaddrs.size() == target_qaddrs.size())
	{
		for (int i = 0; i < control_qaddrs.size(); ++i)
		{
			if (control_qaddrs[i] != target_qaddrs[i])
			{
				cir << iSWAP(get_qubit_by_phyaddr(control_qaddrs[i]), get_qubit_by_phyaddr(target_qaddrs[i]), theta);
			}
			else
			{
				QCERR("double_gate qubit err");
				throw invalid_argument("double_gate qubit");
			}
		}
	}
	else
	{
		QCERR("qubit_vector size err");
		throw invalid_argument("qubit_vector size");
	}

	return cir;
}


QGate  QPanda::CNOT(Qubit* control_qubit, Qubit* target_qubit)
{
	string name = "CNOT";
	return _gs_pGateNodeFactory->getGateNode(name, { control_qubit, target_qubit });
}

QCircuit  QPanda::CNOT(const QVec &control_qubits, const QVec &target_qubits)
{
	if (control_qubits.size() == 0 || target_qubits.size() == 0)
	{
		QCERR("qubit_vector err");
		throw invalid_argument("qubit_vector err");
	}
	string name = "CNOT";
	auto cir = QCircuit();
	if (control_qubits.size() == target_qubits.size()) 
	{
		for (int i = 0; i < control_qubits.size(); ++i) 
		{
			if(control_qubits[i]!=target_qubits[i])
			{
               cir << _gs_pGateNodeFactory->getGateNode(name, { control_qubits[i], target_qubits[i] });
			}
			else
			{
				QCERR("double_gate qubit err");
				throw invalid_argument("double_gate qubit");
			}
		}		
	}
	else
	{
		QCERR("qubit_vector size err");
		throw invalid_argument("qubit_vector size");
	}

	return cir;
}

QGate QPanda::CNOT(int control_qaddr, int target_qaddr)
{
	return CNOT(get_qubit_by_phyaddr(control_qaddr), get_qubit_by_phyaddr(target_qaddr));
}

QCircuit QPanda::CNOT(const std::vector<int>& control_qaddrs, const std::vector<int>& target_qaddrs)
{
	if (control_qaddrs.size() == 0 || target_qaddrs.size() == 0)
	{
		QCERR("qubit_vector err");
		throw invalid_argument("qubit_vector err");
	}

	QCircuit cir = QCircuit();
	if (control_qaddrs.size() == target_qaddrs.size())
	{
		for (int i = 0; i < control_qaddrs.size(); ++i)
		{
			if (control_qaddrs[i] != target_qaddrs[i])
			{
				cir << CNOT(get_qubit_by_phyaddr(control_qaddrs[i]), get_qubit_by_phyaddr(target_qaddrs[i]));
			}
			else
			{
				QCERR("double_gate qubit err");
				throw invalid_argument("double_gate qubit");
			}
		}
	}
	else
	{
		QCERR("qubit_vector size err");
		throw invalid_argument("qubit_vector size");
	}

	return cir;
}


QGate QPanda::CZ(Qubit* control_qubit, Qubit* target_qubit)
{
	string name = "CZ";
	return _gs_pGateNodeFactory->getGateNode(name, { control_qubit, target_qubit });
}

QCircuit QPanda::CZ(const QVec& control_qubits, const QVec &target_qubits)
{
	if (control_qubits.size() == 0 || target_qubits.size() == 0)
	{
		QCERR("qubit_vector err");
		throw invalid_argument("qubit_vector err");
	}

	string name = "CZ";
	QCircuit cir = QCircuit();
	if (control_qubits.size() == target_qubits.size())
	{
		for (int i = 0; i < control_qubits.size(); ++i)
		{
			if (control_qubits[i] != target_qubits[i])
			{
				cir << _gs_pGateNodeFactory->getGateNode(name, { control_qubits[i], target_qubits[i] });
			}
			else
			{
				QCERR("double_gate qubit err");
				throw invalid_argument("double_gate qubit");
			}
		}
	}
	else
	{
		QCERR("qubit_vector size err");
		throw invalid_argument("qubit_vector size");
	}

	return cir;
}

QGate QPanda::CZ(int control_qaddr, int target_qaddr)
{
	return CZ(get_qubit_by_phyaddr(control_qaddr), get_qubit_by_phyaddr(target_qaddr));
}

QCircuit QPanda::CZ(const std::vector<int>& control_qaddrs, const std::vector<int>& target_qaddrs)
{
	if (control_qaddrs.size() == 0 || target_qaddrs.size() == 0)
	{
		QCERR("qubit_vector err");
		throw invalid_argument("qubit_vector err");
	}

	QCircuit cir = QCircuit();
	if (control_qaddrs.size() == target_qaddrs.size())
	{
		for (int i = 0; i < control_qaddrs.size(); ++i)
		{
			if (control_qaddrs[i] != target_qaddrs[i])
			{
				cir << CZ(get_qubit_by_phyaddr(control_qaddrs[i]), get_qubit_by_phyaddr(target_qaddrs[i]));
			}
			else
			{
				QCERR("double_gate qubit err");
				throw invalid_argument("double_gate qubit");
			}
		}
	}
	else
	{
		QCERR("qubit_vector size err");
		throw invalid_argument("qubit_vector size");
	}

	return cir;
}

QGate QPanda::CP(Qubit* control_qubit, Qubit* target_qubit, double theta)
{
	string name = "CP";
	return _gs_pGateNodeFactory->getGateNode(name, { control_qubit, target_qubit }, theta);
}

QCircuit QPanda::CP(const QVec& control_qubits, const QVec &target_qubits, double theta)
{
	if (control_qubits.size() == 0 || target_qubits.size() == 0)
	{
		QCERR("qubit_vector err");
		throw invalid_argument("qubit_vector err");
	}

	string name = "CP";
	QCircuit cir = QCircuit();
	if (control_qubits.size() == target_qubits.size())
	{
		for (int i = 0; i < control_qubits.size(); ++i)
		{
			if (control_qubits[i] != target_qubits[i])
			{
				cir << _gs_pGateNodeFactory->getGateNode(name, { control_qubits[i], target_qubits[i] },theta);
			}
			else
			{
				QCERR("double_gate qubit err");
				throw invalid_argument("double_gate qubit");
			}
		}
	}
	else
	{
		QCERR("qubit_vector size err");
		throw invalid_argument("qubit_vector size");
	}

	return cir;
}

QGate QPanda::CP(int control_qaddr, int target_qaddr, double theta)
{
	return CP(get_qubit_by_phyaddr(control_qaddr), get_qubit_by_phyaddr(target_qaddr), theta);
}

QCircuit QPanda::CP(const std::vector<int>& control_qaddrs, const std::vector<int>& target_qaddrs, double theta)
{
	if (control_qaddrs.size() == 0 || target_qaddrs.size() == 0)
	{
		QCERR("qubit_vector err");
		throw invalid_argument("qubit_vector err");
	}

	QCircuit cir = QCircuit();
	if (control_qaddrs.size() == target_qaddrs.size())
	{
		for (int i = 0; i < control_qaddrs.size(); ++i)
		{
			if (control_qaddrs[i] != target_qaddrs[i])
			{
				cir << CP(get_qubit_by_phyaddr(control_qaddrs[i]), get_qubit_by_phyaddr(target_qaddrs[i]), theta);
			}
			else
			{
				QCERR("double_gate qubit err");
				throw invalid_argument("double_gate qubit");
			}
		}
	}
	else
	{
		QCERR("qubit_vector size err");
		throw invalid_argument("qubit_vector size");
	}

	return cir;
}


QGate QPanda::U4(double alpha, double beta, double gamma, double delta, Qubit* qubit)
{
	string name = "U4";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit }, alpha, beta, gamma, delta);
}

QGate QPanda::U4(QStat& matrix, Qubit* qubit)
{
	string name = "U4";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit }, matrix);
}

/* new interface */
QGate QPanda::U4(Qubit* qubit, QStat& matrix)
{
	string name = "U4";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit }, matrix);
}

QCircuit QPanda::U4(const QVec &qubits, QStat& martix)
{
	string name = "U4";
	QCircuit cir = QCircuit();
	for (auto &qubit : qubits)
	{
		cir << _gs_pGateNodeFactory->getGateNode(name, { qubit });
	}

	return cir;
}

QGate QPanda::U4(Qubit* qubit, double alpha, double beta, double gamma, double delta)
{
	string name = "U4";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit }, alpha, beta, gamma, delta);
}

QCircuit QPanda::U4(const QVec& qubits, double alpha, double beta, double gamma, double delta)
{
	string name = "U4";
	QCircuit cir = QCircuit();
	for (auto &qubit : qubits)
	{
		cir << _gs_pGateNodeFactory->getGateNode(name, { qubit });
	}

	return cir;
}

/** Construct QGate by Qubit physics addr */

QGate QPanda::CU(double alpha,
	double beta,
	double gamma,
	double delta,
	Qubit* control_qubit,
	Qubit* target_qubit)
{
	string name = "CU";
	return _gs_pGateNodeFactory->getGateNode(name, { control_qubit, target_qubit }, alpha, beta, gamma, delta);
}

QCircuit QPanda::CU(double alpha, 
	double beta, 
	double gamma, 
	double delta, 
	const QVec& control_qubits,
	const QVec& target_qubits)
{
	if (control_qubits.size() == 0 || target_qubits.size() == 0)
	{
		QCERR("qubit_vector err");
		throw invalid_argument("qubit_vector err");
	}

	string name = "CU";
	QCircuit cir = QCircuit();
	if (control_qubits.size() == target_qubits.size())
	{
		for (int i = 0; i < control_qubits.size(); ++i)
		{
			if (control_qubits[i] != target_qubits[i])
			{
				cir << _gs_pGateNodeFactory->getGateNode(name, { control_qubits[i], target_qubits[i] }, alpha, beta, gamma, delta);
			}
			else
			{
				QCERR("double_gate qubit err");
				throw invalid_argument("double_gate qubit");
			}
		}
	}
	else
	{
		QCERR("qubit_vector size err");
		throw invalid_argument("qubit_vector size");
	}

	return cir;
}

QGate QPanda::CU(QStat& matrix, Qubit* control_qubit, Qubit* target_qubit)
{
	string name = "CU";
	return _gs_pGateNodeFactory->getGateNode(name, { control_qubit, target_qubit }, matrix);
}

QCircuit QPanda::CU(QStat& matrix, const QVec& control_qubits, const QVec& target_qubits)
{
	if (control_qubits.size() == 0 || target_qubits.size() == 0)
	{
		QCERR("qubit_vector err");
		throw invalid_argument("qubit_vector err");
	}
	string name = "CU";
	QCircuit cir = QCircuit();
	if (control_qubits.size() == target_qubits.size())
	{
		for (int i = 0; i < control_qubits.size(); ++i)
		{
			if (control_qubits[i] != target_qubits[i])
			{
				cir << _gs_pGateNodeFactory->getGateNode(name, { control_qubits[i], target_qubits[i] }, matrix);
			}
			else
			{
				QCERR("double_gate qubit err");
				throw invalid_argument("double_gate qubit");
			}
		}
	}
	else
	{
		QCERR("qubit_vector size err");
		throw invalid_argument("qubit_vector size");
	}

	return cir;
}

QGate QPanda::QDouble(QStat& matrix, Qubit* qubit1, Qubit* qubit2)
{
	string name = ":QDoubleGate";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit1, qubit2 }, matrix);
}

QGate QPanda::Toffoli(Qubit * control_fisrt, Qubit * control_second, Qubit * target)
{
    auto gate = X(target);
    gate.setControl({ control_fisrt, control_second });
    return gate;
}

QGate QPanda::oracle(QVec qubits, std::string oracle_name)
{
	string name = "OracularGate";
	return _gs_pGateNodeFactory->getGateNode(name, qubits, oracle_name);
}

QGate QPanda::QDouble(Qubit* qubit1, Qubit* qubit2, QStat& matrix)
{
	string name = "QDoubleGate";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit1, qubit2 }, matrix);
}

QCircuit QPanda::QDouble(const QVec& qubit1, const QVec& qubit2, QStat& matrix)
{
	if (qubit1.size() == 0 || qubit2.size() == 0)
	{
		QCERR("qubit_vector err");
		throw invalid_argument("qubit_vector err");
	}

	string name = "QDoubleGate";
	QCircuit cir = QCircuit();
	if (qubit1.size() == qubit2.size())
	{
		for (int i = 0; i < qubit1.size(); ++i)
		{
			if (qubit1[i] != qubit2[i])
			{
				cir << _gs_pGateNodeFactory->getGateNode(name, { qubit1[i] }, qubit2[i], matrix);
			}
			else
			{
				QCERR("double_gate qubit err");
				throw invalid_argument("double_gate qubit");
			}
		}
	}
	else
	{
		QCERR("qubit_vector size err");
		throw invalid_argument("qubit_vector size");
	}

	return cir;
}

QGate QPanda::QDouble(int control_qaddr, int target_qaddr, QStat& matrix)
{
	return QDouble(get_qubit_by_phyaddr(control_qaddr), get_qubit_by_phyaddr(target_qaddr), matrix);
}

QCircuit QPanda::QDouble(const std::vector<int>& control_qaddrs, const std::vector<int>& target_qaddrs, QStat& matrix)
{
	if (control_qaddrs.size() == 0 || target_qaddrs.size() == 0)
	{
		QCERR("qubit_vector err");
		throw invalid_argument("qubit_vector err");
	}

	QCircuit cir = QCircuit();
	if (control_qaddrs.size() == target_qaddrs.size())
	{
		for (int i = 0; i < control_qaddrs.size(); ++i)
		{
			if (control_qaddrs[i] != target_qaddrs[i])
			{
				cir << QDouble(get_qubit_by_phyaddr(control_qaddrs[i]), get_qubit_by_phyaddr(target_qaddrs[i]), matrix);
			}
			else
			{
				QCERR("double_gate qubit err");
				throw invalid_argument("double_gate qubit");
			}
		}
	}
	else
	{
		QCERR("qubit_vector size err");
		throw invalid_argument("qubit_vector size");
	}

	return cir;
}


QGate QPanda::CU(Qubit* control_qubit,
	Qubit* target_qubit,
	double alpha, double beta,
	double gamma, double delta)
{
	string name = "CU";
	return _gs_pGateNodeFactory->getGateNode(name, { control_qubit, target_qubit }, alpha, beta, gamma, delta);
}

QCircuit QPanda::CU(const QVec& control_qubits,
	const QVec& target_qubits, 
	double alpha, double beta, 
	double gamma, double delta)
{
	if (control_qubits.size() == 0 || target_qubits.size() == 0)
	{
		QCERR("qubit_vector err");
		throw invalid_argument("qubit_vector err");
	}

	string name = "CU";
	QCircuit cir = QCircuit();
	if (control_qubits.size() == target_qubits.size())
	{
		for (int i = 0; i < control_qubits.size(); i++)
		{
			if (control_qubits[i] != target_qubits[i])
			{
				cir << _gs_pGateNodeFactory->getGateNode(name, { control_qubits[i], target_qubits[i] }, alpha, beta, gamma, delta);
			}
			else
			{
				QCERR("double_gate qubit err");
				throw invalid_argument("double_gate qubit");
			}
		}
	}
	else
	{
		QCERR("qubit_vector size err");
		throw invalid_argument("qubit_vector size");
	}

	return cir;
}

QGate QPanda::CU(Qubit* control_qubit, Qubit* target_qubit, QStat& matrix)
{
	string name = "CU";
	return _gs_pGateNodeFactory->getGateNode(name, { control_qubit, target_qubit }, matrix);
}

QCircuit QPanda::CU(const QVec& control_qubits, const QVec& target_qubits, QStat& matrix)
{
	if (control_qubits.size() == 0 || target_qubits.size() == 0)
	{
		QCERR("qubit_vector err");
		throw invalid_argument("qubit_vector err");
	}

	string name = "CU";
	QCircuit cir = QCircuit();
	if (control_qubits.size() == target_qubits.size())
	{
		for (int i = 0; i < control_qubits.size(); i++)
		{
			if (control_qubits[i] != target_qubits[i])
			{
				cir << _gs_pGateNodeFactory->getGateNode(name, { control_qubits[i], target_qubits[i] }, matrix);
			}
			else
			{
				QCERR("double_gate qubit err");
				throw invalid_argument("double_gate qubit");
			}
		}
	}
	else
	{
		QCERR("qubit_vector size err");
		throw invalid_argument("qubit_vector size");
	}

	return cir;
}

QGate QPanda::CU(int control_qaddr, int target_qaddr, double alpha, double beta, double gamma, double delta)
{
	return CU(get_qubit_by_phyaddr(control_qaddr), get_qubit_by_phyaddr(target_qaddr), alpha, beta, gamma, delta);
}

QCircuit QPanda::CU(const std::vector<int>& control_qaddrs, const std::vector<int>& target_qaddrs, double alpha, double beta, double gamma, double delta)
{
	if (control_qaddrs.size() == 0 || target_qaddrs.size() == 0)
	{
		QCERR("qubit_vector err");
		throw invalid_argument("qubit_vector err");
	}

	QCircuit cir = QCircuit();
	if (control_qaddrs.size() == target_qaddrs.size())
	{
		for (int i = 0; i < control_qaddrs.size(); ++i)
		{
			if (control_qaddrs[i] != target_qaddrs[i])
			{
				cir << CU(get_qubit_by_phyaddr(control_qaddrs[i]), get_qubit_by_phyaddr(target_qaddrs[i]), alpha, beta, gamma, delta);
			}
			else
			{
				QCERR("double_gate qubit err");
				throw invalid_argument("double_gate qubit");
			}
		}
	}
	else
	{
		QCERR("qubit_vector size err");
		throw invalid_argument("qubit_vector size");
	}

	return cir;
}

QGate QPanda::CU(int control_qaddr, int target_qaddr, QStat& matrix)
{
	return CU(get_qubit_by_phyaddr(control_qaddr), get_qubit_by_phyaddr(target_qaddr), matrix);
}

QCircuit QPanda::CU(const std::vector<int>& control_qaddrs, const std::vector<int>& target_qaddrs, QStat& matrix)
{
	if (control_qaddrs.size() == 0 || target_qaddrs.size() == 0)
	{
		QCERR("qubit_vector err");
		throw invalid_argument("qubit_vector err");
	}
	QCircuit cir = QCircuit();
	if (control_qaddrs.size() == target_qaddrs.size())
	{
		for (int i = 0; i < control_qaddrs.size(); ++i)
		{
			if (control_qaddrs[i] != target_qaddrs[i])
			{
				cir << CU(get_qubit_by_phyaddr(control_qaddrs[i]), get_qubit_by_phyaddr(target_qaddrs[i]), matrix);
			}
			else
			{
				QCERR("double_gate qubit err");
				throw invalid_argument("double_gate qubit");
			}
		}
	}
	else
	{
		QCERR("qubit_vector size err");
		throw invalid_argument("qubit_vector size");
	}

	return cir;
}

QGate QPanda::U4(int qaddr, double alpha, double beta, double gamma, double delta)
{
	return U4(get_qubit_by_phyaddr(qaddr), alpha, beta, gamma, delta);
}

QCircuit QPanda::U4(const std::vector<int>& qaddrs, double alpha, double beta, double gamma, double delta)
{
	QCircuit cir = QCircuit();
	for (auto &qaddr : qaddrs)
	{
		cir << U4(get_qubit_by_phyaddr(qaddr), alpha, beta, gamma, delta);
	}
	
	return cir;
}

QGate QPanda::U4(int qaddr, QStat& matrix)
{
	return U4(get_qubit_by_phyaddr(qaddr), matrix);
}

QCircuit QPanda::U4(const std::vector<int>& qaddrs, QStat& martix)
{
	QCircuit cir = QCircuit();
	for (auto &qaddr : qaddrs)
	{
		cir << U4(get_qubit_by_phyaddr(qaddr), martix);
	}

	return cir;
}


QGate QPanda::Toffoli(int qaddr0, int qaddr1, int target_qaddr)
{
    auto qpool = OriginQubitPool::get_instance();
    auto gate = X(qpool->get_qubit_by_addr(target_qaddr));
    gate.setControl({ qpool->get_qubit_by_addr(qaddr0), qpool->get_qubit_by_addr(qaddr1) });
    return gate;
}


QGate QPanda::QOracle(const QVec &qubits, const QStat &matrix)
{
	if (!is_unitary_matrix_by_eigen(matrix))
	{
		QCERR_AND_THROW_ERRSTR(invalid_argument, "Non-unitary matrix for QOracle-gate.");
	}

    auto value = matrix.size();
    for (size_t i = 0; i < qubits.size(); i++)
    {
        value >>= 2;
    }
    QPANDA_ASSERT(1 != value, "Error: QOracle matrix size");


    string name = "OracularGate";
    return _gs_pGateNodeFactory->getGateNode(name, qubits, matrix);
}


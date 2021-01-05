#include "QGate.h"
#include "Core/Utilities/QProgInfo/ConfigMap.h"
#include "Core/Utilities/Tools/QPandaException.h"
#include <type_traits>
using namespace QGATE_SPACE;
using namespace std;
USING_QPANDA
static QGateNodeFactory * _gs_pGateNodeFactory = QGateNodeFactory::getInstance();
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


QGate::QGate(const QGate & old_Gate)
{
    m_qgate_node = old_Gate.m_qgate_node;
}


QGate::QGate(QVec& qs, QuantumGate *QGate)
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


QuantumGate * QGate::getQGate() const
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

OriginQGate::OriginQGate(QVec &qubit_vector, QuantumGate *qgate) :m_Is_dagger(false)
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

Qubit * OriginQGate::popBackQuBit()
{
    auto temp = m_qubit_vector.back();
    m_qubit_vector.pop_back();
    return temp;
}

QuantumGate * OriginQGate::getQGate() const
{
    if (nullptr == m_qgate)
    {
        QCERR("m_qgate is null");
        throw runtime_error("m_qgate is null");
    }
    return m_qgate;
}

void OriginQGate::setQGate(QuantumGate * qgate)
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

void OriginQGate::PushBackQuBit(Qubit * qubit)
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

void QGateParseSingleBit(QuantumGate * qgate, 
    QVec & qubit_vector, 
    QPUImpl* qgates,
    bool is_dagger,
    QVec & control_qubit_vector,
    GateType type)
{
    if (nullptr == qgate)
    {
        QCERR("param error");
        throw invalid_argument("param error");
    }

    QStat matrix;
    qgate->getMatrix(matrix);
    Qubit * qubit = *(qubit_vector.begin());
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

void QGateParseDoubleBit(QuantumGate * qgate, 
    QVec & qubit_vector,
    QPUImpl* qgates,
    bool is_dagger,
    QVec & control_qubit_vector,
    GateType type)
{
    QStat matrix;
    qgate->getMatrix(matrix);
    auto aiter = qubit_vector.begin();
    Qubit * qubit = *aiter;
    aiter++;
    Qubit * qubit2 = *aiter;
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

QGate QPanda::I(Qubit* qubit)
{
	string name = "I";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit });
}

QGate QPanda::X(Qubit * qubit)
{
    string name = "X";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit });
}

QGate QPanda::X1(Qubit * qubit)
{
    string name = "X1";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit });
}

QGate QPanda::RX(Qubit * qubit, double angle)
{
    string name = "RX";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit }, angle);
}

QGate QPanda::U1(Qubit * qubit, double angle)
{
    string name = "U1";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit }, angle);
}

QGate QPanda::U2(Qubit * qubit, double phi, double lambda)
{
    string name = "U2";
    return _gs_pGateNodeFactory->getGateNode(name, { qubit }, phi, lambda);
}

QGate QPanda::U3(Qubit * qubit, double theta, double phi, double lambda)
{
    string name = "U3";
    return _gs_pGateNodeFactory->getGateNode(name, { qubit }, theta, phi, lambda);
}

QGate QPanda::Y(Qubit * qubit)
{
    string name = "Y";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit });
}

QGate QPanda::Y1(Qubit * qubit)
{
    string name = "Y1";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit });
}

QGate QPanda::RY(Qubit * qubit, double angle)
{
    string name = "RY";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit }, angle);
}
QGate QPanda::Z(Qubit * qubit)
{
    string name = "Z";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit });
}
QGate QPanda::Z1(Qubit * qubit)
{
    string name = "Z1";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit });
}

QGate QPanda::RZ(Qubit * qubit, double angle)
{
    string name = "RZ";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit }, angle);
}

QGate QPanda::RPhi(Qubit * qubit, double angle, double phi)
{
	string name = "RPhi";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit }, angle, phi);
}

QGate QPanda::iSWAP(Qubit * targitBit_fisrt, Qubit * targitBit_second)
{
    string name = "ISWAP";
	return _gs_pGateNodeFactory->getGateNode(name, { targitBit_fisrt, targitBit_second });
}

QGate QPanda::iSWAP(Qubit * targitBit_fisrt, Qubit * targitBit_second, double theta)
{
    string name = "ISWAPTheta";
    return _gs_pGateNodeFactory->getGateNode(name,
		{ targitBit_fisrt,targitBit_second },
        theta);
}

QGate QPanda::CR(Qubit * control_qubit, Qubit * targit_qubit, double theta)
{
    string name = "CPHASE";
	return _gs_pGateNodeFactory->getGateNode(name, { control_qubit, targit_qubit }, theta);
}

QGate QPanda::SqiSWAP(Qubit * targitBit_fisrt, Qubit * targitBit_second)
{
    string name = "SQISWAP";
    return _gs_pGateNodeFactory->getGateNode(name,
		{ targitBit_fisrt,targitBit_second });
}

QGate QPanda::SWAP(Qubit * targitBit_fisrt, Qubit * targitBit_second)
{
    string name = "SWAP";
    return _gs_pGateNodeFactory->getGateNode(name,
		{ targitBit_fisrt,targitBit_second });
}

QGate QPanda::S(Qubit * qubit)
{
    string name = "S";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit });
}

QGate QPanda::T(Qubit * qubit)
{
    string name = "T";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit });
}

QGate  QPanda::H(Qubit * qubit)
{
    string name = "H";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit });
}

QGate  QPanda::ECHO(Qubit * qubit)
{
	string name = "ECHO";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit });
}

QGate  QPanda::BARRIER(Qubit * qubit)
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

QGate  QPanda::CNOT(Qubit * control_qubit, Qubit * target_qubit)
{
    string name = "CNOT";
	return _gs_pGateNodeFactory->getGateNode(name, { control_qubit, target_qubit });
}

QGate QPanda::CZ(Qubit * control_qubit, Qubit *target_qubit)
{
    string name = "CZ";
	return _gs_pGateNodeFactory->getGateNode(name, { control_qubit, target_qubit });
}

QGate QPanda::U4(double alpha, double beta, double gamma, double delta, Qubit * qubit)
{
	string name = "U4";
    return _gs_pGateNodeFactory->getGateNode(name, {qubit}, alpha, beta, gamma, delta);
}

QGate QPanda::U4(QStat & matrix, Qubit *qubit)
{
    string name = "U4";
	return _gs_pGateNodeFactory->getGateNode(name, { qubit }, matrix);
}

QGate QPanda::CU(double alpha, 
    double beta,
    double gamma,
    double delta,
    Qubit * control_qubit, 
    Qubit * target_qubit)
{
	string name = "CU";
	return _gs_pGateNodeFactory->getGateNode(name, { control_qubit, target_qubit }, alpha, beta, gamma, delta);
}

QGate QPanda::CU(QStat & matrix, Qubit * control_qubit, Qubit * target_qubit)
{
    string name = "CU";
	return _gs_pGateNodeFactory->getGateNode(name, { control_qubit, target_qubit }, matrix);
}

QGate QPanda::QDouble(QStat& matrix, Qubit * qubit1, Qubit * qubit2)
{
    string name = ":QDoubleGate";
	return _gs_pGateNodeFactory->getGateNode(name,{ qubit1, qubit2 }, matrix);
}

QGate QPanda::oracle(QVec qubits, std::string oracle_name)
{
	string name = "OracularGate";
	return _gs_pGateNodeFactory->getGateNode(name, qubits, oracle_name);
}


/* new interface */
QGate QPanda::U4(Qubit *qubit, QStat & matrix)
{
    string name = "U4";
    return _gs_pGateNodeFactory->getGateNode(name, { qubit }, matrix);
}

QGate QPanda::U4(Qubit * qubit, double alpha, double beta, double gamma, double delta)
{
    string name = "U4";
    return _gs_pGateNodeFactory->getGateNode(name, {qubit}, alpha, beta, gamma, delta);
}

QGate QPanda::QDouble(Qubit * qubit1, Qubit * qubit2, QStat& matrix)
{
    string name = ":QDoubleGate";
    return _gs_pGateNodeFactory->getGateNode(name,{ qubit1, qubit2 }, matrix);
}

QGate QPanda::CU(Qubit * control_qubit,
                 Qubit * target_qubit,
                 double alpha,double beta,
                 double gamma, double delta)
{
    string name = "CU";
    return _gs_pGateNodeFactory->getGateNode(name, { control_qubit, target_qubit }, alpha, beta, gamma, delta);
}

QGate QPanda::CU(Qubit * control_qubit, Qubit * target_qubit, QStat & matrix)
{
    string name = "CU";
    return _gs_pGateNodeFactory->getGateNode(name, { control_qubit, target_qubit }, matrix);
}

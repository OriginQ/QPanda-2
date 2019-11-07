/*! \file QGate.h */
#ifndef _QGATE_H
#define _QGATE_H

#include <complex>
#include <vector>
#include <iterator>
#include <map>

#include "Core/QuantumCircuit/QNode.h"
#include "Core/QuantumCircuit/QuantumGate.h"
#include "Core/QuantumMachine/QubitFactory.h"
#include "Core/QuantumMachine/QVec.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"
#include "Core/Utilities/Tools/TranformQGateTypeStringAndEnum.h"

QPANDA_BEGIN
/**
* @namespace QPanda
* @namespace QGATE_SPACE
*/
using QGATE_SPACE::QuantumGate;
using QGATE_SPACE::QGateFactory;

/**
* @class  AbstractQGateNode
* @brief   Quantum gate basic abstract class
* @ingroup Core
*/
class AbstractQGateNode
{
public:
    virtual size_t getQuBitVector(QVec &) const = 0;
    virtual Qubit * popBackQuBit() = 0;
    virtual void PushBackQuBit(Qubit *) = 0;
    virtual size_t getTargetQubitNum() const = 0;
    virtual size_t getControlQubitNum() const = 0;
    virtual QuantumGate * getQGate() const = 0;
    virtual void setQGate(QuantumGate *) = 0;
    virtual bool isDagger() const = 0;
    virtual size_t getControlVector(QVec &) const = 0;
    virtual bool setDagger(bool) = 0;
    virtual bool setControl(QVec) = 0;
    virtual ~AbstractQGateNode() {}
};


class QGateNodeFactory;



/**
* @class QGate
* @brief    QPanda2 quantum gate  basic classs
* @ingroup  Core
*/
class QGate : public AbstractQGateNode
{
private:
    std::shared_ptr<AbstractQGateNode>  m_qgate_node;

public:
    ~QGate();
    QGate(const QGate&);
	QGate(QVec &, QuantumGate*);
    QGate(std::shared_ptr<AbstractQGateNode> node);

    /**
    * @brief  Get current node type
    * @return     NodeType  current node type
    * @see  NodeType
    */
    NodeType getNodeType() const;


    /**
    * @brief  Get qubit vector inside this quantum gate
    * @param[in]  QVec&  qubit vector
    * @return     size_t  
    * @see   GateType
    */
    size_t getQuBitVector(QVec &) const;

    /**
    * @brief  Get qubit num inside this quantum gate
    * @return     size_t  qubit num
    */
    size_t getTargetQubitNum() const;

    size_t getControlQubitNum() const;

    QuantumGate *getQGate() const;

    /**
    * @brief  Set dagger to current quantum gate
    * @param[in]  bool is dagger
    * @retval   0  set failed
    * @retval   1  set success
    */
    bool setDagger(bool);

    /**
    * @brief  Set control qubits to current quantum gate
    * @param[in]  QVec  control qubits  vector
    * @retval   0  set failed
    * @retval   1  set success
    * @see QVec
    */
    bool setControl(QVec);
    std::shared_ptr<AbstractQGateNode> getImplementationPtr();


    /**
    * @brief  Get a dagger quantumgate  base on current quantum gate node
    * @return     QPanda::QGate  quantum gate
    */
    QGate dagger();
    /**
    * @brief  Get a control quantumgate  base on current quantum gate node
    * @param[in]  QVec control qubits  vector
    * @return     QPanda::QGate  quantum gate
    * @see QVec
    */
    QGate control(QVec);


    /**
    * @brief  Judge current quantum gate is dagger
    * @retval   0  true
    * @retval   1  false
    */
    bool isDagger() const;

    /**
    * @brief  Get control vector fron current quantum gate node
    * @param[in]  QVec& qubits  vector
    * @return     size_t  
    * @see QVec
    */
    size_t getControlVector(QVec &) const;
private:
    Qubit * popBackQuBit() { return nullptr; };
    void setQGate(QuantumGate *) {};
    void PushBackQuBit(Qubit *) {};
};

class OriginQGate : public QNode, public AbstractQGateNode
{
private:
    QVec m_qubit_vector;
    QuantumGate *m_qgate;
    NodeType m_node_type;
    bool m_Is_dagger;
    std::vector<Qubit*> m_control_qubit_vector;
public:
    ~OriginQGate();
    OriginQGate(QVec &, QuantumGate *);
    NodeType getNodeType() const;
    size_t getQuBitVector(QVec &) const;
    size_t getTargetQubitNum() const;
    size_t getControlQubitNum() const;
    Qubit *popBackQuBit();
    QuantumGate *getQGate() const;
    void setQGate(QuantumGate *);
    bool setDagger(bool);
    bool setControl(QVec);
    bool isDagger() const;
    size_t getControlVector(QVec &) const;
    void PushBackQuBit(Qubit *);
};

class QGateNodeFactory
{
public:
    static QGateNodeFactory * getInstance()
    {
        static QGateNodeFactory s_gateNodeFactory;
        return &s_gateNodeFactory;
    }
	template<typename ...Targs>
	QGate getGateNode(const std::string & name, QVec qs, Targs&&... args)
	{
		QuantumGate * pGate = QGATE_SPACE::create_quantum_gate(name, std::forward<Targs>(args)...);
		try
		{
			QGate  QGateNode(qs, pGate);
			return QGateNode;
		}
		catch (const std::exception& e)
		{
			QCERR(e.what());
			throw std::runtime_error(e.what());
		}
	}

private:
};

typedef void(*QGATE_FUN)(QuantumGate *,
    QVec &,
    QPUImpl*,
    bool,
    QVec &,
    GateType);
typedef std::map<int, QGATE_FUN> QGATE_FUN_MAP;

class QGateParseMap
{

    static QGATE_FUN_MAP m_qgate_function_map;
public:

    static void insertMap(int opNum, QGATE_FUN function)
    {
        m_qgate_function_map.insert(std::pair<int, QGATE_FUN>(opNum, function));
    }

    static QGATE_FUN getFunction(int iOpNum)
    {
        auto aiter = m_qgate_function_map.find(iOpNum);
        if (aiter == m_qgate_function_map.end())
        {
            return nullptr;
        }

        return aiter->second;
    }


};

/**
* @brief  Construct a new I gate
* @param[in]  Qubit* qubit target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup Core
*/
QGate I(Qubit* qubit);

/**
* @brief  Construct a new quantum X gate
* @param[in]  Qubit* qubit target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup Core
*/
QGate X(Qubit* qubit);
/**
* @brief  Construct a new quantum X1 gate
* @param[in]  Qubit* qubit target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup Core
*/
QGate X1(Qubit* qubit);
/**
* @brief  Construct a new quantum RX gate
* @param[in]  Qubit* target qubit
* @param[in]  double angle
* @return     QPanda::QGate  quantum gate
* @ingroup Core
*/
QGate RX(Qubit*, double angle);
/**
* @brief  Construct a new quantum U1 gate
* @param[in]  Qubit* target qubit
* @param[in]  double angle
* @return     QPanda::QGate  quantum gate
* @ingroup Core
*/
QGate U1(Qubit*, double angle);

/**
* @brief  Construct a new quantum U1 gate
* @param[in]  Qubit* target qubit
* @param[in]  double phi
* @param[in]  double lambda
* @return     QPanda::QGate  quantum gate
* @ingroup Core
*/
QGate U2(Qubit * qubit, double phi, double lambda);

/**
* @brief  Construct a new quantum U1 gate
* @param[in]  Qubit* target qubit
* @param[in]  double theta
* @param[in]  double phi
* @param[in]  double lambda
* @return     QPanda::QGate  quantum gate
* @ingroup Core
*/
QGate U3(Qubit * qubit, double theta, double phi, double lambda);

/**
* @brief  Construct a new quantum Y gate
* @param[in]  Qubit* qubit target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup Core
*/

QGate Y(Qubit* qubit);
/**
* @brief  Construct a new quantum Y1 gate
* @param[in]  Qubit* qubit target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup Core
*/
QGate Y1(Qubit* qubit);
/**
* @brief  Construct a new quantum RY gate
* @param[in]  Qubit* target qubit
* @param[in]  double angle target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup Core
*/
QGate RY(Qubit*, double angle);
/**
* @brief  Construct a new quantum Z gate
* @param[in]  Qubit* qubit target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup Core
*/
QGate Z(Qubit* qubit);
/**
* @brief  Construct a new quantum Z1 gate
* @param[in]  Qubit* qubit target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup Core
*/
QGate Z1(Qubit* qubit);
/**
* @brief  Construct a new quantum RZ gate
* @param[in]  Qubit* target qubit
* @param[in]  double angle
* @return     QPanda::QGate  quantum gate
* @ingroup Core
*/
QGate RZ(Qubit*, double angle);
/**
* @brief  Construct a new quantum S gate
* @param[in]  Qubit* qubit target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup Core
*/
QGate S(Qubit* qubit);
/**
* @brief  Construct a new quantum T gate
* @param[in]  Qubit* target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup Core
*/
QGate T(Qubit*);
/**
* @brief  Construct a new quantum H gate
* @param[in]  Qubit* qubit target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup Core
*/
QGate H(Qubit* qubit);

/**
* @brief  Construct a new quantum CNOT gate
* @param[in]  Qubit* control qubit
* @param[in]  Qubit* target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup Core
*/
QGate CNOT(Qubit* control_qubit, Qubit* target_qubit);
/**
* @brief  Construct a new quantum CZ gate
* @param[in]  Qubit* control qubit
* @param[in]  Qubit* target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup Core
*/
QGate CZ(Qubit*  control_qubit, Qubit* target_qubit);
/**
* @brief  Construct a new quantum U4 gate
* @param[in]  double alpha
* @param[in]  double beta
* @param[in]  double gamma
* @param[in]  double delta
* @param[in]  Qubit* target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup Core
*/
QGate U4(double alpha, double beta, double gamma, double delta, Qubit*);
/**
* @brief  Construct a new quantum U4 gate
* @param[in]  QStat& matrix
* @param[in]  Qubit* target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup Core
*/
QGate U4(QStat& matrix, Qubit*);
/**
* @brief  Construct a new quantum QDouble gate
* @param[in]  QStat matrix
* @param[in]  Qubit* control qubit
* @param[in]  Qubit* target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup Core
*/
QGate QDouble(QStat& matrix, Qubit * qubit1, Qubit * qubit2);
/**
* @brief  Construct a new quantum CU gate
* @param[in]  double alpha
* @param[in]  double beta
* @param[in]  double gamma
* @param[in]  double delta
* @param[in]  Qubit*   control qubit
* @param[in]  Qubit*   target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup Core
*/
QGate CU(double alpha, double beta, double gamma, double delta, Qubit *, Qubit *);
/**
* @brief  Construct a new quantum CU gate
* @param[in]  QStat & matrix
* @param[in]  Qubit*  target qubit
* @param[in]  Qubit*  control qubit
* @return     QPanda::QGate  quantum gate
* @ingroup Core
*/
QGate CU(QStat& matrix, Qubit*, Qubit*);
/**
* @brief  Construct a new quantum iSWAP gate
* @param[in]  Qubit* control qubit
* @param[in]  Qubit* target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup Core
*/
QGate iSWAP(Qubit * targitBit_fisrt, Qubit * targitBit_second);
/**
* @brief  Construct a new quantum iSWAP gate
* @param[in]  Qubit* control qubit
* @param[in]  Qubit* target qubit
* @param[in]  double theta
* @return     QPanda::QGate  quantum gate
* @ingroup Core
*/
QGate iSWAP(Qubit * targitBit_fisrt, Qubit * targitBit_second, double theta);
/**
* @brief  Construct a new quantum CR gate
* @param[in]  Qubit* control qubit
* @param[in]  Qubit* targit qubit
* @param[in]  double theta
* @return     QPanda::QGate  quantum gate
* @ingroup Core
*/
QGate CR(Qubit * control_qubit, Qubit * targit_qubit, double theta);
/**
* @brief  Construct a new quantum SqiSWAP gate
* @param[in]  Qubit* control qubit
* @param[in]  Qubit* target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup Core
*/
QGate SqiSWAP(Qubit * targitBit_fisrt, Qubit * targitBit_second);
/**
* @brief  Construct a new quantum SWAP gate
* @param[in]  Qubit* control qubit
* @param[in]  Qubit* target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup Core
*/
QGate SWAP(Qubit * targitBit_fisrt, Qubit * targitBit_second);

QGate oracle(QVec qubits, std::string oracle_name);

inline QGate copy_qgate(QuantumGate *  qgate_old,QVec qubit_vector)
{
    if(nullptr == qgate_old)
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }
    auto gate_type = (GateType)qgate_old->getGateType();
    auto class_name = TransformQGateType::getInstance()[gate_type];

    auto temp_gate = QGateNodeFactory::getInstance()->getGateNode(class_name,qubit_vector, std::move(qgate_old));
    return temp_gate;
}

inline QGate copy_qgate(QGate &qgate,QVec qubit_vector)
{
	return copy_qgate(qgate.getQGate(), qubit_vector);
}

inline QGate copy_qgate(QGate *qgate,QVec qubit_vector)
{
	return copy_qgate(qgate->getQGate(), qubit_vector);
}

QPANDA_END
#endif // !_QGATE_H

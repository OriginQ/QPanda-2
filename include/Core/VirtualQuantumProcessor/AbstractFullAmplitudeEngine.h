#ifndef _ABSTRACTFULLAMPLITUDEENGINE_H_
#define _ABSTRACTFULLAMPLITUDEENGINE_H_
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QuantumMeasure.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"
#include "Core/QuantumCircuit/ControlFlow.h"
#include "Core/QuantumMachine/QuantumMachineInterface.h"


/** 
* @brief Distributed full amplitude engine  abstract class
* @ingroup VirtualQuantumProcessor
*/
class AbstractDistributedFullAmplitudeEngine
{
public:
    virtual void initState(int head_rank, int rank_size, int qubit_num) = 0;
    virtual QStat getQState() = 0;
    virtual void singleQubitOperation(const int &iQn, QStat U, bool isConjugate) = 0;
    virtual void controlsingleQubitOperation(const int &iQn, Qnum& qnum, QStat U, bool isConjugate) = 0;

    virtual void doubleQubitOperation(const int &iQn1, const int &iQn2, QStat U, bool isConjugate) = 0;
    virtual void controldoubleQubitOperation(const int &iQn1, const int &iQn2, Qnum& qnum, QStat U, bool isConjugate) = 0;

    virtual int  measureQubitOperation(const int &qn) = 0;
    virtual void PMeasureQubitOperation(Qnum& qnum, prob_vec &mResult) = 0;

	virtual void reset_qubit_operation(const int &qn) = 0;

private:
    virtual void distributeOneRank_doubleQubitOperation(const int &iQn1, const int &iQn2, QStat U, bool isConjugate) = 0;
    virtual void distributeTwoRank_doubleQubitOperation(const int &iQn1, const int &iQn2, QStat U, bool isConjugate) = 0;
    virtual void distributeFourRank_doubleQubitOperation(const int &iQn1, const int &iQn2, QStat U, bool isConjugate) = 0;

    virtual qstate_type distributeAllRank_measureQubitOperation(const int &qn) = 0;
    virtual qstate_type distributeHalfRank_measureQubitOperation(const int &qn) = 0;

    virtual void distributeAllRank_handleMeasureState(const int &qn, int &result, const qstate_type &prob) = 0;
    virtual void distributeHalfRank_handleMeasureState(const int &qn, int &result, const qstate_type &prob) = 0;
};

/**
* @brief Distributed full amplitude engine 
* @ingroup VirtualQuantumProcessor
*/
class DistributedFullAmplitudeEngine :public QPUImpl
{
public:
    bool qubitMeasure(size_t qn);

    QError pMeasure(Qnum& qnum, prob_vec &mResult);

    QError initState(size_t head_rank, size_t rank_size, size_t qubit_num);

    QError unitarySingleQubitGate(size_t qn, QStat& matrix,
        bool isConjugate,
        GateType);

    QError controlunitarySingleQubitGate(size_t qn, Qnum& qnum,
        QStat& matrix,
        bool isConjugate,
        GateType);

    QError unitaryDoubleQubitGate(size_t qn_0, size_t qn_1,
        QStat& matrix,
        bool isConjugate,
        GateType);

    QError controlunitaryDoubleQubitGate(size_t qn_0,
        size_t qn_1,
        Qnum& qnum,
        QStat& matrix,
        bool isConjugate,
        GateType);

    QStat getQState();

	QError Reset(size_t qn);

private:
    AbstractDistributedFullAmplitudeEngine * _PQGates = nullptr;
};




#endif
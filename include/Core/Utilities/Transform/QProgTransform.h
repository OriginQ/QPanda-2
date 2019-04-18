#ifndef QPROG_TRANSFORM_H
#define QPROG_TRANSFORM_H

#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QuantumMeasure.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"
#include "Core/QuantumMachine/QuantumMachineInterface.h"
#include "Core/Utilities/Transform/QProgTransform.h"
#include "Core/QuantumCircuit/ControlFlow.h"
#include <map>
#include <string>


QPANDA_BEGIN

using QGATE_SPACE::angleParameter;
using QGATE_SPACE::QuantumGate;

const unsigned short kUshortMax = 65535;
const int kCountMoveBit = 16;
#define DEF_QPROG_FILENAME        "QProg.dat"

enum QProgStoredNodeType {
    QPROG_PAULI_X_GATE = 1u,
    QPROG_PAULI_Y_GATE,
    QPROG_PAULI_Z_GATE,
    QPROG_X_HALF_PI,
    QPROG_Y_HALF_PI,
    QPROG_Z_HALF_PI,
    QPROG_HADAMARD_GATE,
    QPROG_T_GATE,
    QPROG_S_GATE,
    QPROG_RX_GATE,
    QPROG_RY_GATE,
    QPROG_RZ_GATE,
    QPROG_U1_GATE,
    QPROG_U2_GATE,
    QPROG_U3_GATE,
    QPROG_U4_GATE,
    QPROG_CU_GATE,
    QPROG_CNOT_GATE,
    QPROG_CZ_GATE,
    QPROG_CPHASE_GATE,
    QPROG_ISWAP_GATE,
    QPROG_ISWAP_THETA_GATE,
    QPROG_SQISWAP_GATE,
    QPROG_GATE_ANGLE,
    QPROG_MEASURE_GATE,
    QPROG_QIF_NODE,
    QPROG_QWHILE_NODE,
    QPROG_CEXPR_CBIT,
    QPROG_CEXPR_OPERATOR,
    QPROG_CEXPR_CONSTVALUE,
    QPROG_CEXPR_EVAL
};



class QProgTransform
{
public:
    virtual void transform(QProg& prog) = 0;
    virtual std::string getInsturctions() = 0;

private:
    virtual void transformQProg(AbstractQuantumProgram *) = 0;
    virtual void transformQGate(AbstractQGateNode*) = 0;
    virtual void transformQCircuit(AbstractQuantumCircuit*) = 0;
    virtual void transformQMeasure(AbstractQuantumMeasure*) = 0;
    virtual void transformQNode(QNode*) = 0;
    virtual void transformQControlFlow(AbstractControlFlowNode *) = 0;
};

QPANDA_END

#endif // QPROG_TRANSFORM_H

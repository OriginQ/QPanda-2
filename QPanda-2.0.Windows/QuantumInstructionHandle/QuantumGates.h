/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef _QUANTUM_GATE_H
#define _QUANTUM_GATE_H
#include <iostream>
#include <stdio.h>
#include <vector>
#include <complex>
#include <map>
#include "QuantumGateParameter.h"
#include "QError.h"
using std::map;
using std::vector;
using std::complex;
using std::string;
using std::pair;
using std::size_t;

typedef complex <double> COMPLEX;
typedef vector<size_t>  Qnum;
typedef vector <complex<double>> QStat;


/*****************************************************************************************************************
QuantumGates:quantum gate  
*****************************************************************************************************************/
class QuantumGates
{

public:
     QuantumGates();
     virtual ~QuantumGates() = 0;
    
    /*************************************************************************************************************
    Name:        getQState
    Description: get quantum state
    Argin:       pQuantumProParam      quantum program param.
    Argout:      sState                state string
    return:      quantum error
    *************************************************************************************************************/
     virtual bool getQState(string & sState,QuantumGateParam *pQuantumProParam) = 0;

    /*************************************************************************************************************
    Name:        Hadamard
    Description: Hadamard gate,the matrix is:[1/sqrt(2),1/sqrt(2);1/sqrt(2),-1/sqrt(2)]
    Argin:       qn          qubit number that the Hadamard gate operates on.
                 error_rate  the errorrate of the gate
    Argout:      None
    return:      quantum error
    *************************************************************************************************************/
    virtual QError Hadamard(size_t qn, double error_rate) = 0;

    /*************************************************************************************************************
    Name:        Reset
    Description: reset bit gate
    Argin:       qn          qubit number that the Hadamard gate operates on.
    Argout:      None
    return:      quantum error
    *************************************************************************************************************/
    virtual QError Reset(size_t qn) = 0;

    /*************************************************************************************************************
    Name:        pMeasure
    Description: pMeasure gate
    Argin:       qnum        qubit bit number vector
                 mResult     reuslt vector
    Argout:      None
    return:      quantum error
    *************************************************************************************************************/
    //virtual QError pMeasure(Qnum& qnum, vector<pair<size_t, double>> &mResult) = 0;

    /*************************************************************************************************************
    Name:        Hadamard
    Description: Hadamard gate,the matrix is:[1/sqrt(2),1/sqrt(2);1/sqrt(2),-1/sqrt(2)]
    Argin:       qn                 qubit number that the Hadamard gate operates on.
                 error_rate         the errorrate of the gate
                 vControlBit        control bit vector
                 stQuantumBitNumber quantum bit number
    Argout:      None
    return:      quantum error
    *************************************************************************************************************/
    virtual QError Hadamard(size_t  qn,
                            double  error_rate, 
                            Qnum    vControlBit,
                            size_t  stQuantumBitNumber) = 0;

    /*************************************************************************************************************
    Name:        RX_GATE
    Description: RX_GATE gate,quantum state rotates ¦È by x axis.The matric is:
                 [cos(¦È/2),-i*sin(¦È/2);i*sin(¦È/2),cos(¦È/2)]
    Argin:       qn                 qubit number that the Hadamard gate operates on.
                 theta              rotation angle
                 error_rate         the errorrate of the gate
    Argout:      None
    return:      quantum error
    *************************************************************************************************************/
    virtual QError RX_GATE(size_t qn, double theta, double error_rate) = 0;

    /*************************************************************************************************************
    Name:        RX_GATE
    Description: RX_GATE dagger gate,quantum state rotates ¦È by x axis.The matric is:
                 [cos(¦È/2),-i*sin(¦È/2);i*sin(¦È/2),cos(¦È/2)]
    Argin:       qn          qubit number that the Hadamard gate operates on.
                 theta       rotation angle
                 error_rate  the errorrate of the gate
                 iDagger     is dagger
    Argout:      None
    return:      quantum error
    *************************************************************************************************************/
    virtual QError RX_GATE(size_t qn, double theta, double error_rate, int iDagger) = 0;

    /*************************************************************************************************************
    Name:        RX_GATE
    Description: controled-RX_GATE gate
    Argin:       qn                 qubit number that the Hadamard gate operates on.
                 theta              rotation angle
                 error_rate         the errorrate of the gate
                 vControlBitNumber  control bit number
                 stQuantumBitNumber quantum bit number
    Argout:      None
    return:      quantum error
    *************************************************************************************************************/
    virtual QError RX_GATE(size_t  qn,
                      double  theta,
                      double  error_rate,
                      Qnum    vControlBitNumber,
                      size_t  stQuantumBitNumber) = 0;

    /*************************************************************************************************************
    Name:        RX_GATE
    Description: controled-RX_GATE dagger gate
    Argin:       qn                 qubit number that the Hadamard gate operates on.
                 theta              rotation angle
                 error_rate         the errorrate of the gate
                 vControlBitNumber  control bit number
                 stQuantumBitNumber quantum bit number
                 iDagger            is dagger
    Argout:      None
    return:      quantum error
    *************************************************************************************************************/
    virtual QError RX_GATE(size_t  qn,
                      double  theta,
                      double  error_rate,
                      Qnum    vControlBitNumber,
                      size_t  stQuantumBitNumber,
                      int     iDagger) = 0;

    /*************************************************************************************************************
    Name:        RY_GATE
    Description: RY_GATE gate,quantum state rotates ¦È by y axis.The matric is
                 [cos(¦È/2),-sin(¦È/2);sin(¦È/2),cos(¦È/2)]
    Argin:       qn          qubit number that the Hadamard gate operates on.
                 theta       rotation angle
                 error_rate  the errorrate of the gate
    Argout:      None
    return:      quantum error
    *************************************************************************************************************/
    virtual QError RY(size_t qn, double theta, double error_rate,int iDagger) = 0;

    /*************************************************************************************************************
    Name:        RY_GATE
    Description: RY_GATE control gate
    Argin:       qn                 qubit number that the Hadamard gate operates on.
                 theta              rotation angle
                 error_rate         the errorrate of the gate
                 vControlBit        control bit vector
                 stQuantumBitNumber quantum bit number
    Argout:      None
    return:      quantum error
    *************************************************************************************************************/
    virtual QError RY(size_t   qn,
                      double   theta,
                      double   error_rate,
                      Qnum     vControlBit,
                      size_t   stQuantumBitNumber,
                      int      iDagger) = 0;

    /*************************************************************************************************************
    Name:        RZ_GATE
    Description: RZ_GATE gate,quantum state rotates ¦È by z axis.The matric is
                 [1 0;0 exp(i*¦È)]
    Argin:       qn          qubit number that the Hadamard gate operates on.
                 theta       rotation angle
                 error_rate  the errorrate of the gate
                 iDagger     is dagger
    Argout:      None
    return:      quantum error
    *************************************************************************************************************/
    virtual QError RZ(size_t   qn,
                      double   theta,
                      double   error_rate,
                      int      iDagger) = 0;

    /*************************************************************************************************************
    Name:        RZ_GATE
    Description: RZ_GATE control gate 
    Argin:       qn                 qubit number that the Hadamard gate operates on.
                 theta              rotation angle
                 error_rate         the errorrate of the gate
                 vControlBitNumber  control bit number
                 stQuantumBitNumber quantum bit number
                 iDagger            is dagger
    Argout:      None
    return:      quantum error
    *************************************************************************************************************/
    virtual QError RZ(size_t   qn,
                      double   theta,
                      double   error_rate,
                      Qnum     vControlBitNumber,
                      size_t   stQuantumBitNumber,
                      int      iDagger) = 0;

    /*************************************************************************************************************
    Name:        CNOT
    Description: CNOT gate,when control qubit is |0>,goal qubit does flip,
                 when control qubit is |1>,goal qubit flips.the matric is:
                 [1 0 0 0;0 1 0 0;0 0 0 1;0 0 1 0]
    Argin:       qn_1        control qubit number
                 qn_2        goal qubit number
                 error_rate  the errorrate of the gate
    Argout:      None
    return:      quantum error
    *************************************************************************************************************/
    virtual QError CNOT(size_t qn_1, size_t qn_2, double error_rate) = 0;

    /*************************************************************************************************************
    Name:        CNOT
    Description: CNOT control gate
    Argin:       qn_1               control qubit number
                 qn_2               goal qubit number
                 error_rate         the errorrate of the gate
                 vControlBitNumber  control bit number
                 stQuantumBitNumber quantum bit number
    Argout:      None
    return:      quantum error
    *************************************************************************************************************/
    virtual QError CNOT(size_t  qn_1,
                        size_t  qn_2,
                        double  error_rate,
                        Qnum    vControlBitNumber,
                        int     stQuantumBitNumber) = 0;

    /*************************************************************************************************************
    Name:        CR
    Description: CR gate,when control qubit is |0>,goal qubit does not rotate,
                 when control qubit is |1>,goal qubit rotate ¦È by z axis.the matric is:
                 [1 0 0 0;0 1 0 0;0 0 1 0;0 0 0 exp(i*¦È)]
    Argin:       qn_1        control qubit number
                 qn_2        goal qubit number
                 theta       roration angle
                 error_rate  the errorrate of the gate
                 iDagger     is dagger
    Argout:      None
    return:      quantum error
    *************************************************************************************************************/
    virtual QError CR(size_t qn_1, size_t qn_2, double theta, double error_rate, int iDagger) = 0;

    /*************************************************************************************************************
    Name:        iSWAP
    Description: iSWAP gate,both qubits swap and rotate ¦Ð by z-axis,the matric is:
                 [1 0 0 0;0 0 -i 0;0 -i 0 0;0 0 0 1]
    Argin:       qn_1        first qubit number
                 qn_2        second qubit number
                 error_rate  the errorrate of the gate
                 iDagger     is dagger
    Argout:      None
    return:      quantum error
    *************************************************************************************************************/
    virtual QError iSWAP(size_t  qn_1,
                         size_t  qn_2,
                         double  error_rate,
                         int     iDagger) = 0;

    /*************************************************************************************************************
    Name:        iSWAP
    Description: iSWAP gate,both qubits swap and rotate ¦Ð by z-axis,the matric is:
    [1 0 0 0;0 0 -i 0;0 -i 0 0;0 0 0 1]
    Argin:       qn_1        first qubit number
                 qn_2        second qubit number
                 error_rate  the errorrate of the gate
                 vControlBitNumber  control bit number
                 stQuantumBitNumber quantum bit number
                 iDagger     is dagger
    Argout:      None
    return:      quantum error
    *************************************************************************************************************/
    virtual QError iSWAP(size_t  qn_1,
                         size_t  qn_2,
                         double  error_rate,
                         Qnum    vControlBitNumber,
                         int     stQuantumBitNumber,
                         int     iDagger) = 0;

    /*************************************************************************************************************
    Name:        sqiSWAP
    Description: sqiSWAP gate,both qubits swap and rotate ¦Ð by z-axis,the matrix is:
                 [1 0 0 0;0 1/sqrt(2) -i/sqrt(2) 0;0 -i/sqrt(2) 1/sqrt(2) 0;0 0 0 1]
    Argin:       qn_1        first qubit number
                 qn_2        second qubit number
                 error_rate  the errorrate of the gate
                 iDagger     is dagger
    Argout:      None
    return:      quantum error
    *************************************************************************************************************/
    virtual QError sqiSWAP(size_t  qn_1,
                           size_t  qn_2,
                           double  error_rate,
                           int     iDagger) = 0;

    /*************************************************************************************************************
    Name:        sqiSWAP
    Description: sqiSWAP gate,both qubits swap and rotate ¦Ð by z-axis,the matrix is:
                 [1 0 0 0;0 1/sqrt(2) -i/sqrt(2) 0;0 -i/sqrt(2) 1/sqrt(2) 0;0 0 0 1]
    Argin:       qn_1        first qubit number
                 qn_2        second qubit number
                 error_rate  the errorrate of the gate
                 vControlBitNumber  control bit number
                 stQuantumBitNumber quantum bit number
                 iDagger     is dagger
    Argout:      None
    return:      quantum error
    *************************************************************************************************************/
    virtual QError sqiSWAP(size_t  qn_1,
                           size_t  qn_2,
                           double  error_rate,
                           Qnum    vControlBitNumber,
                           int     stQuantumBitNumber,
                           int     iDagger) = 0;

    /*************************************************************************************************************
    Name:        controlSwap
    Description: c-swap gate
    Argin:       qn_1        control qubit number
                 qn_2        1st swap qubit number
                 qn_3        2nd swap qubit number
                 error_rate  the errorrate of the gate
    Argout:      None
    return:      quantum error
    *************************************************************************************************************/
    virtual QError controlSwap(size_t qn_1, size_t qn_2, size_t qn_3, double error_rate) = 0;



    /*************************************************************************************************************
    Name:        qubitMeasure
    Description: measure qubit and the state collapsed
    Argin:       qn    qubit number of the measurement
    Argout:      None
    return:      quantum error
    *************************************************************************************************************/
    virtual int qubitMeasure(size_t qn) = 0;

    /*************************************************************************************************************
    Name:        initState
    Description: initialize the quantum state
    Argin:       stNumber  Quantum number
    Argout:      None
    return:      quantum error
    *************************************************************************************************************/
    virtual QError initState(QuantumGateParam *pQuantumProParam) = 0;

    /*************************************************************************************************************
    Name:        compareCalculationUnitType
    Description: compare calculation unit type
    Argin:       sCalculationUnitType   external calculation unit type
    Argout:      None
    return:      comparison results
    *************************************************************************************************************/
    virtual bool compareCalculationUnitType(string& sCalculationUnitType) = 0;

    /*************************************************************************************************************
    Name:    NOT
    Description: NOT gate,invert the state.The matrix is
                 [0 1;1 0]
    Argin:       qn                 qubit number that the Hadamard gate operates on.
                 error_rate         the errorrate of the gate
    Argout:      None
    return:      quantum error
    *************************************************************************************************************/
    virtual QError NOT(size_t  qn, double  error_rate) = 0;

    /*************************************************************************************************************
    Name:    NOT
    Description: NOT gate,invert the state.The matrix is
                 [0 1;1 0]
    Argin:       qn                 qubit number that the Hadamard gate operates on.
                 error_rate         the errorrate of the gate
                 vControlBit        control bit vector
                 stQuantumBitNumber quantum bit number
    Argout:      None
    return:      quantum error
    *************************************************************************************************************/
    virtual QError NOT(size_t  qn,
                       double  error_rate,
                       Qnum    vControlBit,
                       int     stQuantumBitNumber) = 0;

    /*************************************************************************************************************
    Name:        toffoli
    Description: toffoli dagger gate,the same as toffoli gate
    Argin:       stControlBit1       first control qubit
                 stControlBit2       the second control qubit
                 stQuantumBit        target qubit
                 errorRate           the errorrate of the gate
                 stQuantumBitNumber  quantum bit number
    Argout:      None
    return:      quantum error
    *************************************************************************************************************/
    virtual QError toffoli(size_t stControlBit1,
                           size_t stControlBit2,
                           size_t stQuantumBit,
                           double errorRate,
                           int    stQuantumBitNumber) = 0;

    /*************************************************************************************************************
    Name:        destroyState
    Description: dwstroy quantum state
    Argin:       stQNum      the number of quantum bit.
    Argout:      None
    return:      quantum error
    *************************************************************************************************************/
    virtual QError destroyState(size_t stQNum) { return qErrorNone; };

    /*************************************************************************************************************
    Name:        endGate
    Description: end gate
    Argin:       pQuantumProParam       quantum program param pointer
                 pQGate                 quantum gate
    Argout:      None
    return:      quantum error
    *************************************************************************************************************/
    virtual QError endGate(QuantumGateParam *pQuantumProParam, QuantumGates * pQGate) = 0;

    /*************************************************
    Name:      unitarySingleQubitGate
    Description:        unitary single qubit gate
    Argin:            psi        reference of vector<complex<double>> which contains quantum state information.
    qn        target qubit number
    matrix  matrix of the gate
    error_rate  the errorrate of the gate
    Argout:    the state after measurement
    return:    None
    *************************************************/
    virtual void unitarySingleQubitGate(size_t qn, QStat& matrix, double error_rate) = 0;

    /*************************************************
    Name:      unitarySingleQubitGateDagger
    Description:        unitary single qubit gate
    Argin:            psi        reference of vector<complex<double>> which contains quantum state information.
    qn        target qubit number
    matrix  matrix of the gate
    error_rate  the errorrate of the gate
    Argout:    the state after measurement
    return:    None
    *************************************************/
    virtual void unitarySingleQubitGateDagger(size_t qn, QStat& matrix, double error_rate) = 0;

    virtual void controlunitarySingleQubitGate(Qnum& qnum, QStat& matrix, double error_rate) = 0;

    virtual void controlunitarySingleQubitGateDagger(Qnum& qnum, QStat& matrix, double error_rate) = 0;

    virtual void unitaryDoubleQubitGate(size_t qn_1, size_t qn_2, QStat& matrix, double error_rate) = 0;

    virtual void unitaryDoubleQubitGateDagger(size_t qn_1, size_t qn_2, QStat& matrix, double error_rate) = 0;

    virtual void controlunitaryDoubleQubitGate(Qnum& qnum, QStat& matrix, double error_rate) = 0;

    virtual void controlunitaryDoubleQubitGateDagger(Qnum& qnum, QStat& matrix, double error_rate) = 0;

protected:
    //string sCalculationUnitType;
    /*************************************************************************************************************
    Name:        randGenerator
    Description: 16807 random number generator
    Argin:       None
    Argout:      None
    return:      random number in the region of [0,1]
    *************************************************************************************************************/
    double randGenerator();

    /*************************************************************************************************************
    Name:        matReverse
    Description: change the position of the qubits in 2-qubit gate
    Argin:       (*mat)[4]       pointer of the origin 2D array
    (*mat_rev)[4]   pointer of the changed 2D array
    Argout:      2D array
    return:      quantum error
    *************************************************************************************************************/
    virtual QError matReverse(COMPLEX(*)[4], COMPLEX(*)[4]) = 0;


};

#endif

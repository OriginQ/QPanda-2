/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.

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

#ifndef _QPUIMPL_H
#define _QPUIMPL_H
#include <iostream>
#include <stdio.h>
#include <vector>
#include <complex>
#include <map>
#include "Core/VirtualQuantumProcessor/QuantumGateParameter.h"
#include "Core/VirtualQuantumProcessor/QError.h"
#include "Core/VirtualQuantumProcessor/RandomEngine/RandomEngine.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"

typedef std::vector<QGateParam> vQParam;

/**
* @brief QPU implementation  base class
* @ingroup VirtualQuantumProcessor
*/
class QPUImpl
{
private:
	RandomEngine* random_engine = nullptr;
public:
    QPUImpl();
    virtual ~QPUImpl() = 0;

    virtual bool qubitMeasure(size_t qn) = 0;

    virtual QError pMeasure(Qnum& qnum, prob_vec &mResult) = 0;

    virtual QError initState(size_t head_rank, size_t rank_size, size_t qubit_num) = 0;

    virtual QError initState(size_t qubit_num, const QStat &state = {}) = 0;
    
	/**
	* @brief  unitary single qubit gate 
	* @param[in]  size_t  qubit address
	* @param[in]  QStat&  matrix
	* @param[in]  bool   state of conjugate
	* @param[in]  GateType    gate type
	* @return    QError
	*/
    virtual QError unitarySingleQubitGate(size_t qn, QStat& matrix, 
                        bool isConjugate, 
                        GateType) = 0;

	/**
	* @brief  controlunitary single qubit gate
	* @param[in]  size_t  qubit address
	* @param[in]  Qnum&  control qubit addresses 
	* @param[in]  QStat &  matrix
	* @param[in]  bool   state of conjugate
	* @param[in]  GateType    gate type
	* @return    QError
	*/
    virtual QError controlunitarySingleQubitGate(size_t qn, Qnum& qnum,
                        QStat& matrix, 
                        bool isConjugate, 
                        GateType) = 0;
    
	/**
	* @brief unitary double qubit gate
	* @param[in]  size_t  first qubit address
	* @param[in]  size_t  second qubit address
	* @param[in]  QStat&  matrix
	* @param[in]  bool   state of conjugate
	* @param[in]  GateType    gate type
	* @return    QError
	*/
    virtual QError unitaryDoubleQubitGate(size_t qn_0, size_t qn_1,
                        QStat& matrix,
                        bool isConjugate, 
                        GateType) = 0;

	/**
	* @brief  controlunitary double qubit gate
	* @param[in]  size_t  first qubit address
	* @param[in]  size_t  second qubit address
	* @param[in]  Qnum&  control qubit addresses
	* @param[in]  QStat&  quantum states
	* @param[in]  bool   state of conjugate
	* @param[in]  GateType    gate type
	* @return    QError
	*/
    virtual QError controlunitaryDoubleQubitGate(size_t qn_0,
                        size_t qn_1,
                        Qnum& qnum,
                        QStat& matrix,
                        bool isConjugate,
                        GateType) = 0;
	/**
	* @brief get quantum states
	*/
    virtual QStat getQState() = 0;

	virtual inline void set_random_engine(RandomEngine* rng) {
		random_engine = rng;
	}
	virtual inline double get_random_double() {
		if (!random_engine)
			return _default_random_generator();
		else
			return (*random_engine)();
	}

	/**
	* @brief reset qubit
	* @param[in]  size_t  qubit address
	*/
	virtual QError Reset(size_t qn) = 0;

};

/**
* @brief Quantum Gates Abstract Class
* @ingroup VirtualQuantumProcessor
*/
class AbstractQuantumGates
{
public:
    virtual QError DiagonalGate(Qnum& vQubit,
        QStat & matrix,
        bool isConjugate,
        double error_rate) = 0;
    virtual QError controlDiagonalGate(Qnum& vQubit,
        QStat & matrix,
        Qnum& vControlBit,
        bool isConjugate,
        double error_rate) = 0;

    virtual QError Reset(size_t qn) = 0;

    virtual QError Hadamard(size_t qn, bool isConjugate,
        double error_rate) = 0;
    virtual QError Hadamard(size_t qn, Qnum& vControlBit,
        bool isConjugate, double error_rate) = 0;
    virtual QError X(size_t qn, bool isConjugate,
        double error_rate) = 0;
    virtual QError X(size_t qn, Qnum& vControlBit,
        bool isConjugate, double error_rate) = 0;
    virtual QError P0(size_t qn, bool isConjugate,
        double error_rate) = 0;
    virtual QError P0(size_t qn, Qnum& vControlBit,
        bool isConjugate, double error_rate) = 0;
    virtual QError P1(size_t qn, bool isConjugate,
        double error_rate) = 0;
    virtual QError P1(size_t qn, Qnum& vControlBit,
        bool isConjugate, double error_rate) = 0;
    virtual QError Y(size_t qn, bool isConjugate,
        double error_rate) = 0;
    virtual QError Y(size_t qn, Qnum& vControlBit,
        bool isConjugate, double error_rate) = 0;
    virtual QError Z(size_t qn, bool isConjugate,
        double error_rate) = 0;
    virtual QError Z(size_t qn, Qnum& vControlBit,
        bool isConjugate, double error_rate) = 0;
    virtual QError T(size_t qn, bool isConjugate,
        double error_rate) = 0;
    virtual QError T(size_t qn, Qnum& vControlBit,
        bool isConjugate, double error_rate) = 0;
    virtual QError S(size_t qn, bool isConjugate,
        double error_rate) = 0;
    virtual QError S(size_t qn, Qnum& vControlBit,
        bool isConjugate, double error_rate) = 0;

    virtual QError U1_GATE(size_t qn, double theta,
        bool isConjugate, double error_rate) = 0;

    virtual QError RX_GATE(size_t qn, double theta,
        bool isConjugate, double error_rate) = 0;
    virtual QError RX_GATE(size_t qn, double theta,
        Qnum& vControlBit,
        bool isConjugate,
        double error_rate) = 0;
    virtual QError RY_GATE(size_t qn, double theta,
        bool isConjugate, double error_rate) = 0;
    virtual QError RY_GATE(size_t qn, double theta,
        Qnum& vControlBit,
        bool isConjugate,
        double error_rate) = 0;
    virtual QError RZ_GATE(size_t qn, double theta,
        bool isConjugate, double error_rate) = 0;
    virtual QError RZ_GATE(size_t qn, double theta,
        Qnum& vControlBit,
        bool isConjugate,
        double error_rate) = 0;
    virtual QError CNOT(size_t qn_0, size_t qn_1,
        bool isConjugate, double error_rate) = 0;
    virtual QError CNOT(size_t qn_0, size_t qn_1,
        Qnum& vControlBit,
        bool isConjugate,
        double error_rate) = 0;
    virtual QError CR(size_t qn_0, size_t qn_1, double theta,
        bool isConjugate, double error_rate) = 0;
    virtual QError CR(size_t qn_0, size_t qn_1,
        Qnum& vControlBit,
        double theta,
        bool isConjugate,
        double error_rate) = 0;

    virtual QError CZ(size_t qn_0, size_t qn_1,
        bool isConjugate, double error_rate) = 0;
    virtual QError CZ(size_t qn_0, size_t qn_1,
        Qnum& vControlBit,
        bool isConjugate,
        double error_rate) = 0;

    virtual QError iSWAP(size_t qn_0, size_t qn_1,
        double theta,
        bool isConjugate,
        double error_rate) = 0;
    virtual QError iSWAP(size_t qn_0, size_t qn_1,
        Qnum& vControlBit,
        double theta,
        bool isConjugate,
        double error_rate) = 0;

    virtual QError iSWAP(size_t qn_0, size_t qn_1,
        bool isConjugate,
        double error_rate) = 0;
    virtual QError iSWAP(size_t qn_0, size_t qn_1,
        Qnum& vControlBit,
        bool isConjugate,
        double error_rate) = 0;

    virtual QError SqiSWAP(size_t qn_0, size_t qn_1,
        bool isConjugate,
        double error_rate) = 0;
    virtual QError SqiSWAP(size_t qn_0, size_t qn_1,
        Qnum& vControlBit,
        bool isConjugate,
        double error_rate) = 0;
};


#endif


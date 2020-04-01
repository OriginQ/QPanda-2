/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

ArithmeticUnit.h

Author: LiYe
Created in 2020-03-24


*/


#ifndef ARITHNERC_UNIT_H_
#define ARITHNERC_UNIT_H_


#include "Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumCircuit/QCircuit.h"

QPANDA_BEGIN


/**
* @brief Quantum adder MAJ module.
*    a ------o---x----- c xor a
*    b --o---|---x----- c xor b
*    c --x---x---o----- ((c xor a) and (c xor b)) xor c = R
* @ingroup ArithmeticUnit
* @param[in] a  carry qubit
* @param[in] b  adder qubit
* @param[in] c  adder qubit
* @return QCircuit
*/
QCircuit MAJ(Qubit* a, Qubit* b, Qubit* c);

/**
* @brief Quantum adder UMA module.
*    a --x---o---x----- ((a and b) xor c) xor a
*    b --x---|---o----- (((a and b) xor c) xor a) xor b
*    c --o---x--------- (a and b) xor c
* If use MAJ module output is the input of UMA, the MAJ module carry qubit a
* and adder qubit c remain unchanged, the MAJ input adder qubit b save the
* result of b+c.
*   c xor a --x---o---x----- a
*   c xor b --x---|---o----- c xor b xor a
*   R       --o---x--------- c
* @ingroup ArithmeticUnit
* @param[in] a  target qubit
* @param[in] b  target qubit
* @param[in] c  auxiliary qubit
* @return QCircuit
*/
QCircuit UMA(Qubit* a, Qubit* b, Qubit* c);

/**
* @brief Quantum adder MAJ2 module.
* @ingroup ArithmeticUnit
* @param[in] adder1  adder qubits
* @param[in] adder2  adder qubits
* @param[in] c  carry qubit
* @return QCircuit
*/
QCircuit MAJ2(QVec &adder1, QVec &adder2, Qubit* c);

/**
* @brief Construct a circuit to determine if there is a carry
* @ingroup ArithmeticUnit
* @param[in] adder1  adder qubits
* @param[in] adder2  adder qubits
* @param[in] c  input carry qubit
* @param[in] is_carry  auxiliary qubit for determine adder1+adder2 if
                       there is a carry
* @return QCircuit
*/
QCircuit isCarry(
    QVec &adder1,
    QVec &adder2,
    Qubit* c,
    Qubit* is_carry);

/**
* @brief Quantum adder
* @ingroup ArithmeticUnit
* @param[in] adder1  adder qubits
* @param[in] adder2  adder qubits
* @param[in] c  input carry qubit
* @param[in] is_carry  auxiliary qubit for determine adder1+adder2 if
                       there is a carry
* @return QCircuit
* @note the result of QAdder is saved in is_carry and adder1,adder2 is not changed
*/
QCircuit QAdder(
    QVec &adder1,
    QVec &adder2,
    Qubit* c,
    Qubit* is_carry);

/**
* @brief Quantum adder ignore carry
* @ingroup ArithmeticUnit
* @param[in] adder1  adder qubits
* @param[in] adder2  adder qubits
* @param[in] c  input carry qubit
* @return QCircuit
* @note the result of QAdderIgnoreCarry is saved in adder1, carry is ignored, adder2 is not changed
*/
QCircuit QAdderIgnoreCarry(
    QVec &adder1,
    QVec &adder2,
    Qubit* c);

QPANDA_END


#endif // !ARITHNERC_UNIT_H_



/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

ArithmeticUnit.h

Author: LiYe
Created in 2020-03-24
Modified in 2020-10-10
Author: LiuYan
Modified in 2020-08-07

*/


#ifndef ARITHNERC_UNIT_H_
#define ARITHNERC_UNIT_H_


#include "Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QProgram.h"

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
* @note Only supports positive addition. The result of QAdder is saved in is_carry and adder1,adder2 is not changed
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
* @note The result of QAdder is saved in adder1, carry is ignored, adder2 is not changed
*/
QCircuit QAdder(
    QVec &adder1,
    QVec &adder2,
    Qubit* c);

/**
* @brief Quantum adder that supports signed operations, but ignore carry
* @ingroup ArithmeticUnit
* @param[in] a  adder qubits
* @param[in] b  adder qubits
* @param[in] k  auxiliary qubits
* @return QCircuit
* @note The size of k is a.size()+2.
*       The highest position of a and b is sign bit.
*       The result of QAdd is saved in a, carry is ignored, b and k is not changed.
*/
QCircuit QAdd(
    QVec& a,
    QVec& b,
    QVec& k);

/**
* @brief Convert quantum state to binary complement representation
* @ingroup ArithmeticUnit
* @param[in] a  adder qubits
* @param[in] k  auxiliary qubits
* @return QCircuit
* @note The size of k is a.size()+2.
*       The result of complement is saved in a, k is not changed.
*       And the initial value of k are all |0>.
*/
QCircuit QComplement(
    QVec& a,
    QVec& k);

/**
* @brief Quantum subtraction
* @ingroup ArithmeticUnit
* @param[in] a  minuend qubits
* @param[in] b  subtrahend qubits
* @param[in] k  auxiliary qubit
* @return QCircuit
* @note The highest position of a and b is sign bit.
*       The size of k is a.size()+2.
*       The result of QSub is saved in a, b and k is not changed.
*/
QCircuit QSub(
    QVec& a,
    QVec& b,
    QVec& k);
/**
* @brief Quantum multiplication
* @ingroup ArithmeticUnit
* @param[in] a  mul qubits
* @param[in] b  mul qubits
* @param[in] k  auxiliary qubit
* @param[in] d  ans qubits
* @return QCircuit
* @note Only supports positive multiplication.
*       The size of k is a.size()+1, the size of d is 2*a.size().
*       The result of QMul is saved in d, a and b and k is not changed.
*/
QCircuit QMultiplier(
    QVec& a,
    QVec& b,
    QVec& k,
    QVec& d);

/**
* @brief Quantum multiplication
* @ingroup ArithmeticUnit
* @param[in] a  mul qubits
* @param[in] b  mul qubits
* @param[in] k  auxiliary qubits
* @param[in] d  ans qubits
* @return QCircuit
* @note The size of k is a.size(), the size of d is 2*a.size()-1.
*       The highest position of a and b and d is sign bit.
*       The result of QMul is saved in d, a and b and k is not changed.
*/
QCircuit QMul(
    QVec& a,
    QVec& b,
    QVec& k,
    QVec& d);

/**
* @brief Quantum division
* @ingroup ArithmeticUnit
* @param[in] a  div qubits
* @param[in] b  div qubits
* @param[in] c  ans qubits
* @param[in] k  auxiliary qubits
* @param[in] t  control cbit
* @return QProg
* @note The highest position of a and b and c is sign bit, but a and b must be positive number.
*       The size of k is 2*a.size()+2.
*       The result of QDiv is saved in c, a saved the remainder, b and k is not changed.
*/
QProg QDivider(
    QVec& a,
    QVec& b,
    QVec& c,
    QVec& k,
    ClassicalCondition& t);

/**
* @brief Quantum division
* @ingroup ArithmeticUnit
* @param[in] a  div qubits
* @param[in] b  div qubits
* @param[in] c  ans qubits
* @param[in] k  auxiliary qubits
* @param[in] t  control cbit
* @return QProg
* @note The highest position of a and b and c is sign bit.
*       The size of k is 2*a.size()+4.
*       The result of QDiv is saved in c, a saved the remainder,b and k is not changed.
*/
QProg QDiv(
    QVec& a,
    QVec& b,
    QVec& c,
    QVec& k,
    ClassicalCondition& t);

/**
* @brief Quantum division
* @ingroup ArithmeticUnit
* @param[in] a  div qubits
* @param[in] b  div qubits
* @param[in] c  ans qubits
* @param[in] k  auxiliary qubits
* @param[in] f  accuracy qubits
* @param[in] s  control cbits
* @return QProg
* @note The highest position of a and b and c is sign bit ,but a and b must be positive number.
*       The size of k is 3*a.size()+5, s is f.size()+2.
*       The result of QDiv is saved in c, accuracy is saved in f, a and b and k is not changed.
*/
QProg QDivider(QVec& a,
    QVec& b,
    QVec& c,
    QVec& k,
    QVec& f,
    std::vector<ClassicalCondition>& s);

/**
* @brief Quantum division
* @ingroup ArithmeticUnit
* @param[in] a  div qubits
* @param[in] b  div qubits
* @param[in] c  ans qubits
* @param[in] k  auxiliary qubits
* @param[in] f  accuracy qubits
* @param[in] s  control cbits
* @return QProg
* @note The highest position of a and b and c is sign bit.
*       The size of k is 3*a.size()+7, s is f.size()+2.
*       The result of QDiv is saved in c, accuracy is saved in f, a and b and k is not changed.
*/
QProg QDiv(
    QVec& a,
    QVec& b,
    QVec& c,
    QVec& k,
    QVec& f,
    std::vector<ClassicalCondition>& s);

/**
* @brief Quantum bind data
* @ingroup ArithmeticUnit
* @param[in] value classical data
* @param[in] qvec  qubits
* @return QCircuit
* @note The highest position of qvec is sign bit,
*       and the initial value of qvec are all |0>
*/
QCircuit bind_data(int value, QVec &qvec);

/**
* @brief Quantum bind nonnegative integer
* @ingroup ArithmeticUnit
* @param[in] value classical data
* @param[in] qvec  qubits
* @return QCircuit
* @note The initial value of qvec are all |0>, and it does't consider the sign bit.
*/
QCircuit bind_nonnegative_data(size_t value, QVec& qvec);

/**
* @brief Quantum modular adder
* @ingroup ArithmeticUnit
* @param[in] qvec  adder qubit and result qubit
* @param[in] base  adder integer
* @param[in] module_Num  modular  integer
* @param[in] qvec1  auxiliary qubit
* @param[in] qvec2  input carry qubit and is_carry qubit
* @note qvec is both the input and output qubit
*/
QCircuit constModAdd(
    QVec &qvec,
    int base,
    int module_Num,
    QVec &qvec1,
    QVec &qvec2);

/**
* @brief Quantum modular multiplier
* @ingroup ArithmeticUnit
* @param[in] qvec  multi qubit
* @param[in] base  multi integer
* @param[in] module_Num  modular  integer
* @param[in] qvec1  multi auxiliary qubit
* @param[in] qvec2  adder auxiliary qubit
* @param[in] qvec3  input carry qubit and is_carry qubit
* @note qvec is both the input and output qubit
*/
QCircuit constModMul(
    QVec &qvec,
    int base,
    int module_Num,
    QVec &qvec1,
    QVec &qvec2,
    QVec &qvec3);

/**
* @brief Quantum modular exponents
* @ingroup ArithmeticUnit
* @param[in] qvec  exponents qubit
* @param[in] result  result qubit
* @param[in] base  base integer
* @param[in] module_Num  modular  integer
* @param[in] qvec1  multi auxiliary qubit
* @param[in] qvec2  adder auxiliary qubit
* @param[in] qvec3  input carry qubit and is_carry qubit
* @note qvec can be divided into binary, rlt should be one state at the beginning
* @note multi exponent is equal to exponent multi
*/
QCircuit constModExp(
    QVec &qvec,
    QVec &result,
    int base,
    int module_Num,
    QVec &qvec1,
    QVec &qvec2,
    QVec &qvec3);

QPANDA_END


#endif // !ARITHNERC_UNIT_H_



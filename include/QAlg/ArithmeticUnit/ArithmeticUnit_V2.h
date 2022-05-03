#pragma once

#include "Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QProgram.h"

QPANDA_BEGIN

enum class ADDER
{
    CDKM_RIPPLE,
    DRAPER_QFT,
    VBE_RIPPLE,
    DRAPER_QCLA
};
/**
 * @brief get required aux bit size
 * 
 * @param[in] t calculation destation qubits width 
 */
size_t auxBitSize(size_t t, ADDER adder);

/**
 * @brief Quantum adder that supports signed operations, but ignore carry
 * 
 * @param[inout] a  adder qubits
 * @param[in] b  adder qubits
 * @param[in] aux  auxiliary help qubits, should be |0...0>
 * @param[in] adder adder imply type
 * 
 * @return QCircuit
 * @note The size of aux is at least 2 * n - floor(log2(n)). n is bit length of a except sign bit 
 *       The highest position of a and b is sign bit.
 *       result is saved in a, b not changed
 *       if addtition overflow, aux will contains |1> at aux[0] or aux.back(), otherwise not changed
 */
QCircuit QAdd_V2(QVec &a, QVec &b, QVec &aux, ADDER adder = ADDER::CDKM_RIPPLE);

/**
 * @brief Quantum subtraction
 * @ingroup ArithmeticUnit
 * @param[in] a  minuend qubits
 * @param[in] b  subtrahend qubits
 * @param[in] aux  auxiliary qubit
 * @return QCircuit
 * @note similar like QAdd_V2
 */
QCircuit QSub_V2(QVec &a, QVec &b, QVec &aux, ADDER adder = ADDER::CDKM_RIPPLE);

/**
 * @brief Convert quantum state to binary complement representation
 * @ingroup ArithmeticUnit
 * @param[in] a qubits represented in binary number, highest bit is sign
 * @param[in] aux  auxiliary qubits, should be |0...0>
 * @return QCircuit
 * @note The size of aux is at least 2 * n - floor(log2(n)). n is bit length of a except sign bit 
 *       And the initial value of aux are all |0>.
 *       We used propesed circuit based on Draper QCLA adder in_place mode
 *       The result of complement is saved in a
 *       input negative zero like |100..00> is illegal
 *       if origin a is negative zero, output will be |000...00>, but aux[0] will be tagged |1>. Else aux not changed.
 *
 * [1] Vedral et al., Quantum Networks for Elementary Arithmetic Operations, 1995.
 * arXiv:quant-ph/0406142v1 <https://arxiv.org/pdf/quant-ph/0406142v1.pdf>
 *
 * [2] F. Orts et al., An optimized quantum circuit for converting from sign–magnitude to two’s complement
 */
QCircuit QComplement_V2(QVec &a, QVec &aux);

/**
 * @brief Convert quantum state to binary complement representation
 * @ingroup ArithmeticUnit
 * @param[in] a qubits represented in binary number, highest bit is sign
 * @param[in] aux  auxiliary qubit, should be |0>
 * @return QCircuit
 * @note The initial value of aux are all |0>.
 *       We used propesed circuit based on Draper QFT adder
 *       The result of complement is saved in a
 *       input negative zero like |100..00> is illegal
 *       if origin a is negative zero, output will be |000...00>, but aux will be tagged |1>. Else aux not changed.
 */
QCircuit QComplement_V2(QVec &a, Qubit* aux);
// TODO: need to implement interface below
#if 0
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
    QVec &a,
    QVec &b,
    QVec &k);

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
    QVec &a,
    QVec &b,
    QVec &k,
    QVec &d);

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
    QVec &a,
    QVec &b,
    QVec &c,
    QVec &k,
    ClassicalCondition &t);

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
    QVec &a,
    QVec &b,
    QVec &c,
    QVec &k,
    QVec &f,
    std::vector<ClassicalCondition> &s);



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
#endif
QPANDA_END

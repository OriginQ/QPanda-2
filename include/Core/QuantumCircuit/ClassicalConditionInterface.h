/*
Copyright (c) 2017-2019 Origin Quantum Computing. All Right Reserved.

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
/*! \file ClassicalConditionInterface.h */
#ifndef _CLASSICAL_CONDITION_INTERFACE_H
#define _CLASSICAL_CONDITION_INTERFACE_H
#include <string>
#include <memory>
#include "Core/QuantumMachine/CBitFactory.h"
#include "Core/QuantumCircuit/CExprFactory.h"
#include "Core/QuantumCircuit/QNode.h"

QPANDA_BEGIN
/**
* @namespace QPanda
*/

/**
* @enum ContentSpecifier
* @brief Content specifier
 */
enum ContentSpecifier
{
    CBIT,      ///< cbit type
    OPERATOR,  ///< operator type
    CONSTVALUE ///< const value type
};

/**
 * @brief Operator specifier
 */
enum OperatorSpecifier
{
    PLUS,    ///< Add operator type
    MINUS,   ///< Minus operator type
    MUL,     ///< Multiply operation type
    DIV,     ///< Division operation type
    GT,      ///< Greater than operation type
    EGT,     ///< Greater than or equal to operation type
    LT,      ///< Less than operation type
    ELT,     ///< Less than or equal to operation type
    EQUAL,   ///< Equal operator type
    NE,      ///< Not equal to operation type
    AND,     ///< And operation type
    OR,      ///< OR operation type
    NOT,     ///< NOT operation type
    ASSIGN   ///< ASSIGN operation type
};


/**
* @class ClassicalCondition
* @brief Classical condition class  Proxy class of cexpr class
* @ingroup Core
*/
class ClassicalCondition
{
    std::shared_ptr<CExpr> expr; ///< CExpr share ptr
    ClassicalCondition();
public:
    /**
     * @brief Get the Expr Ptr 
     * @return std::shared_ptr<CExpr> 
     */
    inline std::shared_ptr<CExpr>  getExprPtr() const { return expr; }

    /**
     * @brief Get the value of the current object
     * @return cbit_size_t 
     */
    cbit_size_t eval();

    /**
     * @brief Set the Value of the current object
     */
    void setValue(cbit_size_t);

    /**
     * @brief Check validity
     * 
     * @return true check validity ture
     * @return false  check validity false
     */
    bool checkValidity() const;

    /**
     * @brief Construct a new Classical Condition object by cbit
     * @param cbit target cbit ptr
     */
    ClassicalCondition(CBit* cbit);

    /**
     * @brief Construct a new Classical Condition object by CExpr
     * @param cexpr target cexpr ptr
     */
    ClassicalCondition(CExpr* cexpr);
    /**
     * @brief Construct a new Classical Condition object by ClassicalCondition
     * @param old target ClassicalCondition object
     */
    ClassicalCondition(const ClassicalCondition& old);

    /**
     * @brief ClassicalCondition assgen function by ClassicalCondition
     * @param old target ClassicalCondition object
     * @return ClassicalCondition 
     */
    ClassicalCondition operator=(ClassicalCondition old);

    /**
     * @brief ClassicalCondition assgen function by value
     * @param value target value 
     * @return ClassicalCondition 
     */
    ClassicalCondition operator=(const cbit_size_t value);
    ~ClassicalCondition();
};

/**
 * @brief Overload operator +
 * @param value cbit_size_t type left operand
 * @param class_cond  ClassicalCondition type right operand 
 * @return ClassicalCondition 
 */
inline ClassicalCondition operator+(cbit_size_t value, ClassicalCondition class_cond)
{
    auto &fac = CExprFactory::GetFactoryInstance();
    auto expr = fac.GetCExprByValue(value);
    if (expr == nullptr)
    {
        QCERR("CExpr factory fails");
        throw std::runtime_error("CExpr factory fails");
    }

    return
        CExprFactory::
        GetFactoryInstance().GetCExprByOperation
        (
            expr->deepcopy(),
            class_cond.getExprPtr()->deepcopy(),
            PLUS
        );
}
/**
 * @brief Uverload operator +
 * @param left_operand left operand
 * @param right_operand cbit_size_t type right operand 
 * @return ClassicalCondition 
 */
ClassicalCondition operator+(ClassicalCondition left_operand, ClassicalCondition right_operand);

/**
 * @brief Overload operator +
 * 
 * @param left_operand left operand
 * @param right_operand cbit_size_t type right operand 
 * @return ClassicalCondition 
 */
ClassicalCondition operator+(ClassicalCondition left_operand, cbit_size_t right_operand);

/**
 * @brief Overload operator -
 * 
 * @param left_operand left operand
 * @param right_operand right operand 
 * @return ClassicalCondition 
 */
ClassicalCondition operator-(ClassicalCondition left_operand, ClassicalCondition right_operand);

/**
 * @brief Overload operator -
 * 
 * @param left_operand left operand
 * @param right_operand cbit_size_t type right operand 
 * @return ClassicalCondition 
 */
ClassicalCondition operator-(ClassicalCondition left_operand, cbit_size_t right_operand);

/**
 * @brief Overload operator -
 * 
 * @param value cbit_size_t type left operand
 * @param class_cond  right operand 
 * @return ClassicalCondition 
 */
inline ClassicalCondition operator-(cbit_size_t value,
    ClassicalCondition class_cond)
{
    auto &fac = CExprFactory::GetFactoryInstance();
    auto expr = fac.GetCExprByValue(value);
    if (expr == nullptr)
    {
        QCERR("CExpr factory fails");
        throw std::runtime_error("CExpr factory fails");
    }

    return CExprFactory::GetFactoryInstance().GetCExprByOperation(
        expr->deepcopy(),
        class_cond.getExprPtr()->deepcopy(),
        MINUS);
}

/**
 * @brief Overload operator *
 * @param left_operand left operand
 * @param right_operand right operand 
 * @return ClassicalCondition 
 */
ClassicalCondition operator*(ClassicalCondition, ClassicalCondition);

/**
 * @brief Overload operator *
 * @param left_operand left operand
 * @param right_operand cbit_size_t type right operand 
 * @return ClassicalCondition 
 */
ClassicalCondition operator*(ClassicalCondition, cbit_size_t);

/**
 * @brief Overload operator -
 * @param value cbit_size_t type left operand
 * @param class_cond  right operand 
 * @return ClassicalCondition 
 */
inline ClassicalCondition operator*(cbit_size_t value,
    ClassicalCondition class_cond)
{
    auto &fac = CExprFactory::GetFactoryInstance();
    auto expr = fac.GetCExprByValue(value);
    if (expr == nullptr)
    {
        QCERR("CExpr factory fails");
        throw std::runtime_error("CExpr factory fails");
    }

    return CExprFactory::GetFactoryInstance().GetCExprByOperation(
        expr->deepcopy(),
        class_cond.getExprPtr()->deepcopy(),
        MUL);
}

/**
 * @brief Overload operator /
 * @param left_operand left operand
 * @param right_operand right operand 
 * @return ClassicalCondition 
 */
ClassicalCondition operator/(ClassicalCondition, ClassicalCondition);

/**
 * @brief Overload operator /
 * @param left_operand left operand
 * @param right_operand cbit_size_t type right operand 
 * @return ClassicalCondition 
 */
ClassicalCondition operator/(ClassicalCondition, cbit_size_t);

/**
 * @brief Overload operator /
 * @param value cbit_size_t type left operand
 * @param class_cond  right operand 
 * @return ClassicalCondition 
 */
inline ClassicalCondition operator/(cbit_size_t value,
    ClassicalCondition class_cond)
{
    auto &fac = CExprFactory::GetFactoryInstance();
    auto expr = fac.GetCExprByValue(value);
    if (expr == nullptr)
    {
        QCERR("CExpr factory fails");
        throw std::runtime_error("CExpr factory fails");
    }

    return CExprFactory::GetFactoryInstance().GetCExprByOperation(
        expr->deepcopy(),
        class_cond.getExprPtr()->deepcopy(),
        DIV);
}

/**
 * @brief Overload operator ==
 * @param left_operand left operand
 * @param right_operand right operand 
 * @return ClassicalCondition 
 */
ClassicalCondition operator==(ClassicalCondition, ClassicalCondition);

/**
 * @brief Overload operator ==
 * @param left_operand left operand
 * @param right_operand cbit_size_t type right operand 
 * @return ClassicalCondition 
 */
ClassicalCondition operator==(ClassicalCondition, cbit_size_t);

/**
 * @brief Overload operator ==
 * @param value cbit_size_t type left operand
 * @param class_cond  right operand 
 * @return ClassicalCondition 
 */
inline ClassicalCondition operator==(cbit_size_t value,
    ClassicalCondition class_cond)
{
    auto &fac = CExprFactory::GetFactoryInstance();
    auto expr = fac.GetCExprByValue(value);
    if (expr == nullptr)
    {
        QCERR("CExpr factory fails");
        throw std::runtime_error("CExpr factory fails");
    }

    return CExprFactory::GetFactoryInstance().GetCExprByOperation(
        expr->deepcopy(),
        class_cond.getExprPtr()->deepcopy(),
        EQUAL);
}
/**
 * @brief Overload operator !=
 * @param left_operand left operand
 * @param right_operand right operand 
 * @return ClassicalCondition 
 */
ClassicalCondition operator!=(ClassicalCondition, ClassicalCondition);
/**
 * @brief Overload operator !=
 * @param left_operand left operand
 * @param right_operand cbit_size_t type right operand 
 * @return ClassicalCondition 
 */
ClassicalCondition operator!=(ClassicalCondition, cbit_size_t);

/**
 * @brief Overload operator !=
 * @param value cbit_size_t type left operand
 * @param class_cond  right operand 
 * @return ClassicalCondition 
 */
inline ClassicalCondition operator!=(cbit_size_t value, 
    ClassicalCondition classical_cond)
{
    return classical_cond != value;
}

/**
 * @brief Overload operator &&
 * @param left_operand left operand
 * @param right_operand right operand 
 * @return ClassicalCondition 
 */
ClassicalCondition operator&&(ClassicalCondition, ClassicalCondition);

/**
 * @brief Overload operator &&
 * @param left_operand left operand
 * @param right_operand cbit_size_t type right operand 
 * @return ClassicalCondition 
 */
ClassicalCondition operator&&(ClassicalCondition, cbit_size_t);

/**
 * @brief Overload operator && 
 * @param value cbit_size_t type left operand
 * @param class_cond  right operand 
 * @return ClassicalCondition 
 */
inline ClassicalCondition operator&&(cbit_size_t value, 
    ClassicalCondition classical_cond)
{
    return classical_cond && value;
}

/**
 * @brief Overload operator ||
 * @param left_operand left operand
 * @param right_operand right operand 
 * @return ClassicalCondition 
 */
ClassicalCondition operator||(ClassicalCondition, ClassicalCondition);

/**
 * @brief Overload operator ||
 * @param left_operand left operand
 * @param right_operand cbit_size_t type right operand 
 * @return ClassicalCondition 
 */
ClassicalCondition operator||(ClassicalCondition, cbit_size_t);

/**
 * @brief Overload operator ||
 * @param value cbit_size_t type left operand
 * @param class_cond  right operand 
 * @return ClassicalCondition 
 */
inline ClassicalCondition operator||(cbit_size_t value,
    ClassicalCondition classical_cond)
{
    return classical_cond || value;
}

/**
 * @brief Overload operator >
 * @param left_operand left operand
 * @param right_operand right operand 
 * @return ClassicalCondition 
 */
ClassicalCondition operator>(ClassicalCondition, ClassicalCondition);

/**
 * @brief Overload operator >
 * @param left_operand left operand
 * @param right_operand cbit_size_t type right operand 
 * @return ClassicalCondition 
 */
ClassicalCondition operator>(ClassicalCondition, cbit_size_t);

/**
 * @brief Overload operator >
 * @param value cbit_size_t type left operand
 * @param class_cond  right operand 
 * @return ClassicalCondition 
 */
inline ClassicalCondition operator>(cbit_size_t value,
    ClassicalCondition class_cond)
{
    auto &fac = CExprFactory::GetFactoryInstance();
    auto expr = fac.GetCExprByValue(value);
    if (expr == nullptr)
    {
        QCERR("CExpr factory fails");
        throw std::runtime_error("CExpr factory fails");
    }

    return CExprFactory::GetFactoryInstance().GetCExprByOperation(
        expr->deepcopy(),
        class_cond.getExprPtr()->deepcopy(),
        GT);
}

/**
 * @brief Overload operator >=
 * @param left_operand left operand
 * @param right_operand right operand 
 * @return ClassicalCondition 
 */
ClassicalCondition operator>=(ClassicalCondition, ClassicalCondition);

/**
 * @brief Overload operator >=
 * @param left_operand left operand
 * @param right_operand cbit_size_t type right operand 
 * @return ClassicalCondition 
 */
ClassicalCondition operator>=(ClassicalCondition, cbit_size_t);

/**
 * @brief Overload operator >=
 * @param value cbit_size_t type left operand
 * @param class_cond  right operand 
 * @return ClassicalCondition 
 */
inline ClassicalCondition operator>=(cbit_size_t value,
    ClassicalCondition class_cond)
{
    auto &fac = CExprFactory::GetFactoryInstance();
    auto expr = fac.GetCExprByValue(value);
    if (expr == nullptr)
    {
        QCERR("CExpr factory fails");
        throw std::runtime_error("CExpr factory fails");
    }

    return CExprFactory::GetFactoryInstance().GetCExprByOperation(
        expr->deepcopy(),
        class_cond.getExprPtr()->deepcopy(),
        EGT);
}

/**
 * @brief Overload operator <
 * @param left_operand left operand
 * @param right_operand right operand 
 * @return ClassicalCondition 
 */
ClassicalCondition operator<(ClassicalCondition, ClassicalCondition);

/**
 * @brief Overload operator <
 * @param left_operand left operand
 * @param right_operand cbit_size_t type right operand 
 * @return ClassicalCondition 
 */
ClassicalCondition operator<(ClassicalCondition, cbit_size_t);

/**
 * @brief Overload operator <
 * @param value cbit_size_t type left operand
 * @param class_cond  right operand 
 * @return ClassicalCondition 
 */
inline ClassicalCondition operator<(cbit_size_t value,
    ClassicalCondition class_cond)
{
    auto &fac = CExprFactory::GetFactoryInstance();
    auto expr = fac.GetCExprByValue(value);
    if (expr == nullptr)
    {
        QCERR("CExpr factory fails");
        throw std::runtime_error("CExpr factory fails");
    }

    return CExprFactory::GetFactoryInstance().GetCExprByOperation(
        expr->deepcopy(),
        class_cond.getExprPtr()->deepcopy(),
        LT);
}

/**
 * @brief Overload operator <=
 * @param left_operand left operand
 * @param right_operand right operand 
 * @return ClassicalCondition 
 */
ClassicalCondition operator<=(ClassicalCondition, ClassicalCondition);

/**
 * @brief Overload operator <=
 * @param left_operand left operand
 * @param right_operand cbit_size_t type right operand 
 * @return ClassicalCondition 
 */
ClassicalCondition operator<=(ClassicalCondition, cbit_size_t);

/**
 * @brief Overload operator <=
 * @param value cbit_size_t type left operand
 * @param class_cond  right operand 
 * @return ClassicalCondition 
 */
inline ClassicalCondition operator<=(cbit_size_t value,
    ClassicalCondition class_cond)
{
    auto &fac = CExprFactory::GetFactoryInstance();
    auto expr = fac.GetCExprByValue(value);
    if (expr == nullptr)
    {
        QCERR("CExpr factory fails");
        throw std::runtime_error("CExpr factory fails");
    }

    return CExprFactory::GetFactoryInstance().GetCExprByOperation(
        expr->deepcopy(),
        class_cond.getExprPtr()->deepcopy(),
        ELT);
}

/**
 * @brief Overload operator !
 * @param right_operand right operand 
 * @return ClassicalCondition 
 */
ClassicalCondition operator!(ClassicalCondition);
QPANDA_END

#endif
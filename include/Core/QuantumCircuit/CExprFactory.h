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

#ifndef CEXPR_FACTORY_H
#define CEXPR_FACTORY_H

#include "Core/QuantumMachine/CBitFactory.h"
#include <functional>
#include <stack>
#include <map>

QPANDA_BEGIN

/**
 * @brief classical expr
 * 
 */
class CExpr 
{
    // classical expression interface    
public:

    /**
     * @brief Get the Left Expr pointer
     * 
     * @return CExpr* 
     */
    virtual CExpr* getLeftExpr() const = 0;

    /**
     * @brief Get the Right Expr pointer
     * 
     * @return CExpr* 
     */
    virtual CExpr* getRightExpr() const = 0;

    /**
     * @brief Set the Left Expr pointer
     * @param left_expr left expr
     */
    virtual void setLeftExpr(CExpr* left_expr) = 0;

    /**
     * @brief Set the Right Expr pointer
     * @param right_expr right expr
     */
    virtual void setRightExpr(CExpr* right_expr) = 0;

    /**
     * @brief Get the Name object
     * 
     * @return std::string 
     */
    virtual std::string getName() const = 0;

    /**
     * @brief get classical bit pointer
     * 
     * @return CBit* 
     */
    virtual CBit* getCBit() const = 0;

    /**
     * @brief check validity
     * 
     * @return true check validity ture
     * @return false  check validity false
     */
    virtual bool checkValidity() const = 0;
    /**
     * @brief Destroy the CExpr object
     * 
     */
    virtual ~CExpr() {}

    /**
     * @brief get value
     * 
     * @return cbit_size_t 
     */
    virtual cbit_size_t eval() const = 0;
    /**
    * @brief get specifier of this cexpr
    *
    * @return int
    */
    virtual int getContentSpecifier() const = 0;

    /**
     * @brief deep copy this cexpr
     * 
     * @return CExpr* 
     */
    virtual CExpr* deepcopy() const = 0;
};

/**
 * @brief classical expr class factory
 * 
 */
class CExprFactory
{
    /**
     * @brief Construct a new CExprFactory object
     * 
     */
    CExprFactory();
public:
    /**
     * @brief Get the Factory Instance object
     * 
     * @return CExprFactory& 
     */
    static CExprFactory & GetFactoryInstance();

    /**
     * @brief We typedef the cbit_constructor_t is a constructor that  use cbit create cexpr
     */
    typedef std::function<CExpr*(CBit*)> cbit_constructor_t;

    /**
     * @brief We typedef the cbit_constructor_map_t is a collection of constructors that use cbit create cexpr 
     */
    typedef std::map<std::string, cbit_constructor_t> cbit_constructor_map_t;

    cbit_constructor_map_t _CExpr_CBit_Constructor; ///<A collection of constructors that use cbit create cexpr 

    /**
     * @brief Get cexpr by cbit
     * @param cbit target cbit
     * @return CExpr* 
     */
    CExpr* GetCExprByCBit(CBit* cbit);

    /**
     * @brief Registration function
     *  This function can be used to register constructors that inherit subclasses of the CExpr class.
     * @param name subclass name
     * @param constructor subclass constructor
     */
    void registerclass_CBit_(std::string & name, cbit_constructor_t constructor);

    /**
     * @brief  We typedef the value_constructor_t is a constructor that  use value create cexpr
     */
    typedef std::function<CExpr*(cbit_size_t)> value_constructor_t;

    /**
     * @brief We typedef the value_constructor_map_t is a collection of constructors that use value create cexpr 
     */
    typedef std::map<std::string, value_constructor_t> value_constructor_map_t;

    value_constructor_map_t _CExpr_Value_Constructor;  ///<A collection of constructors that use value create cexpr 
    /**
     * @brief Get cexpr by value
     * @param value target value
     * @return CExpr* 
     */
    CExpr* GetCExprByValue(cbit_size_t value);

    /**
     * @brief Registration function
     *  This function can be used to register constructors that inherit subclasses of the CExpr class.
     * @param name subclass name
     * @param constructor subclass constructor
     */
    void registerclass_Value_(std::string &, value_constructor_t);
    /**
     * @brief  We typedef the operator_constructor_t is a constructor that use operator create cexpr
     */
    typedef std::function<CExpr*(CExpr*, CExpr*, int)> operator_constructor_t;

    /**
     * @brief We typedef the operator_constructor_map_t is a collection of constructors that use operator create cexpr 
     */
    typedef std::map<std::string, operator_constructor_t> operator_constructor_map_t;
    operator_constructor_map_t _CExpr_Operator_Constructor;///<A collection of constructors that use operator create cexpr
   
    /**
     * @brief Get cexpr by Operation
     * @param left_cexpr left CExpr
     * @param right_cexpr right CExpr
     * @param operat  target operator
     * @return CExpr* 
     */
    CExpr* GetCExprByOperation(CExpr* left_cexpr, CExpr* right_cexpr, int operat);

    /**
     * @brief Registration function
     *  This function can be used to register constructors that inherit subclasses of the CExpr class.
     * @param name subclass name
     * @param constructor subclass constructor
     */
    void registerclass_operator_(std::string &, operator_constructor_t);
};

/**
 * @brief CExpr factory helper
 * Provide CExprFactory class registration interface for the outside
 */
class CExprFactoryHelper
{
    typedef CExprFactory::cbit_constructor_t
        cbit_constructor_t;
    typedef CExprFactory::value_constructor_t
        value_constructor_t;
    typedef CExprFactory::operator_constructor_t
        operator_constructor_t;
public:
    /**
     * @brief Construct a new CExprFactoryHelper object
     * Call the CExprFactory class registration interface for register the CExpr subclass
     * @param CExpr subclass name 
     * @param _Constructor cbit_constructor_t  function 
     */
    CExprFactoryHelper(std::string name, cbit_constructor_t _Constructor);
    
    /**
     * @brief Construct a new CExprFactoryHelper object
     * Call the CExprFactory class registration interface for register the CExpr subclass
     * @param CExpr subclass name 
     * @param _Constructor value_constructor_t  function 
     */
    CExprFactoryHelper(std::string, value_constructor_t _Constructor);

    /**
     * @brief Construct a new CExprFactoryHelper object
     * Call the CExprFactory class registration interface for register the CExpr subclass
     * @param CExpr subclass name 
     * @param _Constructor operator_constructor_t  function 
     */
    CExprFactoryHelper(std::string, operator_constructor_t _Constructor);

};

#define REGISTER_CEXPR(classname)\
CExpr* classname##_CBit_Constructor(CBit* cbit)\
{\
    return new classname(cbit);\
}\
CExpr* classname##_Value_Constructor(cbit_size_t value)\
{\
    return new classname(value);\
}\
CExpr* classname##_Operator_Constructor(\
    CExpr* leftexpr,\
    CExpr* rightexpr,\
    int op\
)\
{\
    return new classname(leftexpr, rightexpr, op);\
}\
static CExprFactoryHelper _CBit_Constructor_Helper_##classname( \
    #classname,\
    classname##_CBit_Constructor\
);\
static CExprFactoryHelper _Value_Constructor_Helper_##classname( \
    #classname,\
    classname##_Value_Constructor\
);\
static CExprFactoryHelper \
_Operator_Constructor_Helper_##classname\
(\
#classname,\
classname##_Operator_Constructor)
QPANDA_END

#endif
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

#include "OriginClassicalExpression.h"
#include <functional>
#include "ClassicalConditionInterface.h"
using namespace std;
USING_QPANDA
inline bool isBinary(int OperatorSpecifier)
{
    return OperatorSpecifier <= OR;
}

inline bool isUnary(int OperatorSpecifier)
{
    return
        OperatorSpecifier > OR
        &&
        OperatorSpecifier <= NOT;
}


inline bool isEqual(int OperatorSpecifier)
{
    return OperatorSpecifier == ASSIGN;
}

inline bool isOperator(int OperatorSpecifier)
{
    return OperatorSpecifier <= ASSIGN;
}

static map<int, function<cbit_size_t(cbit_size_t &, cbit_size_t &)>> _Binary_Operation =
{
    {PLUS,[](cbit_size_t & a,cbit_size_t &b) {return a + b; }},
    {MINUS,[](cbit_size_t& a,cbit_size_t & b) {return a - b; } },
    {MUL,[](cbit_size_t& a,cbit_size_t & b) {return a * b; } },
    {DIV,[](cbit_size_t& a,cbit_size_t & b) {return a / b; } },
    {EQUAL,[](cbit_size_t& a,cbit_size_t & b) {return a == b;}},
    { NE,[](cbit_size_t& a,cbit_size_t & b) {return a != b; } },
    { GT,[](cbit_size_t& a,cbit_size_t & b) {return a > b; } },
    { EGT,[](cbit_size_t& a,cbit_size_t & b) {return a >= b; } },
    { LT,[](cbit_size_t& a,cbit_size_t & b) {return a < b; } },
    { ELT,[](cbit_size_t& a,cbit_size_t & b) {return a <= b; } },
    { AND,[](cbit_size_t &a,cbit_size_t &b) {return a && b; } },
    { OR,[](cbit_size_t &a,cbit_size_t &b) {return a || b; } },
    { ASSIGN,[](cbit_size_t &a,cbit_size_t &b) { a = b; return a; } }
};

static map<int, string> _Operator_Name =
{
    {PLUS,"+"},
    {MINUS,"-"},
    {MUL,"*"},
    {DIV,"/"},
    {EQUAL,"==" },
    { NE,"!=" },
    { GT,">" },
    { EGT,">=" },
    { LT,"<" },
    { ELT,"<=" },
    {AND,"&&"},
    {OR,"||"},
    {NOT,"!"},
    {ASSIGN,"=" }
};

static map<int, function<cbit_size_t(cbit_size_t)>> _Unary_Operation=
{
    {NOT,[](cbit_size_t a) {return !a; }}
};

OriginCExpr::OriginCExpr(CBit* cbit)
{
    content.cbit = cbit;
    contentSpecifier = CBIT;
}

OriginCExpr::OriginCExpr(CExpr *_leftExpr, 
    CExpr *_rightExpr, int op)
{
    leftExpr = _leftExpr;
    rightExpr = _rightExpr;
    contentSpecifier = OPERATOR;
    content.iOperatorSpecifier = op;
}

OriginCExpr::OriginCExpr(cbit_size_t value)
{
    content.const_value = value;
    contentSpecifier = CONSTVALUE;
}

CExpr *OriginCExpr::getLeftExpr() const
{
    return leftExpr;
}

CExpr *OriginCExpr::getRightExpr() const
{
    return rightExpr;
}

string OriginCExpr::getName() const
{
    switch (contentSpecifier)
    {
    case CBIT:
        return this->content.cbit->getName();
    case OPERATOR:
        if (isOperator(this->content.iOperatorSpecifier))
            return
            _Operator_Name[this->content.iOperatorSpecifier];
        else
        {
            QCERR("Bad operator specifier");
            throw invalid_argument("Bad operator specifier");
        }
    case CONSTVALUE:
        return std::to_string(content.const_value);
    default:
        QCERR("Bad operator specifier");
        throw invalid_argument("Bad content specifier");
    }
}

CBit *OriginCExpr::getCBit() const
{
    switch (contentSpecifier)
    {
    case CBIT:
        return content.cbit;
    case OPERATOR:
    case CONSTVALUE:
        return nullptr;
    default:
        QCERR("Bad content specifier");
        throw invalid_argument("Bad content specifier");
    }
}

void OriginCExpr::setLeftExpr(CExpr *leftexpr)
{
    leftExpr = leftexpr;
}

void OriginCExpr::setRightExpr(CExpr* rightexpr)
{
    rightExpr = rightexpr;
}

cbit_size_t OriginCExpr::eval() const
{
    if (contentSpecifier == CBIT)
    {
        auto cbit =  getCBit();
        return cbit->getValue();
    }
    else if (this->contentSpecifier==OPERATOR)
    {
        if (isBinary(this->content.iOperatorSpecifier))
        {
            auto left = this->leftExpr->eval();
            auto right = this->rightExpr->eval();
            return _Binary_Operation[
                this->content.iOperatorSpecifier
            ](left, right);
        }
        else if (isEqual(this->content.iOperatorSpecifier))
        {
            auto left = this->leftExpr->eval();
            auto right = this->rightExpr->eval();
            _Binary_Operation[this->content.iOperatorSpecifier](left, right);
            auto left_cbit = this->leftExpr->getCBit();
            left_cbit->setValue(left);
            return left;
        }
        else if (isUnary(this->content.iOperatorSpecifier))
        {
            return _Unary_Operation[
                this->content.iOperatorSpecifier
            ](this->leftExpr->eval());
        }

        else
        {
            QCERR("Bad operator specifier");
            throw invalid_argument("Bad operator specifier");
        }
    }
    else if(CONSTVALUE == this->contentSpecifier)
    {
        return content.const_value;
    }
    else
    {
        QCERR("Bad operator specifier");
        throw invalid_argument("Bad operator specifier");
    }
}

CExpr *OriginCExpr::deepcopy() const
{
    if (contentSpecifier == CBIT)
    {
        return
            CExprFactory::GetFactoryInstance().
            GetCExprByCBit(this->content.cbit);
    }
    if (contentSpecifier == OPERATOR)
    {
        if (isBinary(this->content.iOperatorSpecifier))
        {
            return
                CExprFactory::GetFactoryInstance().
                GetCExprByOperation(
                    this->leftExpr->deepcopy(),
                    this->rightExpr->deepcopy(),
                    this->content.iOperatorSpecifier
                );
        }
        else if (isUnary(this->content.iOperatorSpecifier))
        {
            return
                CExprFactory::GetFactoryInstance().
                GetCExprByOperation(
                    this->leftExpr->deepcopy(),
                    nullptr,
                    this->content.iOperatorSpecifier
                );
        }
        else if (isEqual(this->content.iOperatorSpecifier))
        {
            return
                CExprFactory::GetFactoryInstance().
                GetCExprByOperation(
                    this->leftExpr->deepcopy(),
                    this->rightExpr->deepcopy(),
                    this->content.iOperatorSpecifier
                );
        }
        else
        {
            QCERR("Bad content specifier");
            throw invalid_argument("Bad content specifier");
        }
    }
    else if (CONSTVALUE == contentSpecifier)
    {
        return
            CExprFactory::GetFactoryInstance().
            GetCExprByValue(this->content.const_value);
    }
    else
    {
        QCERR("Bad content specifier");
        throw invalid_argument("Bad content specifier");
    }
}

bool OriginCExpr::checkValidity() const
{
    if (contentSpecifier == OPERATOR)
    {
        bool leftValidity, rightValidity;
        if (leftExpr == nullptr)
        {
            leftValidity = true;
        }
        else
        {
            leftValidity = leftExpr->checkValidity();
        }
        if (rightExpr == nullptr)
        {
            rightValidity = true;
        }
        else
        {
            rightValidity = rightExpr->checkValidity();
        }
        return leftValidity && rightValidity;
    }
    else if (contentSpecifier == CBIT)
    {
        return content.cbit->getOccupancy();
    }
    else
    {
        QCERR("Bad content specifier");
        throw invalid_argument("Bad content specifier");
    }
}

NodeType OriginCExpr::getNodeType() const
{
    return m_node_type;
}

qmap_size_t OriginCExpr::getPosition() const
{
    return m_postion;
}

void OriginCExpr::setPosition(qmap_size_t  postion)
{
    m_postion = postion;
}

int OriginCExpr::getContentSpecifier() const
{
    return contentSpecifier;
}

void OriginCExpr::getCBitsName(std::vector<std::string>& names)
{
    if (CBIT == contentSpecifier)
    {
        names.push_back(getName());
    }
    else if (OPERATOR == contentSpecifier)
    {
        if (leftExpr != nullptr)
        {
            leftExpr->getCBitsName(names);
        }

        if (rightExpr != nullptr)
        {
            rightExpr->getCBitsName(names);
        }
    }
    else
    {

    }
}

OriginCExpr::~OriginCExpr()
{
    if (contentSpecifier == CBIT)
    {
        if (leftExpr == nullptr && rightExpr == nullptr)
        {
            return;
        }

        return;
    }
    else if (contentSpecifier == OPERATOR)
    {
        if (leftExpr == nullptr)
        {
        }
        else
        {
            delete leftExpr;
        }
        if (rightExpr == nullptr)
        {
            if (isUnary(this->content.iOperatorSpecifier))
            {
                return;
            }
        }
        else
        {
            delete rightExpr;
        }
    }
}

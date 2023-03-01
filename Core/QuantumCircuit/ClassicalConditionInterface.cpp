/*
Copyright (c) 2017-2023 Origin Quantum Computing. All Right Reserved.

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

#include "ClassicalConditionInterface.h"
#include "Factory.h"
USING_QPANDA
using namespace std;
cbit_size_t ClassicalCondition::get_val()
{
    if(nullptr == expr)
    {
        QCERR("expr is null");
        throw invalid_argument("expr is null");
    }
    auto temp  = expr->get_val();
    return temp;
}

void ClassicalCondition::set_val(cbit_size_t value)
{
    auto cbit = expr->getCBit();
    if (nullptr == cbit)
    {
        QCERR("cbit is null");
        throw runtime_error("cbit is null");
    }
    cbit->set_val(value);
}

bool ClassicalCondition::checkValidity() const
{
    return expr->checkValidity();
}

ClassicalCondition::ClassicalCondition(CBit *cbit)
{
    auto &fac = CExprFactory::GetFactoryInstance();
    expr= shared_ptr<CExpr>(fac.GetCExprByCBit(cbit));
    if (expr == nullptr)
    {
        QCERR("CExpr factory fails");
        throw runtime_error("CExpr factory fails");
    }
}

ClassicalCondition::ClassicalCondition(cbit_size_t value)
{
    auto &fac = CExprFactory::GetFactoryInstance();
    auto value_expr = fac.GetCExprByValue(value);
    if (expr == nullptr)
    {
        QCERR("CExpr factory fails");
        throw runtime_error("CExpr factory fails");
    }
}


ClassicalCondition::ClassicalCondition(CExpr *_Expr)
{
    expr = shared_ptr<CExpr>(_Expr);
}

ClassicalCondition&
ClassicalCondition::operator=(const ClassicalCondition& old)
{
    if (this == &old)
    {
        return *this;
    }

    auto type = old.getExprPtr()->getContentSpecifier();
    if (0 == type)
    {
        expr = shared_ptr<CExpr>(old.expr->deepcopy());
    }
    else
    {
        auto &fac = CExprFactory::GetFactoryInstance();
        fac.GetCExprByOperation(expr->deepcopy(),
            old.expr->deepcopy(),
            ASSIGN);
    }
    
    return *this;
}

ClassicalCondition &
ClassicalCondition::operator=(const cbit_size_t value)
{
    //auto cbit = expr->getCBit();
    //cbit->set_val(value);

    auto &fac = CExprFactory::GetFactoryInstance();
    auto value_expr = fac.GetCExprByValue(value);
    //expr = shared_ptr<CExpr>(value_expr);
    CExprFactory::GetFactoryInstance().GetCExprByOperation(expr->deepcopy(),
                        value_expr->deepcopy(),
                        ASSIGN);

    return *this;
}




ClassicalCondition::ClassicalCondition
(const ClassicalCondition &cc)
{
    expr = cc.expr;
}

ClassicalCondition::~ClassicalCondition()
{
    expr.reset();
}

ClassicalCondition QPanda::operator+(
    ClassicalCondition leftcc,
    ClassicalCondition rightcc)
{
    return
        CExprFactory::
        GetFactoryInstance().GetCExprByOperation
        (
            leftcc.getExprPtr()->deepcopy(),
            rightcc.getExprPtr()->deepcopy(),
            PLUS
        );
}

ClassicalCondition  QPanda::operator+(ClassicalCondition class_cond, 
    cbit_size_t value)
{
    auto &fac = CExprFactory::GetFactoryInstance();
    auto expr = fac.GetCExprByValue(value);
    if (expr == nullptr)
    {
        QCERR("CExpr factory fails");
        throw runtime_error("CExpr factory fails");
    }

    return
        CExprFactory::
        GetFactoryInstance().GetCExprByOperation
        (
            class_cond.getExprPtr()->deepcopy(),
            expr->deepcopy(),
            PLUS
        );

}


ClassicalCondition QPanda::operator-(
    ClassicalCondition leftcc,
    ClassicalCondition rightcc)
{
    return
        CExprFactory::
        GetFactoryInstance().GetCExprByOperation
        (
            leftcc.getExprPtr()->deepcopy(),
            rightcc.getExprPtr()->deepcopy(),
            MINUS
        );
}

ClassicalCondition  QPanda::operator-(ClassicalCondition class_cond,
    cbit_size_t value)
{
    auto &fac = CExprFactory::GetFactoryInstance();
    auto expr = fac.GetCExprByValue(value);
    if (expr == nullptr)
    {
        QCERR("CExpr factory fails");
        throw runtime_error("CExpr factory fails");
    }

    return CExprFactory::GetFactoryInstance().GetCExprByOperation(
            class_cond.getExprPtr()->deepcopy(),
            expr->deepcopy(),
            MINUS);

}


ClassicalCondition QPanda::operator*(
    ClassicalCondition leftcc,
    ClassicalCondition rightcc)
{
    return
        CExprFactory::
        GetFactoryInstance().GetCExprByOperation
        (
            leftcc.getExprPtr()->deepcopy(),
            rightcc.getExprPtr()->deepcopy(),
            MUL
        );
}

ClassicalCondition  QPanda::operator*(ClassicalCondition class_cond,
    cbit_size_t value)
{
    auto &fac = CExprFactory::GetFactoryInstance();
    auto expr = fac.GetCExprByValue(value);
    if (expr == nullptr)
    {
        QCERR("CExpr factory fails");
        throw runtime_error("CExpr factory fails");
    }

    return CExprFactory::GetFactoryInstance().GetCExprByOperation(
        class_cond.getExprPtr()->deepcopy(),
        expr->deepcopy(),
        MUL);

}




ClassicalCondition QPanda::operator/(
    ClassicalCondition leftcc,
    ClassicalCondition rightcc)
{
    return
        CExprFactory::
        GetFactoryInstance().GetCExprByOperation
        (
            leftcc.getExprPtr()->deepcopy(),
            rightcc.getExprPtr()->deepcopy(),
            DIV
        );
}

ClassicalCondition  QPanda::operator/(ClassicalCondition class_cond,
     cbit_size_t value)
{
    if (0 == value)
    {
        QCERR("you can't have a dividend of 0");
        throw invalid_argument("you can't have a dividend of 0");
    }
    auto &fac = CExprFactory::GetFactoryInstance();
    auto expr = fac.GetCExprByValue(value);
    if (expr == nullptr)
    {
        QCERR("CExpr factory fails");
        throw runtime_error("CExpr factory fails");
    }

    return CExprFactory::GetFactoryInstance().GetCExprByOperation(
        class_cond.getExprPtr()->deepcopy(),
        expr->deepcopy(),
        DIV);

}




ClassicalCondition QPanda::operator==(
    ClassicalCondition leftcc,
    ClassicalCondition rightcc)
{
    return
        CExprFactory::
        GetFactoryInstance().GetCExprByOperation
        (
            leftcc.getExprPtr()->deepcopy(),
            rightcc.getExprPtr()->deepcopy(),
            EQUAL
        );
}

ClassicalCondition  QPanda::operator==(ClassicalCondition class_cond,
    cbit_size_t value)
{
    auto &fac = CExprFactory::GetFactoryInstance();
    auto expr = fac.GetCExprByValue(value);
    if (expr == nullptr)
    {
        QCERR("CExpr factory fails");
        throw runtime_error("CExpr factory fails");
    }

    return CExprFactory::GetFactoryInstance().GetCExprByOperation(
        class_cond.getExprPtr()->deepcopy(),
        expr->deepcopy(),
        EQUAL);

}



ClassicalCondition QPanda::operator!=(
    ClassicalCondition leftcc,
    ClassicalCondition rightcc)
{
    return
        CExprFactory::
        GetFactoryInstance().GetCExprByOperation
        (
            leftcc.getExprPtr()->deepcopy(),
            rightcc.getExprPtr()->deepcopy(),
            NE
        );
}

ClassicalCondition  QPanda::operator!=(ClassicalCondition class_cond,
    cbit_size_t value)
{
    auto &fac = CExprFactory::GetFactoryInstance();
    auto expr = fac.GetCExprByValue(value);
    if (expr == nullptr)
    {
        QCERR("CExpr factory fails");
        throw runtime_error("CExpr factory fails");
    }

    return CExprFactory::GetFactoryInstance().GetCExprByOperation(
        class_cond.getExprPtr()->deepcopy(),
        expr->deepcopy(),
        NE);

}


ClassicalCondition QPanda::operator>(
    ClassicalCondition leftcc,
    ClassicalCondition rightcc)
{
    return
        CExprFactory::
        GetFactoryInstance().GetCExprByOperation
        (
            leftcc.getExprPtr()->deepcopy(),
            rightcc.getExprPtr()->deepcopy(),
            GT
        );
}

ClassicalCondition  QPanda::operator>(ClassicalCondition class_cond,
    cbit_size_t value)
{
    auto &fac = CExprFactory::GetFactoryInstance();
    auto expr = fac.GetCExprByValue(value);
    if (expr == nullptr)
    {
        QCERR("CExpr factory fails");
        throw runtime_error("CExpr factory fails");
    }

    return CExprFactory::GetFactoryInstance().GetCExprByOperation(
        class_cond.getExprPtr()->deepcopy(),
        expr->deepcopy(),
        GT);

}




ClassicalCondition QPanda::operator>=(
    ClassicalCondition leftcc,
    ClassicalCondition rightcc)
{
    return
        CExprFactory::
        GetFactoryInstance().GetCExprByOperation
        (
            leftcc.getExprPtr()->deepcopy(),
            rightcc.getExprPtr()->deepcopy(),
            EGT
        );
}

ClassicalCondition  QPanda::operator>=(ClassicalCondition class_cond,
    cbit_size_t value)
{
    auto &fac = CExprFactory::GetFactoryInstance();
    auto expr = fac.GetCExprByValue(value);
    if (expr == nullptr)
    {
        QCERR("CExpr factory fails");
        throw runtime_error("CExpr factory fails");
    }

    return CExprFactory::GetFactoryInstance().GetCExprByOperation(
        class_cond.getExprPtr()->deepcopy(),
        expr->deepcopy(),
        EGT);

}



ClassicalCondition QPanda::operator<(
    ClassicalCondition leftcc,
    ClassicalCondition rightcc)
{
    return
        CExprFactory::
        GetFactoryInstance().GetCExprByOperation
        (
            leftcc.getExprPtr()->deepcopy(),
            rightcc.getExprPtr()->deepcopy(),
            LT
        );
}

ClassicalCondition  QPanda::operator<(ClassicalCondition class_cond,
    cbit_size_t value)
{
    auto &fac = CExprFactory::GetFactoryInstance();
    auto expr = fac.GetCExprByValue(value);
    if (expr == nullptr)
    {
        QCERR("CExpr factory fails");
        throw runtime_error("CExpr factory fails");
    }

    return CExprFactory::GetFactoryInstance().GetCExprByOperation(
        class_cond.getExprPtr()->deepcopy(),
        expr->deepcopy(),
        LT);

}



ClassicalCondition QPanda::operator<=(
    ClassicalCondition leftcc,
    ClassicalCondition rightcc)
{
    return
        CExprFactory::
        GetFactoryInstance().GetCExprByOperation
        (
            leftcc.getExprPtr()->deepcopy(),
            rightcc.getExprPtr()->deepcopy(),
            ELT
        );
}

ClassicalCondition  QPanda::operator<=(ClassicalCondition class_cond,
    cbit_size_t value)
{
    auto &fac = CExprFactory::GetFactoryInstance();
    auto expr = fac.GetCExprByValue(value);
    if (expr == nullptr)
    {
        QCERR("CExpr factory fails");
        throw runtime_error("CExpr factory fails");
    }

    return CExprFactory::GetFactoryInstance().GetCExprByOperation(
        class_cond.getExprPtr()->deepcopy(),
        expr->deepcopy(),
        ELT);

}





ClassicalCondition QPanda::operator&&(
    ClassicalCondition leftcc,
    ClassicalCondition rightcc)
{
    return
        CExprFactory::
        GetFactoryInstance().GetCExprByOperation
        (
            leftcc.getExprPtr()->deepcopy(),
            rightcc.getExprPtr()->deepcopy(),
            AND
        );
}

ClassicalCondition  QPanda::operator&&(ClassicalCondition class_cond,
    cbit_size_t value)
{
    auto &fac = CExprFactory::GetFactoryInstance();
    auto expr = fac.GetCExprByValue(value);
    if (expr == nullptr)
    {
        QCERR("CExpr factory fails");
        throw runtime_error("CExpr factory fails");
    }

    return CExprFactory::GetFactoryInstance().GetCExprByOperation(
        class_cond.getExprPtr()->deepcopy(),
        expr->deepcopy(),
        AND);

}

ClassicalCondition QPanda::operator||(
    ClassicalCondition leftcc,
    ClassicalCondition rightcc)
{
    return
        CExprFactory::
        GetFactoryInstance().GetCExprByOperation
        (
            leftcc.getExprPtr()->deepcopy(),
            rightcc.getExprPtr()->deepcopy(),
            OR
        );
}

ClassicalCondition  QPanda::operator||(ClassicalCondition class_cond,
    cbit_size_t value)
{
    auto &fac = CExprFactory::GetFactoryInstance();
    auto expr = fac.GetCExprByValue(value);
    if (expr == nullptr)
    {
        QCERR("CExpr factory fails");
        throw runtime_error("CExpr factory fails");
    }

    return CExprFactory::GetFactoryInstance().GetCExprByOperation(
        class_cond.getExprPtr()->deepcopy(),
        expr->deepcopy(),
        OR);

}


ClassicalCondition QPanda::operator!(
    ClassicalCondition leftcc)
{
    return
        CExprFactory::

        GetFactoryInstance().GetCExprByOperation
        (
            leftcc.getExprPtr()->deepcopy(),
            nullptr,
            NOT
        );
}

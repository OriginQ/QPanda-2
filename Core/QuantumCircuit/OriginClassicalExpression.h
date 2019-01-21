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

#ifndef ORIGIN_CLASSICAL_EXPRESSION_H
#define ORIGIN_CLASSICAL_EXPRESSION_H

//#include "QuantumCircuit/ClassicalConditionInterface.h"
#include "CBitFactory.h"
#include "CExprFactory.h"
#include "QNode.h"
USING_QPANDA

class OriginCExpr :public QNode,public CExpr
{
public:
    union content_u
    {
        CBit* cbit;
        int iOperatorSpecifier;
        cbit_size_t const_value;
    };

    NodeType m_node_type;
    qmap_size_t m_postion;
private:
    CExpr* leftExpr = nullptr;
    CExpr* rightExpr = nullptr;
    int contentSpecifier;
    content_u content;
    OriginCExpr();
public:
    OriginCExpr(CBit* cbit);
    OriginCExpr(CExpr* leftExpr, CExpr* rightExpr, int);
    OriginCExpr(cbit_size_t);
    CExpr * getLeftExpr() const;
    CExpr *getRightExpr() const;
    std::string getName() const;
    CBit* getCBit() const;
    void setLeftExpr(CExpr*);
    void setRightExpr(CExpr*);
    cbit_size_t eval() const;
    CExpr* deepcopy() const;
    bool checkValidity() const;

    NodeType getNodeType() const;
    qmap_size_t getPosition() const;
    void setPosition(qmap_size_t);
    ~OriginCExpr();
};

#endif
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

#ifndef ORIGIN_CLASSICAL_EXPRESSION_H
#define ORIGIN_CLASSICAL_EXPRESSION_H
#include "Core/QuantumMachine/CBitFactory.h"
#include "Core/QuantumCircuit/CExprFactory.h"
#include "Core/QuantumCircuit/QNode.h"
QPANDA_BEGIN

/**
* @brief Implementation  class  of  CExpr
* @ingroup QuantumCircuit
*/
class OriginCExpr :public CExpr
{
public:
    union content_u
    {
        CBit* cbit;
        int iOperatorSpecifier;
        cbit_size_t const_value;
    };

    NodeType m_node_type;  /**< quantum node type*/
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
	void getCBitsName(std::vector<std::string> & names);

	/**
     * @brief get quantum node type
     * @return NodeType
     */
    NodeType getNodeType() const;
    qmap_size_t getPosition() const;
    void setPosition(qmap_size_t);

	/**
	 * @brief get content specifier
	 * @return NodeType
	 */
    int getContentSpecifier() const;

    ~OriginCExpr();
};
QPANDA_END
#endif
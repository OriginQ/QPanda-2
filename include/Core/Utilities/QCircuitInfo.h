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

add by zhaody

*/

#ifndef _QCIRCUIT_INFO_H
#define _QCIRCUIT_INFO_H

#include "Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/Utilities/Traversal.h"
#include "Core/QuantumCircuit/QNode.h"
#include "Core/Utilities/QProgToDAG/QProgDAG.h"
#include "Core/Utilities/QProgToDAG/GraphMatch.h"

QPANDA_BEGIN

#define QCERR_AND_THROW_ERRSTR(x) {\
    std::cerr<<__FILE__<<" " <<__LINE__<<" "<<__FUNCTION__<<" " <<(x)<<std::endl;\
    throw runtime_error(#x);}

template<class T>
class TraversalNodeIter : public TraversalInterface<bool&>
{
public:
    TraversalNodeIter(QProg &prog, T &t)
        :m_prog(prog), m_T(t)
    {
    }
    ~TraversalNodeIter() {}

public:
    virtual void execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node, bool &) {}
    virtual void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, bool &) {}
    virtual void execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, bool &is_dagger) {
		m_T.handleSubProgNodeIter(cur_node, parent_node, is_dagger);
    }
    virtual void execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node, bool &is_dagger) {
		m_T.handleSubProgNodeIter(cur_node, parent_node, is_dagger);
    }
    virtual void execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node, bool &is_dagger) {
		m_T.handleSubProgNodeIter(cur_node, parent_node, is_dagger);
    }
    virtual void execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node, bool &) {}

    void traversalQProg()
    {
        bool isDagger = false;
        Traversal::traversalByType(m_prog.getImplementationPtr(), nullptr, *this, isDagger);
    }

protected:
    template<typename node_T>
    void continueTraversal(node_T subPgogNode, std::shared_ptr<QNode> parent_node, bool &is_dagger);

private:
    QProg &m_prog;
    T &m_T;
};

class PrintAllNodeType : public TraversalNodeIter<PrintAllNodeType>
{
public:
	PrintAllNodeType(QProg &srcProg)
		:TraversalNodeIter(srcProg, *this)
	{}
	~PrintAllNodeType() {}

	template<typename node_T>
	void handleSubProgNodeIter(node_T subPgogNode, std::shared_ptr<QNode> parent_node, bool &is_dagger) {
		printNodeType(subPgogNode, parent_node, is_dagger);
	}

	template<typename node_T>
	void printNodeType(node_T subPgogNode, std::shared_ptr<QNode> parent_node, bool &is_dagger);

private:

};

class PickUpNodes : public TraversalNodeIter<PickUpNodes>
{
public:
	PickUpNodes(QProg &outputProg, QProg &srcProg, const NodeIter &nodeItrStart, const NodeIter &nodeItrEnd)
		:TraversalNodeIter(srcProg, *this), m_output_prog(outputProg),
		m_start_iter(nodeItrStart),
		m_end_iter(nodeItrEnd),
		m_b_picking(false), m_b_pickup_end(false), m_b_pick_measure_node(false)
	{}
	~PickUpNodes() {}

	template<typename node_T>
	void handleSubProgNodeIter(node_T subPgogNode, std::shared_ptr<QNode> parent_node, bool &is_dagger) {
		pickUp(subPgogNode, parent_node, is_dagger);
	}

	template<typename node_T>
	void pickUp(node_T subPgogNode, std::shared_ptr<QNode> parent_node, bool &is_dagger);

	void setPickUpMeasureNode(bool b) { m_b_pick_measure_node = b; }

private:
	bool m_b_pick_measure_node;
	QProg &m_output_prog;
	const NodeIter m_start_iter;
	const NodeIter m_end_iter;
	bool m_b_picking;
	bool m_b_pickup_end;
};

class JudgeTwoNodeIterIsSwappable : public TraversalNodeIter<JudgeTwoNodeIterIsSwappable>
{
    enum ResultStatue
    {
        INIT = 0,
        NEED_JUDGE_LAYER,
        CAN_NOT_BE_EXCHANGED,
        COULD_BE_EXCHANGED
    };

public:
    JudgeTwoNodeIterIsSwappable(QProg &prog, NodeIter &nodeItr1, NodeIter &nodeItr2)
        : TraversalNodeIter(prog, *this), m_nodeItr1(nodeItr1), m_nodeItr2(nodeItr2),
		m_result(INIT), m_b_found_first_iter(false), m_b_found_second_iter(false), m_need_layer_node_type(NODE_UNDEFINED){}
    ~JudgeTwoNodeIterIsSwappable() {}

    template<typename node_T>
    void handleSubProgNodeIter(node_T subPgogNode, std::shared_ptr<QNode> parent_node, bool &is_dagger) {
        judgeNodeIters(subPgogNode, parent_node, is_dagger);
    }

    bool getResult();

    /**
    * @brief 
    * @param[out] result if the two Iters on the same layer and could be exchanged, result=true, or else false.
    * @return if any any error happened, return <0 ,else return 0
    */
    int judgeLayerInfo(bool &result);

private:
    template<typename node_T>
    void judgeNodeIters(node_T subPgogNode, std::shared_ptr<QNode> parent_node, bool &is_dagger);
    int getNodeIndex(const NodeIter &iter);

private:
	ResultStatue m_result;
    NodeIter m_nodeItr1; 
    NodeIter m_nodeItr2;
    bool m_b_found_first_iter;
    bool m_b_found_second_iter;
    std::shared_ptr<QNode> m_need_layer_node;
    NodeType m_need_layer_node_type;
};

class AdjacentQGates : public TraversalNodeIter<AdjacentQGates>
{
    enum TraversalStatue
    {
        HAVE_NOT_FOUND_TARGET_NODE = 0, // 0: init satue(haven't found the target node)
        TO_FIND_BACK_NODE, // 1: found the target node,
        FOUND_ALL_ADJACENT_NODE //  2: found enough
    };
public:
    AdjacentQGates(QProg &prog, NodeIter &nodeItr)
        :TraversalNodeIter(prog, *this)
        , m_traversal_flag(HAVE_NOT_FOUND_TARGET_NODE)
        , m_target_node_itr(nodeItr)
        , m_prog(prog)
    {}
    ~AdjacentQGates() {}

    void updateFrontIter(NodeIter &itr) { m_front_iter = itr; }
    void updateBackIter(const NodeIter &itr) { m_back_iter = itr; }
    GateType getFrontIterNodeType() {
        if (HAVE_NOT_FOUND_TARGET_NODE == m_traversal_flag)
        {
            return GATE_UNDEFINED;
        }
        return getItrNodeType(m_front_iter);
    }
    GateType getBackIterNodeType() { return getItrNodeType(m_back_iter); }

    GateType getItrNodeType(const NodeIter &ter);

    std::string getItrNodeTypeStr(const NodeIter &ter);
    std::string getBackIterNodeTypeStr() { return getItrNodeTypeStr(m_back_iter); }
    std::string getFrontIterNodeTypeStr() {
        if (HAVE_NOT_FOUND_TARGET_NODE == m_traversal_flag)
        {
            return std::string("Null");
        }
        return getItrNodeTypeStr(m_front_iter);
    }

    static bool isSubProgNode(const std::shared_ptr<QNode> &node) {
        const NodeType t = node->getNodeType();
        return  ((t == CIRCUIT_NODE) || (t == PROG_NODE));
    }
    static bool isFlowCtrlNode(const std::shared_ptr<QNode> &node) {
        const NodeType t = node->getNodeType();
        return  ((t == WHILE_START_NODE) || (t == QIF_START_NODE));
    }

    template<typename node_T>
    bool isSubNodeEmpty(node_T subPgogNode);

    bool isValidNodeType(const NodeIter &itr) { return isValidNodeType((*itr)->getNodeType()); }
    bool isValidNodeType(const NodeType t) { return ((GATE_NODE == t) || (MEASURE_GATE == t)); }

    const NodeIter& getFrontIter() { return m_front_iter; }
    const NodeIter& getBackIter() { return m_back_iter; }

public:
    template<typename node_T>
    void handleSubProgNodeIter(node_T subPgogNode, std::shared_ptr<QNode> parent_node, bool &is_dagger) {
        findTargetNodeItr2(subPgogNode, parent_node, is_dagger);
    }

private:
    template<typename node_T>
    void findTargetNodeItr2(node_T subPgogNode, std::shared_ptr<QNode> parent_node, bool &is_dagger);

    template<typename node_T>
    void onHaveNotFoundTargetIter(node_T subPgogNode, std::shared_ptr<QNode> parent_node, bool &is_dagger);
    template<typename node_T>
    void onToFindBackGateNode(node_T subPgogNode, std::shared_ptr<QNode> parent_node, bool &is_dagger);
    template<typename node_T>
    void onFoundAllAdjacentNode(node_T subPgogNode, std::shared_ptr<QNode> parent_node, bool &is_dagger);

private:
    QProg &m_prog;
    const NodeIter m_target_node_itr;
    NodeIter m_front_iter;
    NodeIter m_cur_iter;
    NodeIter m_back_iter;
	TraversalStatue m_traversal_flag;
    std::shared_ptr<QNode> m_last_parent_node_itr;
};

class MatrixMathFunction
{
public:
	typedef struct _matrix_block
	{
		_matrix_block() 
			:m_row_index(0), m_column_index(0)
		{}

		int m_row_index;
		int m_column_index;
		QStat m_mat;
	}matrixBlock_t;

	typedef struct _blocked_matrix
	{
		_blocked_matrix()
			:m_block_rows(0), m_block_columns(0)
		{}

		int m_block_rows;
		int m_block_columns;
		std::vector<matrixBlock_t> m_vec_block;
	}blockedMatrix_t;

public:
	MatrixMathFunction() {}
	~MatrixMathFunction() {}

	static QStat multip(const QStat& leftMatrix, const QStat& rightMatrix);
	static QStat ZhangMultip(const QStat& leftMatrix, const QStat& rightMatrix);
	static int partition(const QStat& srcMatrix, int partitionNum, int partitionColumnNum, blockedMatrix_t& blockedMat);
	static int blockMultip(const QStat& leftMatrix, const blockedMatrix_t& blockedMat, QStat& resultMatrix);
	static qcomplex_t complexMultip(qcomplex_t c1, qcomplex_t c2) { return c1*=c2; }
};

class QprogToMatrix
{
	using gateQubitInfo_t = std::vector<std::pair<std::shared_ptr<AbstractQGateNode>, std::vector<int>>>;
	using gateAndQubitsItem_t = std::pair<std::shared_ptr<AbstractQGateNode>, std::vector<int>>;
	using calcUnitVec_t = std::vector<std::pair<QStat, std::vector<int>>>;

public:
	QprogToMatrix(QProg& p)
		:m_prog(p), m_mat_I{ 1, 0, 0, 1 }
	{}
	~QprogToMatrix() {}

	QStat getMatrix();
	QStat getMatrixOfOneLayer(SequenceLayer& layer, const QProgDAG& progDag);
	QStat reverseCtrlGateMatrix(QStat& srcMat);
	void getStrideOverQubits(const std::vector<int> &qgateUsedQubits, std::vector<int> &strideOverQubits);
	void mergeToCalcUnit(std::vector<int>& qubits, QStat& gateMat, calcUnitVec_t &calcUnitVec, gateQubitInfo_t &singleQubitGates);
	void zhangMultipMatrix(QStat& srcMat, const QStat& zhangMat);
	void zhangMultipQGate(QStat& srcMat, std::shared_ptr<AbstractQGateNode> &pGate);

private:
	QProg& m_prog;
	GraphMatch m_dag;
	TopologincalSequence m_seq;
	std::vector<int> m_qubits_in_use;
	const QStat m_mat_I;
};


    /**
    * @brief  judge the Qgate if match the target topologic structure of quantum circuit
    * @param[in]  vector<vector<int>>& the target topologic structure of quantum circuit
    * @return     if the Qgate match the target topologic structure return true, or else return false
    * @see XmlConfigParam::readAdjacentMatrix(TiXmlElement *, int&, std::vector<std::vector<int>>&)
    */
    bool isMatchTopology(const QGate& gate, const std::vector<std::vector<int>>& vecTopoSt);

    /**
    * @brief  get the adjacent quantum gates's(the front one and the back one) type
    * @param[in] nodeItr  the specialed NodeIter
    * @param[out] std::vector<NodeIter> frontAndBackIter the front iter and the back iter
    * @return result string.
    * @see
    */
    std::string getAdjacentQGateType(QProg &prog, NodeIter &nodeItr, std::vector<NodeIter>& frontAndBackIter);

    /**
    * @brief  judge the specialed two NodeIters whether can be exchanged
    * @param[in] nodeItr1 the first NodeIter
    * @param[in] nodeItr2 the second NodeIter
    * @return if the two NodeIters can be exchanged, return true, otherwise retuen false.
    * @see
    */
    bool isSwappable(QProg &prog, NodeIter &nodeItr1, NodeIter &nodeItr2);

    /**
    * @brief  judge if the target node is a base QGate type
    * @param[in] nodeItr the target NodeIter
    * @return if the target node is a base QGate type, return true, otherwise retuen false.
    * @see
    */
    bool isSupportedGateType(const NodeIter &nodeItr);

	/**
	* @brief  get the target matrix between the input two Nodeiters
	* @param[in] nodeItrStart the start NodeIter
	* @param[in] nodeItrEnd the end NodeIter
	* @return the target matrix include all the QGate's matrix (multiply).
	* @see
	*/
    QStat getMatrix(QProg srcProg, const NodeIter nodeItrStart = NodeIter(), const NodeIter nodeItrEnd = NodeIter());
	QStat getMatrix(QCircuit srcProg, const NodeIter nodeItrStart = NodeIter(), const NodeIter nodeItrEnd = NodeIter());

	/**
	* @brief  output matrix information to consol
	* @param[in] the target matrix
	*/
	void printMat(const QStat& mat);

	/**
	* @brief  output all the node type of the target prog
	* @param[in] the target prog
	*/
	void printAllNodeType(QProg &prog);

	/**
	* @brief  pick up the nodes of srcProg between nodeItrStart and  nodeItrEnd to outPutProg
	* @param[out] outPutProg  the output prog
	* @param[in] srcProg The source prog
	* @param[in] nodeItrStart The start pos of source prog
	* @param[in] nodeItrEnd The end pos of source prog
	* @ Note: If there are any Qif/Qwhile nodes between nodeItrStart and nodeItrEnd, 
	          Or the nodeItrStart and the nodeItrEnd are in different sub-circuit, an exception will be throw.
	*/
	void pickUpNode(QProg &outPutProg, QProg &srcProg, const NodeIter nodeItrStart = NodeIter(), const NodeIter nodeItrEnd = NodeIter(), bool bPickMeasure = false);

	/**
	* @brief  Get all the used  quantum bits in the input prog
	* @param[in] prog  the input prog
	* @param[out] vecQuBitsInUse The vector of used quantum bits
	* @ Note: All the Qif/Qwhile or other sub-circuit nodes in the input prog will be ignored.
	*/
	void getAllUsedQuBits(QProg &prog, std::vector<int> &vecQuBitsInUse);

	/**
	* @brief  Get all the used  class bits in the input prog
	* @param[in] prog  the input prog
	* @param[out] vecClBitsInUse The vector of used class bits
	* @ Note: All the Qif/Qwhile or other sub-circuit nodes in the input prog will be ignored.
	*/
	void getAllUsedClassBits(QProg &prog, std::vector<int> &vecClBitsInUse);

QPANDA_END
#endif
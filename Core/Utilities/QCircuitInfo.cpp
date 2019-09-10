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

#include "QCircuitInfo.h"
#include "Core/Utilities/QProgToDAG/GraphMatch.h"
#include "Core/Utilities/QuantumMetadata.h"
#include <algorithm>
#include "QPanda.h"

USING_QPANDA
using namespace std;

#define ENUM_TO_STR(x) #x

#define PRINT_TRACE 0
#if PRINT_TRACE
#define PTrace printf
#define PTraceMat(mat) printMat(mat)
#else
#define PTrace
#define PTraceMat(mat)
#endif

template<class T>
template<typename node_T>
void TraversalNodeIter<T>::continueTraversal(node_T subPgogNode, std::shared_ptr<QNode> parent_node, bool &is_dagger)
{
    const NodeType curT = (std::dynamic_pointer_cast<QNode>(subPgogNode))->getNodeType();
    if (GATE_NODE == curT)
    {
        return;
    }
    else if ((curT == CIRCUIT_NODE) || (PROG_NODE == curT))
    {
        Traversal::traversalByType(subPgogNode, std::dynamic_pointer_cast<QNode>(subPgogNode), *this, is_dagger);
    }
    else if ((curT == WHILE_START_NODE) || (curT == QIF_START_NODE))
    {
		PTrace("Enter flow control node\n ");
        Traversal::traversal(std::dynamic_pointer_cast<AbstractControlFlowNode>(subPgogNode), *this, is_dagger);
    }
    else if ((MEASURE_GATE == curT) || (CLASS_COND_NODE == curT))
    {
        //How to deal with these nodes ??

        return;
    }
    else
    {
        // error
        QCERR_AND_THROW_ERRSTR("error node type.");
        return;
    }
}

bool QPanda::isMatchTopology(const QGate& gate, const std::vector<std::vector<int>>& vecTopoSt)
{
    if (0 == vecTopoSt.size())
    {
        return false;
    }
    QVec vec_qubits;
    gate.getQuBitVector(vec_qubits);

    size_t first_qubit_pos = vec_qubits.front()->getPhysicalQubitPtr()->getQubitAddr();
    if (vecTopoSt.size() <= first_qubit_pos)
    {
        return false;
    }

    int pos_in_topology = first_qubit_pos; //the index of qubits in topological structure is start from 1.
    std::vector<int> vec_topology = vecTopoSt[pos_in_topology];
    for (auto iter = ++(vec_qubits.begin()); iter != vec_qubits.end(); ++iter)
    {
        auto target_qubit = (*iter)->getPhysicalQubitPtr()->getQubitAddr();
        if (vecTopoSt.size() <= target_qubit)
        {
            //cout << target_qubit << endl;
            return false;
        }
        if (0 == vec_topology[target_qubit])
        {
            return false;
        }
    }
    return true;
}

std::string QPanda::getAdjacentQGateType(QProg &prog, NodeIter &nodeItr, std::vector<NodeIter>& frontAndBackIter)
{
    std::shared_ptr<AdjacentQGates> p_adjacent_QGates = std::make_shared<AdjacentQGates>(prog, nodeItr);
    if (nullptr == p_adjacent_QGates)
    {
        QCERR_AND_THROW_ERRSTR("Failed to create adjacent object, memory error.");
        return std::string("Error");
    }

    //Judging whether the target nodeItr is Qgate or no
    if ((GATE_UNDEFINED == p_adjacent_QGates->getItrNodeType(nodeItr)))
    {
        // target node type error
        QCERR_AND_THROW_ERRSTR("The target node is not a Qgate.");
        return std::string("Error");
    }

    std::shared_ptr<TraversalNodeIter<AdjacentQGates>> p_traversal = std::make_shared<TraversalNodeIter<AdjacentQGates>>(prog, *p_adjacent_QGates);
	p_traversal->traversalQProg();
    
    frontAndBackIter.clear();
    frontAndBackIter.push_back(p_adjacent_QGates->getFrontIter());
    frontAndBackIter.push_back(p_adjacent_QGates->getBackIter());

    std::string ret = std::string("frontNodeType = ") + p_adjacent_QGates->getFrontIterNodeTypeStr()
        + std::string(", backNodeType = ") + p_adjacent_QGates->getBackIterNodeTypeStr();

    return ret;
}

template<typename node_T>
void AdjacentQGates::onHaveNotFoundTargetIter(node_T subPgogNode, std::shared_ptr<QNode> parent_node, bool &is_dagger)
{
    //update front, if QWhileNode clear front
    NodeType parent_node_type = NODE_UNDEFINED;
    if (nullptr != parent_node.get())
    {
		parent_node_type = parent_node->getNodeType();
        if (WHILE_START_NODE == parent_node_type)
        {
            m_front_iter.setPCur(nullptr);
        }
    }

    std::shared_ptr<QNode> cur_node = std::dynamic_pointer_cast<QNode>(subPgogNode);
    const NodeType cur_node_type = cur_node->getNodeType();

    // if flowCtrl node, traversal flowCtrlNode
    if ((cur_node_type == WHILE_START_NODE) || (cur_node_type == QIF_START_NODE))
    {
		PTrace("Enter flow control node\n ");
        Traversal::traversal(std::dynamic_pointer_cast<AbstractControlFlowNode>(subPgogNode), *this, is_dagger);
    }
    else if ((cur_node_type == CIRCUIT_NODE) || (cur_node_type == PROG_NODE))
    {
        // if subNode, handle subNode
        NodeIter cur_iter, end_itr;
        if (cur_node_type == CIRCUIT_NODE)
        {
			cur_iter = (std::dynamic_pointer_cast<AbstractQuantumCircuit>(subPgogNode))->getFirstNodeIter();
			end_itr = (std::dynamic_pointer_cast<AbstractQuantumCircuit>(subPgogNode))->getEndNodeIter();
        }
        else
        {
			cur_iter = (std::dynamic_pointer_cast<AbstractQuantumProgram>(subPgogNode))->getFirstNodeIter();
			end_itr = (std::dynamic_pointer_cast<AbstractQuantumProgram>(subPgogNode))->getEndNodeIter();
        }

        while (cur_iter != end_itr)
        {
            if (FOUND_ALL_ADJACENT_NODE == m_traversal_flag)
            {
                return;
            }
            else if (TO_FIND_BACK_NODE == m_traversal_flag)
            {
                /*
                   if gateNode, update back, flag = 2, return;
                */
                NodeIter tmp_itr = cur_iter;
				cur_iter = cur_iter.getNextIter();
                if (isValidNodeType(tmp_itr))
                {
                    updateBackIter(tmp_itr);
                    m_traversal_flag = FOUND_ALL_ADJACENT_NODE;
					PTrace("Found all nodes\n ");
                    return;
                }
                else
                {
                    //continue traversal
                    continueTraversal(*tmp_itr, parent_node, is_dagger);
                }
            }
            else if (HAVE_NOT_FOUND_TARGET_NODE == m_traversal_flag)
            {
                /* find target
                   if found target, flage = 1; continue
                */
                if (m_target_node_itr == cur_iter)
                {
                    m_traversal_flag = TO_FIND_BACK_NODE;
					cur_iter = cur_iter.getNextIter();
                    continue;
                }

                //if gate type
                NodeIter tmp_itr = cur_iter;
				cur_iter = cur_iter.getNextIter();
                if (isValidNodeType(tmp_itr))
                {
                    updateFrontIter(tmp_itr);

                    //test
                    if ((*tmp_itr)->getNodeType() == MEASURE_GATE)
                    {
						PTrace(">>measureGate ");
                    }
                    else
                    {
                        GateType gt = getItrNodeType(tmp_itr);
						PTrace(">>gatyT=%d ", gt);
                    }
                }
                else
                {
                    //continue traversal
                    continueTraversal(*tmp_itr, parent_node, is_dagger);
                }
            }
        }
    }
    else
    {
        // error type
        return;
    }

    if (HAVE_NOT_FOUND_TARGET_NODE == m_traversal_flag)
    {
        if ((WHILE_START_NODE == parent_node_type))
        {
            m_front_iter.setPCur(nullptr);
        }
    }
    else if (TO_FIND_BACK_NODE == m_traversal_flag)
    {
        if ((WHILE_START_NODE == parent_node_type))
        {
            m_back_iter.setPCur(nullptr);
            m_traversal_flag = FOUND_ALL_ADJACENT_NODE;
        }
    }

    m_last_parent_node_itr = parent_node;
}

template<typename node_T>
void AdjacentQGates::onToFindBackGateNode(node_T subPgogNode, std::shared_ptr<QNode> parent_node, bool &is_dagger)
{
    //if ctrlNode , clear back, flag = 2, return;
    if (nullptr != parent_node.get())
    {
        const NodeType parent_node_type = parent_node->getNodeType();
        if ((WHILE_START_NODE == parent_node_type) || (QIF_START_NODE == parent_node_type))
        {
            if (QIF_START_NODE == parent_node_type)
            {
                // The false_branch_node of Qif 
                if (m_last_parent_node_itr.get() == parent_node.get())
                {
                    return;
                }
            }
            m_back_iter.setPCur(nullptr);
            m_traversal_flag = FOUND_ALL_ADJACENT_NODE;
            return;
        }
    }


    std::shared_ptr<QNode> cur_node = std::dynamic_pointer_cast<QNode>(subPgogNode);
    NodeType cur_node_type = cur_node->getNodeType();
    
    //if subNode, traversal
    NodeIter cur_iter, end_itr;
    if (cur_node_type == CIRCUIT_NODE)
    {
		cur_iter = (std::dynamic_pointer_cast<AbstractQuantumCircuit>(subPgogNode))->getFirstNodeIter();
		end_itr = (std::dynamic_pointer_cast<AbstractQuantumCircuit>(subPgogNode))->getEndNodeIter();
    }
    else if (cur_node_type == PROG_NODE)
    {
		cur_iter = (std::dynamic_pointer_cast<AbstractQuantumProgram>(subPgogNode))->getFirstNodeIter();
		end_itr = (std::dynamic_pointer_cast<AbstractQuantumProgram>(subPgogNode))->getEndNodeIter();
    }
    else
    {
        QCERR_AND_THROW_ERRSTR("error node type.");
        return;
    }

    while (cur_iter != end_itr)
    {
        NodeIter tmp_itr = cur_iter;
		cur_iter = cur_iter.getNextIter();

        const NodeType curT = (*tmp_itr)->getNodeType();
        if (isValidNodeType(curT))
        {
            updateBackIter(tmp_itr);
            m_traversal_flag = FOUND_ALL_ADJACENT_NODE;
            return;
        }
        else if ((curT == WHILE_START_NODE) || (curT == QIF_START_NODE))
        {
            updateBackIter(NodeIter());
            m_traversal_flag = FOUND_ALL_ADJACENT_NODE;
            return;
        }
        else if ((curT == CIRCUIT_NODE) || (curT == PROG_NODE))
        {
            //if subNode, continue traversalByType
            continueTraversal(*tmp_itr, parent_node, is_dagger);
        }
        else
        {
            QCERR_AND_THROW_ERRSTR("error node type.");
            return;
        }

        if (FOUND_ALL_ADJACENT_NODE == m_traversal_flag)
        {
            return;
        }
    }

    //leave subNode
}

template<typename node_T>
void AdjacentQGates::onFoundAllAdjacentNode(node_T subPgogNode, std::shared_ptr<QNode> parent_node, bool &is_dagger)
{
    //do nothing
}

template<typename node_T>
bool AdjacentQGates::isSubNodeEmpty(node_T subPgogNode)
{
    std::shared_ptr<QNode> cur_node = std::dynamic_pointer_cast<QNode>(subPgogNode);
    const NodeType t = cur_node->getNodeType();

    if ((t == WHILE_START_NODE) || (t == QIF_START_NODE))
    {
        return false;
    }
    else if ((t == CIRCUIT_NODE))
    {
        std::shared_ptr<AbstractQuantumCircuit> circuit = std::dynamic_pointer_cast<AbstractQuantumCircuit>(subPgogNode);
        auto cur_iter = circuit->getFirstNodeIter();
        auto end_itr = circuit->getEndNodeIter();
        if (cur_iter == end_itr)
        {
            return true;
        }
    }
    else if (t == PROG_NODE)
    {
        std::shared_ptr<AbstractQuantumProgram> prog = std::dynamic_pointer_cast<AbstractQuantumProgram>(subPgogNode);
        auto cur_iter = prog->getFirstNodeIter();
        auto end_itr = prog->getEndNodeIter();
        if (cur_iter == end_itr)
        {
            return true;
        }
    }
    else
    {
        //unknow node type
        QCERR_AND_THROW_ERRSTR("Unknown node type.");
    }

    return false;
}

template<typename node_T>
void AdjacentQGates::findTargetNodeItr2(node_T subPgogNode, std::shared_ptr<QNode> parent_node, bool &is_dagger)
{
    //if subPgogNode is empty  ,return
    if (isSubNodeEmpty(subPgogNode))
        return;

    switch (m_traversal_flag)
    {
    case HAVE_NOT_FOUND_TARGET_NODE:
        onHaveNotFoundTargetIter(subPgogNode, parent_node, is_dagger);
        break;

    case TO_FIND_BACK_NODE:
        onToFindBackGateNode(subPgogNode, parent_node, is_dagger);
        break;

    case FOUND_ALL_ADJACENT_NODE:
        onFoundAllAdjacentNode(subPgogNode, parent_node, is_dagger);
        break;

    default:
        break;
    }
}

GateType AdjacentQGates::getItrNodeType(const NodeIter &ter)
{
    std::shared_ptr<QNode> tmp_node = *(ter);
    if (nullptr != tmp_node)
    {
        if (GATE_NODE == tmp_node->getNodeType())
        {
            std::shared_ptr<OriginQGate> gate = std::dynamic_pointer_cast<OriginQGate>(tmp_node);
            return (GateType)(gate->getQGate()->getGateType());
        }
    }
    else
    {
        cout << "nullptr" << endl;
    }


    return GATE_UNDEFINED;
}

std::string AdjacentQGates::getItrNodeTypeStr(const NodeIter &ter)
{
    std::shared_ptr<QNode> tmp_node = *(ter);
    if (nullptr != tmp_node.get())
    {
        const NodeType t = tmp_node->getNodeType();
        if (t == GATE_NODE)
        {
            std::shared_ptr<OriginQGate> gate = std::dynamic_pointer_cast<OriginQGate>(tmp_node);
			return TransformQGateType::getInstance()[(GateType)(gate->getQGate()->getGateType())];
        }
        else if (t == MEASURE_GATE)
        {
            return std::string("MEASURE_GATE");
        }
    }

    return std::string("Null");
}

bool QPanda::isSwappable(QProg &prog, NodeIter &nodeItr1, NodeIter &nodeItr2)
{
	if (nodeItr1 == nodeItr2)
	{
		QCERR("Error: the two nodeIter is equivalent.");
		return false;
	}

    std::shared_ptr<JudgeTwoNodeIterIsSwappable> p_judge_node_iters = std::make_shared<JudgeTwoNodeIterIsSwappable>(prog, nodeItr1, nodeItr2);
    if (nullptr == p_judge_node_iters.get())
    {
        QCERR_AND_THROW_ERRSTR("Failed to create JudgeNodeIter object, memory error.");
        return false;
    }

    std::shared_ptr<TraversalNodeIter<JudgeTwoNodeIterIsSwappable>> pTraversal = std::make_shared<TraversalNodeIter<JudgeTwoNodeIterIsSwappable>>(prog, *p_judge_node_iters);
    pTraversal->traversalQProg();

    return p_judge_node_iters->getResult();
}

template<typename node_T>
void JudgeTwoNodeIterIsSwappable::judgeNodeIters(node_T subPgogNode, std::shared_ptr<QNode> parent_node, bool &is_dagger)
{
    std::shared_ptr<QNode> cur_node = std::dynamic_pointer_cast<QNode>(subPgogNode);
    const NodeType cur_node_type = cur_node->getNodeType();

    // if flowCtrl node, traversal flowCtrlNode
    if ((cur_node_type == WHILE_START_NODE) || (cur_node_type == QIF_START_NODE))
    {
        //enter flow control node
        Traversal::traversal(std::dynamic_pointer_cast<AbstractControlFlowNode>(subPgogNode), *this, is_dagger);
    }
    else if ((cur_node_type == CIRCUIT_NODE) || (cur_node_type == PROG_NODE))
    {
        // processing Nested nodes
        NodeIter cur_iter, end_itr;
        if (cur_node_type == CIRCUIT_NODE)
        {
			cur_iter = (std::dynamic_pointer_cast<AbstractQuantumCircuit>(subPgogNode))->getFirstNodeIter();
			end_itr = (std::dynamic_pointer_cast<AbstractQuantumCircuit>(subPgogNode))->getEndNodeIter();
        }
        else
        {
			cur_iter = (std::dynamic_pointer_cast<AbstractQuantumProgram>(subPgogNode))->getFirstNodeIter();
			end_itr = (std::dynamic_pointer_cast<AbstractQuantumProgram>(subPgogNode))->getEndNodeIter();
        }

        while (cur_iter != end_itr)
        {
            if (m_result > INIT)
            {
                return;
            }
            const NodeIter tmp_itr = cur_iter;
			cur_iter = cur_iter.getNextIter();
			const NodeType curT = (*tmp_itr)->getNodeType();
			if ((GATE_NODE == curT) || (MEASURE_GATE == curT))
            {
                if (tmp_itr == m_nodeItr1)
                {
					m_b_found_first_iter = true;
                }
                else if (tmp_itr == m_nodeItr2)
                {
                    m_b_found_second_iter = true;
                }

                if (m_b_found_first_iter && m_b_found_second_iter)
                {
					m_result = NEED_JUDGE_LAYER;
                    m_need_layer_node = cur_node;
                    m_need_layer_node_type = cur_node_type;
                }
            }
            else
            {
                if ((m_b_found_first_iter || m_b_found_second_iter)
                    && (m_b_found_first_iter != m_b_found_second_iter))
                {
					m_result = CAN_NOT_BE_EXCHANGED;
                    return;
                }

                //continue to traversal
                continueTraversal(*tmp_itr, parent_node, is_dagger);
            }
        }
    }
}

bool JudgeTwoNodeIterIsSwappable::getResult()
{
    bool ret = false;

    switch (m_result)
    {
    case QPanda::JudgeTwoNodeIterIsSwappable::INIT:
        //Error: cann't found the target nodeIter
        QCERR_AND_THROW_ERRSTR("Error: Cann't found the target nodeIter.");
        break;

    case QPanda::JudgeTwoNodeIterIsSwappable::NEED_JUDGE_LAYER:
        //judge layer info
        judgeLayerInfo(ret);
        break;

    case QPanda::JudgeTwoNodeIterIsSwappable::CAN_NOT_BE_EXCHANGED:
        ret = false;
        break;

    case QPanda::JudgeTwoNodeIterIsSwappable::COULD_BE_EXCHANGED:
        ret = true;
        break;

    default:
        QCERR_AND_THROW_ERRSTR("Error: unknow type.");
        break;
    }

    return ret;
}

int JudgeTwoNodeIterIsSwappable::judgeLayerInfo(bool &result)
{
    //get subNode
    QProg tmp_prog = QProg();
    NodeIter tmp_iter = m_nodeItr1;
    NodeType tmp_node_type = (*tmp_iter)->getNodeType();
    if (GATE_NODE == tmp_node_type)
    {
		tmp_prog << QGate(std::dynamic_pointer_cast<AbstractQGateNode>(*tmp_iter));
    }
    else if (MEASURE_GATE == tmp_node_type)
    {
		tmp_prog << QMeasure(std::dynamic_pointer_cast<AbstractQuantumMeasure>(*tmp_iter));
    }
    else
    {
        QCERR_AND_THROW_ERRSTR("Error: unknow type.");
        return -1;
    }

    auto sub_pickup_func = [&tmp_node_type, &tmp_iter, &tmp_prog](bool bForward = true) {
        while (true)
        {
			if (bForward)
			{
				if (NodeIter() == (--tmp_iter))
				{
					break;
				}
				tmp_node_type = (*tmp_iter)->getNodeType();
			}
			else
			{
				if (NodeIter() == (++tmp_iter).getNextIter())
				{
					break;
				}
				tmp_node_type = (*tmp_iter)->getNodeType();
			}
            
            if (GATE_NODE == tmp_node_type)
            {
                auto gate = QGate(std::dynamic_pointer_cast<AbstractQGateNode>(*tmp_iter));
                if (bForward)
                {
					tmp_prog.insertQNode(tmp_prog.getFirstNodeIter(), &gate);
                }
                else
                {
					tmp_prog << gate;
                }
            }
            else if (MEASURE_GATE == tmp_node_type)
            {
                auto measure = QMeasure(std::dynamic_pointer_cast<AbstractQuantumMeasure>(*tmp_iter));
                if (bForward)
                {
					tmp_prog.insertQNode(tmp_prog.getFirstNodeIter(), &measure);
                }
                else
                {
					tmp_prog << measure;
                }
            }
            else if ((CIRCUIT_NODE == tmp_node_type) || (PROG_NODE == tmp_node_type)
                || (QIF_START_NODE == tmp_node_type) || (WHILE_START_NODE == tmp_node_type)
                || (CLASS_COND_NODE == tmp_node_type))
            {
                break;
            }
        }
    };

	sub_pickup_func(false);
	tmp_iter = m_nodeItr1;
	sub_pickup_func();

    //get layer info
    QNodeMatch dag;
    TopologincalSequence seq;
    dag.getMainGraphSequence(tmp_prog, seq);

    int iter1_index = getNodeIndex(m_nodeItr1);
    int iter2_index = getNodeIndex(m_nodeItr2);
    int found_cnt = 0;
    for (auto &seq_item : seq)
    {
        for (auto &seq_node_item : seq_item)
        {
            if (iter1_index == (seq_node_item.first.m_vertex_num))
            {
                ++found_cnt;
            }

            if (iter2_index == (seq_node_item.first.m_vertex_num))
            {
                ++found_cnt;
            }
        }

        if (2 == found_cnt)
        {
			m_result = COULD_BE_EXCHANGED;
            result = true;
            return 0;
        }
        else if (1 == found_cnt)
        {
			m_result = CAN_NOT_BE_EXCHANGED;
            result = false;
            return 0;
        }
        else if (0 == found_cnt)
        {
            continue;
        }
        else
        {
            QCERR_AND_THROW_ERRSTR("Error: unknow error.");
            return -1;
        }
    }

    QCERR_AND_THROW_ERRSTR("Error: get layer error.");
    return -1;
}

int JudgeTwoNodeIterIsSwappable::getNodeIndex(const NodeIter &iter)
{
    int ret = 0;
    NodeIter tmp_iter = iter;
    while (nullptr != *(--tmp_iter))
    {
        const NodeType t = (*tmp_iter)->getNodeType();
        if ((CIRCUIT_NODE == t) || (PROG_NODE == t)
            || (WHILE_START_NODE == t) || (QIF_START_NODE == t))
        {
            break;
        }
        ++ret;
    }

    return ret;
}

bool QPanda::isSupportedGateType(const NodeIter &nodeItr)
{
    //read meta data
    QuantumMetadata meta_data;
    std::vector<std::string> vec_single_gates;
    std::vector<string> vec_double_gates;
	meta_data.getQGate(vec_single_gates, vec_double_gates);

    //judge
    string gate_type_str;
    NodeType tmp_node_type = (*nodeItr)->getNodeType();
    if (GATE_NODE == tmp_node_type)
    {
        std::shared_ptr<OriginQGate> gate = std::dynamic_pointer_cast<OriginQGate>(*nodeItr);
		gate_type_str = TransformQGateType::getInstance()[(GateType)(gate->getQGate()->getGateType())];
    }
    else
    {
        QCERR_AND_THROW_ERRSTR("Error: The target node is NOT a QGate.");
        return false;
    }

    std::transform(gate_type_str.begin(), gate_type_str.end(), gate_type_str.begin(), ::tolower);
    for (auto itr : vec_single_gates)
    {
        std::transform(itr.begin(), itr.end(), itr.begin(), ::tolower);
        if (0 == strcmp(gate_type_str.c_str(), itr.c_str()))
        {
            return true;
        }
    }

    for (auto itr : vec_double_gates)
    {
        std::transform(itr.begin(), itr.end(), itr.begin(), ::tolower);
        if (0 == strcmp(gate_type_str.c_str(), itr.c_str()))
        {
            return true;
        }
    }

    return false;
}

QStat MatrixMathFunction::multip(const QStat& leftMatrix, const QStat& rightMatrix)
{
	if (leftMatrix.size() == 0)
	{
		return rightMatrix;
	}

	QStat result_matrix;

	//get rows  and columns
	double left_rows = sqrt(leftMatrix.size());
	double right_columns = sqrt(rightMatrix.size());
	if (abs(left_rows - right_columns) > 0.8)
	{
		QCERR_AND_THROW_ERRSTR("Error: The two input matrixs cann't multip, they have different rows and columns.");
		return result_matrix;
	}

	QStat tmp_row;
	QStat tmp_column;
	qcomplex_t tmp_num;
	for (size_t row = 0; row < left_rows; ++row)
	{
		for (size_t column = 0; column < right_columns; ++column)
		{
			tmp_row.clear();
			tmp_column.clear();
			for (size_t i = 0; i < right_columns; ++i)
			{
				tmp_row.push_back(leftMatrix[row*left_rows + i]); //get the row items of leftMatrix
				tmp_column.push_back(rightMatrix[i*right_columns + column]); //get the column items of rightMatrix
			}

			tmp_num = 0;
			for (size_t tmpRowIndex = 0; tmpRowIndex < right_columns; tmpRowIndex++)
			{
				tmp_num += complexMultip(tmp_row[tmpRowIndex], tmp_column[tmpRowIndex]);
			}

			result_matrix.push_back(tmp_num);
		}
	}

	return result_matrix;
}

QStat MatrixMathFunction::ZhangMultip(const QStat& leftMatrix, const QStat& rightMatrix)
{
	QStat result_matrix;

	//get rows  and columns
	double left_rows = sqrt(leftMatrix.size());
	double right_columns = sqrt(rightMatrix.size());

	result_matrix.resize(leftMatrix.size()*rightMatrix.size());
	int left_row = 0, left_column = 0, right_row = 0, right_column = 0, target_row = 0, target_column = 0;
	for (size_t left_index = 0; left_index < leftMatrix.size(); ++left_index)
	{
		for (size_t right_index = 0; right_index < rightMatrix.size(); ++right_index)
		{
			left_row = left_index / left_rows;
			left_column = left_index % ((int)left_rows);

			right_row = right_index / right_columns;
			right_column = right_index % ((int)right_columns);

			target_row = right_row + (left_row * right_columns);
			target_column = right_column + (left_column * right_columns);
			result_matrix[(target_row)*(left_rows*right_columns) + target_column] = complexMultip(leftMatrix[left_index], rightMatrix[right_index]);
		}
	}

	PTrace("ZhangMultip result: ");
	PTraceMat(result_matrix);
	return result_matrix;
}

int MatrixMathFunction::partition(const QStat& srcMatrix, int partitionRowNum, int partitionColumnNum, blockedMatrix_t& blockedMat)
{
	blockedMat.m_vec_block.clear();

	PTrace("partition:\nsrcMatrix: ");
	PTraceMat(srcMatrix);

	size_t mat_size = srcMatrix.size();
	int src_mat_rows = sqrt(mat_size); // same to the Columns of the srcMatrix
	if ((0 != src_mat_rows %partitionRowNum) || (0 != src_mat_rows %partitionColumnNum))
	{
		QCERR_AND_THROW_ERRSTR("Error: Failed to partition.");
		return -1;
	}

	blockedMat.m_block_rows = partitionRowNum;
	blockedMat.m_block_columns = partitionColumnNum;

	int row_cnt_in_block = src_mat_rows / partitionRowNum;
	int col_cnt_in_block = src_mat_rows / partitionColumnNum;

	blockedMat.m_vec_block.resize(partitionRowNum*partitionColumnNum);
	for (size_t block_row = 0; block_row < partitionRowNum; ++block_row)
	{
		for (size_t block_col = 0; block_col < partitionColumnNum; ++block_col)
		{
			matrixBlock_t& block = blockedMat.m_vec_block[block_row*partitionColumnNum + block_col];
			block.m_row_index = block_row;
			block.m_column_index = block_col;

			for (size_t row_in_block = 0; row_in_block < row_cnt_in_block; row_in_block++)
			{
				for (size_t col_in_block = 0; col_in_block < col_cnt_in_block; col_in_block++)
				{
					int row_in_src_mat = block_row * row_cnt_in_block + row_in_block;
					int col_in_src_mat = block_col * col_cnt_in_block + col_in_block;
					block.m_mat.push_back(srcMatrix[row_in_src_mat*src_mat_rows + col_in_src_mat]);
				}
			}
		}
	}

	return 0;
}

int MatrixMathFunction::blockMultip(const QStat& leftMatrix, const blockedMatrix_t& blockedMat, QStat& resultMatrix)
{
	if ( (0 == leftMatrix.size()) || (blockedMat.m_vec_block.size() == 0))
	{
		QCERR_AND_THROW_ERRSTR("Error: parameter error.");
		return -1;
	}

	std::vector<matrixBlock_t> tmp_Block_Vec;
	tmp_Block_Vec.resize(blockedMat.m_vec_block.size());
	for (auto &itr : blockedMat.m_vec_block)
	{
		matrixBlock_t &tmp_block = tmp_Block_Vec[itr.m_row_index*(blockedMat.m_block_columns) + itr.m_column_index];
		tmp_block.m_row_index = itr.m_row_index;
		tmp_block.m_column_index = itr.m_column_index;
		tmp_block.m_mat = (ZhangMultip(leftMatrix, itr.m_mat));
	}

	int row_cnt_in_block = sqrt(tmp_Block_Vec[0].m_mat.size());
	int col_cnt_in_block = row_cnt_in_block; //square matrix
	size_t block_index = 0;
	size_t item_in_block_index = 0;
	for (size_t block_row = 0; block_row < blockedMat.m_block_rows; block_row++)
	{
		for (size_t row_in_block = 0; row_in_block < row_cnt_in_block; row_in_block++)
		{
			for (size_t block_col = 0; block_col < blockedMat.m_block_columns; block_col++)
			{
				for (size_t col_in_block = 0; col_in_block < col_cnt_in_block; col_in_block++)
				{
					block_index = block_row * blockedMat.m_block_columns + block_col;
					item_in_block_index = row_in_block * col_cnt_in_block + col_in_block;
					resultMatrix.push_back(tmp_Block_Vec[block_index].m_mat[item_in_block_index]);
				}
			}
		}
	}

	PTrace("blockMultip result: ");
	PTraceMat(resultMatrix);

	return 0;
}

QStat QPanda::getMatrix(QProg srcProg, const NodeIter nodeItrStart , const NodeIter nodeItrEnd)
{
	QProg tmp_prog;

	//fill the prog through traversal 
	PickUpNodes pick_handle(tmp_prog, srcProg,
		nodeItrStart == NodeIter() ? srcProg.getFirstNodeIter() : nodeItrStart, 
		nodeItrEnd == NodeIter() ? srcProg.getLastNodeIter() : nodeItrEnd);

	pick_handle.traversalQProg();

	QprogToMatrix calc_matrix(tmp_prog);

	return calc_matrix.getMatrix();
}

template<typename node_T>
void PrintAllNodeType::printNodeType(node_T subPgogNode, std::shared_ptr<QNode> parent_node, bool &is_dagger)
{
	std::shared_ptr<QNode> cur_node = std::dynamic_pointer_cast<QNode>(subPgogNode);
	const NodeType cur_node_type = cur_node->getNodeType();

	// if flowCtrl node, traversal flowCtrlNode
	if ((cur_node_type == WHILE_START_NODE) || (cur_node_type == QIF_START_NODE))
	{
		PTrace("Enter flow control node\n ");
		Traversal::traversal(std::dynamic_pointer_cast<AbstractControlFlowNode>(subPgogNode), *this, is_dagger);
	}
	else if ((cur_node_type == CIRCUIT_NODE) || (cur_node_type == PROG_NODE))
	{
		// if subNode, handle subNode
		NodeIter cur_iter, end_itr;
		if (cur_node_type == CIRCUIT_NODE)
		{
			cur_iter = (std::dynamic_pointer_cast<AbstractQuantumCircuit>(subPgogNode))->getFirstNodeIter();
			end_itr = (std::dynamic_pointer_cast<AbstractQuantumCircuit>(subPgogNode))->getEndNodeIter();
		}
		else
		{
			cur_iter = (std::dynamic_pointer_cast<AbstractQuantumProgram>(subPgogNode))->getFirstNodeIter();
			end_itr = (std::dynamic_pointer_cast<AbstractQuantumProgram>(subPgogNode))->getEndNodeIter();
		}

		while (cur_iter != end_itr)
		{
			//if gate type
			NodeIter tmp_itr = cur_iter;
			cur_iter = cur_iter.getNextIter();

			//if gateNode
			const NodeType curT = (*tmp_itr)->getNodeType();
			if ((GATE_NODE == curT) || (MEASURE_GATE == curT))
			{
				if ((*tmp_itr)->getNodeType() == MEASURE_GATE)
				{
					PTrace(">>measureGate ");
				}
				else
				{
					GateType gt = GATE_UNDEFINED;
					std::shared_ptr<QNode> tmp_node = *(tmp_itr);
					if (nullptr != tmp_node)
					{
						if (GATE_NODE == tmp_node->getNodeType())
						{
							std::shared_ptr<OriginQGate> gate = std::dynamic_pointer_cast<OriginQGate>(tmp_node);
							gt = (GateType)(gate->getQGate()->getGateType());
						}
					}

					PTrace(">>gateType=%d ", gt);
				}
			}
			else
			{
				//continue traversal
				continueTraversal(*tmp_itr, parent_node, is_dagger);
			}
		}
	}
	else
	{
		// error type
		PTrace(">>OtherNodeType \n");
		return;
	}

	if (nullptr != parent_node.get())
	{
		auto parent_node_type = (parent_node)->getNodeType();
		if (((WHILE_START_NODE == parent_node_type)) || ((QIF_START_NODE == parent_node_type)))
		{
			PTrace("Leave flow control node\n ");
		}
	}
}

template<typename node_T>
void PickUpNodes::pickUp(node_T subPgogNode, std::shared_ptr<QNode> parent_node, bool &is_dagger)
{
	if (m_b_pickup_end)
	{
		return;
	}

	std::shared_ptr<QNode> cur_node = std::dynamic_pointer_cast<QNode>(subPgogNode);
	const NodeType cur_node_type = cur_node->getNodeType();

	// if flowCtrl node, traversal flowCtrlNode
	if ((cur_node_type == WHILE_START_NODE) || (cur_node_type == QIF_START_NODE))
	{
		//enter flow control node
		Traversal::traversal(std::dynamic_pointer_cast<AbstractControlFlowNode>(subPgogNode), *this, is_dagger);
	}
	else if ((cur_node_type == CIRCUIT_NODE) || (cur_node_type == PROG_NODE))
	{
		// processing Nested nodes
		NodeIter cur_iter, end_itr;
		if (cur_node_type == CIRCUIT_NODE)
		{
			cur_iter = (std::dynamic_pointer_cast<AbstractQuantumCircuit>(subPgogNode))->getFirstNodeIter();
			end_itr = (std::dynamic_pointer_cast<AbstractQuantumCircuit>(subPgogNode))->getEndNodeIter();
		}
		else
		{
			cur_iter = (std::dynamic_pointer_cast<AbstractQuantumProgram>(subPgogNode))->getFirstNodeIter();
			end_itr = (std::dynamic_pointer_cast<AbstractQuantumProgram>(subPgogNode))->getEndNodeIter();
		}

		while (cur_iter != end_itr)
		{
			if (m_b_pickup_end)
			{
				return;
			}
			
			const NodeIter tmp_itr = cur_iter;
			cur_iter = cur_iter.getNextIter();
			const NodeType curT = (*tmp_itr)->getNodeType();
			if ((GATE_NODE == curT))
			{
				if (m_b_picking)
				{
					auto gate = QGate(std::dynamic_pointer_cast<AbstractQGateNode>(*tmp_itr));
					m_output_prog.pushBackNode(&gate);
					if (tmp_itr == m_end_iter)
					{
						m_b_pickup_end = true;
						return;
					}
				}
				else
				{
					if (tmp_itr == m_start_iter)
					{
						m_b_picking = true;
						auto gate = QGate(std::dynamic_pointer_cast<AbstractQGateNode>(*tmp_itr));
						m_output_prog.pushBackNode(&gate);

						if (tmp_itr == m_end_iter)
						{
							//On case for _startIter == _endIter
							m_b_pickup_end = true;
							return;
						}

						continue;
					}
				}
			}
			else
			{
				//if there are measure/Qif/Qwhile node, throw an exception 
				if ((MEASURE_GATE == curT) || (WHILE_START_NODE == curT) || (QIF_START_NODE == curT))
				{
					m_b_pickup_end = true;
					QCERR_AND_THROW_ERRSTR("Error: There are some illegal nodes, failed to calc the target matrix between the specialed nodeIters.");
					m_output_prog.clear();
					return;
				}

				//continue to traversal
				continueTraversal(*tmp_itr, parent_node, is_dagger);
			}
		}
	}
}

QStat QprogToMatrix::getMatrix()
{
	QStat result_matrix;

	//get quantumBits number
	NodeIter itr = m_prog.getFirstNodeIter();
	NodeIter itr_end = m_prog.getEndNodeIter();
	QVec qubits_vector;
	do
	{
		if (GATE_NODE != (*itr)->getNodeType())
		{
			QCERR_AND_THROW_ERRSTR("Error: Qprog node type error.");
			return QStat();
		}

		std::shared_ptr<AbstractQGateNode> p_QGate = std::dynamic_pointer_cast<AbstractQGateNode>(*itr);
		qubits_vector.clear();
		p_QGate->getQuBitVector(qubits_vector);
		for (auto _val : qubits_vector)
		{
			m_qubits_in_use.push_back(_val->getPhysicalQubitPtr()->getQubitAddr());
		}

	} while ((++itr) != itr_end);
	sort(m_qubits_in_use.begin(), m_qubits_in_use.end());
	m_qubits_in_use.erase(unique(m_qubits_in_use.begin(), m_qubits_in_use.end()), m_qubits_in_use.end());

	//layer
	m_dag.getMainGraphSequence(m_prog, m_seq);
	const QProgDAG& prog_dag = m_dag.getProgDAG();
	for (auto &seqItem : m_seq)
	{
		//each layer
		result_matrix = MatrixMathFunction::multip(result_matrix, getMatrixOfOneLayer(seqItem, prog_dag));
	}

	return result_matrix;
}

QStat QprogToMatrix::getMatrixOfOneLayer(SequenceLayer& layer, const QProgDAG& progDag)
{
	QStat final_result_mat;
	gateQubitInfo_t double_qubit_gates;//double qubit gate vector
	gateQubitInfo_t single_qubit_gates;//single qubit gate vector
	calcUnitVec_t calc_unit_vec;

	for (auto &layer_item : layer)
	{
		auto p_node = progDag.getVertex(layer_item.first.m_vertex_num);
		auto p_gate = std::dynamic_pointer_cast<AbstractQGateNode>(p_node);
		QVec qubits_vector;
		p_gate->getQuBitVector(qubits_vector);
		if (qubits_vector.size() == 2)
		{
			std::vector<int> quBits;
			for (auto _val : qubits_vector)
			{
				quBits.push_back(_val->getPhysicalQubitPtr()->getQubitAddr());
			}

			double_qubit_gates.push_back(std::pair<std::shared_ptr<AbstractQGateNode>, std::vector<int>>(p_gate, quBits));
		}
		else if (qubits_vector.size() == 1)
		{
			std::vector<int> quBits;
			quBits.push_back(qubits_vector.front()->getPhysicalQubitPtr()->getQubitAddr());
			single_qubit_gates.push_back(std::pair<std::shared_ptr<AbstractQGateNode>, std::vector<int>>(p_gate, quBits));
		}
		else
		{
			QCERR_AND_THROW_ERRSTR("Error: QGate type error.");
			return QStat();
		}
	}

	//sort by qubit address spacing
	auto sorfFun = [](gateAndQubitsItem_t &a, gateAndQubitsItem_t &b) { return (abs(a.second.front() - a.second.back())) < (abs(b.second.front() - b.second.back())); };
	std::sort(double_qubit_gates.begin(), double_qubit_gates.end(), sorfFun);
	std::sort(single_qubit_gates.begin(), single_qubit_gates.end(), sorfFun);

	//calc double qubit gate first
	GateType gate_T = GATE_UNDEFINED;
	for (auto &double_gate : double_qubit_gates)
	{
		QStat gate_mat;
		gate_T = (GateType)(double_gate.first->getQGate()->getGateType());
		if ((CNOT_GATE == gate_T) || (CU_GATE == gate_T) || (CZ_GATE == gate_T) || (CPHASE_GATE == gate_T))
		{
			if (2 == double_gate.second.size())
			{
				auto &qubits = double_gate.second;
				
				if (qubits[0] > qubits[1])
				{
					// transf base matrix
					double_gate.first->getQGate()->getMatrix(gate_mat);
					auto transformed_mat = reverseCtrlGateMatrix(gate_mat);
					gate_mat.swap(transformed_mat);
				}
				else
				{
					//get matrix directly
					double_gate.first->getQGate()->getMatrix(gate_mat);
				}
			}
			else
			{
				QCERR_AND_THROW_ERRSTR("Error: Qubits number error.");
				return QStat();
			}
		}
		else
		{
		    //swap gate
			double_gate.first->getQGate()->getMatrix(gate_mat);
		}

		mergeToCalcUnit(double_gate.second, gate_mat, calc_unit_vec, single_qubit_gates);
	}

	for (auto &itr_calc_unit_vec : calc_unit_vec)
	{
		//calc all the qubits to get the final matrix
		QStat final_mat_of_one_calc_unit;
		for (auto &in_used_qubit_val : m_qubits_in_use)
		{
			bool b_no_gate_on_this_qubit = true;
			for (auto itr_single_gate = single_qubit_gates.begin(); itr_single_gate < single_qubit_gates.end();)
			{
				const int qubit_val = itr_single_gate->second.front();
				if (qubit_val == in_used_qubit_val)
				{
					b_no_gate_on_this_qubit = false;
					zhangMultipQGate(final_mat_of_one_calc_unit, itr_single_gate->first);

					itr_single_gate = single_qubit_gates.erase(itr_single_gate);
					continue;
				}

				++itr_single_gate;
			}

			if (itr_calc_unit_vec.second.front() == in_used_qubit_val)
			{
				b_no_gate_on_this_qubit = false;
				zhangMultipMatrix(final_mat_of_one_calc_unit, itr_calc_unit_vec.first);
			}

			if ((itr_calc_unit_vec.second.front() <= in_used_qubit_val) && (itr_calc_unit_vec.second.back() >= in_used_qubit_val))
			{
				continue;
			}

			if (b_no_gate_on_this_qubit)
			{
				zhangMultipMatrix(final_mat_of_one_calc_unit, m_mat_I);
			}
		}

		//Multiply, NOT ZhangMultip
		if (final_result_mat.empty())
		{
			final_result_mat = final_mat_of_one_calc_unit;
		}
		else
		{
			final_result_mat = MatrixMathFunction::multip(final_result_mat, final_mat_of_one_calc_unit);
		}
	}

	if (single_qubit_gates.size() > 0)
	{
		for (auto &in_used_qubit_val : m_qubits_in_use)
		{
			bool b_no_gate_on_this_qubit = true;
			for (auto itr_single_gate = single_qubit_gates.begin(); itr_single_gate < single_qubit_gates.end();)
			{
				const int qubit_val = itr_single_gate->second.front();
				if (qubit_val == in_used_qubit_val)
				{
					b_no_gate_on_this_qubit = false;
					zhangMultipQGate(final_result_mat, itr_single_gate->first);

					itr_single_gate = single_qubit_gates.erase(itr_single_gate);
					continue;
				}

				++itr_single_gate;
			}

			if (b_no_gate_on_this_qubit)
			{
				zhangMultipMatrix(final_result_mat, m_mat_I);
			}
		}
	}

	return final_result_mat;
}

void QprogToMatrix::zhangMultipQGate(QStat& srcMat, std::shared_ptr<AbstractQGateNode> &pGate)
{
	if (nullptr == pGate)
	{
		return;
	}

	if (srcMat.empty())
	{
		pGate->getQGate()->getMatrix(srcMat);
	}
	else
	{
		QStat single_gate_mat;
		pGate->getQGate()->getMatrix(single_gate_mat);
		srcMat = MatrixMathFunction::ZhangMultip(srcMat, single_gate_mat);
	}
}

void QprogToMatrix::zhangMultipMatrix(QStat& srcMat, const QStat& zhangMat)
{
	if (srcMat.empty())
	{
		srcMat = zhangMat;
	}
	else
	{
		srcMat = MatrixMathFunction::ZhangMultip(srcMat, zhangMat);
	}
}

void QprogToMatrix::mergeToCalcUnit(std::vector<int>& qubits, QStat& gateMat, calcUnitVec_t &calcUnitVec, gateQubitInfo_t &singleQubitGates)
{
	//auto &qubits = curGareItem.second;
	std::sort(qubits.begin(), qubits.end(), [](int &a, int &b) {return a < b; });
	std::vector<int> stride_over_qubits;
	getStrideOverQubits(qubits, stride_over_qubits);
	if (stride_over_qubits.empty())
	{
		//serial qubits
		calcUnitVec.insert(calcUnitVec.begin(), std::pair<QStat, std::vector<int>>(gateMat, qubits));
	}
	else
	{
		//get crossed CalcUnits;
		calcUnitVec_t crossed_calc_units;
		for (auto itr_calc_unit = calcUnitVec.begin(); itr_calc_unit < calcUnitVec.end();)
		{
			if ((qubits[0] < itr_calc_unit->second.front()) && (qubits[1] > itr_calc_unit->second.back()))
			{
				//The current double qubit gate has crossed the itr_calc_unit
				crossed_calc_units.push_back(*itr_calc_unit);

				itr_calc_unit = calcUnitVec.erase(itr_calc_unit);
				continue;
			}

			++itr_calc_unit;
		}

		//get crossed SingleQubitGates;
		gateQubitInfo_t crossed_single_qubit_gates;
		for (auto itr_single_gate = singleQubitGates.begin(); itr_single_gate < singleQubitGates.end();)
		{
			const int qubit_val = itr_single_gate->second.front();
			if ((qubit_val > qubits[0]) && (qubit_val < qubits[1]))
			{
				crossed_single_qubit_gates.push_back(*itr_single_gate);

				itr_single_gate = singleQubitGates.erase(itr_single_gate);
				continue;
			}

			++itr_single_gate;
		}

		//zhang multiply
		QStat filled_matrix;
		for (auto &in_used_qubit_val : m_qubits_in_use)
		{
			if (in_used_qubit_val > qubits[0])
			{
				if (in_used_qubit_val >= qubits[1])
				{
					break;
				}

				bool b_no_qGate_on_this_qubit = true;

				//find current qubit_val in crossed_single_qubit_gates
				std::shared_ptr<AbstractQGateNode> pGate;
				for (auto itr_crossed_single_gate = crossed_single_qubit_gates.begin();
					itr_crossed_single_gate != crossed_single_qubit_gates.end(); itr_crossed_single_gate++)
				{
					if (in_used_qubit_val == itr_crossed_single_gate->second.front())
					{
						b_no_qGate_on_this_qubit = false;

						zhangMultipQGate(filled_matrix, itr_crossed_single_gate->first);
						crossed_single_qubit_gates.erase(itr_crossed_single_gate);
						break;
					}
				}

				//find current qubit_val in crossed_calc_units
				for (auto itr_crossed_calc_unit = crossed_calc_units.begin();
					itr_crossed_calc_unit != crossed_calc_units.end(); itr_crossed_calc_unit++)
				{
					if ((in_used_qubit_val >= itr_crossed_calc_unit->second.front()) && (in_used_qubit_val <= itr_crossed_calc_unit->second.back()))
					{
						b_no_qGate_on_this_qubit = false;

						if (in_used_qubit_val == itr_crossed_calc_unit->second.front())
						{
							zhangMultipMatrix(filled_matrix, itr_crossed_calc_unit->first);
							//just break, CANN'T erase itr_crossed_calc_unit here
							break;
						}
					}
				}

				//No handle on this qubit
				if (b_no_qGate_on_this_qubit)
				{
					zhangMultipMatrix(filled_matrix, m_mat_I);
				}
			}
		}

		//blockMultip
		QStat filled_double_gate_matrix;
		MatrixMathFunction::blockedMatrix_t blocked_mat;
		MatrixMathFunction::partition(gateMat, pow(2, qubits.size() - 1), pow(2, qubits.size() - 1), blocked_mat);
		MatrixMathFunction::blockMultip(filled_matrix, blocked_mat, filled_double_gate_matrix);

		//insert into calcUnitVec
		calcUnitVec.insert(calcUnitVec.begin(), std::pair<QStat, std::vector<int>>(filled_double_gate_matrix, qubits));
	}
}

void QprogToMatrix::getStrideOverQubits(const std::vector<int> &qgateUsedQubits, std::vector<int> &strideOverQubits)
{
	strideOverQubits.clear();

	for (auto &qubit_val : m_qubits_in_use)
	{
		if ((qubit_val > qgateUsedQubits.front()) && (qubit_val < qgateUsedQubits.back()))
		{
			strideOverQubits.push_back(qubit_val);
		}
	}
}

QStat QprogToMatrix::reverseCtrlGateMatrix(QStat& srcMat)
{
	init(QMachineType::CPU);
	auto q = qAllocMany(6);
	QGate gate_H = H(q[0]);
	QStat mat_H;
	gate_H.getQGate()->getMatrix(mat_H);
	finalize();

	QStat result_mat;
	QStat mat_of_zhang_multp_two_H = MatrixMathFunction::ZhangMultip(mat_H, mat_H);

	result_mat = MatrixMathFunction::multip(mat_of_zhang_multp_two_H, srcMat);
	result_mat = MatrixMathFunction::multip(result_mat, mat_of_zhang_multp_two_H);

	PTrace("reverseCtrlGateMatrix: ");
	PTraceMat(result_mat);
	return result_mat;
}

void QPanda::printMat(const QStat& mat)
{
	int rows = 0;
	int columns = 0;
	rows = columns = sqrt(mat.size());
	printf("Matrix:\n");
	int index = 0;
	float imag_val = 0.0;
	float real_val = 0.0;
    const int max_width = 12;
	char outputBuf[32] = "";
	string outputStr;
	for (size_t i = 0; i < rows; i++)
	{
		for (size_t j = 0; j < columns; j++)
		{
			memset(outputBuf, ' ', sizeof(outputBuf));
			index = i * columns + j;
			imag_val = mat[index].imag();
			real_val = mat[index].real();
			if ((abs(real_val) < 0.000000001) || (abs(imag_val) < 0.000000001))
			{
				if ((abs(real_val) < 0.000000001) && (abs(imag_val) < 0.000000001))
				{
					snprintf(outputBuf, sizeof(outputBuf), " 0");
				}
				else if (abs(imag_val) < 0.000000001)
				{
					if (real_val < 0)
					{
						snprintf(outputBuf, sizeof(outputBuf), "%05.06f", (real_val));
					}
					else
					{
						snprintf(outputBuf, sizeof(outputBuf), " %05.06f", abs(real_val));
					}
				}
				else
				{
					//only imag_val
					if (imag_val < 0)
					{
						snprintf(outputBuf, sizeof(outputBuf), "%05.06fi", (imag_val));
					}
					else
					{
						snprintf(outputBuf, sizeof(outputBuf), " %05.06fi", abs(imag_val));
					}
				}
			}
			else if (imag_val < 0)
			{
				if (real_val < 0)
				{
					snprintf(outputBuf, sizeof(outputBuf), "%05.06f%05.06fi", real_val, imag_val);
				}
				else
				{
					snprintf(outputBuf, sizeof(outputBuf), " %05.06f%05.06fi", abs(real_val), imag_val);
				}
				
			}
			else
			{
				if (real_val < 0)
				{
					snprintf(outputBuf, sizeof(outputBuf), "%05.06f+%05.06fi  ", real_val, imag_val);
				}
				else
				{
					snprintf(outputBuf, sizeof(outputBuf), " %05.06f+%05.06fi  ", abs(real_val), imag_val);
				}
			}

			outputStr = outputBuf;
			size_t valLen = outputStr.size();
			outputBuf[valLen] = ' ';
			outputStr = outputBuf;
			outputStr = outputStr.substr(0, (max_width < valLen ? valLen :max_width) + 5);
			printf(outputStr.c_str());
		}
		printf("\n");
	}
}

void QPanda::printAllNodeType(QProg &prog)
{
	PrintAllNodeType print_node_type(prog);
	print_node_type.traversalQProg();
}
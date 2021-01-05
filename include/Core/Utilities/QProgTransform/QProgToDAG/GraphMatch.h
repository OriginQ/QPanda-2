/*! \file GraphMatch.h */
#ifndef _QNODE_MATCH_H_
#define _QNODE_MATCH_H_
#include <algorithm>
#include <functional>
#include "Core/Utilities/Tools/Traversal.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/QProgToDAG.h"
#include "Core/QuantumMachine/QuantumMachineInterface.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/TopologSequence.h"

QPANDA_BEGIN

using LayerVector = std::vector<SequenceNode>;
using ResultVector = std::vector<LayerVector>;

enum GraphType
{
    MAIN_GRAPH = 0,
    QUERY_GRAPH,
    REPLACE_GRAPH
};

/**
* @class GraphMatch
* @ingroup Utilities
* @brief Graph Match and Replace
* @note
*/
class GraphMatch
{
public:
    /**
    * @brief  get topological sequence
    * @param[in]  _Ty & node
    * @param[out]  TopologSequence<SequenceNode> & seq
    * @param[in]  GraphType graph_type
    * @return     void
    */
    template <typename _Ty>
    void get_topological_sequence(_Ty &node, TopologSequence<SequenceNode> &seq, GraphType graph_type = MAIN_GRAPH)
    {
        QProgToDAG dag;
		DAGToTopologSequence<SequenceNode>::tranf_fun<GateNodeInfo> tmp_tranf_fun = SequenceNode::construct_sequence_node;
        if (GraphType::MAIN_GRAPH == graph_type)
        {
            dag.traversal(node, m_graph_dag);
			DAGToTopologSequence<SequenceNode>(seq, m_graph_dag, tmp_tranf_fun);
        }
        else if (GraphType::QUERY_GRAPH == graph_type)
        {
            dag.traversal(node, m_query_dag);
			DAGToTopologSequence<SequenceNode>(seq, m_query_dag, tmp_tranf_fun);
        }
        else if (GraphType::REPLACE_GRAPH == graph_type)
        {
            dag.traversal(node, m_replace_dag);
			DAGToTopologSequence<SequenceNode>(seq, m_replace_dag, tmp_tranf_fun);
        }
        else
        {}
    }  

    template <typename _Ty1, typename _Ty2>
    void replace(_Ty1 &query_node, _Ty2 &replace_node,
        ResultVector &result, TopologSequence<SequenceNode> &graph_seq, QProg &prog, QuantumMachine*qvm)
    {
		TopologSequence<SequenceNode> replace_seq;
        get_topological_sequence(replace_node, replace_seq, GraphType::REPLACE_GRAPH);

        if (!_compare_qnum(m_query_dag.m_qubit_vec, m_replace_dag.m_qubit_vec))
        {
            QCERR("qubits compare error");
            return;
        }
        else
        {
            _convert_node(result, replace_seq, graph_seq, qvm);
            _convert_prog(graph_seq, prog);
        }
    }
    
    bool query(TopologSequence<SequenceNode> &, TopologSequence<SequenceNode> &, ResultVector &);
    const QProgDAG<GateNodeInfo>& getProgDAG() { return m_graph_dag; }
    
private:
    QProgDAG<GateNodeInfo> m_graph_dag;
    QProgDAG<GateNodeInfo> m_query_dag;
    QProgDAG<GateNodeInfo> m_replace_dag;
    std::vector<size_t> m_compare_vec;

    bool _compare_qnum(Qnum, Qnum);
    bool _compare_node(SeqNode<SequenceNode> &, SeqNode<SequenceNode> &);
    bool _compare_edge(const LayerVector &, const LayerVector &);
    bool _compare_parm(SequenceNode &, SequenceNode &);

    void _replace_node(const ResultVector &, TopologSequence<SequenceNode> &);
    void _get_pre_node(size_t, TopologSequence<SequenceNode>&, LayerVector&);

    void _convert_prog(TopologSequence<SequenceNode> &, QProg &);
    void _convert_gate(SequenceNode&, QuantumMachine*, std::map<size_t, size_t> &, SequenceNode&);
    void _convert_node(ResultVector &, TopologSequence<SequenceNode> &,
		TopologSequence<SequenceNode> &, QuantumMachine*);

    Qnum _get_qubit_vector(const SequenceNode &, QProgDAG<GateNodeInfo> &);
};



/**
* @brief  graph query and replace
* @ingroup Utilities
* @param[in]  _Ty1 & graph_node
* @param[in]  _Ty2 & query_node
* @param[in]  _Ty3 & replace_node
* @param[out]  QProg & prog
* @param[in]  QuantumMachine * qvm
* @return     void
*/
template <typename _Ty1, typename _Ty2, typename _Ty3>
void graph_query_replace(_Ty1 &graph_node, _Ty2 &query_node, _Ty3 &replace_node,
                         QProg &prog, QuantumMachine *qvm)
{
    GraphMatch match;
    ResultVector query_result;

	TopologSequence<SequenceNode> graph_seq;
    match.get_topological_sequence(graph_node, graph_seq);

	TopologSequence<SequenceNode> query_seq;
    match.get_topological_sequence(query_node, query_seq, GraphType::QUERY_GRAPH);

    if (match.query(graph_seq, query_seq, query_result))
    {
        match.replace(query_node, replace_node, query_result, graph_seq, prog, qvm);
    }
    else
    {
        std::cout << "Unable to find matching query graph" << std::endl;
    }
}


/**
* @brief  graph query
* @ingroup Utilities
* @param[in]  _Ty1 & graph_node
* @param[in]  _Ty2 & query_node
* @param[out]  ResultVector & query_result
* @return     bool true or false
*/
template <typename _Ty1, typename _Ty2>
bool graph_query(_Ty1 &graph_node, _Ty2 &query_node, ResultVector &query_result)
{
    GraphMatch match;

	TopologSequence<SequenceNode> graph_seq;
    match.get_topological_sequence(graph_node, graph_seq);

	TopologSequence<SequenceNode> query_seq;
    match.get_topological_sequence(query_node, query_seq, GraphType::QUERY_GRAPH);

    if (match.query(graph_seq, query_seq, query_result))
    {
        return true;
    }
    else
    {
        std::cout << "Unable to find matching query graph" << std::endl;
        return false;
    }
}

QPANDA_END
#endif
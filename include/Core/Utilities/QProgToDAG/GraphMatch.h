/*! \file GraphMatch.h */
#ifndef _QNODE_MATCH_H_
#define _QNODE_MATCH_H_
#include "Core/Utilities/QProgToDAG/QProgToDAG.h"
#include "Core/QuantumMachine/QuantumMachineInterface.h"
#include "Core/Utilities/Traversal.h"
#include <functional>
#include <algorithm>

QPANDA_BEGIN

using LayerVector = std::vector<SequenceNode>;
using ResultVector = std::vector<LayerVector>;
using MatchVector = std::vector<ResultVector>;

enum QNodeGateType
{
    SINGLE_OR_MEASURE_GATE = 0,
    DOUBLE_GATE
};


class QubitsCompare :public TraversalInterface<>
{
private:
    std::vector<size_t> m_qubit_vec;

public:
    std::map<int, std::function<QGate(Qubit *)> > m_singleGateFunc;
    std::map<int, std::function<QGate(Qubit *, Qubit*)> > m_doubleGateFunc;
    std::map<int, std::function<QGate(Qubit *, double)> > m_angleGateFunc;
    std::map<int, std::function<QGate(Qubit *, Qubit*, double)> > m_doubleAngleGateFunc;

    template <typename _Ty>
    std::vector<size_t> traversal(_Ty &node)
    {
        m_qubit_vec.clear();
        static_assert(std::is_base_of<QNode, _Ty>::value, "node type is error");
        Traversal::traversalByType(node.getImplementationPtr(), nullptr, *this);
        return m_qubit_vec;
    }

    QubitsCompare();
    void execute(std::shared_ptr<AbstractQGateNode>, std::shared_ptr<QNode>);
    void execute(std::shared_ptr<AbstractClassicalProg>, std::shared_ptr<QNode>);
    void execute(std::shared_ptr<AbstractQuantumMeasure>, std::shared_ptr<QNode>);
    void execute(std::shared_ptr<AbstractQuantumCircuit>, std::shared_ptr<QNode>);
    void execute(std::shared_ptr<AbstractQuantumProgram>, std::shared_ptr<QNode>);
    void execute(std::shared_ptr<AbstractControlFlowNode>, std::shared_ptr<QNode>);
};

class GraphMatch
{
public:
    template <typename _Ty>
    void getMainGraphSequence(_Ty &node, TopologincalSequence &seq)
    {
        static_assert(std::is_base_of<QNode, _Ty>::value, "node type error");
        QProgToDAG dag;
        dag.traversal(node, m_graph_dag);
        m_graph_dag.getTopologincalSequence(seq);
    }

    template <typename _Ty>
    void getQueryGraphSequence(_Ty &node, TopologincalSequence &seq)
    {
        static_assert(std::is_base_of<QNode, _Ty>::value, "node type error");
        QProgToDAG dag;
        dag.traversal(node, m_query_dag);
        m_query_dag.getTopologincalSequence(seq);
    }

    template <typename _Ty1, typename _Ty2, typename _Ty3>
    void graphQueryReplace(_Ty1 &graph_node, _Ty2 &query_node,
        _Ty3 &replace_node, QProg &prog, QuantumMachine *qvm)
    {
        static_assert(std::is_base_of<QNode, _Ty1>::value, "node type error");
        static_assert(std::is_base_of<QNode, _Ty2>::value, "node type error");
        static_assert(std::is_base_of<QNode, _Ty3>::value, "node type error");

        TopologincalSequence graph_seq;
        getMainGraphSequence(graph_node, graph_seq);

        TopologincalSequence query_seq;
        getQueryGraphSequence(query_node, query_seq);

        MatchVector result;
        if (!graphQuery(graph_seq, query_seq, result))
        {
            std::cout << "Unable to find matching QueryMap" << std::endl;
            return;
        }

        graphReplace(query_node, replace_node, result, graph_seq, prog, qvm);
    }

    template <typename _Ty1, typename _Ty2>
    bool graphReplace(_Ty1 &query_node, _Ty2 &replace_node,
        MatchVector &result, TopologincalSequence &seq, QProg &prog, QuantumMachine*qvm)
    {
        static_assert(std::is_base_of<QNode, _Ty1>::value, "node type error");
        static_assert(std::is_base_of<QNode, _Ty2>::value, "node type error");

        QubitsCompare count;
        if (!qubitsCompare(count.traversal(query_node), count.traversal(replace_node)))
        {
            std::cout << "Qubits does not compare" << std::endl;
            return false;
        }
        else
        {
            TopologincalSequence replace_seq;
            QProgToDAG dag;
            dag.traversal(replace_node, m_replace_dag);
            m_replace_dag.getTopologincalSequence(replace_seq);

            insertQNodes(result, replace_seq, seq, qvm);
            SequenceToQProg(seq, prog, m_graph_dag);

            return true;
        }
    }

    bool graphQuery(TopologincalSequence &, TopologincalSequence &, MatchVector &);

    void SequenceToQProg(SequenceLayer &, QProg &, QProgDAG &);
    void SequenceToQProg(TopologincalSequence &, QProg &, QProgDAG &);

    const QProgDAG& getProgDAG() { return m_graph_dag; }
    
private:
    QProgDAG m_graph_dag;
    QProgDAG m_query_dag;
    QProgDAG m_replace_dag;

    std::vector<size_t> m_compare_vec;

    bool compareCurLayer(SequenceLayer &, SequenceLayer &, MatchVector &, 
                         ResultVector &, std::vector<size_t> &);
    bool compareCurQNode(std::pair<SequenceNode, LayerVector> &,
                         std::pair<SequenceNode, LayerVector> &);

    bool compareGateParm(SequenceNode &, QProgDAG &,
                         SequenceNode &, QProgDAG &);

    size_t getTarQubit(SequenceNode &, QProgDAG &);
    size_t getCtrQubit(SequenceNode &, QProgDAG &);

    bool qnodeContain(LayerVector &, LayerVector &);
    bool qubitContain(SequenceNode &, std::vector<size_t> &);

    void updateQubits(LayerVector &, std::vector<size_t> &);
    void insertQNodes(MatchVector &, TopologincalSequence &,
        TopologincalSequence &, QuantumMachine*);

    bool qubitsCompare(std::vector<size_t>,std::vector<size_t>);

    SequenceNode contsructQNode(const std::map<size_t, size_t> &,
        SequenceNode &, QuantumMachine*, QubitsCompare &);

};


QPANDA_END
#endif
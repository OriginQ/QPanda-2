#ifndef  QPROG_TO_DAG_H_
#define  QPROG_TO_DAG_H_

#include <map>
#include "Core/Utilities/Traversal.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"
#include "Core/Utilities/QPandaException.h"
#include "Core/Utilities/QProgToDAG/QProgDAG.h"
QPANDA_BEGIN

class QProgToDAG : public TraversalInterface<QProgDAG &, NodeIter&>
{
public:
    template <typename _Ty>
    void traversal(_Ty &node,QProgDAG & prog_dag)
    {
        static_assert(std::is_base_of<QNode,_Ty>::value,"node type is error");
		NodeIter NullItr;
        Traversal::traversalByType(node.getImplementationPtr(), nullptr, *this, prog_dag, NullItr);
    }

    void execute(std::shared_ptr<AbstractQGateNode>, std::shared_ptr<QNode>,QProgDAG &, NodeIter&);
    void execute(std::shared_ptr<AbstractClassicalProg>, std::shared_ptr<QNode>, QProgDAG &, NodeIter&);
    void execute(std::shared_ptr<AbstractQuantumMeasure>, std::shared_ptr<QNode>,QProgDAG &, NodeIter&);
    void execute(std::shared_ptr<AbstractQuantumCircuit>, std::shared_ptr<QNode>, QProgDAG &, NodeIter&);
    void execute(std::shared_ptr<AbstractQuantumProgram>, std::shared_ptr<QNode>, QProgDAG &, NodeIter&);
    void execute(std::shared_ptr<AbstractControlFlowNode>, std::shared_ptr<QNode>, QProgDAG &, NodeIter&);

protected:
    void transformQGate(std::shared_ptr<AbstractQGateNode>, QProgDAG &, NodeIter&);
    void transformQMeasure(std::shared_ptr<AbstractQuantumMeasure>, QProgDAG &, NodeIter&);
    void construct(size_t, size_t, QProgDAG &);


private:
    std::map<size_t,size_t> qubit_vertices_map;
};

QPANDA_END


#endif

/*
Copyright (c) 2017-2019 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

QProgDAG.h
Author: doumenghan
Updated in 2019/08/06 

Classes for QProgDAG.

*/
/*! \file QProgDAG.h */
#ifndef  QPROGDAG_H_
#define  QPROGDAG_H_

#include <vector>
#include <memory>
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/QuantumMeasure.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"
#include "ThirdParty/Eigen/Sparse"
#include "ThirdParty/Eigen/Dense"

QPANDA_BEGIN

/**
* @namespace QPanda
*/

struct SequenceNode
{
    int m_node_type; //Enum GateType ，if Measure Node ： -1
    size_t m_vertex_num;
};

using edges_vec = std::vector<std::pair<size_t, size_t>>; 
//using vertices_map = std::map<size_t, std::shared_ptr<QNode>>; 
using vertices_map = std::map<size_t, NodeIter>;

using AdjacencyMatrix = Eigen::MatrixXi;   
using SequenceLayer = std::vector<std::pair<SequenceNode,std::vector<SequenceNode>>>; 
using TopologincalSequence = std::vector<SequenceLayer>;  

class QProgDAG
{
public:
    QProgDAG() {}
    void getTopologincalSequence(TopologincalSequence &); 
    size_t addVertex(const NodeIter& iter); 
    size_t addEgde(size_t,size_t);
    
    void constructAdjacencyMatrix(const vertices_map &, AdjacencyMatrix &);

    SequenceNode constructSequenceNode(size_t);
    std::shared_ptr<QNode> getVertex(size_t) const;
	NodeIter getVertexNodeIter(size_t) const;

    size_t addVertex(QNode *);

private:
    QProg m_dag_prog;
    vertices_map m_vertices_map;
    edges_vec m_edges_vector;

    void getCurLayerVertices(AdjacencyMatrix &,SequenceLayer &);
};

QPANDA_END
#endif

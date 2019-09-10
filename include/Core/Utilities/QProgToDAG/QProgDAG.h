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
#include "Core/QuantumCircuit/QuantumMeasure.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"
#include "ThirdParty/Eigen/Sparse"
#include "ThirdParty/Eigen/Dense"
#include "Core/QPanda.h"

QPANDA_BEGIN

/**
* @namespace QPanda
*/

//拓扑序列顶点信息
struct SequenceNode
{
    int m_node_type; //Enum GateType ，if Measure Node ： -1
    size_t m_vertex_num;
};

using edges_vec = std::vector<std::pair<size_t, size_t>>; //边容器
//using vertices_map = std::map<size_t, std::shared_ptr<QNode>>; //顶点容器
using vertices_map = std::map<size_t, NodeIter>;

using AdjacencyMatrix = Eigen::MatrixXi;   //邻接矩阵
using SequenceLayer = std::vector<std::pair<SequenceNode,std::vector<SequenceNode>>>; //一层拓扑序列的顶点信息以及与其有关的下一层顶点的信息
using TopologincalSequence = std::vector<SequenceLayer>;  //分层量子拓扑序列

class QProgDAG
{
public:
    QProgDAG() {}
    void getTopologincalSequence(TopologincalSequence &); //获取分层量子拓扑序列信息
    size_t addVertex(const NodeIter& iter); //添加顶点，返回顶点编号；
    size_t addEgde(size_t,size_t);//添加边 参数是 入顶点的编号 和出顶点的编号，返回的是边的编号；
    
    void constructAdjacencyMatrix(const vertices_map &, AdjacencyMatrix &);

    SequenceNode constructSequenceNode(size_t);
    std::shared_ptr<QNode> getVertex(size_t) const; //获取顶点的QNode
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

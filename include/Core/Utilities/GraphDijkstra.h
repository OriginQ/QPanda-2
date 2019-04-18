/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

GraphDijkstra.h
Author: Wangjing
Created in 2018-8-31

Classes for get the shortes path of graph

*/
/*! \file GraphDijkstra.h */
#ifndef GRAPH_DIJKSTAR_H
#define GRAPH_DIJKSTAR_H

#include "Core/Utilities/QPandaNamespace.h"
#include <iostream>
#include <vector>
QPANDA_BEGIN
/**
* @namespace QPanda
*/

/**
* @brief   Int infinite 
*/
const int kInfinite = 0xffff;

/**
* @brief   Error identifier
*/
const int kError = -1;


/**
* @class Dist
* @brief Dijkstra graph node
*/
struct Dist{
    Dist() : value(0), visit(false){}

    std::vector<int> path_vec;
    int value;
    bool visit;
};

/**
* @class   GraphDijkstra
* @brief   Solutions for Dijkstra  algorithm
* @ingroup Utilities
*/
class GraphDijkstra
{
public:
    GraphDijkstra() = delete;
    GraphDijkstra(const std::vector<std::vector<int> > &matrix);

    /**
    * @brief  Get the shortest path of the graph between begin with end
    * @param[in]  int Begin: starting point  
    * @param[in]  int End: end point
    * @param[in]  std::vector<int>& path_vec:  The points at which the shortes path passes
    * @return     int  The length of the shortes path
    */
    int getShortestPath(int begin, int end, std::vector<int> &path_vec);

    /**
    * @brief  Determine if the graph is connected
    * @return   bool  Result of the judgement  
    * @see
    */
    bool is_connective();

    virtual ~GraphDijkstra();

protected:
    /**
    * @brief  Dijkstra algorithm
    * @param[in]  int Begin: stating point
    * @return     bool  If the execution is correct
    */
    bool dijkstra(int begin);
    
private:
    std::vector<std::vector<int>> m_matrix;
    std::vector<Dist> m_dist_vec;
    int m_vertex_count;
    int m_edge;
};
QPANDA_END
#endif // GRAPH_DIJKSTAR_H

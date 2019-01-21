/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

GraphDijkstra.h
Author: Wangjing
Created in 2018-8-31

Classes for get the shortes path of graph

*/

#ifndef GRAPH_DIJKSTAR_H
#define GRAPH_DIJKSTAR_H

#include "QPandaNamespace.h"
#include <iostream>
#include <vector>
QPANDA_BEGIN
const int kInfinite = 0xffff;
const int kError = -1;
struct Dist{
    Dist() : value(0), visit(false){}

    std::vector<int> path_vec;
    int value;
    bool visit;
};

class GraphDijkstra
{
public:
    GraphDijkstra() = delete;
    GraphDijkstra(const std::vector<std::vector<int> > &matrix);

    /*
    get the shortest path of the graph between begin with end
    param:
        begin: starting point 
        end:   end point
        path_vec:  the points at which the shortes path passes
    return:
        Return the length of the shortes path

    Note:
    */
    int getShortestPath(int begin, int end, std::vector<int> &path_vec);

    /*
    Determine if the graph is connected
    param:
        None
    return:
        Return result of the judgement

    Note:
    */
    bool is_connective();

    virtual ~GraphDijkstra();

protected:
    /*
    Dijkstra algorithm
    param:
        begin: stating point
    return:
        if the execution is correct

    Note:
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

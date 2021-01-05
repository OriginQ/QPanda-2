#include "Core/Utilities/Tools/Graph.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include "Core//Utilities/QPandaNamespace.h"
#include "Core/Utilities/Tools/QPandaException.h"

using namespace QPanda;

/**********************************************************************
*                     class Graph
**********************************************************************/
Graph::Graph(Kind k, uint32_t n, Type ty) : mK(k), mN(n), mTy(ty) 
{
    mSuccessors.assign(n, std::set<uint32_t>());
    mPredecessors.assign(n, std::set<uint32_t>());
}

Graph::Graph(uint32_t n, Type ty) : mK(K_GRAPH), mN(n), mTy(ty) 
{
    mSuccessors.assign(n, std::set<uint32_t>());
    mPredecessors.assign(n, std::set<uint32_t>());
}

std::string Graph::vertexToString(uint32_t i) const
{
    return std::to_string(i);
}

std::string Graph::edgeToString(uint32_t i, uint32_t j, std::string op) const 
{
    return vertexToString(i) + " " + op + " " + vertexToString(j);
}

uint32_t Graph::inDegree(uint32_t i) const 
{
    return mPredecessors[i].size();
}

uint32_t Graph::outDegree(uint32_t i) const 
{
    return mSuccessors[i].size();
}

uint32_t Graph::size() const 
{
    return mN;
}

std::set<uint32_t>& Graph::succ(uint32_t i) 
{
    return mSuccessors[i];
}

const std::set<uint32_t>& Graph::c_succ(uint32_t i) const
{
	return mSuccessors[i];
}

std::set<uint32_t>& Graph::pred(uint32_t i) 
{
    return mPredecessors[i];
}

std::set<uint32_t> Graph::adj(uint32_t i) const 
{
    std::set<uint32_t> adj;

    auto& succ = mSuccessors[i];
    auto& pred = mPredecessors[i];

    adj.insert(pred.begin(), pred.end());
    adj.insert(succ.begin(), succ.end());
    return adj;
}

bool Graph::hasEdge(uint32_t i, uint32_t j) const 
{
    const std::set<uint32_t>& succ = this->c_succ(i);
    return succ.find(j) != succ.end();
}

void Graph::putEdge(uint32_t i, uint32_t j) 
{
    mSuccessors[i].insert(j);
    mPredecessors[j].insert(i);

    if (!isDirectedGraph()) {
        mSuccessors[j].insert(i);
        mPredecessors[i].insert(j);
    }
}

std::string Graph::dotify(std::string name) const 
{
    bool isDirected = isDirectedGraph();
    std::string edgeOp, graphTy, dot;

    if (isDirected) { edgeOp = "->"; graphTy = "digraph"; }
    else { edgeOp = "--"; graphTy = "graph"; }

    dot = graphTy + " " + name + " {\n";
    for (uint32_t i = 0; i < mN; ++i) {
        dot += "    " + vertexToString(i) + ";\n";

        auto adjacent = mSuccessors[i];
        if (!isDirected) adjacent = adj(i);

        for (uint32_t j : adjacent) {
            if (isDirected || (!isDirected && j >= i))
                dot += "    " + edgeToString(i, j, edgeOp) + ";\n";
        }
    }
    dot += "}";
    return dot;
}

/**********************************************************************
*                     class JsonFields
**********************************************************************/
const std::string JsonFields<Graph>::_VerticesLabel_ = "vertices";
const std::string JsonFields<Graph>::_AdjListLabel_ = "adj";
const std::string JsonFields<Graph>::_TypeLabel_ = "type";
const std::string JsonFields<Graph>::_VLabel_ = "v";

/**********************************************************************
*                     class JsonBackendParser
**********************************************************************/
std::unique_ptr<Graph> JsonBackendParser<Graph>::Parse(const rapidjson::Value& root)
{
	if (!root[JsonFields<Graph>::_VerticesLabel_.c_str()].IsInt())
	{
		QCERR_AND_THROW(run_fail, "Error: Graph json error.");
	}

	if (!root[JsonFields<Graph>::_TypeLabel_.c_str()].IsString())
	{
		QCERR_AND_THROW(run_fail, "Error: Graph json error.");
	}

    auto vertices = root[JsonFields<Graph>::_VerticesLabel_.c_str()].GetUint();
    auto typeStr = root[JsonFields<Graph>::_TypeLabel_.c_str()].GetString();
    auto ty = Graph::Type::Undirected;

    if (typeStr == "Directed") 
	{
        ty = Graph::Type::Directed;
    } 
	else if (typeStr != "Undirected")
	{
        std::cout << "Warning: Graph parsing warning: defaulting to `Undirected type`." << std::endl;
    }

    auto graph = Graph::Create(vertices, ty);
    auto &adj = root[JsonFields<Graph>::_AdjListLabel_.c_str()];

    for (uint32_t i = 0; i < vertices; ++i) {
        auto &iList = adj[i];
		if (!iList.IsArray())
		{
			QCERR_AND_THROW(run_fail, "Error: Graph json error.");
		}
        for (uint32_t j = 0, f = iList.Size(); j < f; ++j) 
		{
            auto &jElem = iList[j];
			if (!jElem[JsonFields<Graph>::_VLabel_.c_str()].IsInt())
			{
				QCERR_AND_THROW(run_fail, "Error: Graph json error.");
			}
            graph->putEdge(i, jElem[JsonFields<Graph>::_VLabel_.c_str()].GetUint());
        }
    }

    return graph;
}

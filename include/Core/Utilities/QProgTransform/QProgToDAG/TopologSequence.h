#ifndef TOPOLOG_SEQUENCE_H
#define TOPOLOG_SEQUENCE_H

#include "Core/Utilities/QPandaNamespace.h"
#include <iostream>
#include <complex>
#include <vector>
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
//#include "Core/Utilities/QProgTransform/QProgToDAG/QProgToDAG.h"
#include <functional>

QPANDA_BEGIN

/*
* store the sequence node and the next sequence node, 
* for double qubits gate, control qubit first, and then target qubit
*/
//template <class T>
//struct SeqNode
//{
//	T m_cur_node;
//	std::vector<T> m_successor_nodes;
//};

template <class T>
using SeqNode = std::pair<T, std::vector<T>>;

template <class T>
using SeqLayer = std::vector<SeqNode<T>>;

template <class T>
class TopologSequence : public std::vector<SeqLayer<T>>
{
public:
	TopologSequence()
		:m_cur_layer(0)
	{}
	virtual ~TopologSequence() {}

private:
	size_t m_cur_layer;
};

QPANDA_END

#endif // TOPOLOG_SEQUENCE_H
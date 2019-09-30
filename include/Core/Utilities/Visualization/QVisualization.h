#pragma once
#include "QCircuitToTextPic.h"
#include "Core/Utilities/QPandaNamespace.h"

/**
* @class QVisuallization
* @brief output matrix, Qprog, Qcircuit, Realization of Background Data Visualization.
*/

QPANDA_BEGIN

/**
	* @brief output a quantum prog/circuit to console by text-pic(UTF-8 code), 
	          and will save the text-pic in file named QCircuitTextPic.txt in the same time in current path.
	* @param[in] prog  the source prog
	* @param[in] nodeItrStart The start pos, default is the first node of the prog
	* @param[in] nodeItrEnd The end pos, default is the end node of the prog
	* @ Note: All the output characters are UTF-8 code.
	*/
inline void printProg(QProg &prog, const NodeIter nodeItrStart = NodeIter(), const NodeIter nodeItrEnd = NodeIter())
{
	QCircuitToTextPic test_text_pic(prog, nodeItrStart, nodeItrEnd);
	test_text_pic.textDraw();
}

inline void printProg(QCircuit &cir, const NodeIter nodeItrStart = NodeIter(), const NodeIter nodeItrEnd = NodeIter())
{
	QProg prog(cir);
	printProg(prog,
		nodeItrStart == NodeIter() ? cir.getFirstNodeIter() : nodeItrStart,
		nodeItrEnd == NodeIter() ? cir.getLastNodeIter() : nodeItrEnd);
}

QPANDA_END
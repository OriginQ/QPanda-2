#pragma once
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/GraphMatch.h"
#include "Core/Utilities/QProgInfo/Visualization/Draw.h"

QPANDA_BEGIN

/**
* @class DrawQProg
* @brief Do some preparation for draw text picture.
*/

class DrawQProg
{
public:
	DrawQProg(QProg &prg, const NodeIter node_itr_start, const NodeIter node_itr_end);
	~DrawQProg();

	std::string textDraw(TEXT_PIC_TYPE t);

private:
	QProg m_prog;
	DrawPicture* m_p_text;
	std::vector<int> m_quantum_bits_in_use;
	std::vector<int> m_class_bits_in_use;
};

QPANDA_END
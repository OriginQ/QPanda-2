#pragma once
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/Utilities/QProgToDAG/GraphMatch.h"

QPANDA_BEGIN

/**
* @class QCircuitToTextPic
* @brief Do some preparation for draw text picture.
*/

class TextPic;
class QCircuitToTextPic
{
public:
	QCircuitToTextPic(QProg &prg, const NodeIter nodeItrStart, const NodeIter nodeItrEnd);
	~QCircuitToTextPic();

	void textDraw();
	void layer();

private:
	QProg m_prog;
	TextPic* m_p_text;
	std::vector<int> m_quantum_bits_in_use;
	std::vector<int> m_class_bits_in_use;
	GraphMatch grapth_dag;
	TopologincalSequence seq;
};

QPANDA_END
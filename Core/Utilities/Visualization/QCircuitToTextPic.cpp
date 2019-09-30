#include "Core/Utilities/Visualization/QCircuitToTextPic.h"
#include "Core/Utilities/Visualization/TextPic.h"
#include <stdexcept>
#include "Core/Utilities/QCircuitInfo.h"

USING_QPANDA
using namespace std;

QCircuitToTextPic::QCircuitToTextPic(QProg &prg, const NodeIter nodeItrStart, const NodeIter nodeItrEnd)
	: m_p_text(nullptr)
{
	pickUpNode(m_prog, prg, nodeItrStart == NodeIter() ? prg.getFirstNodeIter() : nodeItrStart,
		nodeItrEnd == NodeIter() ? prg.getLastNodeIter() : nodeItrEnd, true);

	//get all the used Qubits and classBits
	getAllUsedQuBits(m_prog, m_quantum_bits_in_use);
	getAllUsedClassBits(m_prog, m_class_bits_in_use);
}

QCircuitToTextPic::~QCircuitToTextPic()
{
	if (nullptr != m_p_text)
	{
		delete m_p_text;
	}
}

void QCircuitToTextPic::textDraw()
{
	/*Do some preparations*/
	//layer
	layer();

	//draw
	if (nullptr != m_p_text)
	{
		delete m_p_text;
		m_p_text = nullptr;
	}

	m_p_text = new(std::nothrow) TextPic(seq, grapth_dag.getProgDAG());
	if (nullptr == m_p_text)
	{
		throw runtime_error("Memory error, failed to create TextPic.");
		return;
	}

	m_p_text->init(m_quantum_bits_in_use, m_class_bits_in_use);

	m_p_text->drawBack();

	m_p_text->present();

	delete m_p_text;
	m_p_text = nullptr;
}

void QCircuitToTextPic::layer()
{
	grapth_dag.getMainGraphSequence(m_prog, seq);
}

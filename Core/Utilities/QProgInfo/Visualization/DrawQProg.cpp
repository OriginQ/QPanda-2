#include "Core/Utilities/QProgInfo/Visualization/DrawQProg.h"
#include "Core/Utilities/QProgInfo/Visualization/Draw.h"
#include <stdexcept>
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"

USING_QPANDA
using namespace std;
using namespace DRAW_TEXT_PIC;

#define PRINT_TRACE 0

DrawQProg::DrawQProg(QProg &prg, const NodeIter node_itr_start, const NodeIter node_itr_end)
	: m_p_text(nullptr)
{
	pickUpNode(m_prog, prg, {},
		node_itr_start == NodeIter() ? prg.getFirstNodeIter() : node_itr_start,
		node_itr_end == NodeIter() ? prg.getEndNodeIter() : node_itr_end);

#if PRINT_TRACE
	cout << "got the target tmp-prog:" << endl;
	printAllNodeType(m_prog);
#endif

	//get all the used Qubits and classBits
	get_all_used_qubits(m_prog, m_quantum_bits_in_use);
	get_all_used_class_bits(m_prog, m_class_bits_in_use);
}

DrawQProg::~DrawQProg()
{
	if (nullptr != m_p_text)
	{
		delete m_p_text;
	}
}

string DrawQProg::textDraw(const TEXT_PIC_TYPE t, const std::string config_data /*= CONFIG_PATH*/)
{
	/*Do some preparations*/
	if (m_quantum_bits_in_use.size() == 0)
	{
		return "Null";
	}

	//draw
	if (nullptr != m_p_text)
	{
		delete m_p_text;
		m_p_text = nullptr;
	}

	m_layer_info = prog_layer(m_prog);

	m_p_text = new(std::nothrow) DrawPicture(m_prog, m_layer_info);
	if (nullptr == m_p_text)
	{
		QCERR_AND_THROW(runtime_error, "Memory error, failed to create DrawPicture obj.");
	}

	m_p_text->init(m_quantum_bits_in_use, m_class_bits_in_use);

	if (t == LAYER)
	{
		m_p_text->draw_by_layer();
	}
	else if (t == TIME_SEQUENCE)
	{
		m_p_text->draw_by_time_sequence(config_data);
	}
	else
	{
		throw runtime_error("Unknow text-pic type, failed to draw Text-Pic.");
	}
	
	string outputStr = m_p_text->present();

	delete m_p_text;
	m_p_text = nullptr;

	return outputStr;
}
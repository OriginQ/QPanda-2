#include "Core/Utilities/QProgInfo/Visualization/DrawQProg.h"
#include "Core/Utilities/QProgInfo/Visualization/DrawTextPic.h"
#include "Core/Utilities/QProgInfo/Visualization/DrawLatex.h"
#include <stdexcept>
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"

USING_QPANDA
using namespace std;
using namespace DRAW_TEXT_PIC;

#define PRINT_TRACE 0

DrawQProg::DrawQProg(QProg &prg, const NodeIter node_itr_start, const NodeIter node_itr_end, const std::string& output_file /*= ""*/)
	: m_drawer(nullptr), m_output_file(output_file)
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
	if (nullptr != m_drawer)
	{
		delete m_drawer;
	}
}

string DrawQProg::textDraw(const LAYER_TYPE t, PIC_TYPE p /*= PIC_TYPE::TEXT*/, bool with_logo /* = false */, uint32_t length /*= 100*/, const std::string config_data /*= CONFIG_PATH*/)
{
	/*Do some preparations*/
	if (m_quantum_bits_in_use.size() == 0)
	{
		return "Null";
	}

	//draw
	if (nullptr != m_drawer)
	{
		delete m_drawer;
		m_drawer = nullptr;
	}

	if (t == LAYER)
	{
		m_layer_info = prog_layer(m_prog);
	}
	else if (t == TIME_SEQUENCE)
	{
		m_layer_info = get_clock_layer(m_prog, config_data);
	}

	if (PIC_TYPE::TEXT == p)
	{
		m_drawer = new(std::nothrow) DrawPicture(m_prog, m_layer_info, length);
	}else if(PIC_TYPE::LATEX  == p){
		m_drawer = new(std::nothrow) DrawLatex(m_prog, m_layer_info, length);
	}else
	{
		QCERR_AND_THROW(runtime_error, "Unknow text-pic type, failed to draw Pic.")
	}
	
	if (nullptr == m_drawer)
	{
		QCERR_AND_THROW(runtime_error, "Memory error, failed to create DrawPicture obj.");
	}

	m_drawer->init(m_quantum_bits_in_use, m_class_bits_in_use);

	if (t == LAYER)
	{
		m_drawer->draw_by_layer();
	}
	else if (t == TIME_SEQUENCE)
	{
		m_drawer->draw_by_time_sequence(config_data);
	}
	else
	{
		throw runtime_error("Unknow text-pic type, failed to draw Text-Pic.");
	}

	if (PIC_TYPE::LATEX == p && with_logo)
	{
		dynamic_cast<DrawLatex *>(m_drawer)->setLogo();
	}
	
	
	string outputStr = m_drawer->present(m_output_file);

	delete m_drawer;
	m_drawer = nullptr;

	return outputStr;
}
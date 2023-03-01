#include "Core/Utilities/QProgInfo/Visualization/QVisualization.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"

USING_QPANDA
using namespace std;
using namespace DRAW_TEXT_PIC;

string QPanda::draw_qprog(QProg prog, PIC_TYPE p /* = PIC_TYPE::TEXT */, bool with_logo /* = false */, uint32_t length /*= 100*/, const std::string &output_file /*= ""*/,
						  const NodeIter itr_start /* = NodeIter()*/, const NodeIter itr_end /* = NodeIter()*/)
{
	DrawQProg test_text_pic(prog, itr_start, itr_end, output_file);
	return test_text_pic.textDraw(LAYER, p, with_logo, length);
}

std::string QPanda::draw_qprog(QProg prog, LayeredTopoSeq &m_layer_info, PIC_TYPE p /* = PIC_TYPE::TEXT */, bool with_logo /* = false */, uint32_t length /*= 100*/,
							   const std::string &output_file /*= ""*/)
{
	std::vector<int> quantum_bits_in_use;
	std::vector<int> class_bits_in_use;
	get_all_used_qubits(prog, quantum_bits_in_use);
	get_all_used_class_bits(prog, class_bits_in_use);
	if (quantum_bits_in_use.size() == 0)
	{
		return "Null";
	}

	AbstractDraw *drawer = nullptr;
	if (PIC_TYPE::TEXT == p)
	{
		drawer = new DrawPicture(prog, m_layer_info, length);
	}
	else if (PIC_TYPE::LATEX == p)
	{
		drawer = new DrawLatex(prog, m_layer_info, length);
	}
    else
    {
        throw std::invalid_argument("Error: PIC_TYPE");
    }

    drawer->init(quantum_bits_in_use, class_bits_in_use);
    drawer->draw_by_layer();

	if (PIC_TYPE::LATEX == p && with_logo)
	{
        dynamic_cast<DrawLatex *>(drawer)->set_logo();
	}

	auto text_pic_str = drawer->present(output_file);

	delete drawer;
	drawer = nullptr;
	return text_pic_str;
}

std::string QPanda::draw_qprog_with_clock(QProg prog, PIC_TYPE p /* = PIC_TYPE::TEXT */, const std::string config_data /*= CONFIG_PATH*/, bool with_logo /* = false */,
										  uint32_t length /*= 100*/, const std::string &output_file /*= ""*/, const NodeIter itr_start /* = NodeIter()*/, const NodeIter itr_end /*= NodeIter()*/)
{
	DrawQProg test_text_pic(prog, itr_start, itr_end, output_file);
	return test_text_pic.textDraw(TIME_SEQUENCE, p, with_logo, length, config_data);
}

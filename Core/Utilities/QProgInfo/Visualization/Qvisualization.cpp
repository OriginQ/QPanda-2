#include "Core/Utilities/QProgInfo/Visualization/QVisualization.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"

USING_QPANDA
using namespace std;
using namespace DRAW_TEXT_PIC;

string QPanda::draw_qprog(QProg prog, const NodeIter itr_start/* = NodeIter()*/, const NodeIter itr_end/* = NodeIter()*/)
{
	DrawQProg test_text_pic(prog, itr_start, itr_end);
	return test_text_pic.textDraw(LAYER);
}

std::string QPanda::draw_qprog(QProg prog, LayeredTopoSeq& m_layer_info)
{
	std::vector<int> quantum_bits_in_use;
	std::vector<int> class_bits_in_use;
	get_all_used_qubits(prog, quantum_bits_in_use);
	get_all_used_class_bits(prog, class_bits_in_use);
	if (quantum_bits_in_use.size() == 0)
	{
		return "Null";
	}

	DrawPicture text_drawer(prog, m_layer_info);

	text_drawer.init(quantum_bits_in_use, class_bits_in_use);

	text_drawer.draw_by_layer();

	auto text_pic_str = text_drawer.present();

	return text_pic_str;
}

std::string QPanda::draw_qprog_with_clock(QProg prog, const std::string config_data /*= CONFIG_PATH*/, 
	const NodeIter itr_start/* = NodeIter()*/, const NodeIter itr_end /*= NodeIter()*/)
{
	DrawQProg test_text_pic(prog, itr_start, itr_end);
	return test_text_pic.textDraw(TIME_SEQUENCE, config_data);
}

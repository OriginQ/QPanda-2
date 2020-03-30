#include "Core/Utilities/QProgInfo/Visualization/QVisualization.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"

USING_QPANDA
using namespace std;

string QPanda::draw_qprog(QProg prog, const NodeIter itr_start/* = NodeIter()*/, const NodeIter itr_end/* = NodeIter()*/)
{
	DrawQProg test_text_pic(prog, itr_start, itr_end);
	return test_text_pic.textDraw(LAYER);
}

std::string QPanda::draw_qprog_with_clock(QProg prog, const NodeIter itr_start/* = NodeIter()*/, const NodeIter itr_end /*= NodeIter()*/)
{
	DrawQProg test_text_pic(prog, itr_start, itr_end);
	return test_text_pic.textDraw(TIME_SEQUENCE);
}
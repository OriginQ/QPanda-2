#include "Core/Utilities/Tools/ProcessOnTraversing.h"
#include "Core/Utilities/QProgInfo/Visualization/QVisualization.h"

USING_QPANDA

#define PRINT_TRACE 0
#if PRINT_TRACE
#define PTrace printf
#define PTraceMat(mat) (std::cout << (mat) << endl)
#define PTraceCircuit(cir) (std::cout << cir << endl)
#else
#define PTrace
#define PTraceMat(mat)
#define PTraceCircuit(cir)
#endif

#define MAX_INCLUDE_LAYERS 8192
#define MIN_INCLUDE_LAYERS 10

/*******************************************************************
*                      class ProcessOnTraversing
********************************************************************/
void ProcessOnTraversing::run_traversal(QProg src_prog, const QVec qubits /*= {}*/)
{
	if (qubits.size() == 0)
	{
		get_all_used_qubits(src_prog, m_qubits);
	}
	else
	{
		m_qubits = qubits;
	}

	init_gate_buf();

	traverse_qprog(src_prog);
	PTrace("finished traverse_qprog.");

	//At the end of the traversal, call process again and clear all the gate-buf
	do_process(true);
}

void ProcessOnTraversing::execute(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node,
	QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	// push_back cur_node_iter to related qubits
	add_gate_to_buffer(cur_node_iter, cir_param, parent_node, m_cur_gates_buffer);

	//optimizer
	size_t min_include_layers = get_min_include_layers();
	if (min_include_layers > MAX_INCLUDE_LAYERS)
	{
		do_process(false);
	}
}

void ProcessOnTraversing::execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node,
	QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	//handle measure node
	add_non_gate_to_buffer(cur_node_iter, MEASURE_GATE, { cur_node->getQuBit() }, cir_param, m_cur_gates_buffer);
}

void ProcessOnTraversing::execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node,
	QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	//handle reset node
	add_non_gate_to_buffer(cur_node_iter, RESET_NODE, { cur_node->getQuBit() }, cir_param, m_cur_gates_buffer);
}

void ProcessOnTraversing::execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node,
	QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	// handle flow control node
	Traversal::traversal(cur_node, *this, cir_param, cur_node_iter);
}

void ProcessOnTraversing::execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node,
	QCircuitParam &cir_param, NodeIter& cur_node_iter)
{}

void ProcessOnTraversing::execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node,
	QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	TraverseByNodeIter::execute(cur_node, parent_node, cir_param, cur_node_iter);
}

void ProcessOnTraversing::execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node,
	QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	TraverseByNodeIter::execute(cur_node, parent_node, cir_param, cur_node_iter);
}

void ProcessOnTraversing::add_gate_to_buffer(NodeIter iter, QCircuitParam &cir_param, std::shared_ptr<QNode> parent_node, OptimizerSink& gates_buffer)
{
	auto gate_node = std::dynamic_pointer_cast<AbstractQGateNode>(*iter);
	QVec gate_qubits;
	gate_node->getQuBitVector(gate_qubits);

	QVec target_qubits_int;
	QVec control_qubits_int;
	for (auto i : gate_qubits)
	{
		target_qubits_int.push_back(i);
	}

	QVec control_qubits;
	gate_node->getControlVector(control_qubits);
	for (auto i : control_qubits)
	{
		control_qubits_int.push_back(i);
	}
	for (auto i : cir_param.m_control_qubits)
	{
		control_qubits_int.push_back(i);
	}

	QVec total_qubits;
	total_qubits.insert(total_qubits.end(), target_qubits_int.begin(), target_qubits_int.end());
	total_qubits.insert(total_qubits.end(), control_qubits_int.begin(), control_qubits_int.end());
	size_t layer = get_node_layer(total_qubits, gates_buffer);
	//PTrace("On layer: %lld\n", layer);

	pOptimizerNodeInfo tmp_node = std::make_shared<OptimizerNodeInfo>(iter, layer,
		target_qubits_int, control_qubits_int, (GateType)(gate_node->getQGate()->getGateType()), parent_node, cir_param.m_is_dagger);
	for (const auto& i : total_qubits)
	{
		gates_buffer.at(i->get_phy_addr()).push_back(tmp_node);
	}

	//PTrace("Finished add_gate_to_buffer.\n");
}

void ProcessOnTraversing::add_non_gate_to_buffer(NodeIter iter, NodeType node_type, QVec gate_qubits, QCircuitParam &cir_param,
	OptimizerSink& gates_buffer, std::shared_ptr<QNode> parent_node/* = nullptr*/)
{
	switch (node_type)
	{
	case CIRCUIT_NODE:
	case PROG_NODE:
	case MEASURE_GATE:
	case RESET_NODE:
	{
		QVec tmp_control_qubits;
		for (auto& qubit : cir_param.m_control_qubits)
		{
			tmp_control_qubits.push_back(qubit);
		}
		size_t layer = get_node_layer(gate_qubits, gates_buffer);
		int t = -1;
		if (MEASURE_GATE == node_type)
		{
			t = SequenceNodeType::MEASURE;
		}
		else if (RESET_NODE == node_type)
		{
			t = SequenceNodeType::RESET;
		}
		else
		{
			t = -3;
		}
		pOptimizerNodeInfo tmp_node = std::make_shared<OptimizerNodeInfo>(iter, layer, gate_qubits, tmp_control_qubits, (GateType)t, parent_node, false);
		for (const auto& i : gate_qubits)
		{
			gates_buffer.at(i->get_phy_addr()).push_back(tmp_node);
		}
	}
	break;

	case WHILE_START_NODE:
	case QIF_START_NODE:
		//do nothing
		break;

	case CLASS_COND_NODE:
	case QWAIT_NODE:
		//do nothing
		break;

	default:
		QCERR_AND_THROW_ERRSTR(run_fail, "Error: Node type error.");
		break;
	}
}

size_t ProcessOnTraversing::get_node_layer(QVec gate_qubits, OptimizerSink& gate_buffer)
{
	size_t next_layer = 0;
	for (auto i : gate_qubits)
	{
		std::list<pOptimizerNodeInfo> &vec = gate_buffer.at(i->get_phy_addr());
		if (!(vec.empty()))
		{
			size_t layer_increase = 1;
			/*if (BARRIER_GATE == vec.back()->m_type)
			{
				layer_increase = 0;
			}*/
			size_t tmp_layer = vec.back()->m_layer + layer_increase;
			if (tmp_layer > next_layer)
			{
				next_layer = tmp_layer;
			}
		}
	}

	return next_layer;
}

size_t ProcessOnTraversing::get_min_include_layers()
{
	size_t include_min_layers = MAX_LAYER;
	m_min_layer = MAX_LAYER;
	for (auto item : m_cur_gates_buffer)
	{
		const auto &vec = item.second;
		if (vec.empty())
		{
			include_min_layers = 0;
			m_min_layer = 0;
		}
		else
		{
			const size_t tmp_include_layer = vec.back()->m_layer - vec.front()->m_layer + 1;
			if (tmp_include_layer < include_min_layers)
			{
				include_min_layers = tmp_include_layer;
			}

			if (m_min_layer > vec.back()->m_layer)
			{
				m_min_layer = vec.back()->m_layer;
			}
		}
	}

	return include_min_layers;
}

void ProcessOnTraversing::gates_sink_to_topolog_sequence(OptimizerSink& gate_buf, TopologSequence<pOptimizerNodeInfo>& seq, const size_t max_output_layer /*= MAX_SIZE*/)
{
	if (gate_buf.begin()->second.size() == 0)
	{
		QCERR_AND_THROW_ERRSTR(run_fail, "Error: unknown error on gates_sink_to_topolog_sequence.");
	}
	size_t min_layer = gate_buf.begin()->second.front()->m_layer;
	for (auto &item : gate_buf)
	{
		if (item.second.size() == 0)
		{
			QCERR_AND_THROW_ERRSTR(run_fail, "Error: unknown error on gates_sink_to_topolog_sequence.");
		}
		if (item.second.front()->m_layer < min_layer)
		{
			min_layer = item.second.front()->m_layer;
		}
	}

	seq.clear();
	for (auto &item : gate_buf)
	{
		const size_t max_layer = item.second.back()->m_layer - min_layer + 1;
		if (seq.size() < max_layer)
		{
			seq.resize(max_layer);
		}

		//Traverse the gates of each qubit
		for (auto itr = item.second.begin(); itr != item.second.end(); ++itr)
		{
			if (max_output_layer <= (*itr)->m_layer)
			{
				break;
			}

			const size_t cur_layer = (*itr)->m_layer - min_layer;
			std::vector<pOptimizerNodeInfo> next_adja_nodes;
			auto tmp_iter = itr;
			if (item.second.end() != (++tmp_iter))
			{
				//append next addacent node
				next_adja_nodes.push_back(*tmp_iter);
			}

			bool b_already_exist = false;
			for (auto& node_item : seq.at(cur_layer))
			{
				if (node_item.first == *itr)
				{
					b_already_exist = true;
					if (node_item.second.size() == 0)
					{
						/* A null pointer is inserted here to distinguish the correspondence between two qubits of the double gate
						*/
						node_item.second.push_back(nullptr);
					}

					if (next_adja_nodes.size() > 0)
					{
						node_item.second.insert(node_item.second.end(), next_adja_nodes.begin(), next_adja_nodes.end());
					}
					else
					{
						/* A null pointer is inserted here to distinguish the correspondence between two qubits of the double gate
						*/
						node_item.second.push_back(nullptr);
					}
				}
			}

			if (!b_already_exist)
			{
				seq.at(cur_layer).push_back(std::pair<pOptimizerNodeInfo, std::vector<pOptimizerNodeInfo>>(*itr, next_adja_nodes));
			}
		}
	}

	while (seq.size() > 0)
	{
		if (seq.back().size() == 0)
		{
			seq.pop_back();
		}
		else
		{
			break;
		}
	}
}

void ProcessOnTraversing::clean_gate_buf_to_cir(QProg &cir, bool b_clean_all_buf/* = false*/)
{
	get_min_include_layers();

	size_t drop_max_layer = 0;
	if (b_clean_all_buf)
	{
		drop_max_layer = MAX_LAYER;
	}
	else
	{
		if (m_min_layer <= MIN_INCLUDE_LAYERS)
		{
			return;
		}
		drop_max_layer = m_min_layer - MIN_INCLUDE_LAYERS;
	}

	TopologSequence<std::pair<size_t, NodeIter>> tmp_seq;

	for (auto &item : m_cur_gates_buffer)
	{
		auto &vec = item.second;
		while ((0 != vec.size()))
		{
			if ((vec.front()->m_layer < drop_max_layer))
			{
				add_node_to_seq(tmp_seq, vec.front()->m_iter, vec.front()->m_layer);
				vec.erase(vec.begin());
			}
			else
			{
				break;
			}
		}
	}

	seq_to_cir(tmp_seq, cir);
}

void ProcessOnTraversing::clean_gate_buf(bool b_clean_all_buf /*= false*/)
{
	get_min_include_layers();

	size_t drop_max_layer = 0;
	if (b_clean_all_buf)
	{
		drop_max_layer = MAX_LAYER;
	}
	else
	{
		if (m_min_layer <= MIN_INCLUDE_LAYERS)
		{
			return;
		}
		drop_max_layer = m_min_layer - MIN_INCLUDE_LAYERS;
	}

	drop_gates(drop_max_layer);
}

void ProcessOnTraversing::drop_gates(const size_t max_drop_layer)
{
	for (auto &item : m_cur_gates_buffer)
	{
		auto &vec = item.second;
		while ((0 != vec.size()))
		{
			if ((vec.front()->m_layer < max_drop_layer))
			{
				vec.erase(vec.begin());
			}
			else
			{
				break;
			}
		}
	}
}

void ProcessOnTraversing::add_node_to_seq(layer_iter_seq &tmp_seq, NodeIter node_iter, const size_t layer)
{
	using tmp_seq_node_type = std::pair<size_t, NodeIter>;
	if ((tmp_seq.size() == 0) || (layer > tmp_seq.back().front().first.first))
	{
		SeqLayer<tmp_seq_node_type> tmp_layer;
		tmp_layer.push_back(SeqNode<tmp_seq_node_type>(tmp_seq_node_type(layer, node_iter), std::vector<tmp_seq_node_type>()));
		tmp_seq.push_back(tmp_layer);
		return;
	}
	if (layer < tmp_seq.at(0).front().first.first)
	{
		SeqLayer<tmp_seq_node_type> tmp_layer;
		tmp_layer.push_back(SeqNode<tmp_seq_node_type>(tmp_seq_node_type(layer, node_iter), std::vector<tmp_seq_node_type>()));
		tmp_seq.insert(tmp_seq.begin(), tmp_layer);
		return;
	}

	for (size_t i = 0; i < tmp_seq.size(); ++i)
	{
		if (layer == tmp_seq.at(i).front().first.first)
		{
			bool b_repeat_exist = false;
			for (auto& itr_tmp : tmp_seq.at(i))
			{
				if ((itr_tmp).first.second == node_iter)
				{
					b_repeat_exist = true;
				}
			}
			if (!b_repeat_exist)
			{
				tmp_seq.at(i).push_back(SeqNode<tmp_seq_node_type>(tmp_seq_node_type(layer, node_iter), std::vector<tmp_seq_node_type>()));
			}
		}
		else if ((layer < tmp_seq.at(i).front().first.first) && (layer > tmp_seq.at(i - 1).front().first.first))
		{
			SeqLayer<tmp_seq_node_type> tmp_layer;
			tmp_layer.push_back(SeqNode<tmp_seq_node_type>(tmp_seq_node_type(layer, node_iter), std::vector<tmp_seq_node_type>()));
			tmp_seq.insert(tmp_seq.begin() + i, tmp_layer);
		}
	}
}

void ProcessOnTraversing::seq_to_cir(layer_iter_seq &tmp_seq, QProg& prog)
{
	for (auto& layer_iter : tmp_seq)
	{
		for (auto& node_iter : layer_iter)
		{
			prog.pushBackNode(*(node_iter.first.second));
		}
	}
}

void ProcessOnTraversing::seq_to_cir(layer_iter_seq &tmp_seq, QProg& prog, const size_t start_layer_to_cir, const size_t max_output_layer)
{
	for (auto layer_iter = tmp_seq.begin(); layer_iter != tmp_seq.end();)
	{
		const size_t cur_layer = layer_iter->front().first.first;
		if (cur_layer > max_output_layer)
		{
			break;
		}

		for (auto& node_iter : *layer_iter)
		{
			prog.pushBackNode(*(node_iter.first.second));
		}

		layer_iter = tmp_seq.erase(layer_iter);
	}
}

/*******************************************************************
*                      class QProgLayer
********************************************************************/
void QProgLayer::process(const bool on_travel_end/* = false*/)
{
	if (m_cur_gates_buffer.size() == 0)
	{
		return;
	}

	get_min_include_layers();

	size_t drop_max_layer = 0;
	if (on_travel_end)
	{
		drop_max_layer = MAX_LAYER;
	}
	else
	{
		if (m_min_layer <= MIN_INCLUDE_LAYERS)
		{
			return;
		}
		drop_max_layer = m_min_layer - MIN_INCLUDE_LAYERS;
	}


	//transfer gate sink to topolog sequence
	TopologSequence<pOptimizerNodeInfo> tmp_topolog_sequence;
	gates_sink_to_topolog_sequence(m_cur_gates_buffer, tmp_topolog_sequence, drop_max_layer);
	

	//update gate sink
	append_topolog_seq(tmp_topolog_sequence);

	drop_gates(drop_max_layer);
}

void QProgLayer::append_topolog_seq(TopologSequence<pOptimizerNodeInfo>& tmp_seq)
{
	m_topolog_sequence.insert(m_topolog_sequence.end(), tmp_seq.begin(), tmp_seq.end());
}

/*******************************************************************
*                      public interface
********************************************************************/
static void divide_layer_by_double_gate(TopologSequence<pOptimizerNodeInfo>& seq)
{
	for (auto itr_single_layer = seq.begin(); itr_single_layer != seq.end(); ++itr_single_layer)
	{
		auto& cur_layer = *itr_single_layer;
		if (cur_layer.size() == 0)
		{
			continue;
		}

		size_t echo_gate_cnt = 0;
		for (auto itr_seq_node_item = cur_layer.begin(); itr_seq_node_item != cur_layer.end(); ++itr_seq_node_item)
		{
			if (ECHO_GATE == itr_seq_node_item->first->m_type)
			{
				++echo_gate_cnt;
			}
		}

		if ((cur_layer.size() - echo_gate_cnt) < 2)
		{
			continue;
		}

		std::vector<SeqLayer<pOptimizerNodeInfo>> new_layer_vec;
		for (auto itr_seq_node_item = cur_layer.begin(); itr_seq_node_item != cur_layer.end(); )
		{
			if ((itr_seq_node_item->first->m_target_qubits.size() > 1) && (cur_layer.size() > 1))
			{
				SeqLayer<pOptimizerNodeInfo> new_layer;
				new_layer.push_back(*itr_seq_node_item);
				itr_seq_node_item = cur_layer.erase(itr_seq_node_item);
				new_layer_vec.push_back(new_layer);
			}
			else
			{
				++itr_seq_node_item;
			}
		}

		for (auto& new_layer : new_layer_vec)
		{
			itr_single_layer = seq.insert(itr_single_layer, new_layer);
			++itr_single_layer;
		}
	}
}

static bool get_cur_layer_used_qubits(const SeqLayer<pOptimizerNodeInfo>& cur_layer, QVec& used_qubits)
{
	used_qubits.clear();
	for (auto itr_seq_node_item = cur_layer.begin(); itr_seq_node_item != cur_layer.end(); ++itr_seq_node_item)
	{
		auto& n = itr_seq_node_item->first;
		if (SequenceNodeType::MEASURE == n->m_type)
		{
			std::shared_ptr<AbstractQuantumMeasure> p_measure = std::dynamic_pointer_cast<AbstractQuantumMeasure>(*(n->m_iter));
			used_qubits.push_back(p_measure->getQuBit());
		}
		else if (SequenceNodeType::RESET == n->m_type)
		{
			std::shared_ptr<AbstractQuantumReset> p_reset = std::dynamic_pointer_cast<AbstractQuantumReset>(*(n->m_iter));
			used_qubits.push_back(p_reset->getQuBit());
		}
		else
		{
			std::shared_ptr<AbstractQGateNode> p_gate = std::dynamic_pointer_cast<AbstractQGateNode>(*(n->m_iter));
			QGate tmp_gate_node(p_gate);
			QVec gate_qubits;
			p_gate->getQuBitVector(gate_qubits);
			p_gate->getControlVector(gate_qubits);
			used_qubits.insert(used_qubits.end(), gate_qubits.begin(), gate_qubits.end());

			if ((gate_qubits.size() > 1) && (BARRIER_GATE != p_gate->getQGate()->getGateType()))
			{
				return true; // Returns true if there are double gates
			}
		}
	}

	return false;
}

static void fill_layer_by_next_layer_gate(QVec& unused_qubits, SeqLayer<pOptimizerNodeInfo>& cur_layer, SeqLayer<pOptimizerNodeInfo>& next_layer)
{
	QVec filled_qubits;
	bool b_filled = false;
	for (auto unused_qubit_iter = unused_qubits.begin(); unused_qubit_iter != unused_qubits.end(); ++unused_qubit_iter)
	{
		b_filled = false;
		for (auto filled_q : filled_qubits)
		{
			if ((*unused_qubit_iter)->get_phy_addr() == filled_q->get_phy_addr())
			{
				b_filled = true;
				break;
			}
		}

		if (b_filled)
		{
			continue;
		}

		for (auto itr_seq_node_item = next_layer.begin(); itr_seq_node_item != next_layer.end(); ++itr_seq_node_item)
		{
			auto& n = itr_seq_node_item->first;
			if (SequenceNodeType::MEASURE == n->m_type)
			{
				std::shared_ptr<AbstractQuantumMeasure> p_measure = std::dynamic_pointer_cast<AbstractQuantumMeasure>(*(n->m_iter));
				if (p_measure->getQuBit()->get_phy_addr() == (*unused_qubit_iter)->get_phy_addr())
				{
					filled_qubits.push_back(*unused_qubit_iter);
				}
			}
			else if (SequenceNodeType::RESET == n->m_type)
			{
				std::shared_ptr<AbstractQuantumReset> p_reset = std::dynamic_pointer_cast<AbstractQuantumReset>(*(n->m_iter));
				if (p_reset->getQuBit()->get_phy_addr() == (*unused_qubit_iter)->get_phy_addr())
				{
					filled_qubits.push_back(*unused_qubit_iter);

					cur_layer.push_back(*itr_seq_node_item);
					next_layer.erase(itr_seq_node_item);
					break;
				}
			}
			else
			{
				std::shared_ptr<AbstractQGateNode> p_gate = std::dynamic_pointer_cast<AbstractQGateNode>(*(n->m_iter));
				QGate tmp_gate_node(p_gate);
				QVec gate_qubits;
				p_gate->getQuBitVector(gate_qubits);
				p_gate->getControlVector(gate_qubits);
				if ((gate_qubits.size() > 1) && (BARRIER_GATE != p_gate->getQGate()->getGateType()))
				{
					QCERR_AND_THROW_ERRSTR(run_fail, "Error: unsupportedMulti-control-gate on fill_layer_by_next_layer_gate.");
				}

				filled_qubits += gate_qubits;
				QVec clash_qubits = gate_qubits - unused_qubits;
				if (clash_qubits.size() == 0)
				{
					// No conflict qubits, indicates that qu-gates can be filled
					//filled_qubits.insert(filled_qubits.end(), gate_qubits.begin(), gate_qubits.end());

					cur_layer.push_back(*itr_seq_node_item);
					next_layer.erase(itr_seq_node_item);
					break;
				}
			}
		}
	}

	unused_qubits -= filled_qubits;
}

static void fill_layer(TopologSequence<pOptimizerNodeInfo>& seq, QProg src_prog)
{
	QVec all_used_qubits;
	get_all_used_qubits(src_prog, all_used_qubits);

	QVec vec_qubits_used_in_layer;
	for (auto itr_single_layer = seq.begin(); itr_single_layer != seq.end(); )
	{
		auto& cur_layer = *itr_single_layer;

		/** 
		*  Get the qubit used by the current layer, exit when it encounters double gates,
		*  and directly process the data of the next layer
		*/
		vec_qubits_used_in_layer.clear();
		if (!get_cur_layer_used_qubits(cur_layer, vec_qubits_used_in_layer))
		{
			if (vec_qubits_used_in_layer.size() == 0)
			{
				itr_single_layer = seq.erase(itr_single_layer);
				continue;
			}

			QVec unused_qubits = all_used_qubits - vec_qubits_used_in_layer;
			auto next_layer_itr = itr_single_layer + 1;
			if (seq.end() == next_layer_itr)
			{
				break;
			}

			while (unused_qubits.size() > 0)
			{
				QVec vec_qubits_used_in_next_layer;
				bool b_exist_double_gate = get_cur_layer_used_qubits(*next_layer_itr, vec_qubits_used_in_next_layer);
				if (b_exist_double_gate)
				{
					unused_qubits -= vec_qubits_used_in_next_layer;
				}
				else
				{
					QVec tmp_qubits = unused_qubits - vec_qubits_used_in_next_layer;
					if (tmp_qubits.size() != unused_qubits.size())
					{
						fill_layer_by_next_layer_gate(unused_qubits, cur_layer, *next_layer_itr);
					}
				}

				if (seq.end() == ++next_layer_itr)
				{
					break;
				}
			}
		}

		++itr_single_layer;
	}
}

static void move_measure_to_last_layer(TopologSequence<pOptimizerNodeInfo>& seq) {
	if (seq.size() == 0)
	{
		return;
	}

	auto& last_layer = seq.back();
	bool b_exist_measure_node = false;
	bool b_exist_gate_node = false;
	for (const auto& gate_item : last_layer)
	{
		const NodeType tmp_node_type = (*gate_item.first->m_iter)->getNodeType();
		if (tmp_node_type == MEASURE_GATE)
		{
			b_exist_measure_node = true;
		}
		else if (tmp_node_type == GATE_NODE)
		{
			b_exist_gate_node = true;
		}
		else
		{
			QCERR_AND_THROW_ERRSTR(run_fail, "Error: error node type in last layer.");
		}
	}

	if (b_exist_gate_node)
	{
		seq.push_back(SeqLayer<pOptimizerNodeInfo>());
	}

	auto& real_last_layer = seq.back();

	for (auto layer_iter = seq.begin(); layer_iter != (seq.end() - 1); ++layer_iter)
	{
		auto& cur_layer = *layer_iter;
		for (auto gate_itr = cur_layer.begin(); gate_itr != cur_layer.end();)
		{
			const NodeType t = (*(gate_itr->first->m_iter))->getNodeType();
			if (t == MEASURE_GATE)
			{
				real_last_layer.push_back(*gate_itr);
				gate_itr = cur_layer.erase(gate_itr);
			}
			else
			{
				++gate_itr;
			}
		}
	}
}

const TopologSequence<pOptimizerNodeInfo> QPanda::prog_layer(QProg src_prog, const bool b_double_gate_one_layer/* = false*/)
{
	QProgLayer q_layer;
	q_layer.layer(src_prog);

	TopologSequence<pOptimizerNodeInfo> seq = q_layer.get_topo_seq();

	if (b_double_gate_one_layer)
	{
		divide_layer_by_double_gate(seq);
#if PRINT_TRACE
		std::cout << draw_qprog(src_prog, seq) << std::endl;
#endif
		fill_layer(seq, src_prog);
#if PRINT_TRACE
		std::cout << draw_qprog(src_prog, seq) << std::endl;
#endif

		move_measure_to_last_layer(seq);
	}

	return seq;
}

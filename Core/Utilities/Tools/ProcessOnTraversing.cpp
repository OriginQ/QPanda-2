#include "Core/Utilities/Tools/ProcessOnTraversing.h"
#include "Core/Utilities/QProgInfo/Visualization/QVisualization.h"
#include <chrono>

USING_QPANDA
using namespace std;

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

#define MAX_INCLUDE_LAYERS 1024
#define MIN_INCLUDE_LAYERS 10
#define COMPENSATE_GATE_TYPE 0XFFFF

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
		target_qubits_int, control_qubits_int, (GateType)(gate_node->getQGate()->getGateType()), parent_node, 
		check_dagger(gate_node, (gate_node->isDagger() ^ cir_param.m_is_dagger)));
	for (const auto& i : total_qubits)
	{
		append_data_to_gate_buf(gates_buffer.at(i->get_phy_addr()), tmp_node, i->get_phy_addr());
	}

	//PTrace("Finished add_gate_to_buffer.\n");
}

void ProcessOnTraversing::append_data_to_gate_buf(std::vector<pOptimizerNodeInfo>& gate_buf,
	pOptimizerNodeInfo p_node, const size_t qubit_i) {
	auto &tmp_pos = m_cur_buffer_pos.at(qubit_i);
	if (gate_buf.size() <= (tmp_pos))
	{
		gate_buf.push_back(p_node);
	}
	else
	{
		gate_buf[tmp_pos] = p_node;
	}
	++tmp_pos;
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
		QCERR_AND_THROW(run_fail, "Error: Node type error.");
		break;
	}
}

size_t ProcessOnTraversing::get_node_layer(QVec gate_qubits, OptimizerSink& gate_buffer)
{
	std::vector<int> gate_qubits_i;
	for (auto i : gate_qubits)
	{
		gate_qubits_i.push_back(i->get_phy_addr());
	}

	return get_node_layer(gate_qubits_i, gate_buffer);
}

size_t ProcessOnTraversing::get_node_layer(const std::vector<int>& gate_qubits, OptimizerSink& gate_buffer)
{
	size_t next_layer = 0;
	for (auto i : gate_qubits)
	{
		std::vector<pOptimizerNodeInfo> &vec = gate_buffer.at(i);
		const auto& tmp_pos = m_cur_buffer_pos.at(i);
		if (tmp_pos > 0)
		{
			size_t layer_increase = 1;
			/*if (BARRIER_GATE == vec.back()->m_type)
			{
				layer_increase = 0;
			}*/
			size_t tmp_layer = vec[tmp_pos-1]->m_layer + layer_increase;
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
		const auto& tmp_pos = m_cur_buffer_pos.at(item.first);
		if (tmp_pos == 0)
		{
			include_min_layers = 0;
			m_min_layer = 0;
		}
		else
		{
			const size_t tmp_include_layer = vec[tmp_pos-1]->m_layer - vec.front()->m_layer + 1;
			if (tmp_include_layer < include_min_layers)
			{
				include_min_layers = tmp_include_layer;
			}

			if (m_min_layer > vec[tmp_pos - 1]->m_layer)
			{
				m_min_layer = vec[tmp_pos - 1]->m_layer;
			}
		}
	}

	return include_min_layers;
}

void ProcessOnTraversing::gates_sink_to_topolog_sequence(OptimizerSink& gate_buf, LayeredTopoSeq& seq, const size_t max_output_layer /*= MAX_SIZE*/)
{
	if (gate_buf.begin()->second.size() == 0)
	{
		QCERR_AND_THROW(run_fail, "Error: unknown error on gates_sink_to_topolog_sequence.");
	}
	size_t min_layer = gate_buf.begin()->second.front()->m_layer;
	for (auto &item : gate_buf)
	{
		if (item.second.size() == 0)
		{
			QCERR_AND_THROW(run_fail, "Error: unknown error on gates_sink_to_topolog_sequence.");
		}
		if (item.second.front()->m_layer < min_layer)
		{
			min_layer = item.second.front()->m_layer;
		}
	}

	seq.clear();
	for (auto &item : gate_buf)
	{
		const auto& tmp_pos = m_cur_buffer_pos.at(item.first);
		const size_t max_layer = item.second.at(tmp_pos-1)->m_layer - min_layer + 1;
		if (seq.size() < max_layer)
		{
			seq.resize(max_layer);
		}

		//Traverse the gates of each qubit
		for (size_t i = 0; i < tmp_pos; ++i)
		{
			const auto& n = item.second.at(i);
			if (max_output_layer <= n->m_layer)
			{
				break;
			}

			if (COMPENSATE_GATE_TYPE == n->m_type)
			{
				continue;
			}

			const size_t cur_layer = n->m_layer - min_layer;
			std::vector<pOptimizerNodeInfo> next_adja_nodes;
			while (i + 1 < tmp_pos)
			{
				if (COMPENSATE_GATE_TYPE != n->m_type)
				{
					//append next addacent node
					next_adja_nodes.push_back(item.second.at(i + 1));
					break;
				}
			}

			bool b_already_exist = false;
			for (auto& node_item : seq.at(cur_layer))
			{
				if (node_item.first == n)
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
				seq.at(cur_layer).push_back(std::pair<pOptimizerNodeInfo, std::vector<pOptimizerNodeInfo>>(n, next_adja_nodes));
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
		auto& tmp_pos = m_cur_buffer_pos.at(item.first);
		while ((0 < tmp_pos))
		{
			if ((vec.front()->m_layer < drop_max_layer))
			{
				add_node_to_seq(tmp_seq, vec.front()->m_iter, vec.front()->m_layer);
				vec.erase(vec.begin());
				--tmp_pos;
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
		const auto& tmp_pos = m_cur_buffer_pos.at(item.first);
		size_t i = 0;
		size_t j = 0;
		for (; i < tmp_pos; ++i)
		{
			if ((vec.at(i)->m_layer >= max_drop_layer))
			{
				vec.at(j) = vec.at(i);
				++j;
			}
			vec.at(i) = nullptr;
		}

		//// 有没有 vector的块操作？？？？？？


		/*if (item.first == 9)
		{
			cout << "on drop_gates on qubit_9..........., size = "<< j << endl;
		}*/
		m_cur_buffer_pos.at(item.first) = j;
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
	if (layer < tmp_seq.front().front().first.first)
	{
		SeqLayer<tmp_seq_node_type> tmp_layer;
		tmp_layer.push_back(SeqNode<tmp_seq_node_type>(tmp_seq_node_type(layer, node_iter), std::vector<tmp_seq_node_type>()));
		tmp_seq.insert(tmp_seq.begin(), tmp_layer);
		return;
	}

	for (auto layer_itr = tmp_seq.begin(); layer_itr != tmp_seq.end(); ++layer_itr)
	{
		if (layer == layer_itr->front().first.first)
		{
			bool b_repeat_exist = false;
			for (auto& itr_tmp : *layer_itr)
			{
				if ((itr_tmp).first.second == node_iter)
				{
					b_repeat_exist = true;
				}
			}
			if (!b_repeat_exist)
			{
				layer_itr->push_back(SeqNode<tmp_seq_node_type>(tmp_seq_node_type(layer, node_iter), std::vector<tmp_seq_node_type>()));
			}

			return;
		}
		else if ((layer < layer_itr->front().first.first))
		{
			auto pre_layer_iter = layer_itr;
			--pre_layer_iter;
			if ((layer > pre_layer_iter->front().first.first))
			{
				SeqLayer<tmp_seq_node_type> tmp_layer;
				tmp_layer.push_back(SeqNode<tmp_seq_node_type>(tmp_seq_node_type(layer, node_iter), std::vector<tmp_seq_node_type>()));
				tmp_seq.insert(layer_itr, tmp_layer);
				return;
			}
			else
			{
				QCERR_AND_THROW(run_fail, "Error: failed to add_node_to_seq.");
			}
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

void ProcessOnTraversing::init_gate_buf()
{
	for (const auto& item : m_qubits)
	{
		m_cur_gates_buffer.insert(GatesBufferType(item->getPhysicalQubitPtr()->getQubitAddr(), 
			std::vector<pOptimizerNodeInfo>(MAX_INCLUDE_LAYERS)));
		m_cur_buffer_pos.insert(std::make_pair(item->getPhysicalQubitPtr()->getQubitAddr(), 0));
	}
}

/*******************************************************************
*                      class QProgLayer
********************************************************************/
class QProgLayer : protected ProcessOnTraversing
{
public:
	QProgLayer(const bool b_double_gate_one_layer = false, const std::string& config_data = CONFIG_PATH)
		:m_b_double_gate_one_layer(b_double_gate_one_layer), m_config_data(config_data)
	{
		//read compensate-qubit
		init();
	}
	~QProgLayer() {}

	void init() {
		if (m_b_double_gate_one_layer)
		{
			QuantumChipConfig config_reader;
			config_reader.load_config(m_config_data);
			if (!(config_reader.read_adjacent_matrix(m_qubit_size, m_qubit_topo_matrix)))
			{
				QCERR_AND_THROW(run_fail, "Error: failed to read virtual_Z_config.");
			}

			m_high_frequency_qubits = config_reader.read_high_frequency_qubit();
		}
	}

	void layer(QProg src_prog) { run_traversal(src_prog); }

	void process(const bool on_travel_end = false) override {
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
		LayeredTopoSeq tmp_topolog_sequence;
		gates_sink_to_topolog_sequence(m_cur_gates_buffer, tmp_topolog_sequence, drop_max_layer);

		//update gate sink
		append_topolog_seq(tmp_topolog_sequence);

		drop_gates(drop_max_layer);
	}
	void append_topolog_seq(LayeredTopoSeq& tmp_seq) {
		m_topolog_sequence.insert(m_topolog_sequence.end(), tmp_seq.begin(), tmp_seq.end());
	}

	const LayeredTopoSeq& get_topo_seq() { return m_topolog_sequence; }

protected:
	void add_gate_to_buffer(NodeIter iter, QCircuitParam &cir_param, std::shared_ptr<QNode> parent_node, OptimizerSink& gates_buffer) override {
		auto gate_node = std::dynamic_pointer_cast<AbstractQGateNode>(*iter);
		QVec gate_qubits;
		gate_node->getQuBitVector(gate_qubits);
		QVec control_qubits;
		gate_node->getControlVector(control_qubits);
		control_qubits += cir_param.m_control_qubits;

		QVec total_qubits = gate_qubits + control_qubits;
		std::vector<int> relation_qubits;
		if (m_b_double_gate_one_layer && (total_qubits.size() > 1))
		{
			//append relation qubits
			for (auto  tmp_qubit : total_qubits)
			{
				for (const auto& h : m_high_frequency_qubits)
				{
					if (h == tmp_qubit->get_phy_addr())
					{
						//get adjacent qubit
						for (int a = 0; a < m_qubit_topo_matrix[h].size(); ++a)
						{
							if (0 != m_qubit_topo_matrix[h][a])
							{
								if (gates_buffer.end() != gates_buffer.find(a))
								{
									relation_qubits.push_back(a);
								}
							}
						}
					}
				}
			}
		}

		for (auto relation_qubit_itr = relation_qubits.begin(); relation_qubit_itr != relation_qubits.end(); )
		{
			bool b = false;
			for (auto &q : total_qubits)
			{
				if (q->get_phy_addr() == (*relation_qubit_itr))
				{
					b = true;
					break;
				}
			}

			if (b)
			{
				relation_qubit_itr = relation_qubits.erase(relation_qubit_itr);
			}
			else
			{
				++relation_qubit_itr;
			}
		}

		std::vector<int> total_qubits_i;
		for (auto q : total_qubits)
		{
			total_qubits_i.push_back(q->get_phy_addr());
		}
		total_qubits_i.insert(total_qubits_i.end(), relation_qubits.begin(), relation_qubits.end());
		size_t layer = get_node_layer(total_qubits_i, gates_buffer);
		//PTrace("On layer: %lld\n", layer);

		pOptimizerNodeInfo tmp_node = std::make_shared<OptimizerNodeInfo>(iter, layer,
			gate_qubits, control_qubits, (GateType)(gate_node->getQGate()->getGateType()), parent_node, 
			check_dagger(gate_node, (gate_node->isDagger() ^ cir_param.m_is_dagger)));
		for (const auto& i : total_qubits)
		{
			const auto qubit_i = i->get_phy_addr();
			append_data_to_gate_buf(gates_buffer.at(qubit_i), tmp_node, qubit_i);
		}

		if (relation_qubits.size() > 0)
		{
			tmp_node = std::make_shared<OptimizerNodeInfo>(NodeIter(), layer,
				QVec(), QVec(), (GateType)(COMPENSATE_GATE_TYPE), nullptr, false);
			for (const auto& i : relation_qubits)
			{
				append_data_to_gate_buf(gates_buffer.at(i), tmp_node, i);
			}
		}
	}

private:
	bool m_b_double_gate_one_layer;
	const std::string m_config_data;
	LayeredTopoSeq m_topolog_sequence;
	std::vector<std::vector<int>> m_qubit_topo_matrix;
	std::vector<int> m_high_frequency_qubits;
	size_t m_qubit_size;
};
#if 1
/*******************************************************************
*                      class QPressedLayer
********************************************************************/
class QPressedLayer : protected ProcessOnTraversing
{
public:
	QPressedLayer() {}
	~QPressedLayer() {}

	void layer(QProg src_prog) { run_traversal(src_prog); }

	void process(const bool on_travel_end = false) override {
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
		PressedTopoSeq tmp_topolog_sequence;
		gates_sink_to_topolog_sequence(m_cur_gates_buffer, tmp_topolog_sequence, drop_max_layer);

		//update gate sink
		append_topolog_seq(tmp_topolog_sequence);

		drop_gates(drop_max_layer);
	}
	void append_topolog_seq(PressedTopoSeq& tmp_seq) {
		m_topolog_sequence.insert(m_topolog_sequence.end(), tmp_seq.begin(), tmp_seq.end());
	}

	const PressedTopoSeq& get_topo_seq() { return m_topolog_sequence; }

protected:
	void gates_sink_to_topolog_sequence(OptimizerSink& gate_buf, PressedTopoSeq& seq,
		const size_t max_output_layer = MAX_LAYER) {
		size_t min_layer = MAX_LAYER;
		for (auto &item : gate_buf)
		{
			if (item.second.size() == 0)
			{
				QCERR_AND_THROW(run_fail, "Error: unknown error on QPressedLayer::gates_sink_to_topolog_sequence.");
			}
			if (item.second.front()->m_layer < min_layer)
			{
				min_layer = item.second.front()->m_layer;
			}
		}

		seq.clear();
		std::map<size_t, size_t> cur_gate_buf_pos;
		std::map<size_t, std::vector<pOptimizerNodeInfo>> qubit_relation_pre_nodes; // only single QGate
		for (auto &item : gate_buf)
		{
			cur_gate_buf_pos.insert(std::make_pair(item.first, 0));
			qubit_relation_pre_nodes.insert(std::make_pair(item.first, std::vector<pOptimizerNodeInfo>()));
		}

		std::vector<std::pair<pOptimizerNodeInfo, std::pair<size_t, size_t>>> candidate_double_gate;
		while (true)
		{
			bool b_finished = true;
			candidate_double_gate.clear();
			for (auto &item : gate_buf)
			{
				const auto& tmp_size = m_cur_buffer_pos.at(item.first);

				//Traverse the gates of each qubit
				auto& i = cur_gate_buf_pos.at(item.first);
				for (; i < tmp_size; ++i)
				{
					const auto& n = item.second.at(i);
					if (max_output_layer <= n->m_layer)
					{
						break;
					}
					b_finished = false;

					if (COMPENSATE_GATE_TYPE == n->m_type)
					{
						++(cur_gate_buf_pos.at(item.first));
						continue;
					}

					if (n->m_control_qubits.size() > 0)
					{
						QCERR_AND_THROW(run_fail, "Error: It is not allowed to have multiple control gates during QPressedLayer.");
					}

					if (n->m_target_qubits.size() == 1)
					{
						qubit_relation_pre_nodes.at(item.first).emplace_back(n);
						continue;
					}
					else if (n->m_target_qubits.size() == 2)
					{
						auto q_1 = n->m_target_qubits.front()->get_phy_addr();
						auto q_2 = n->m_target_qubits.back()->get_phy_addr();
						if (q_1 > q_2) 
						{ 
							auto _q = q_2;
							q_2 = q_1;
							q_1 = _q;
						}
						candidate_double_gate.push_back(std::make_pair(n, std::make_pair(q_1, q_2)));
						break;
					}
				}
			}

			if (b_finished)
			{
				break;
			}

			// make-pair
			PressedLayer tmp_layer;
			for (size_t i = 0; i < candidate_double_gate.size() - 1; ++i)
			{
				for (size_t j = i + 1; j < candidate_double_gate.size(); ++j)
				{
					if (candidate_double_gate[i].second == candidate_double_gate[j].second)
					{
						PressedCirNode _node;
						_node.m_cur_node = candidate_double_gate[i].first;
						++cur_gate_buf_pos.at(candidate_double_gate[i].second.first);
						++cur_gate_buf_pos.at(candidate_double_gate[i].second.second);
						for (auto& relation_qubits : qubit_relation_pre_nodes)
						{
							if (relation_qubits.second.size() == 0)
							{
								continue;
							}

							if ((relation_qubits.first == candidate_double_gate[i].second.first) 
								|| (relation_qubits.first == candidate_double_gate[i].second.second))
							{
								_node.m_relation_pre_nodes.insert(_node.m_relation_pre_nodes.end(), 
									relation_qubits.second.begin(), relation_qubits.second.end());
								relation_qubits.second.clear();
							}
						}
						

						//get seccessor nodes
						append_successor_nodes(_node, candidate_double_gate[i].second, tmp_layer, gate_buf, cur_gate_buf_pos);
						candidate_double_gate.erase(candidate_double_gate.begin() + j);
						break;
					}
				}
			}
			seq.emplace_back(tmp_layer);
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

	void append_successor_nodes(PressedCirNode& pressed_node, std::pair<size_t, size_t>& qubits, PressedLayer &cur_seq_layer,
		OptimizerSink& gate_buf, std::map<size_t, size_t>& cur_gate_buf_pos) {
		auto& gate_vec_1 = gate_buf.at(qubits.first);
		auto& cur_pos_1 = cur_gate_buf_pos.at(qubits.first);
		const auto& max_pos_1 = m_cur_buffer_pos.at(qubits.first);
		auto& gate_vec_2 = gate_buf.at(qubits.second);
		auto& cur_pos_2 = cur_gate_buf_pos.at(qubits.second);
		const auto& max_pos_2 = m_cur_buffer_pos.at(qubits.second);

		auto get_successor_single_gate = [&, this](std::vector<pOptimizerNodeInfo>& gates_vec, size_t& p, const size_t& max_pos) {
			while (true)
			{
				if (p >= max_pos) { return; }

				auto& g = gates_vec[p];
				if (g->m_target_qubits.size() == 1){ pressed_node.m_relation_successor_nodes.push_back(g); }
				else { break; }
		
				++p;
			}
		};

		/*while (true)
		{
			auto& gate_1 = gate_vec_1[_pos_1];
			if (gate_1->m_target_qubits.size() == 1)
			{
				pressed_node.m_relation_successor_nodes.push_back(gate_1);
			}
			else
			{
				break;
			}
			++_pos_1;
		}*/
		/*get_successor_single_gate(gate_vec_1, cur_pos_1, max_pos_1);
		get_successor_single_gate(gate_vec_2, cur_pos_2, max_pos_2);*/

		/*while (true)
		{
			auto& gate_2 = gate_vec_2[_pos_2];
			if (gate_2->m_target_qubits.size() == 1)
			{
				pressed_node.m_relation_successor_nodes.push_back(gate_2);
			}
			else
			{
				break;
			}
			++_pos_2;
		}*/

		while (true)
		{
			get_successor_single_gate(gate_vec_1, cur_pos_1, max_pos_1);
			get_successor_single_gate(gate_vec_2, cur_pos_2, max_pos_2);

			//check double-gate
			auto& gate_1 = gate_vec_1[cur_pos_1];
			auto& gate_2 = gate_vec_2[cur_pos_2];
			if ((gate_1 == gate_2) && 
				((max_pos_1 != cur_pos_1) && (max_pos_2 != cur_pos_2))){ 
				pressed_node.m_relation_successor_nodes.push_back(gate_1); 
				++cur_pos_1, ++cur_pos_2;
			}
			else{ break; }
		}

		std::vector<PressedCirNode> tail_vec(2);
		if (max_pos_1 != cur_pos_1){ tail_vec.front().m_cur_node = gate_vec_1[cur_pos_1]; }
		if (max_pos_2 != cur_pos_2) { tail_vec.back().m_cur_node = gate_vec_2[cur_pos_2]; }
		

		cur_seq_layer.emplace_back(std::make_pair(pressed_node, tail_vec));
	}
#if 0
	void add_gate_to_buffer(NodeIter iter, QCircuitParam &cir_param, std::shared_ptr<QNode> parent_node, OptimizerSink& gates_buffer) override {
		auto gate_node = std::dynamic_pointer_cast<AbstractQGateNode>(*iter);
		QVec gate_qubits;
		gate_node->getQuBitVector(gate_qubits);
		QVec control_qubits;
		gate_node->getControlVector(control_qubits);
		control_qubits += cir_param.m_control_qubits;

		QVec total_qubits = gate_qubits + control_qubits;
		std::vector<int> relation_qubits;
		if (m_b_double_gate_one_layer && (total_qubits.size() > 1))
		{
			//append relation qubits
			for (auto tmp_qubit : total_qubits)
			{
				for (const auto& h : m_high_frequency_qubits)
				{
					if (h == tmp_qubit->get_phy_addr())
					{
						//get adjacent qubit
						for (int a = 0; a < m_qubit_topo_matrix[h].size(); ++a)
						{
							if (0 != m_qubit_topo_matrix[h][a])
							{
								if (gates_buffer.end() != gates_buffer.find(a))
								{
									relation_qubits.push_back(a);
								}
							}
						}
					}
				}
			}
		}

		for (auto relation_qubit_itr = relation_qubits.begin(); relation_qubit_itr != relation_qubits.end(); )
		{
			bool b = false;
			for (auto &q : total_qubits)
			{
				if (q->get_phy_addr() == (*relation_qubit_itr))
				{
					b = true;
					break;
				}
			}

			if (b)
			{
				relation_qubit_itr = relation_qubits.erase(relation_qubit_itr);
			}
			else
			{
				++relation_qubit_itr;
			}
		}

		std::vector<int> total_qubits_i;
		for (auto q : total_qubits)
		{
			total_qubits_i.push_back(q->get_phy_addr());
		}
		total_qubits_i.insert(total_qubits_i.end(), relation_qubits.begin(), relation_qubits.end());
		size_t layer = get_node_layer(total_qubits_i, gates_buffer);
		//PTrace("On layer: %lld\n", layer);

		pOptimizerNodeInfo tmp_node = std::make_shared<OptimizerNodeInfo>(iter, layer,
			gate_qubits, control_qubits, (GateType)(gate_node->getQGate()->getGateType()), parent_node,
			check_dagger(gate_node, (gate_node->isDagger() ^ cir_param.m_is_dagger)));
		for (const auto& i : total_qubits)
		{
			const auto qubit_i = i->get_phy_addr();
			append_data_to_gate_buf(gates_buffer.at(qubit_i), tmp_node, qubit_i);
		}

		if (relation_qubits.size() > 0)
		{
			tmp_node = std::make_shared<OptimizerNodeInfo>(NodeIter(), layer,
				QVec(), QVec(), (GateType)(COMPENSATE_GATE_TYPE), nullptr, false);
			for (const auto& i : relation_qubits)
			{
				append_data_to_gate_buf(gates_buffer.at(i), tmp_node, i);
			}
		}
	}
#endif
private:
	bool m_b_double_gate_one_layer;
	const std::string m_config_data;
	PressedTopoSeq m_topolog_sequence;
	std::vector<std::vector<int>> m_qubit_topo_matrix;
	std::vector<int> m_high_frequency_qubits;
	size_t m_qubit_size;
};
#endif
/*******************************************************************
*                      public interface
********************************************************************/
static void move_measure_to_last_layer(LayeredTopoSeq& seq) {
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
			QCERR_AND_THROW(run_fail, "Error: error node type in last layer.");
		}
	}

	if (b_exist_gate_node)
	{
		seq.push_back(SeqLayer<pOptimizerNodeInfo>());
	}

	auto& real_last_layer = seq.back();

	for (auto layer_iter = seq.begin(); layer_iter != (--seq.end()); ++layer_iter)
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

LayeredTopoSeq QPanda::prog_layer(QProg src_prog, const bool b_enable_qubit_compensation/* = false*/, const std::string config_data /*= CONFIG_PATH*/)
{
	QProgLayer q_layer(b_enable_qubit_compensation, config_data);

	//auto start = std::chrono::system_clock::now();
	q_layer.layer(src_prog);
	/*auto end = chrono::system_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
	cout << "The QProgLayer::layer() takes "
		<< double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den
		<< "seconds" << endl;*/

	LayeredTopoSeq seq = q_layer.get_topo_seq();

	if (b_enable_qubit_compensation)
	{
		move_measure_to_last_layer(seq);
	}

	return seq;
}

PressedTopoSeq QPanda::get_pressed_layer(QProg src_prog)
{
	QPressedLayer layer_obj;
	layer_obj.layer(src_prog);

	return layer_obj.get_topo_seq();
}
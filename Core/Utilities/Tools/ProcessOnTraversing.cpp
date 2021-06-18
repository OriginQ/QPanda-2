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
#define MAX_BUF_SIZE 5000
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

	// do_process
	if ((get_min_include_layers() > MAX_INCLUDE_LAYERS) || (get_max_buf_size() > MAX_BUF_SIZE))
	{
		do_process(false);
	}
}

void ProcessOnTraversing::execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node,
	QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	//handle measure node
	add_non_gate_to_buffer(cur_node_iter, MEASURE_GATE, { cur_node->getQuBit() }, cir_param, m_cur_gates_buffer, parent_node);
}

void ProcessOnTraversing::execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node,
	QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	//handle reset node
	add_non_gate_to_buffer(cur_node_iter, RESET_NODE, { cur_node->getQuBit() }, cir_param, m_cur_gates_buffer, parent_node);
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
		check_dagger(gate_node, cir_param.m_is_dagger));
	for (const auto& i : total_qubits)
	{
		gates_buffer.append_data(tmp_node, i->get_phy_addr());
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
		int t = DAGNodeType::NUKNOW_SEQ_NODE_TYPE;
		if (MEASURE_GATE == node_type)
		{
			t = DAGNodeType::MEASURE;
		}
		else if (RESET_NODE == node_type)
		{
			t = DAGNodeType::RESET;
		}
		else
		{
			QCERR_AND_THROW(run_fail, "Error: unknow node-type.");
		}

		pOptimizerNodeInfo tmp_node = std::make_shared<OptimizerNodeInfo>(iter, layer, gate_qubits, tmp_control_qubits, (GateType)t, parent_node, false);
		for (const auto& i : gate_qubits)
		{
			//gates_buffer.at(i->get_phy_addr()).push_back(tmp_node);
			gates_buffer.append_data(tmp_node, i->get_phy_addr());
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
		const auto& tmp_pos = gate_buffer.get_target_qubit_sink_size(i);
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

size_t ProcessOnTraversing::get_max_buf_size()
{
	size_t ret = 0;
	for (const auto& item : m_cur_gates_buffer){
		const auto& _s = m_cur_gates_buffer.get_target_qubit_sink_size(item.first);
		ret = ret < _s ? _s : ret;
	}

	return ret;
}

size_t ProcessOnTraversing::get_min_include_layers()
{
	size_t include_min_layers = MAX_LAYER;
	m_min_layer = MAX_LAYER;
	for (const auto& item : m_cur_gates_buffer)
	{
		const auto &vec = item.second;
		const auto& tmp_pos = m_cur_gates_buffer.get_target_qubit_sink_size(item.first);
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
		const auto& tmp_pos = gate_buf.get_target_qubit_sink_size(item.first);
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
				++i;
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
	size_t drop_max_layer = 0;
	TopologSequence<std::pair<size_t, NodeIter>> tmp_seq;
	for (auto &item : m_cur_gates_buffer)
	{
		auto &vec = item.second;
		auto& _pos = m_cur_gates_buffer.get_target_qubit_sink_size(item.first);
		if ((0 < _pos))
		{
			if (b_clean_all_buf)
			{
				drop_max_layer = MAX_LAYER;
			}
			else
			{
				const auto& _cur_max_layer = vec[_pos - 1]->m_layer;
				const size_t tmp_include_layer = _cur_max_layer - vec.front()->m_layer + 1;
				if ((tmp_include_layer < MIN_INCLUDE_LAYERS) || (_cur_max_layer <= MIN_INCLUDE_LAYERS)) {
					continue;
				}

				drop_max_layer = _cur_max_layer - MIN_INCLUDE_LAYERS;
			}
			
			size_t _i = 0;
			while ((_i < _pos) && (vec[_i]->m_layer < drop_max_layer))
			{
				add_node_to_seq(tmp_seq, vec[_i]->m_iter, vec[_i]->m_layer);
				vec[_i].reset();
				++_i;
			}

			size_t _j = 0;
			for (; _i < _pos; ++_j, ++_i){
				vec[_j] = vec[_i];
			}
			_pos = _j;
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
		auto& tmp_pos = m_cur_gates_buffer.get_target_qubit_sink_size(item.first);
		size_t i = 0;
		size_t j = 0;
		for (; i < tmp_pos; ++i)
		{
			if ((vec.at(i)->m_layer >= max_drop_layer))
			{
				if (i == j)
				{
					j = tmp_pos;
					break;
				}
				vec.at(j) = vec.at(i);
				++j;
			}
			vec.at(i) = nullptr;
		}

		tmp_pos = j;
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
	{}
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

	void layer(QProg src_prog) { 
		//read compensate-qubit
		init();

		run_traversal(src_prog); }

	const LayeredTopoSeq& get_topo_seq() { return m_topolog_sequence; }

protected:
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
			gates_buffer.append_data(tmp_node, qubit_i);
		}

		if (relation_qubits.size() > 0)
		{
			tmp_node = std::make_shared<OptimizerNodeInfo>(NodeIter(), layer,
				QVec(), QVec(), (GateType)(COMPENSATE_GATE_TYPE), nullptr, false);
			for (const auto& i : relation_qubits)
			{
				gates_buffer.append_data(tmp_node, i);
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

	const PressedTopoSeq& get_topo_seq() { return m_topolog_sequence; }

protected:
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
		std::vector<std::pair<pOptimizerNodeInfo, QVec>> candidate_special_gate;
		while (true)
		{
			bool b_finished = true;
			candidate_double_gate.clear();
			candidate_special_gate.clear();
			for (auto &item : gate_buf)
			{
				const auto& tmp_size = gate_buf.get_target_qubit_sink_size(item.first);

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
						if (BARRIER_GATE == n->m_gate_type)
						{
							candidate_special_gate.push_back(std::make_pair(n, n->m_control_qubits + n->m_target_qubits));
							break;
						}

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

			// make-pair
			PressedLayer tmp_layer;
			if (candidate_double_gate.empty() && candidate_special_gate.empty())
			{
				for (const auto& _item : qubit_relation_pre_nodes)
				{
					if (_item.second.size() == 0){
						continue;
					}

					pPressedCirNode _node = std::make_shared<PressedCirNode>();
					_node->m_cur_node = _item.second.back();
					_node->m_relation_pre_nodes.assign(_item.second.begin(), _item.second.end() - 1);

					std::vector<pPressedCirNode> tail_vec;
					tmp_layer.emplace_back(std::make_pair(_node, tail_vec));
				}

				if (tmp_layer.size() > 0){
					seq.emplace_back(tmp_layer);
				}
				
				break; // only one layer(pressed layer)
			}

			if (b_finished)
			{
				break;
			}

			while (candidate_special_gate.size() > 1)
			{
				std::vector<std::pair<pOptimizerNodeInfo, QVec>> tmp_special_gate;
				tmp_special_gate.push_back(candidate_special_gate.back());
				candidate_special_gate.pop_back();
				for (size_t i = 0; i < candidate_special_gate.size();)
				{
					if (tmp_special_gate.front().first == candidate_special_gate[i].first)
					{
						tmp_special_gate.push_back(candidate_special_gate[i]);
						candidate_special_gate.erase(candidate_special_gate.begin() + i);
						continue;
					}

					++i;
				}

				if (tmp_special_gate.front().second.size() == tmp_special_gate.size())
				{
					pPressedCirNode _node = std::make_shared<PressedCirNode>();
					_node->m_cur_node = tmp_special_gate.front().first;

					for (const auto& _q : tmp_special_gate.front().second)
					{
						for (auto& relation_qubits : qubit_relation_pre_nodes)
						{
							if (relation_qubits.second.size() == 0){
								continue;
							}

							if (_q->get_phy_addr() == relation_qubits.first)
							{
								_node->m_relation_pre_nodes.insert(_node->m_relation_pre_nodes.end(),
									relation_qubits.second.begin(), relation_qubits.second.end());
								relation_qubits.second.clear();
								break;
							}
						}

						++(cur_gate_buf_pos.at(_q->get_phy_addr()));
					}

					std::vector<pPressedCirNode> tail_vec;
					tmp_layer.emplace_back(std::make_pair(_node, tail_vec));
				}
			}

			if (candidate_double_gate.size() > 1)
			{
				for (size_t i = 0; i < candidate_double_gate.size() - 1; ++i)
				{
					for (size_t j = i + 1; j < candidate_double_gate.size(); ++j)
					{
						if (candidate_double_gate[i].second == candidate_double_gate[j].second)
						{
							pPressedCirNode _node = std::make_shared<PressedCirNode>();
							_node->m_cur_node = candidate_double_gate[i].first;
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
									_node->m_relation_pre_nodes.insert(_node->m_relation_pre_nodes.end(),
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
			}
			
			if (tmp_layer.size() > 0)
			{
				if (seq.size() == 0)
				{
					seq.emplace_back(tmp_layer);
					continue;
				}

				auto& last_layer = seq.back();
				for (auto& cur_layer_node : tmp_layer)
				{
					if (BARRIER_GATE == cur_layer_node.first->m_cur_node->m_gate_type)
					{
						continue;
					}

					for (auto _iter = last_layer.begin(); _iter != last_layer.end(); )
					{
						auto& last_layer_node = *_iter;
						if (BARRIER_GATE == last_layer_node.first->m_cur_node->m_gate_type)
						{
							auto& barrier_node = last_layer_node.first->m_cur_node;
							QVec barrier_qubits = barrier_node->m_control_qubits + barrier_node->m_target_qubits;
							QVec tmp_qv = cur_layer_node.first->m_cur_node->m_target_qubits - barrier_qubits;
							if (tmp_qv.size() == 0)
							{
								cur_layer_node.first->m_relation_pre_nodes.insert(
									cur_layer_node.first->m_relation_pre_nodes.begin(), barrier_node);
								cur_layer_node.first->m_relation_pre_nodes.insert(
									cur_layer_node.first->m_relation_pre_nodes.begin(),
									last_layer_node.first->m_relation_pre_nodes.begin(), 
									last_layer_node.first->m_relation_pre_nodes.end());
								
								_iter = last_layer.erase(_iter);
								break;
							}
						}

						++_iter;
					}
				}

				seq.emplace_back(tmp_layer);
			}
		}

		for (auto _i = seq.begin(); _i != seq.end(); )
		{
			if ((*_i).size() == 0)
			{
				_i = seq.erase(_i);
				continue;
			}

			++_i;
		}
	}

	void append_successor_nodes(pPressedCirNode pressed_node, std::pair<size_t, size_t>& qubits, PressedLayer &cur_seq_layer,
		OptimizerSink& gate_buf, std::map<size_t, size_t>& cur_gate_buf_pos) {
		auto& gate_vec_1 = gate_buf.at(qubits.first);
		auto& cur_pos_1 = cur_gate_buf_pos.at(qubits.first);
		const auto& max_pos_1 = m_cur_gates_buffer.get_target_qubit_sink_size(qubits.first);
		auto& gate_vec_2 = gate_buf.at(qubits.second);
		auto& cur_pos_2 = cur_gate_buf_pos.at(qubits.second);
		const auto& max_pos_2 = m_cur_gates_buffer.get_target_qubit_sink_size(qubits.second);

		auto get_successor_single_gate = [&, this](std::vector<pOptimizerNodeInfo>& gates_vec, size_t& p, const size_t& max_pos) {
			while (true)
			{
				if (p >= max_pos) { return; }

				auto& g = gates_vec[p];

				if (BARRIER_GATE == g->m_gate_type)
				{
					break;
				}

				if (g->m_target_qubits.size() == 1){ pressed_node->m_relation_successor_nodes.push_back(g); }
				else { break; }
		
				++p;
			}
		};

		while (true)
		{
			get_successor_single_gate(gate_vec_1, cur_pos_1, max_pos_1);
			get_successor_single_gate(gate_vec_2, cur_pos_2, max_pos_2);

			if ((max_pos_1 == cur_pos_1) || (max_pos_2 == cur_pos_2)){
				break;
			}

			//check double-gate
			auto& gate_1 = gate_vec_1[cur_pos_1];
			auto& gate_2 = gate_vec_2[cur_pos_2];
			if (gate_1 == gate_2){ 
				if (BARRIER_GATE == gate_1->m_gate_type)
				{
					break;
				}

				pressed_node->m_relation_successor_nodes.push_back(gate_1); 
				++cur_pos_1, ++cur_pos_2;
			}
			else{ break; }
		}

		std::vector<pPressedCirNode> tail_vec = {std::make_shared<PressedCirNode>(), std::make_shared<PressedCirNode>()};
		if (max_pos_1 != cur_pos_1){ tail_vec.front()->m_cur_node = gate_vec_1[cur_pos_1]; }
		if (max_pos_2 != cur_pos_2) { tail_vec.back()->m_cur_node = gate_vec_2[cur_pos_2]; }
		

		cur_seq_layer.emplace_back(std::make_pair(pressed_node, tail_vec));
	}

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
*                      class QProgLayerByClock
********************************************************************/
class QProgLayerByClock : protected ProcessOnTraversing
{
public:
	QProgLayerByClock()
		:m_b_temp_storage(false), m_tmp_storage_max_clock(0)
	{}
	~QProgLayerByClock() {}

	void layer_by_clock(QProg src_prog, const std::string config_data = CONFIG_PATH) {
		m_time_sequence_conf.load_config(config_data);

		run_traversal(src_prog);
	}

	const LayeredTopoSeq& get_topo_seq() { return m_topolog_sequence; }

protected:
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
		//drop_max_layer = 3;
		//gates_sink_to_topolog_sequence(m_cur_gates_buffer, tmp_topolog_sequence, drop_max_layer);
		////update gate sink
		//append_topolog_seq(tmp_topolog_sequence);

		//drop_gates(drop_max_layer);

		//drop_max_layer = MAX_LAYER;
		gates_sink_to_topolog_sequence(m_cur_gates_buffer, tmp_topolog_sequence, drop_max_layer);

		//update gate sink
		append_topolog_seq(tmp_topolog_sequence);

		drop_gates(drop_max_layer);
	}

	void append_topolog_seq(LayeredTopoSeq& tmp_seq) {
		m_topolog_sequence.insert(m_topolog_sequence.end(), tmp_seq.begin(), tmp_seq.end());
	}

	void gates_sink_to_topolog_sequence(OptimizerSink& gate_buf, LayeredTopoSeq& seq,
		const size_t max_output_layer = MAX_LAYER) override {
		if (gate_buf.begin()->second.size() == 0)
		{
			QCERR_AND_THROW(run_fail, "Error: unknown error on gates_sink_to_topolog_sequence.");
		}
		/*size_t min_layer = gate_buf.begin()->second.front()->m_layer;
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
		}*/

		seq.clear();
		std::map<uint32_t, uint32_t> traversal_pos;
		std::map<uint32_t, uint32_t> cur_qubit_clock;
		for (const auto &item : gate_buf) {
			traversal_pos.insert(std::make_pair(item.first, 0));
			cur_qubit_clock.insert(std::make_pair(item.first, 0));
		}
		
		while (true)
		{
			uint32_t cur_max_clock = 0;
			bool b_first_traver = true;
			SeqLayer<pOptimizerNodeInfo> cur_clock_layer;
			if (m_b_temp_storage)
			{
				cur_qubit_clock = m_tmp_storage_qubit_clock;
				cur_clock_layer.swap(m_tmp_storage_clock_layer);
				cur_max_clock = m_tmp_storage_max_clock;
				b_first_traver = false;
				m_b_temp_storage = false;

			}
			else
			{
				for (auto &_item_clock : cur_qubit_clock) {
					_item_clock.second = 0;
				}
			}

			while (true)
			{
				bool b_no_valid_node = true;
				std::vector<pOptimizerNodeInfo> _node_cache;
				uint32_t temp_storage_qubits = 0;
				for (const auto &item : gate_buf)
				{
					if (gate_buf.get_target_qubit_sink_size(item.first) <= traversal_pos[item.first]){
						continue;
					}
					const auto& n = item.second.at(traversal_pos[item.first]);
					if ((max_output_layer <= n->m_layer))
					{
						if (gate_buf.size() == ++temp_storage_qubits) {
							m_b_temp_storage = true;
						}

						continue;
					}

					const auto _c = get_node_clock(n);
					if ((COMPENSATE_GATE_TYPE == n->m_type) ||
						((!b_first_traver) && (cur_qubit_clock[item.first] + _c > cur_max_clock))) {
						continue;
					}
					
					int already_exist_times = 1;
					const auto cur_node_used_qubits = n->m_control_qubits + n->m_target_qubits;
					if (cur_node_used_qubits.size() > 1)
					{
						for (const auto& _node : _node_cache)
						{
							if (_node == n)
							{
								++already_exist_times;
							}
						}
					}
					
					if (cur_node_used_qubits.size() == already_exist_times)
					{
						cur_clock_layer.push_back(std::pair<pOptimizerNodeInfo,
							std::vector<pOptimizerNodeInfo>>(n, std::vector<pOptimizerNodeInfo>()));

						for (const auto& _q : cur_node_used_qubits){
							++traversal_pos[_q->get_phy_addr()];
							cur_qubit_clock[_q->get_phy_addr()] += _c;
						}

						if (b_first_traver && (_c > cur_max_clock)) {
							cur_max_clock = _c;
						}

						b_no_valid_node = false;
					}
					else
					{
						_node_cache.push_back(n);
					}
				}

				if (b_no_valid_node/* && (0 == temp_storage_qubits)*/){
					break;
				}

				for (auto &_item_clock : cur_qubit_clock) {
					if (0 == _item_clock.second) {
						_item_clock.second += cur_max_clock;
					}
				}

				b_first_traver = false;
			}

			if (m_b_temp_storage)
			{
				if ((cur_clock_layer.size() > 0))
				{
					m_tmp_storage_qubit_clock = cur_qubit_clock;
					m_tmp_storage_clock_layer = cur_clock_layer;
					m_tmp_storage_max_clock = cur_max_clock;
					break;
				}
				else
				{
					m_b_temp_storage = false;
				}
			}

			if (cur_clock_layer.size() > 0){
				seq.emplace_back(cur_clock_layer);
			}
			else{
				break;
			}
		}
	}

	int get_measure_time_sequence(){
		return m_time_sequence_conf.get_measure_time_sequence();
	}

	int get_ctrl_node_time_sequence(){
		return m_time_sequence_conf.get_ctrl_node_time_sequence();
	}

	int get_swap_gate_time_sequence(){
		return m_time_sequence_conf.get_swap_gate_time_sequence();
	}

	int get_single_gate_time_sequence(){
		return m_time_sequence_conf.get_single_gate_time_sequence();
	}

	int get_reset_time_sequence(){
		return m_time_sequence_conf.get_reset_time_sequence();
	}

	int get_node_clock(const pOptimizerNodeInfo& p_node_info)
	{
		const auto& qubits_vector = p_node_info->m_target_qubits;
		const auto&  control_qubits_vec = p_node_info->m_control_qubits;
		const auto gate_type = (GateType)(p_node_info->m_type);

		const auto append_ctrl_qubit_clock = control_qubits_vec.size() * 2;
		if (1 == qubits_vector.size())
		{
			// single gate
			return get_single_gate_time_sequence() + append_ctrl_qubit_clock;
		}
		else if (2 == qubits_vector.size())
		{
			//double gate
			switch (gate_type)
			{
			case ISWAP_THETA_GATE:
			case ISWAP_GATE:
			case SQISWAP_GATE:
			case SWAP_GATE:
				return get_swap_gate_time_sequence() + append_ctrl_qubit_clock;
				break;

			case CU_GATE:
			case CNOT_GATE:
			case CZ_GATE:
			case CPHASE_GATE:
			{
				return get_ctrl_node_time_sequence() + append_ctrl_qubit_clock;
			}
			break;

			default:
				break;
			}

		}
		

		QCERR_AND_THROW(run_fail, "Error: unknow gate type on get_node_clock.");
		return -1;
	}

private:
	TimeSequenceConfig m_time_sequence_conf;
	LayeredTopoSeq m_topolog_sequence;
	bool m_b_temp_storage;

	/** for tmp_storage */
	std::map<uint32_t, uint32_t> m_tmp_storage_qubit_clock;
	SeqLayer<pOptimizerNodeInfo> m_tmp_storage_clock_layer;
	uint32_t m_tmp_storage_max_clock;
};

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

LayeredTopoSeq QPanda::get_clock_layer(QProg src_prog, const std::string config_data /*= CONFIG_PATH*/)
{
	QProgLayerByClock layer_obj;
	layer_obj.layer_by_clock(src_prog, config_data);

	return layer_obj.get_topo_seq();
}
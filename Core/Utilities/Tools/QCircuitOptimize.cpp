#include "Core/Utilities/Tools/QCircuitOptimize.h"
#include <cmath>
#include <string.h>
#include "Core/Utilities/Tools/QProgFlattening.h"
#include "Core/Utilities/QProgInfo/Visualization/QVisualization.h"
#include "Core/Utilities/Tools/QProgFlattening.h"
#include "Core/QuantumCircuit/QuantumGate.h"
#include "Core/Utilities/Tools/JsonConfigParam.h"
#include "Core/Utilities//Tools/ThreadPool.h"
#include <atomic>
#include <ctime>
#include <vector>
#include <thread>

using namespace std;
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

#define MAX_SIZE 0xEFFFFFFFFFFFFFFF
#define MAX_COMPARE_PRECISION 0.000001

class DelQNodeByIter : protected TraverseByNodeIter
{
public:
	DelQNodeByIter() = delete;
	DelQNodeByIter(QProg src_prog, NodeIter node_iter) 
		:m_target_iter(node_iter)
	{
		traverse_qprog(src_prog);

		if (nullptr != m_target_iter.getPCur())
		{
			QCERR_AND_THROW_ERRSTR(run_fail, "Error: failed to delete target QNode, unknow error.");
		}
	}
	~DelQNodeByIter() {}

	void execute(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
		del_QNode(parent_node, cur_node_iter);
	}
	void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
		del_QNode(parent_node, cur_node_iter);
	}
	void execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
		del_QNode(parent_node, cur_node_iter);
	}
	void execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
		del_QNode(parent_node, cur_node_iter);
	}

protected:
	void del_QNode(std::shared_ptr<QNode> parent_node, NodeIter& cur_node_iter) {
		if (m_target_iter == cur_node_iter)
		{
			auto node_type = parent_node->getNodeType();
			switch (node_type)
			{
			case CIRCUIT_NODE:
				(std::dynamic_pointer_cast<AbstractQuantumCircuit>(parent_node))->deleteQNode(m_target_iter);
				break;

			case PROG_NODE:
				(std::dynamic_pointer_cast<AbstractQuantumProgram>(parent_node))->deleteQNode(m_target_iter);
				break;

			default:
				QCERR_AND_THROW_ERRSTR(run_fail, "Error: failed to delete target QNode, Node type error.");
				break;
			}
		}
	}

private:
	NodeIter m_target_iter;
};

/*******************************************************************
*                      class OptimizerSingleGate
********************************************************************/
class OptimizerSingleGate : public AbstractCirOptimizer
{
public:
	OptimizerSingleGate() {
		m_thread_pool.init_thread_pool();
	}
	void do_optimize(QProg src_prog, OptimizerSink &gates_sink, std::vector<QCircuit>& replace_to_cir_vec) override {
		m_sub_cir_cnt = 0;
		m_job_cnt = 0;
		for (auto &item : gates_sink)
		{
			//create thread process
			m_thread_pool.append(std::bind(&OptimizerSingleGate::process_single_gate, this, src_prog, item.second));
		}

		while (m_job_cnt != gates_sink.size()) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); }
		for (auto& item : m_replace_cir_vec)
		{
			replace_to_cir_vec.push_back(item.second);
		}
		m_replace_cir_vec.clear();
	}

	void process_single_gate(QProg &src_prog, std::list<pOptimizerNodeInfo> &node_vec) {
		std::map<size_t, QCircuit> tmp_replace_cir;
		bool m_last_gate_is_X = false;
		bool m_last_gate_is_H = false;
		for (auto iter = node_vec.begin(); iter != node_vec.end(); ++iter)
		{
			check_continuous_same_gate(src_prog, node_vec, m_last_gate_is_X, PAULI_X_GATE, iter, tmp_replace_cir);
			check_continuous_same_gate(src_prog, node_vec, m_last_gate_is_H, HADAMARD_GATE, iter, tmp_replace_cir);
		}

		m_queue_mutex.lock();
		m_replace_cir_vec.insert(tmp_replace_cir.begin(), tmp_replace_cir.end());
		m_queue_mutex.unlock();
		++m_job_cnt;
	}

protected:
	template<typename T>
	void check_continuous_same_gate(QProg &src_prog, std::list<pOptimizerNodeInfo> &node_vec, bool &b_last_type, const T t, 
		std::list<pOptimizerNodeInfo>::iterator iter, std::map<size_t, QCircuit>& replace_to_cir_vec) {
		GateType gt_type = (*iter)->m_type;
		bool is_continous = (b_last_type && (t == gt_type));

		if (node_vec.begin() == iter)
		{
			b_last_type = (t == gt_type);
			return;
		}
		
		auto tmp_iter = iter; //last node
		--tmp_iter;
		if (is_continous && (is_same_controled(*tmp_iter, *iter)))
		{
			size_t cur_sub_graph_index = (m_sub_cir_cnt++);
			(*tmp_iter)->m_sub_graph_index = cur_sub_graph_index;
			(*iter)->m_sub_graph_index = cur_sub_graph_index;
			QGate new_gate = I((*iter)->m_target_qubits.at(0));
			replace_to_cir_vec.insert(std::make_pair(cur_sub_graph_index, QCircuit(new_gate)));
			b_last_type = false;
		}
		else
		{
			b_last_type = (t == gt_type);
		}
	}

private:
	threadPool m_thread_pool;
	std::map<size_t, QCircuit> m_replace_cir_vec;
	std::atomic<size_t> m_sub_cir_cnt;
	std::mutex m_queue_mutex;
	std::atomic<size_t> m_job_cnt;
};

/*******************************************************************
*                      class OptimizerRotationSingleGate
********************************************************************/
class OptimizerRotationSingleGate : public AbstractCirOptimizer
{
public:
	OptimizerRotationSingleGate(GateType t)
		:m_gate_type(t)
	{}
	void do_optimize(QProg src_prog, OptimizerSink &gates_sink, std::vector<QCircuit>& replace_to_cir_vec) override {
		replace_to_cir_vec.clear();
		for (auto &item : gates_sink)
		{
			m_continue_gate = false;
			for (auto iter = item.second.begin(); iter != item.second.end(); ++iter)
			{
				check_continuous_same_gate(src_prog, item.second, iter, replace_to_cir_vec);
			}
			m_continues_iter_vec.clear();
		}
	}

protected:
	bool is_same_controled(std::list<pOptimizerNodeInfo> &node_vec, std::list<pOptimizerNodeInfo>::iterator cur_node_iter) {
		if (m_continues_iter_vec.size() == 0)
		{
			return true;
		}

		return AbstractCirOptimizer::is_same_controled(*(m_continues_iter_vec.back()), *cur_node_iter);
	}
	void check_continuous_same_gate(QProg &src_prog, std::list<pOptimizerNodeInfo> &node_vec, 
		std::list<pOptimizerNodeInfo>::iterator iter, std::vector<QCircuit>& replace_to_cir_vec) {
		GateType gt_type = (*iter)->m_type;
		if (m_gate_type == gt_type && (is_same_controled(node_vec, iter)))
		{
			m_continues_iter_vec.push_back(iter);
		}
		else
		{
			if (m_continues_iter_vec.size() < 2)
			{
				if (m_continues_iter_vec.size() > 0)
				{
					m_continues_iter_vec.clear();
				}
				return;
			}

			double angle = 0.0;
			for (const auto i : m_continues_iter_vec)
			{
				auto p_gate = std::dynamic_pointer_cast<AbstractQGateNode>(*((*i)->m_iter));
				QuantumGate* p = p_gate->getQGate();
				auto p_single_angle_gate = dynamic_cast<QGATE_SPACE::AbstractSingleAngleParameter*>(p);
				if (((*i)->m_dagger) ^ p_gate->isDagger())
				{
					angle -= p_single_angle_gate->getParameter();
				}
				else
				{
					angle += p_single_angle_gate->getParameter();
				}
				(*i)->m_sub_graph_index = replace_to_cir_vec.size();
			}
			
			QGate new_gate = build_new_gate((*(m_continues_iter_vec.back())), angle);
			m_continues_iter_vec.clear();
			replace_to_cir_vec.push_back(new_gate);
		}
	}

	QGate build_new_gate(pOptimizerNodeInfo pos_node, const double angle) {
		std::shared_ptr<AbstractQGateNode> p_new_gate;
		switch (m_gate_type)
		{
		case RX_GATE:
			p_new_gate = RX(pos_node->m_target_qubits[0], angle).getImplementationPtr();
			break;

		case RY_GATE:
			p_new_gate = RY(pos_node->m_target_qubits[0], angle).getImplementationPtr();
			break;

		case RZ_GATE:
			p_new_gate = RZ(pos_node->m_target_qubits[0], angle).getImplementationPtr();
			break;

		default:
			QCERR_AND_THROW_ERRSTR(run_fail, "Error: rotation single gate type error.");
			break;
		}
		
		p_new_gate->setControl(pos_node->m_ctrl_qubits);
		p_new_gate->setDagger((std::dynamic_pointer_cast<AbstractQGateNode>(*pos_node->m_iter))->isDagger());
		return QGate(p_new_gate);
	}

private:
	const GateType m_gate_type;
	bool m_continue_gate;
	QCircuit m_tmp_cir;
	std::vector<std::list<pOptimizerNodeInfo>::iterator> m_continues_iter_vec;
};

/*******************************************************************
*                      class MergeU3Gate
********************************************************************/
class MergeU3Gate : public AbstractCirOptimizer
{
public:
	MergeU3Gate() {
		m_thread_pool.init_thread_pool(8);
	}
	~MergeU3Gate() {}

	void do_optimize(QProg src_prog, OptimizerSink &gates_sink, std::vector<QCircuit>& replace_to_cir_vec) override {
		m_sub_cir_cnt = 0;
		m_job_cnt = 0;
		for (auto &item : gates_sink)
		{
			//create thread process
			m_thread_pool.append(std::bind(&MergeU3Gate::process_single_gate, this, src_prog, item.second));
		}

		while (m_job_cnt != gates_sink.size()) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); }
		for (auto& item : m_replace_cir_vec)
		{
			replace_to_cir_vec.push_back(item.second);
		}
		m_replace_cir_vec.clear();
	}

	void process_single_gate(QProg &src_prog, std::list<pOptimizerNodeInfo> &node_vec) {
		std::map<size_t, QCircuit> tmp_replace_cir;
		std::vector<std::list<pOptimizerNodeInfo>::iterator> continues_itr_vec;
		for (auto itr = node_vec.begin(); itr != node_vec.end(); ++itr)
		{
			check_continuous_single_gate(node_vec, itr, continues_itr_vec, tmp_replace_cir);
		}

		handle_continue_single_gate(node_vec, continues_itr_vec, tmp_replace_cir);

		m_queue_mutex.lock();
		m_replace_cir_vec.insert(tmp_replace_cir.begin(), tmp_replace_cir.end());
		m_queue_mutex.unlock();
		++m_job_cnt;
	}

	QGate build_u3_gate(Qubit* target_qubit, QStat &mat) {
		if (mat.size() != 4)
		{
			QCERR_AND_THROW_ERRSTR(run_fail, "Error: Failed to build U3 gate, the size of input matrix is error.");
		}

		QGate tmp_u4 = U4(mat, target_qubit);
		auto p_gate = dynamic_cast<QGATE_SPACE::AbstractAngleParameter*>(tmp_u4.getQGate());
		double alpha = p_gate->getAlpha();
		double beta = p_gate->getBeta();
		double gamma = p_gate->getGamma();
		double delta = p_gate->getDelta();

		double u3_theta = gamma;
		double u3_phi = beta;
		double u3_lambda = delta;
		return U3(target_qubit, u3_theta, u3_phi, u3_lambda);
	}

protected:
	void check_continuous_single_gate(std::list<pOptimizerNodeInfo> &node_vec, std::list<pOptimizerNodeInfo>::iterator cur_iter,
		std::vector<std::list<pOptimizerNodeInfo>::iterator>& continues_itr_vec, std::map<size_t, QCircuit>& replace_to_cir_vec) {
		GateType gt_type = (*cur_iter)->m_type;
		const bool cur_gate_is_single_gate = is_single_gate(gt_type);

		if (cur_gate_is_single_gate && (is_same_controled(node_vec, continues_itr_vec, cur_iter)))
		{
			//push to index vector
			continues_itr_vec.push_back(cur_iter);
		}
		else
		{
			handle_continue_single_gate(node_vec, continues_itr_vec, replace_to_cir_vec);
		}
	}

	void handle_continue_single_gate(std::list<pOptimizerNodeInfo> &node_vec, std::vector<std::list<pOptimizerNodeInfo>::iterator>& continues_itr_vec,
		std::map<size_t, QCircuit>& replace_to_cir_vec) {
		if (continues_itr_vec.size() == 0)
		{
			return;
		}

		//handle index vector
		QStat mat = get_matrix_of_index_vec(continues_itr_vec, node_vec);

		//build U3 gate
		size_t cur_sub_graph_index = (m_sub_cir_cnt++);
		for (const auto i : continues_itr_vec)
		{
			(*i)->m_sub_graph_index = cur_sub_graph_index;
		}
		QGate new_u3 = build_u3_gate((*(continues_itr_vec.back()))->m_target_qubits.at(0), mat);

		QCircuit cir(new_u3);
		for (const auto& i : continues_itr_vec)
		{
			auto p_node = *((*i)->m_iter);
			auto p_gate = std::dynamic_pointer_cast<AbstractQGateNode>(p_node);
			if (ECHO_GATE == p_gate->getQGate()->getGateType())
			{
				QVec q;
				p_gate->getQuBitVector(q);
				cir << ECHO(q[0]);
			}
		}

		replace_to_cir_vec.insert(std::make_pair(cur_sub_graph_index, cir));

		continues_itr_vec.clear();
	}

	QStat get_matrix_of_index_vec(const std::vector<std::list<pOptimizerNodeInfo>::iterator>& continues_itr_vec, 
		const std::list<pOptimizerNodeInfo> &node_vec) {
		QCircuit cir;
		for (const auto& i : continues_itr_vec)
		{
			auto p_node = *((*i)->m_iter);
			auto p_gate = std::dynamic_pointer_cast<AbstractQGateNode>(p_node);
			if ((*i)->m_dagger)
			{
				cir << QGate(p_gate).dagger();
			}
			else
			{
				cir.pushBackNode(p_node);
			}
		}

		return getCircuitMatrix(cir);
	}

	bool is_same_controled(std::list<pOptimizerNodeInfo> &node_vec,
		const std::vector<std::list<pOptimizerNodeInfo>::iterator>& continues_itr_vec, std::list<pOptimizerNodeInfo>::iterator cur_iter) {
		if (continues_itr_vec.size() == 0)
		{
			return true;
		}

		return AbstractCirOptimizer::is_same_controled(*(continues_itr_vec.back()), *cur_iter);
	}

	bool is_single_gate(const GateType gt_type) {
		if (gt_type < 0)
		{
			return false;
		}

		switch (gt_type)
		{
		case P0_GATE:
		case P1_GATE:
		case PAULI_X_GATE:
		case PAULI_Y_GATE:
		case PAULI_Z_GATE:
		case X_HALF_PI:
		case Y_HALF_PI:
		case Z_HALF_PI:
		case HADAMARD_GATE:
		case T_GATE:
		case S_GATE:
		case RX_GATE:
		case RY_GATE:
		case RZ_GATE:
		case RPHI_GATE:
		case U1_GATE:
		case U2_GATE:
		case U3_GATE:
		case U4_GATE:
		case P00_GATE:
		case P11_GATE:
		case I_GATE:
		case ECHO_GATE:
			return true;
		}

		return false;
	}

private:
	threadPool m_thread_pool;
	std::map<size_t, QCircuit> m_replace_cir_vec;
	std::atomic<size_t> m_sub_cir_cnt;
	std::mutex m_queue_mutex;
	std::atomic<size_t> m_job_cnt;
};

/*******************************************************************
*                      class FindSubCircuit
********************************************************************/
void FindSubCircuit::sub_cir_query(TopologSequence<pOptimizerNodeInfo>& target_sub_sequence)
{
	m_sub_graph_vec.clear();

	for (auto &target_seq_layer : target_sub_sequence)
	{
		for (auto &target_seq_layer_node : target_seq_layer)
		{
			MatchNodeVec<pOptimizerNodeInfo> node_vector;
			for (auto &graph_layer : m_topolog_sequence)
			{
				for (auto &graph_node : graph_layer)
				{
					if (node_match(target_seq_layer_node, graph_node))
					{
						node_vector.emplace_back(graph_node);
					}
				}
			}

			if (node_vector.empty())
			{
				return;
			}
			else
			{
				m_node_match_vector.emplace_back(make_pair(target_seq_layer_node.first, node_vector));
			}
		}
	}

	/*The first layer data of the queried subgraph will be processed firstly,
	* according to the number of matching nodes, the corresponding sequence of subgraphs is created.
	* There are duplicate problems, which will be merged later
	*/
	size_t head_node_size = 0;
	for (size_t i = 0; i < target_sub_sequence.at(0).size(); ++i)
	{
		head_node_size += m_node_match_vector.at(i).second.size();
	}
	m_sub_graph_vec.resize(head_node_size);

	size_t sub_graph_index = 0;
	for (size_t i = 0; i < target_sub_sequence.at(0).size(); ++i)
	{
		for (auto& match_item : m_node_match_vector.front().second)
		{
			SeqLayer<pOptimizerNodeInfo> sub_graph_tmp_layer;
			sub_graph_tmp_layer.push_back(match_item);
			m_sub_graph_vec.at(sub_graph_index).push_back(sub_graph_tmp_layer);
			++sub_graph_index;
		}

		/* Remove the matched nodes that have been processed */
		m_node_match_vector.erase(m_node_match_vector.begin());
	}

	/* Eliminate duplicates */
	for (size_t i = 0; i < (m_sub_graph_vec.size() - 1); ++i)
	{
		for (size_t j = i + 1; j < m_sub_graph_vec.size(); )
		{
			if (m_sub_graph_vec.at(i).front().front().first->m_iter == m_sub_graph_vec.at(j).front().front().first->m_iter)
			{
				m_sub_graph_vec.erase(m_sub_graph_vec.begin() + j);
				continue;
			}
			else
			{
				++j;
			}
		}
	}

	merge_sub_graph_vec(m_sub_graph_vec, 0, target_sub_sequence);

	/* Filter and combine the preliminary matching nodes */
	size_t cur_layer_index = 0;
	/* from the second layer */
	for (auto sub_seq_layer_iter = target_sub_sequence.begin() + 1; sub_seq_layer_iter != target_sub_sequence.end(); ++sub_seq_layer_iter)
	{
		++cur_layer_index;
		for (auto& perhaps_sub_graph : m_sub_graph_vec)
		{
			/* add one layer for all the matched sub-graphs */
			SeqLayer<pOptimizerNodeInfo> sub_graph_tmp_layer;
			perhaps_sub_graph.push_back(sub_graph_tmp_layer);
		}

		/* match layer*/
		match_layer(*sub_seq_layer_iter, cur_layer_index, m_sub_graph_vec);

		/* merge sub graphs */
		merge_sub_graph_vec(m_sub_graph_vec, cur_layer_index, target_sub_sequence);
	}

	/* Eliminating the incomplete sequence of subgraphs */
	clean_sub_graph_vec(m_sub_graph_vec, target_sub_sequence);
}

void FindSubCircuit::merge_sub_graph_vec(std::vector<TopologSequence<pOptimizerNodeInfo>>& sub_graph_vec,
	const size_t match_layer, TopologSequence<pOptimizerNodeInfo>& target_sub_sequence)
{
	bool b_need_merge_check = false;
	if (match_layer < (target_sub_sequence.size() - 1))
	{
		const auto& cur_layer_node_vec = target_sub_sequence.at(match_layer);
		std::vector<pOptimizerNodeInfo> tmp_vec;
		for (const auto& node : cur_layer_node_vec)
		{
			tmp_vec.insert(tmp_vec.end(), node.second.begin(), node.second.end());
		}

		for (size_t i = 0; i < tmp_vec.size() - 1; ++i)
		{
			if (nullptr == tmp_vec[i])
			{
				continue;
			}
			for (size_t j = i + 1; j < tmp_vec.size(); j++)
			{
				if (nullptr == tmp_vec[j])
				{
					continue;
				}
				if (tmp_vec[i]->m_iter == tmp_vec[j]->m_iter)
				{
					b_need_merge_check = true;
					break;
				}
			}
		}
	}

	if (!b_need_merge_check)
	{
		return;
	}

	/* Get the next node iterator set of all subgraphs */
	std::vector<std::vector<NodeIter>> sub_graph_next_layer_iter_vec;
	for (auto& gtaph_item : sub_graph_vec)
	{
		std::vector<NodeIter> tmp_graph_layer_next_iter_vec;
		for (auto& cur_layer_node : gtaph_item.at(match_layer))
		{
			/* Traverse each node of the current layer */
			for (auto& next_layer_node : cur_layer_node.second)
			{
				if (nullptr != next_layer_node)
				{
					tmp_graph_layer_next_iter_vec.push_back(next_layer_node->m_iter);
				}
			}
		}

		sub_graph_next_layer_iter_vec.push_back(tmp_graph_layer_next_iter_vec);
	}

	size_t perhaps_sub_graph_number = sub_graph_vec.size();
	auto sort_fun = [](NodeIter& a, NodeIter& b) {return a.getPCur() < b.getPCur(); };
	for (size_t i = 0; i < perhaps_sub_graph_number; ++i)
	{
		for (size_t j = i + 1; j < (perhaps_sub_graph_number); ++j)
		{
			/* Judge whether there is intersection */
			std::sort(sub_graph_next_layer_iter_vec.at(i).begin(), sub_graph_next_layer_iter_vec.at(i).end(), sort_fun);
			std::sort(sub_graph_next_layer_iter_vec.at(j).begin(), sub_graph_next_layer_iter_vec.at(j).end(), sort_fun);
			std::vector<NodeIter> result_vec;
			set_intersection(sub_graph_next_layer_iter_vec.at(i).begin(), sub_graph_next_layer_iter_vec.at(i).end(),
				sub_graph_next_layer_iter_vec.at(j).begin(), sub_graph_next_layer_iter_vec.at(j).end(), std::back_inserter(result_vec), sort_fun);

			if (result_vec.size() > 0)
			{
				//merge to graph_i
				merge_topolog_sequence(sub_graph_vec.at(j), sub_graph_vec.at(i));
				sub_graph_vec.erase(sub_graph_vec.begin() + j);
				sub_graph_next_layer_iter_vec.erase(sub_graph_next_layer_iter_vec.begin() + j);
				--perhaps_sub_graph_number;
				--j;
			}
		}
	}
}

void FindSubCircuit::merge_topolog_sequence(TopologSequence<pOptimizerNodeInfo>& src_seq, TopologSequence<pOptimizerNodeInfo>& dst_seq)
{
	if (src_seq.size() != dst_seq.size())
	{
		QCERR_AND_THROW_ERRSTR(run_fail, "Error: failed to merge two TopologSequence.");
	}

	for (size_t i = 0; i < dst_seq.size(); ++i)
	{
		dst_seq.at(i).insert(dst_seq.at(i).end(), src_seq.at(i).begin(), src_seq.at(i).end());
	}
}

void FindSubCircuit::clean_sub_graph_vec(std::vector<TopologSequence<pOptimizerNodeInfo>>& sub_graph_vec,
	TopologSequence<pOptimizerNodeInfo>& target_sub_sequence)
{
	/* clean:
	* Match the number of layer elements of each subgraph and the target query subgraph in the subgraph set.
	* If the number of layer elements does not match, the subgraph is considered to be wrong and deleted
	*/
	for (auto gtaph_itr = sub_graph_vec.begin(); gtaph_itr != sub_graph_vec.end(); )
	{
		bool b_is_complete_graph = true;
		for (size_t layer_index = 0; layer_index < target_sub_sequence.size(); ++layer_index)
		{
			if (gtaph_itr->at(layer_index).size() != target_sub_sequence.at(layer_index).size())
			{
				/*If the number of nodes in the current layer is different from the number of nodes in the same layer of the target subgraph,
				* the current graph is considered incomplete
				*/
				gtaph_itr = sub_graph_vec.erase(gtaph_itr);
				b_is_complete_graph = false;
				break;
			}
		}

		if (b_is_complete_graph)
		{
			++gtaph_itr;
		}
	}
}

void FindSubCircuit::match_layer(SeqLayer<pOptimizerNodeInfo>& sub_seq_layer,
	const size_t match_layer, std::vector<TopologSequence<pOptimizerNodeInfo>>& sub_graph_vec)
{
	for (auto &sub_seq_layer_node : sub_seq_layer) /* Query each node of each layer of the subgraph */
	{
		/*Traverse matching node*/
		for (auto match_node_tier = m_node_match_vector.begin(); match_node_tier != m_node_match_vector.end(); ++match_node_tier)
		{
			auto& match_node = *match_node_tier;
			if (sub_seq_layer_node.first == match_node.first) /*Find matching set with target sub-graph node*/
			{
				for (auto match_itr = match_node.second.begin(); match_itr != match_node.second.end(); ++match_itr)
				{
					const auto match_gate_iter = match_itr->first->m_iter;

					/* Insert elements in matching set into the corresponding layers of corresponding subgraphs in sub_graph_vec */
					for (auto& perhaps_sub_graph : sub_graph_vec)
					{
						/* Obtain the set of possible subgraph sequences, then obtain the nodes of the previous layer,
						* and judge whether the current matching node can pair according to the tail node of the previous layer node
						*/
						const auto& last_layer_node_vec = perhaps_sub_graph.at(match_layer - 1);
						for (auto& last_layer_node : last_layer_node_vec)
						{
							/* Multiple tail nodes may exist in multiple qubit gates */
							for (auto& tail_node : last_layer_node.second)
							{
								if (match_gate_iter == tail_node->m_iter)
								{
									bool b_repeated_exist = false;
									for (const auto& cur_match_layer_node : perhaps_sub_graph.at(match_layer))
									{
										if (cur_match_layer_node.first->m_iter == match_itr->first->m_iter)
										{
											b_repeated_exist = true;
											break;
										}
									}
									if (!b_repeated_exist)
									{
										perhaps_sub_graph.at(match_layer).push_back(*match_itr);
									}
								}
							}
						}
					}
				}

				/* Eliminate the matched nodes which has been processed */
				m_node_match_vector.erase(match_node_tier);
				break;
			}
		}
	}
}

bool FindSubCircuit::check_angle(const pOptimizerNodeInfo node_1, const pOptimizerNodeInfo node_2)
{
	if ((node_1 == nullptr) || (node_2 == nullptr))
	{
		return false;
	}

	if (node_1->m_type != node_2->m_type)
	{
		return false;
	}

	auto angle_check_fun = [](const double target_angle, const double matched_angle) {
		if ((target_angle < ANGLE_VAR_BASE) && (abs(target_angle - matched_angle) > MAX_COMPARE_PRECISION))
		{
			return false;
		}

		return true;
	};

	QuantumGate* p1 = (std::dynamic_pointer_cast<AbstractQGateNode>(*(node_1->m_iter)))->getQGate();
	auto p_gate_1 = dynamic_cast<QGATE_SPACE::AbstractSingleAngleParameter*>(p1);
	auto p_mult_angle_gate_1 = dynamic_cast<QGATE_SPACE::AbstractAngleParameter*>(p1);
	if (nullptr != p_gate_1)
	{
		QuantumGate* p2 = (std::dynamic_pointer_cast<AbstractQGateNode>(*(node_2->m_iter)))->getQGate();
		auto p_gate_2 = dynamic_cast<QGATE_SPACE::AbstractSingleAngleParameter*>(p2);
		if (nullptr == p_gate_2)
		{
			QCERR_AND_THROW_ERRSTR(run_fail, "Error: unknow error on check-gate-angle.");
		}

		const auto target_angle_1 = p_gate_1->getParameter();
		const auto matched_angle_2 = p_gate_2->getParameter();
		if (!(angle_check_fun(target_angle_1, matched_angle_2)))
		{
			return false;
		}
	}
	else if (nullptr != p_mult_angle_gate_1)
	{
		const auto gate_type = node_1->m_type;
		switch (gate_type)
		{
		case U3_GATE:
		{
			QGATE_SPACE::U3 *u3_gate = dynamic_cast<QGATE_SPACE::U3*>((std::dynamic_pointer_cast<AbstractQGateNode>(*(node_1->m_iter)))->getQGate());
			double theta = u3_gate->get_theta();
			QGATE_SPACE::U3 *matched_u3_gate = dynamic_cast<QGATE_SPACE::U3*>((std::dynamic_pointer_cast<AbstractQGateNode>(*(node_2->m_iter)))->getQGate());
			double matched_theta = matched_u3_gate->get_theta();
			if (!(angle_check_fun(theta, matched_theta)))
			{
				return false;
			}

			double phi = u3_gate->get_phi();
			double matched_phi = matched_u3_gate->get_phi();
			if (!(angle_check_fun(phi, matched_phi)))
			{
				return false;
			}

			double lambda = u3_gate->get_lambda();
			double matched_lamda = matched_u3_gate->get_lambda();
			if (!(angle_check_fun(lambda, matched_lamda)))
			{
				return false;
			}
		}
			break;

		default:
			break;
		}
	}

	return true;
}

bool FindSubCircuit::node_match(const SeqNode<pOptimizerNodeInfo>& target_seq_node, const SeqNode<pOptimizerNodeInfo>& graph_node)
{
	if ((target_seq_node.first->m_type != graph_node.first->m_type)
		|| (target_seq_node.second.size() > graph_node.second.size()))
	{
		return false;
	}

	if ((graph_node.first->m_ctrl_qubits.size() > 0) || (graph_node.first->m_dagger))
	{
		return false;
	}

	//check angle
	if (!(check_angle(target_seq_node.first, graph_node.first)))
	{
		return false;
	}

	//check next layer 
	auto p_gate = std::dynamic_pointer_cast<AbstractQGateNode>(*(target_seq_node.first->m_iter));
	for (size_t i = 0; i < target_seq_node.second.size(); ++i)
	{
		if (target_seq_node.second.at(i) == nullptr)
		{
			continue;
		}

		if (graph_node.second.at(i) == nullptr)
		{
			return false;
		}
		else if ((target_seq_node.second.at(i)->m_type != (graph_node.second.at(i)->m_type)))
		{
			return false;
		}
		else
		{
			if (!(check_angle(target_seq_node.second.at(i), graph_node.second.at(i))))
			{
				return false;
			}
		}
	}

	return true;
}

/*******************************************************************
*                      class QCircuitOPtimizer
********************************************************************/
QCircuitOPtimizer::QCircuitOPtimizer()
	:m_cur_optimizer_sub_cir_index(0), m_b_enable_I(false), m_sub_cir_finder(m_topolog_sequence)
{
}

QCircuitOPtimizer::~QCircuitOPtimizer()
{
}

void QCircuitOPtimizer::process(const bool on_travel_end /*= false*/)
{
	PTrace("On process...\n");
	do_optimizer();

	//pop some layers to new circuit
	clean_gate_buf_to_cir(m_new_prog, on_travel_end);
	PTrace("process end.\n");
}

void QCircuitOPtimizer::register_single_gate_optimizer(const OptimizerFlag mode)
{
	if (mode & Merge_H_X)
	{
		m_optimizers.push_back(std::make_shared<OptimizerSingleGate>());
	}
	else if (mode & Merge_U3)
	{
		m_optimizers.push_back(std::make_shared<MergeU3Gate>());
	}
	else if (mode & Merge_RX)
	{
		m_optimizers.push_back(std::make_shared<OptimizerRotationSingleGate>(RX_GATE));
	}
	else if (mode & Merge_RY)
	{
		m_optimizers.push_back(std::make_shared<OptimizerRotationSingleGate>(RY_GATE));
	}
	else if (mode & Merge_RZ)
	{
		m_optimizers.push_back(std::make_shared<OptimizerRotationSingleGate>(RZ_GATE));
	}
}

void QCircuitOPtimizer::run_optimize(QProg src_prog, const QVec qubits /*= {}*/, bool b_enable_I /*= false*/)
{
	PTrace("on run_optimize\n");
	m_src_prog = src_prog;
	m_b_enable_I = b_enable_I;

	run_traversal(src_prog, qubits);
}

QProg QCircuitOPtimizer::gate_sink_to_cir(std::vector<QCircuit>& replace_to_cir_vec)
{
	auto cir_fun = [this, &replace_to_cir_vec](const size_t i) ->QCircuit {
		if (i >= replace_to_cir_vec.size())
		{
			QCERR_AND_THROW_ERRSTR(run_fail, "Error: sub-graph index error.");
		}
		return replace_to_cir_vec.at(i);
	};

	return replase_sub_cir(cir_fun);
}

void QCircuitOPtimizer::do_optimizer()
{
	if (m_cur_gates_buffer.size() == 0)
	{
		return;
	}

	for (const auto &optimizer_item : m_optimizers)
	{
		std::vector<QCircuit> replace_to_cir_vec;
		optimizer_item->do_optimize(m_src_prog, m_cur_gates_buffer, replace_to_cir_vec);
		if (replace_to_cir_vec.size() == 0) { continue; }
		
		//gate buf to cir
		QProg tmp_cir = gate_sink_to_cir(replace_to_cir_vec);

		//cir to gate buf
		cir_to_gate_buffer(tmp_cir, m_cur_gates_buffer);
	}

	for (m_cur_optimizer_sub_cir_index = 0; 
		m_cur_optimizer_sub_cir_index < m_optimizer_cir_vec.size(); ++m_cur_optimizer_sub_cir_index)
	{ 
		m_angle_vec.clear();
		sub_cir_optimizer(m_cur_optimizer_sub_cir_index);
	}
}

void QCircuitOPtimizer::check_bit_map(TopologSequence<pOptimizerNodeInfo>& target_sub_graph_seq)
{
	m_sub_graph_qubit_map_vec.clear();
	m_angle_vec.clear();
	/* Traverse each matched subgraph */
	const auto& sub_graph_vec = m_sub_cir_finder.get_sub_graph_vec();
	for (auto gtaph_itr = sub_graph_vec.begin(); gtaph_itr != sub_graph_vec.end(); ++gtaph_itr)
	{
		std::map<size_t, Qubit*> cur_sub_graph_bit_map;
		std::vector<double> cur_sub_graph_angle_var;
		/* Traversing each layer of the target topological sequence */
		for (size_t layer_index = 0; layer_index < target_sub_graph_seq.size(); ++layer_index)
		{
			auto& cur_layer_on_target_sub_graph = target_sub_graph_seq.at(layer_index);
			/* If there are the same logic gates including same angles, 
			* bit mapping is not checked, because mapping cannot be checked for this case
			*/
			if (check_same_gate_type(cur_layer_on_target_sub_graph))
			{
				continue;
			}

			auto& matched_sub_graph_layer = gtaph_itr->at(layer_index);
			/*Traversing every gate in current layer in the target topological sequence*/
			for (size_t gate_index = 0; gate_index < cur_layer_on_target_sub_graph.size(); ++gate_index)
			{
				auto& gate_on_target_sub_graph = cur_layer_on_target_sub_graph.at(gate_index).first;
				/*Traversing every gate in current layer in the matched topological sequence*/
				for (size_t match_gate_index = 0; match_gate_index < matched_sub_graph_layer.size(); ++match_gate_index)
				{
					auto& matched_gate = matched_sub_graph_layer.at(match_gate_index).first;
					if (gate_on_target_sub_graph->m_type == matched_gate->m_type)
					{
						check_angle_param(gate_on_target_sub_graph, matched_gate, cur_sub_graph_angle_var);
						for (size_t i = 0; i < gate_on_target_sub_graph->m_target_qubits.size(); ++i)
						{
							size_t src_qubit = gate_on_target_sub_graph->m_target_qubits.at(i)->get_phy_addr();
							Qubit* maped_qubit = matched_gate->m_target_qubits.at(i);
							auto exist_iter = cur_sub_graph_bit_map.find(src_qubit);
							if (exist_iter == cur_sub_graph_bit_map.end())
							{
								auto val = cur_sub_graph_bit_map.insert(std::pair<size_t, Qubit*>(src_qubit, maped_qubit));
							}
							else
							{
								if (exist_iter->second->get_phy_addr() != (maped_qubit->get_phy_addr()))
								{
									QCERR_AND_THROW_ERRSTR(run_fail, "Error: unknow error on check qubit map.");
								}
							}
						}
					}
				}
			}
		}

		m_angle_vec.push_back(cur_sub_graph_angle_var);
		m_sub_graph_qubit_map_vec.push_back(cur_sub_graph_bit_map);
	}
}

bool QCircuitOPtimizer::check_same_gate_type(SeqLayer<pOptimizerNodeInfo>& layer)
{
	for (size_t i = 0; i < (layer.size() - 1); ++i)
	{
		for (size_t j = i + 1; j < layer.size(); ++j)
		{
			auto& first_gate = layer.at(i).first;
			auto& second_gate = layer.at(j).first;
			if (first_gate->m_type == second_gate->m_type)
			{
				return true;
			}
		}
	}

	return false;
}

void QCircuitOPtimizer::check_angle_param(pOptimizerNodeInfo target_gate, pOptimizerNodeInfo matched_gate, std::vector<double>& angle_vec)
{
	auto add_to_angle_vec = [&](const double angle, const double match_angle) {
		if (angle >= ANGLE_VAR_BASE)
		{
			int i = angle / ANGLE_VAR_BASE;
			if ((angle_vec.size() + 1) != i)
			{
				QCERR_AND_THROW_ERRSTR(run_fail, "Error: unknow error on check angle param.");
			}
			angle_vec.push_back(match_angle);
		}
	};

	const auto gate_type = target_gate->m_type;
	switch (gate_type)
	{
	case RX_GATE:
	case RY_GATE:
	case RZ_GATE:
	case U1_GATE:
	case CPHASE_GATE:
	case ISWAP_THETA_GATE:
	{
		QuantumGate* p = (std::dynamic_pointer_cast<AbstractQGateNode>(*(target_gate->m_iter)))->getQGate();
		auto p_gate = dynamic_cast<QGATE_SPACE::AbstractSingleAngleParameter*>(p);
		const double angle = p_gate->getParameter();

		p = (std::dynamic_pointer_cast<AbstractQGateNode>(*(matched_gate->m_iter)))->getQGate();
		auto p_matched_gate = dynamic_cast<QGATE_SPACE::AbstractSingleAngleParameter*>(p);
		add_to_angle_vec(angle, p_matched_gate->getParameter());
	}
		break;

	case U3_GATE:
	{
		QGATE_SPACE::U3 *u3_gate = dynamic_cast<QGATE_SPACE::U3*>((std::dynamic_pointer_cast<AbstractQGateNode>(*(target_gate->m_iter)))->getQGate());
		double theta = u3_gate->get_theta();
		QGATE_SPACE::U3 *matched_u3_gate = dynamic_cast<QGATE_SPACE::U3*>((std::dynamic_pointer_cast<AbstractQGateNode>(*(matched_gate->m_iter)))->getQGate());
		double matched_theta = matched_u3_gate->get_theta();
		add_to_angle_vec(theta, matched_theta);
		
		double phi = u3_gate->get_phi();
		double matched_phi = matched_u3_gate->get_phi();
		add_to_angle_vec(phi, matched_phi);

		double lambda = u3_gate->get_lambda();
		double matched_lamda = matched_u3_gate->get_lambda();
		add_to_angle_vec(lambda, matched_lamda);
	}
		break;

	default:
		break;
	}
}

void QCircuitOPtimizer::set_angle_param(std::shared_ptr<AbstractQGateNode> p_gate, const size_t target_graph_index)
{
	auto get_map_angle_fun = [this, &target_graph_index](const double angle, double& maped_angle) {
		if (angle >= ANGLE_VAR_BASE)
		{
			maped_angle = m_angle_vec.at(target_graph_index).at((angle / ANGLE_VAR_BASE) - 1);
			return true;
		}

		return false;
	};

	if (m_angle_vec.at(target_graph_index).size() != 0)
	{
		const auto gate_type = p_gate->getQGate()->getGateType();
		if (gate_type == U3_GATE)
		{
			QGATE_SPACE::U3 *u3_gate = dynamic_cast<QGATE_SPACE::U3*>(p_gate->getQGate());
			double theta = u3_gate->get_theta();
			bool b_need_map = false || get_map_angle_fun(theta, theta);
			
			double phi = u3_gate->get_phi();
			b_need_map = b_need_map || get_map_angle_fun(phi, phi);

			double lambda = u3_gate->get_lambda();
			b_need_map = b_need_map || get_map_angle_fun(lambda, lambda);

			if (b_need_map)
			{
				QuantumGate* qgate = new QGATE_SPACE::U3(theta, phi, lambda);
				p_gate->setQGate(qgate);
			}

			return;
		}

		auto p_single_rotation_gate = dynamic_cast<QGATE_SPACE::AbstractSingleAngleParameter*>(p_gate->getQGate());
		if (nullptr != p_single_rotation_gate)
		{
			const double angle = p_single_rotation_gate->getParameter();
			if (angle >= ANGLE_VAR_BASE)
			{
				int i = (angle / ANGLE_VAR_BASE) - 1;
				double real_angle = m_angle_vec.at(target_graph_index).at(i);
				QuantumGate * qgate = nullptr;
				switch (p_gate->getQGate()->getGateType())
				{
				case RX_GATE:
				{
					qgate = new QGATE_SPACE::RX(real_angle);

				}
				break;

				case RY_GATE:
				{
					qgate = new QGATE_SPACE::RY(real_angle);
				}
				break;

				case RZ_GATE:
				{
					qgate = new QGATE_SPACE::RZ(real_angle);
				}
				break;

				case U1_GATE:
				{
					qgate = new QGATE_SPACE::U1(real_angle);
				}
				break;

				case CPHASE_GATE:
				{
					qgate = new QGATE_SPACE::CPHASE(real_angle);
				}
				break;

				case ISWAP_THETA_GATE:
				{
					qgate = new QGATE_SPACE::ISWAPTheta(real_angle);
				}
				break;

				default:
					QCERR_AND_THROW_ERRSTR(run_fail, "Error: unknow error on set angel param.");
					break;
				}

				p_gate->setQGate(qgate);
			}
		}
	}
}

QCircuit QCircuitOPtimizer::remap_cir(QCircuit src_cir, const size_t target_graph_index)
{
	auto& cur_sub_graph_bit_map = m_sub_graph_qubit_map_vec.at(target_graph_index);
	QCircuit new_cir = deepCopy(src_cir);
	for (auto gate_itr = new_cir.getFirstNodeIter(); gate_itr != new_cir.getEndNodeIter(); ++gate_itr)
	{
		auto p_gate = std::dynamic_pointer_cast<AbstractQGateNode>(*gate_itr);
		QVec qubit_vec;
		p_gate->getQuBitVector(qubit_vec);
		for (auto& tmp_qubit : qubit_vec)
		{
			tmp_qubit = cur_sub_graph_bit_map.at(tmp_qubit->get_phy_addr());
		}
		p_gate->remap(qubit_vec);

		//check angle
		set_angle_param(p_gate, target_graph_index);
	}

	return new_cir;
}

QProg QCircuitOPtimizer::replase_sub_cir(std::function<QCircuit(const size_t)> get_cir_fun)
{
	m_tmp_cir_vec.push_back(QProg());
	if (m_tmp_cir_vec.size() > 4)
	{
		m_tmp_cir_vec.erase(m_tmp_cir_vec.begin());
	}
	auto& tmp_cir = m_tmp_cir_vec.back();
	std::map<size_t, std::list<pOptimizerNodeInfo>::iterator> pos_map;/* traversal position for every qubit*/
	for (auto& q : m_cur_gates_buffer)
	{
		pos_map.insert(std::make_pair(q.first, q.second.begin()));
	}

	layer_iter_seq tmp_seq;
	while (true)
	{
		bool b_finished = true;
		bool b_find_sub_graph = false;
		size_t min_layer = MAX_SIZE;
		size_t target_graph_index = MAX_SIZE;
		for (const auto& item : m_cur_gates_buffer)
		{
			const size_t qubit_index = item.first;
			auto j = pos_map.at(qubit_index);
			for (; j != m_cur_gates_buffer.at(qubit_index).end(); ++j)
			{
				b_finished = false;
				//PTrace("On qubit %lld, %lld gate.\n", qubit_index, j);

				auto &gate_node = (*j);
				if (gate_node->m_sub_graph_index >= 0)
				{
					if (gate_node->m_layer < min_layer)
					{
						min_layer = gate_node->m_layer;
						target_graph_index = gate_node->m_sub_graph_index;
					}

					b_find_sub_graph = true;
					break; //break to deal with next qubit
				}
				else
				{
					/* add node to tmp_seq */
					add_node_to_seq(tmp_seq, gate_node->m_iter, gate_node->m_layer);
				}
			}
			pos_map.at(qubit_index) = j;
		}

		if (b_finished)
		{
			break;
		}

		size_t max_output_layer = 0;/* the max layer for output to seq*/
		QCircuit remaped_cir;
		std::vector<int> sub_cir_used_qubits;
		if (b_find_sub_graph)
		{
			remaped_cir = get_cir_fun(target_graph_index);
			/*PTrace("remaped_cir");
		    PTraceCircuit(remaped_cir);*/
			get_all_used_qubits(remaped_cir, sub_cir_used_qubits);
			if (sub_cir_used_qubits.size() == 0)
			{
				QCERR_AND_THROW_ERRSTR(run_fail, "Error: sub_cir is null.");
			}

			for (auto& _qubit : sub_cir_used_qubits)
			{
				const auto& gate_vec_on_target_qubit = m_cur_gates_buffer.at(_qubit);
				if (gate_vec_on_target_qubit.end() != pos_map.at(_qubit))
				{
					const auto& tmp_qubit_layer = (*(pos_map.at(_qubit)))->m_layer;
					if (tmp_qubit_layer > max_output_layer)
					{
						max_output_layer = tmp_qubit_layer;
					}
				}
			}
		}
		else
		{
			max_output_layer = MAX_SIZE;
		}

		//convert tmp_seq to quantum circuit
		seq_to_cir(tmp_seq, tmp_cir, 0, max_output_layer);

		if (!b_find_sub_graph)
		{
			break;
		}
		
		for (auto sub_cir_itr = remaped_cir.getFirstNodeIter(); sub_cir_itr != remaped_cir.getEndNodeIter(); ++sub_cir_itr)
		{
			if (((std::dynamic_pointer_cast<AbstractQGateNode>(*sub_cir_itr))->getQGate()->getGateType() == I_GATE) && (!m_b_enable_I))
			{
				continue;
			}
			tmp_cir.pushBackNode(*sub_cir_itr);
		}

		//skip sub_graph matched gate node
		for (auto& target_qubit : sub_cir_used_qubits)
		{
			for (auto m = pos_map.at(target_qubit); m != m_cur_gates_buffer.at(target_qubit).end(); ++m, ++(pos_map.at(target_qubit)))
			{
				const auto& gate_node = (*m);
				if (gate_node->m_sub_graph_index != target_graph_index)
				{
					break;
				}
			}
		}
	}

	if (tmp_seq.size() > 0)
	{
		seq_to_cir(tmp_seq, tmp_cir, 0, MAX_SIZE);
	}
	return tmp_cir;
}

void QCircuitOPtimizer::mark_sug_graph(const std::vector<TopologSequence<pOptimizerNodeInfo>>& sub_graph_vec)
{
	for (size_t i = 0; i < sub_graph_vec.size(); ++i)
	{
		auto& sub_graph_item = sub_graph_vec.at(i);
		for (auto& sug_graph_layer : sub_graph_item)
		{
			for (auto& sug_graph_layer_node : sug_graph_layer)
			{
				sug_graph_layer_node.first->m_sub_graph_index = i;
			}
		}
	}
}

void QCircuitOPtimizer::sub_cir_optimizer(const size_t optimizer_sub_cir_index)
{
	//transfer gate sink to topolog sequence
	gates_sink_to_topolog_sequence(m_cur_gates_buffer, m_topolog_sequence);

	//query sub circuit
	TopologSequence<pOptimizerNodeInfo> sub_cir_sequence;
	cir_to_topolog_sequence(m_optimizer_cir_vec.at(optimizer_sub_cir_index).target_sub_cir, 
		m_optimizer_cir_vec.at(optimizer_sub_cir_index).m_sub_cir_gates_buffer, sub_cir_sequence);

	m_sub_cir_finder.sub_cir_query(sub_cir_sequence);

	check_bit_map(sub_cir_sequence);

	mark_sug_graph(m_sub_cir_finder.get_sub_graph_vec());

	//replace sub circuit
	auto cir_fun = [this, optimizer_sub_cir_index](const size_t i) ->QCircuit {
		auto sub_cir = m_optimizer_cir_vec.at(optimizer_sub_cir_index).replace_to_sub_cir;
		QCircuit remaped_cir = remap_cir(sub_cir, i);
		return remaped_cir;
	};

	QProg tmp_cir = replase_sub_cir(cir_fun);

	/* clean up the temporary vector after the sub-graph replacement is completed */
	m_sub_cir_finder.clear();

	cir_to_gate_buffer(tmp_cir, m_cur_gates_buffer);

	m_sub_graph_qubit_map_vec.clear();
}

void QCircuitOPtimizer::register_optimize_sub_cir(QCircuit sub_cir, QCircuit replase_to_cir)
{
	OptimizerSubCir tmp_optimizer_sub_cir;
	tmp_optimizer_sub_cir.target_sub_cir = sub_cir;
	tmp_optimizer_sub_cir.replace_to_sub_cir = replase_to_cir;
	m_optimizer_cir_vec.push_back(tmp_optimizer_sub_cir);
}

/*******************************************************************
*                      public interface
********************************************************************/
void QPanda::sub_cir_optimizer(QCircuit& src_cir, std::vector<std::pair<QCircuit, QCircuit>> optimizer_cir_vec, const OptimizerFlag mode/* = Merge_H_X*/)
{
	if (src_cir.is_empty())
	{
		return;
	}

	flatten(src_cir);

	QCircuitOPtimizer tmp_optimizer;
	for (auto& optimizer_cir_pair : optimizer_cir_vec)
	{
		tmp_optimizer.register_optimize_sub_cir(optimizer_cir_pair.first, optimizer_cir_pair.second);
	}
	tmp_optimizer.register_single_gate_optimizer(mode);
	tmp_optimizer.run_optimize(src_cir/*, used_qubits*/);

	flatten(tmp_optimizer.m_new_prog, true);
	src_cir = QProgFlattening::prog_flatten_to_cir(tmp_optimizer.m_new_prog);
}

void QPanda::sub_cir_optimizer(QProg& src_prog, std::vector<std::pair<QCircuit, QCircuit>> optimizer_cir_vec, const OptimizerFlag mode/*= Merge_H_X*/)
{
	if (src_prog.is_empty())
	{
		return;
	}

	flatten(src_prog);

	QCircuitOPtimizer tmp_optimizer;
	for (auto& optimizer_cir_pair : optimizer_cir_vec)
	{
		tmp_optimizer.register_optimize_sub_cir(optimizer_cir_pair.first, optimizer_cir_pair.second);
	}
	tmp_optimizer.register_single_gate_optimizer(mode);
	tmp_optimizer.run_optimize(src_prog/*, used_qubits*/);

	src_prog = tmp_optimizer.m_new_prog;
}
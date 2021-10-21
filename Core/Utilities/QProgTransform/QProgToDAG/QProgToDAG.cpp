#include <memory>
#include "Core/QuantumCircuit/QNodeDeepCopy.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/QProgToDAG.h"
USING_QPANDA
using namespace std;

/*******************************************************************
*                      class QProgToDAG
********************************************************************/
void QProgToDAG::transformQGate(shared_ptr<AbstractQGateNode> gate_node, QCircuitParam& param, NodeIter& cur_iter)
{
    if (nullptr == gate_node || nullptr == gate_node->getQGate())
    {
        QCERR("gate_node is null");
        throw invalid_argument("gate_node is null");
    }

	QCirParamForDAG& cir_param = static_cast<QCirParamForDAG&>(param);
	QProgDAG &prog_dag = cir_param.m_dag;

	auto p_node_info = std::make_shared<QProgDAGNode>();
	gate_node->getQuBitVector(p_node_info->m_qubits_vec);
	gate_node->getControlVector(p_node_info->m_control_vec);
	p_node_info->m_control_vec += param.m_control_qubits;
	p_node_info->m_itr = cur_iter;
	p_node_info->m_angles = get_gate_parameter(gate_node);

	//check control qubits
	const auto total_used_qv = p_node_info->m_qubits_vec + p_node_info->m_control_vec;
	if (total_used_qv.size() != (p_node_info->m_qubits_vec.size() + p_node_info->m_control_vec.size())){
		QCERR_AND_THROW(runtime_error, "Control gate Error: Illegal control qubits.");
	}

	p_node_info->m_dagger = gate_node->isDagger() ^ param.m_is_dagger;

    prog_dag.add_vertex(p_node_info, (DAGNodeType)(gate_node->getQGate()->getGateType()));
}

void QProgToDAG::execute(std::shared_ptr<AbstractQuantumMeasure>  cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam& param, NodeIter& cur_iter)
{
	QCirParamForDAG& cir_param = static_cast<QCirParamForDAG&>(param);
	QProgDAG &prog_dag = cir_param.m_dag;
	transform_non_gate_node(cur_node, prog_dag, cur_iter, DAGNodeType::MEASURE);
}

void QProgToDAG::execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam& param, NodeIter& cur_iter)
{
	QCirParamForDAG& cir_param = static_cast<QCirParamForDAG&>(param);
	QProgDAG &prog_dag = cir_param.m_dag;
	transform_non_gate_node(cur_node, prog_dag, cur_iter, DAGNodeType::RESET);
}

void QProgToDAG::execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam& param, NodeIter& cur_iter)
{
    transformQGate(cur_node, param, cur_iter);
}

void QProgToDAG::execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam&, NodeIter& cur_iter)
{
    QCERR("ignore classical prog node");
}

void QProgToDAG::execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam&, NodeIter& cur_iter)
{
    QCERR_AND_THROW(run_fail, "Error: unsupport controlflow node.");
}

#if 0
/*******************************************************************
*                      class CutQCircuit
********************************************************************/
class CutQCircuit
{
public:
	CutQCircuit(const QProgDAG& prog_dag)
		:m_src_prog_dag(prog_dag), m_cut_prog_dag(prog_dag)
	{}
	virtual ~CutQCircuit() {}

	void cut_circuit(const std::map<uint32_t, std::vector<uint32_t>>& cut_pos){
		const auto& qubit_vertices_map = m_src_prog_dag.get_qubit_vertices_map();
		const std::vector<QProgDAGVertex>& dag_vertices = m_src_prog_dag.get_vertex_c();
		//for (const auto& _cur : cut_pos)
		//{
		//	const auto& qubit_vertices = qubit_vertices_map.at(_cur.first);
		//	for (const auto& _vertex_index : _cur.second)
		//	{
		//		if (qubit_vertices.size() <= _vertex_index + 1) {
		//			QCERR_AND_THROW(run_fail, "Error: cut-pos error; vertices.size() = "
		//				<< qubit_vertices.size() << "_vertex_index= ", _vertex_index);
		//		}

		//		//cut edge
		//		QProgDAGEdge cut_edge(qubit_vertices[_vertex_index], qubit_vertices[_vertex_index + 1], _cur.first);
		//		m_cut_prog_dag.remove_edge(cut_edge);
		//	}
		//}

		//get sub-circuit
		for (const auto& _cur : cut_pos)
		{
			const auto& qubit_vertices = qubit_vertices_map.at(_cur.first);
			uint32_t start_vertex_index = 0;
			for (const auto& _vertex_index : _cur.second)
			{
				if (qubit_vertices.size() <= _vertex_index + 1) {
					QCERR_AND_THROW(run_fail, "Error: cut-pos error; vertices.size() = "
						<< qubit_vertices.size() << "_vertex_index= ", _vertex_index);
				}

				//cut edge
				QProgDAGEdge cut_edge(qubit_vertices[_vertex_index], qubit_vertices[_vertex_index + 1], _cur.first);
				m_cut_prog_dag.remove_edge(cut_edge);

				//check first vertex
				uint32_t _cur_vertex = qubit_vertices[start_vertex_index];
				int sub_cir_index = find_vertex_in_sub_cir(_cur_vertex);
				if (-1 == sub_cir_index)
				{
					m_sub_cir_vec.emplace_back(SubCircuit());
					sub_cir_index = m_sub_cir_vec.size() - 1;
				}
				
				update_sub_cir(m_sub_cir_vec[sub_cir_index], qubit_vertices,
					start_vertex_index, _vertex_index, _cur.first);

			}
		}

	}

#if 0
	void extend_topo_seq(std::vector<std::vector<uint32_t>>& topo_seq, 
		std::vector<std::vector<uint32_t>>::iterator cur_layer_itr,
		std::set<uint32_t>& vertice, const uint32_t cur_vertex_index) {
		if (vertice.size() == 0){
			return;
		}

		const auto& all_vertice_vec = m_cut_prog_dag.get_vertex_c();
		const auto& _vertex = all_vertice_vec[cur_vertex_index];

		// check pre_node
		for (const auto& _vertex_index : _vertex.m_pre_node)
		{
			auto _itr = vertice.find(_vertex_index);
			if (vertice.end() != _itr)
			{
				vertice.erase(_itr);
				if (cur_layer_itr == topo_seq.begin())
				{
					topo_seq.insert(topo_seq.begin(), std::vector<uint32_t>({ _vertex_index }));
					extend_topo_seq(topo_seq, topo_seq.begin(), vertice, _vertex_index);
				}
				else
				{
					(cur_layer_itr - 1)->emplace_back(_vertex_index);
					extend_topo_seq(topo_seq, cur_layer_itr - 1, vertice, _vertex_index);
				}
			}
			else
			{
				QCERR_AND_THROW(run_fail, "Error: sub_cir pre-vertice error.");
			}
		}

		// check succ_node
		for (const auto& _vertex_index : _vertex.m_succ_node)
		{
			auto _itr = vertice.find(_vertex_index);
			if (vertice.end() != _itr)
			{
				vertice.erase(_itr);
				if (cur_layer_itr == topo_seq.end() - 1)
				{
					topo_seq.insert(topo_seq.end(), std::vector<uint32_t>({ _vertex_index }));
					extend_topo_seq(topo_seq, topo_seq.end() - 1, vertice, _vertex_index);
				}
				else
				{
					(cur_layer_itr + 1)->emplace_back(_vertex_index);
					extend_topo_seq(topo_seq, cur_layer_itr + 1, vertice, _vertex_index);
				}
			}
			else
			{
				QCERR_AND_THROW(run_fail, "Error: sub_cir succ-vertice error.");
			}
		}
	}

	void generate_subcircuits() {
		const auto& qv_map = m_src_prog_dag.m_qubits;
		std::vector<std::vector<uint32_t>> topo_seq;
		std::vector<std::vector<uint32_t>>::iterator cur_topo_seq_itr;
		const auto& all_vertice_vec = m_cut_prog_dag.get_vertex_c();
		for (auto& _sub_cir : m_sub_cir_vec)
		{
			topo_seq.clear();
			auto vertice_copy = _sub_cir.m_vertice;

			auto _vertex_index = *(vertice_copy.begin());
			vertice_copy.erase(vertice_copy.begin());
			topo_seq.emplace_back(std::vector<uint32_t>({ _vertex_index }));
			cur_topo_seq_itr = topo_seq.begin();//µ±Ç°²ãµü´úÆ÷
			/*const auto& _vertex = all_vertice_vec[_vertex_index];*/
			extend_topo_seq(topo_seq, cur_topo_seq_itr, vertice_copy, _vertex_index);
		}
	}
#endif

	void generate_subcircuits() {
		const auto& qv_map = m_src_prog_dag.m_qubits;
		std::vector<std::vector<uint32_t>> topo_seq;
		std::vector<std::vector<uint32_t>>::iterator cur_topo_seq_itr;
		const auto& all_vertice_vec = m_cut_prog_dag.get_vertex_c();
		for (auto& _sub_cir : m_sub_cir_vec)
		{
			topo_seq.clear();
			auto vertice_copy = _sub_cir.m_vertice;

			while (vertice_copy.size() > 0)
			{
				//find vertex that the in-degree is 0
				std::vector<uint32_t> _layer;
				for (auto vertex_itr = vertice_copy.begin(); vertex_itr != vertice_copy.end(); )
				{
					const auto _vertex_index = *vertex_itr;
					const auto& cur_vertex_node = all_vertice_vec[_vertex_index];
					if (cur_vertex_node.m_pre_edges.size() == 0)
					{
						_layer.emplace_back(_vertex_index);
						vertex_itr = vertice_copy.erase(vertex_itr);
						continue;
					}

					++vertex_itr;
				}

				for (const auto& _i : _layer)
				{
					const auto& cur_vertex_node = all_vertice_vec[_i];
					for (const auto& _e : cur_vertex_node.m_succ_edges) {
						m_cut_prog_dag.remove_edge(_e);
					}
				}

				topo_seq.emplace_back(_layer);
			}

			// build QCircuit by topo_seq
			for (const auto& _layer : topo_seq)
			{
				for (const auto& _node_index : _layer)
				{
					_sub_cir.m_cir.insertQNode(_sub_cir.m_cir.getEndNodeIter(), *(all_vertice_vec[_node_index].m_node->m_itr));
				}
			}
		}
	}

	const std::vector<SubCircuit>& get_cutted_sub_circuits() { return m_sub_cir_vec; }

protected:
	int find_vertex_in_sub_cir(const uint32_t& vertex)
	{
		for (size_t i = 0; i < m_sub_cir_vec.size(); ++i)
		{
			const SubCircuit& _sub_cir = m_sub_cir_vec[i];
			if (_sub_cir.m_vertice.end() != _sub_cir.m_vertice.find(vertex)) {
				return i;
			}
		}

		return -1;
	}

	void update_sub_cir(SubCircuit& sub_cir, const std::vector<size_t>& qubit_vertices,
		const uint32_t& start_pos, const uint32_t& end_pos, const uint32_t& qubit)
	{
		const std::vector<QProgDAGVertex>& src_dag_vertices = m_src_prog_dag.get_vertex_c();
		std::vector<QProgDAGVertex>& cut_dag_vertices = m_cut_prog_dag.get_vertex();
		for (auto i = start_pos; i <= end_pos; ++i)
		{
			const auto& _cur_vertex_index = qubit_vertices[i];
			
			//update sub-circuit vertice
			sub_cir.m_vertice.emplace(_cur_vertex_index);

			//update sub-circuit qubit
			auto vertex_qubit = src_dag_vertices[_cur_vertex_index].m_node->m_qubits_vec + 
				src_dag_vertices[_cur_vertex_index].m_node->m_control_vec;
			for (const auto& _q : vertex_qubit){
				sub_cir.m_qubit.emplace(_q->get_phy_addr());
			}

			// check preparation-qubit
			if (i == start_pos)
			{
				const auto& src_vertex = src_dag_vertices[_cur_vertex_index];
				const auto& _cur_vertex = cut_dag_vertices[_cur_vertex_index];
				if (src_vertex.m_pre_edges.size() != _cur_vertex.m_pre_edges.size())
				{
					if (src_vertex.m_pre_edges.size() > _cur_vertex.m_pre_edges.size())
					{
						sub_cir.m_pre_qubit.emplace(qubit);
					}
					else
					{
						QCERR_AND_THROW(run_fail, "Error: nuknow error on preparation-qubit.");
					}
				}
			}

			// check measure-qubit
			if (i == end_pos)
			{
				const auto& src_vertex = src_dag_vertices[_cur_vertex_index];
				const auto& _cur_vertex = cut_dag_vertices[_cur_vertex_index];
				if (src_vertex.m_succ_edges.size() != _cur_vertex.m_succ_edges.size())
				{
					if (src_vertex.m_succ_edges.size() > _cur_vertex.m_succ_edges.size())
					{
						sub_cir.m_pre_qubit.emplace(qubit);
					}
					else
					{
						QCERR_AND_THROW(run_fail, "Error: nuknow error on measure-qubit.");
					}
				}
			}
		}
	}

private:
	const QProgDAG& m_src_prog_dag;
	QProgDAG m_cut_prog_dag;
	std::vector<SubCircuit> m_sub_cir_vec;
};
#endif
/*******************************************************************
*                      public interface
********************************************************************/
std::shared_ptr<QProgDAG> QPanda::qprog_to_DAG(QProg prog)
{
	QProgToDAG prog_to_dag;
	auto dag = std::make_shared<QProgDAG>();
	prog_to_dag.traversal(prog, *dag);

	return dag;
}

#if 0
QCircuit QPanda::circuit_stripping(QProg prog)
{
	return QCircuitStripping().get_stripped_cir(prog);
}

std::vector<SubCircuit> QPanda::cut_circuit(const QProgDAG& prog_dag, const std::map<uint32_t, std::vector<uint32_t>>& cut_pos)
{
	CutQCircuit cutter(prog_dag);
	cutter.cut_circuit(cut_pos);
	cutter.generate_subcircuits();
	return cutter.get_cutted_sub_circuits();
#if 0
	QProgDAG tmp_dag = prog_dag;
	const auto& qubit_vertices_map = tmp_dag.get_qubit_vertices_map();
	const std::vector<QProgDAGVertex>& dag_vertices = tmp_dag.get_vertex();
	for (const auto& _cur : cut_pos)
	{
		const auto& qubit_vertices = qubit_vertices_map.at(_cur.first);
		for (const auto& _vertex_index : _cur.second)
		{
			if (qubit_vertices.size() <= _vertex_index + 1) {
				QCERR_AND_THROW(run_fail, "Error: cut-pos error; vertices.size() = " 
					<< qubit_vertices.size() << "_vertex_index= ", _vertex_index);
			}

			//cut edge
			QProgDAGEdge cut_edge(qubit_vertices[_vertex_index], qubit_vertices[_vertex_index + 1], _cur.first);
			tmp_dag.remove_edge(cut_edge);
		}
	}

	//get sub-circuit
	std::vector<SubCircuit> sub_cir_vec;
	for (const auto& _cur : cut_pos)
	{
		const auto& qubit_vertices = qubit_vertices_map.at(_cur.first);
		uint32_t start_vertex_index = 0;
		for (const auto& _vertex_index : _cur.second)
		{
			/*if (qubit_vertices.size() <= _vertex_index + 1) {
				QCERR_AND_THROW(run_fail, "Error: cut-pos error; vertices.size() = "
					<< qubit_vertices.size() << "_vertex_index= ", _vertex_index);
			}*/

			//cut edge
			/*QProgDAGEdge cut_edge(qubit_vertices[_vertex_index], qubit_vertices[_vertex_index + 1], _cur.first);
			tmp_dag.remove_edge(cut_edge);*/

			uint32_t _cur_vertex = qubit_vertices[start_vertex_index];
			const int sub_cir_index = find_vertex_in_sub_cir(sub_cir_vec, _cur_vertex);
			if (sub_cir_index == -1)
			{
				sub_cir_vec.emplace_back(SubCircuit());
				update_sub_cir(sub_cir_vec.back(), qubit_vertices,
					start_vertex_index, _vertex_index, dag_vertices);
			}
			else
			{
				update_sub_cir(sub_cir_vec[sub_cir_index], qubit_vertices,
					start_vertex_index, _vertex_index, dag_vertices);
			}
			
		}
	}
#endif
}
#endif
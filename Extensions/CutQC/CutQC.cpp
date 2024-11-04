#include "CutQC.h"
#include <complex>
#include <algorithm>
#include <numeric>
#include "ThirdParty/Eigen/Dense"
#include "ThirdParty/Eigen/Sparse"
#include "ThirdParty/EigenUnsupported/Eigen/KroneckerProduct"
#include "Core/QuantumCircuit/QNodeDeepCopy.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/Utilities/QProgInfo/GetAllUsedQubitAndCBit.h"
#include "Core/Utilities/Tools/QStatMatrix.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/QProgToDAG.h"

using namespace QPanda;
using namespace std;
using namespace Eigen;

#define PRINT_TRACE 0
#if PRINT_TRACE
#define PTrace(_msg) {\
    std::ostringstream ss;\
    ss << _msg;\
    std::cout<<__FILENAME__<<"," <<__LINE__<<","<<__FUNCTION__<<":"<<ss.str()<<std::endl;}
#define PTraceCircuit(cir) (std::cout << cir << endl)
#define PTraceCircuitMat(cir) { auto m = getCircuitMatrix(cir); std::cout << m << endl; }
#define PTraceMat(mat) (std::cout << (mat) << endl)
#else
#define PTrace(_msg)
#define PTraceCircuit(cir)
#define PTraceCircuitMat(cir)
#define PTraceMat(mat)
#endif

static size_t g_shot = 1e7;

static Qnum get_qubits_addr(const QVec &qvs)
{
    Qnum qubits_addrs;
    for (const auto& qubit : qvs)
    {
        qubits_addrs.emplace_back(qubit->get_phy_addr());
    }

    return qubits_addrs;
}
/*******************************************************************
*                      class CutQCircuit
********************************************************************/
void CutQCircuit::cut_circuit(const std::map<uint32_t, std::vector<uint32_t>>& cut_pos,
	const std::vector<std::vector<uint32_t>>& sub_graph_vertice, QuantumMachine *qvm)
{
	const auto& qubit_vertices_map = m_src_prog_dag.get_qubit_vertices_map();
	m_max_qubit_num = qubit_vertices_map.size();
	const std::vector<QProgDAGVertex>& dag_vertices = m_src_prog_dag.get_vertex_c();
	m_vertex_sub_cir_info.resize(dag_vertices.size(), -1);

	// update vertex-sub-cir info
	for (uint32_t _i = 0; _i < sub_graph_vertice.size(); ++_i)
	{
		for (uint32_t _j = 0; _j < sub_graph_vertice[_i].size(); ++_j)
		{
			m_vertex_sub_cir_info[sub_graph_vertice[_i][_j]] = _i;
		}
	}
	m_cut_fragment_vec.resize(sub_graph_vertice.size());

	for (const auto& _cut : cut_pos)
	{
		const auto& qubit_vertices = qubit_vertices_map.at(_cut.first);
		for (const auto& _cut_index : _cut.second)
		{
			if (qubit_vertices.size() <= _cut_index + 1) {
				QCERR_AND_THROW(run_fail, "Error: cut-pos error; vertices.size() = "
					<< qubit_vertices.size() << "_cut_index= " << _cut_index);
			}

			//cut edge
			QProgDAGEdge cut_edge(qubit_vertices[_cut_index], qubit_vertices[_cut_index + 1], _cut.first);
			m_cut_prog_dag.remove_edge(cut_edge);
		}
	}

	//get sub-circuit
	const std::vector<QProgDAGVertex>& src_dag_vertices = m_src_prog_dag.get_vertex_c();
	const std::vector<QProgDAGVertex>& cut_dag_vertices = m_cut_prog_dag.get_vertex();
	for (const auto& q_vertices : qubit_vertices_map)
	{
		const auto& qubit_vertices = q_vertices.second;
		if (cut_pos.find(q_vertices.first) == cut_pos.end())
		{
			CutFragment tmp_fragment;
			tmp_fragment.m_qubit.emplace(q_vertices.first);
			for (const auto& _vertex_index : qubit_vertices){
				tmp_fragment.m_vertice.emplace(_vertex_index);
			}
			append_sub_cir(tmp_fragment, qvm);
		}
		else
		{
			auto cut_pos_vec = cut_pos.find(q_vertices.first)->second;
			cut_pos_vec.emplace_back(qubit_vertices.size() - 1);//append the remain vertice after the last cut-pos
			uint32_t start_pos_index = 0;
			for (const auto& _cut_pos : cut_pos_vec)
			{
				CutFragment tmp_fragment;
				tmp_fragment.m_qubit.emplace(q_vertices.first);

				// check preparation-qubit
				if (is_pre_qubit(src_dag_vertices[qubit_vertices[start_pos_index]], 
					cut_dag_vertices[qubit_vertices[start_pos_index]], q_vertices.first))
				{
                    tmp_fragment.m_prep_qubit.emplace(q_vertices.first);
                }

				for (; start_pos_index <= _cut_pos; ++start_pos_index)
				{
					const auto& _target_vertex_index = qubit_vertices[start_pos_index];
					tmp_fragment.m_vertice.emplace(_target_vertex_index);

					// check measure-qubit
					if (start_pos_index == _cut_pos){
						if (is_mea_qubit(src_dag_vertices[_target_vertex_index], cut_dag_vertices[_target_vertex_index], q_vertices.first)){
                            tmp_fragment.m_meas_qubit.emplace(q_vertices.first);
                        }
					}
				}

				append_sub_cir(tmp_fragment, qvm);
			}
		}
	}

	for (size_t i = 0; i < m_vertex_sub_cir_info.size(); ++i){
		if (m_vertex_sub_cir_info[i] < 0)
		{
			QCERR_AND_THROW(run_fail, "Error: unknow error on cut_circuit, vertex:" << i << " missing.");
		}
	}
}

bool CutQCircuit::exist_edge_on_target_qubit(const uint32_t& target_qubit, const std::vector<QProgDAGEdge>& edges)
{
	for (const auto& _e : edges)
	{
		if (target_qubit == _e.m_qubit){
			return true;
		}
	}

	return false;
}

bool CutQCircuit::is_pre_qubit(const QProgDAGVertex& src_dag_node, const QProgDAGVertex& cut_dag_node,
	const uint32_t& target_qubit)
{
	if (src_dag_node.m_pre_edges.size() != cut_dag_node.m_pre_edges.size())
	{
		if (src_dag_node.m_succ_edges.size() < cut_dag_node.m_succ_edges.size()) {
			QCERR_AND_THROW(run_fail, "Error: nuknow error on preparation-qubit.");
		}

		if (exist_edge_on_target_qubit(target_qubit, src_dag_node.m_pre_edges)
			&& (!exist_edge_on_target_qubit(target_qubit, cut_dag_node.m_pre_edges))) {
			return true;
		}
	}

	return false;
}

bool CutQCircuit::is_mea_qubit(const QProgDAGVertex& src_dag_node, const QProgDAGVertex& cut_dag_node,
	const uint32_t& target_qubit)
{
	if (src_dag_node.m_succ_edges.size() != cut_dag_node.m_succ_edges.size())
	{
		if (src_dag_node.m_succ_edges.size() < cut_dag_node.m_succ_edges.size()){
			QCERR_AND_THROW(run_fail, "Error: nuknow error on measure-qubit.");
		}

		if (exist_edge_on_target_qubit(target_qubit, src_dag_node.m_succ_edges)
			&& (!exist_edge_on_target_qubit(target_qubit, cut_dag_node.m_succ_edges))){
			return true;
		}
		
	}

	return false;
}

static long _check_same_item(const std::set<uint32_t>& set_1, const std::set<uint32_t>& set_2)
{
	for (const auto& _q_1 : set_1)
	{
		for (const auto& _q_2 : set_2)
		{
			if (_q_1 == _q_2) {
				return _q_1;
			}
		}
	}

	return -1;
}

void CutQCircuit::append_sub_cir(CutFragment& sub_cir, QuantumMachine *qvm)
{
	// find possible index of sub-circuit
	int sub_cir_index = -1;
	for (const auto& vertex_index : sub_cir.m_vertice)
	{
		if (m_vertex_sub_cir_info[vertex_index] > -1)
		{
			sub_cir_index = m_vertex_sub_cir_info[vertex_index];
			break;
		}
	}

	if (-1 == sub_cir_index)
	{
		m_cut_fragment_vec.emplace_back(sub_cir);
		sub_cir_index = m_cut_fragment_vec.size() - 1;
	}
	else
	{
		auto& target_sub_cir = m_cut_fragment_vec[sub_cir_index];
		while (true)
		{
			const auto _ret_q = _check_same_item(sub_cir.m_prep_qubit, target_sub_cir.m_meas_qubit);
			if (_ret_q == -1){
				break;
			}
			else{
				// add auxiliary bit
				const auto auxi_qubit = ++m_max_qubit_num;
				sub_cir.m_auxi_qubit_map.insert(std::make_pair(_ret_q, auxi_qubit));
				auto _itr = sub_cir.m_prep_qubit.find(_ret_q);
				sub_cir.m_prep_qubit.erase(_itr);
				sub_cir.m_prep_qubit.emplace(auxi_qubit);

				_itr = sub_cir.m_meas_qubit.find(_ret_q);
				sub_cir.m_meas_qubit.erase(_itr);
				sub_cir.m_meas_qubit.emplace(auxi_qubit);

				sub_cir.m_qubit.insert(auxi_qubit);
				for (auto& _v : sub_cir.m_vertice)
				{
					auto& all_vertice_vec = m_cut_prog_dag.get_vertex();
					for (auto& _q : all_vertice_vec[_v].m_node->m_qubits_vec)
					{
						if (_q->get_phy_addr() == _ret_q)
						{
							_q = qvm->allocateQubitThroughPhyAddress(auxi_qubit);
						}
					}
				}
			}
		}

		target_sub_cir.m_vertice.insert(sub_cir.m_vertice.begin(), sub_cir.m_vertice.end());
		target_sub_cir.m_qubit.insert(sub_cir.m_qubit.begin(), sub_cir.m_qubit.end());
        target_sub_cir.m_prep_qubit.insert(sub_cir.m_prep_qubit.begin(), sub_cir.m_prep_qubit.end());
        target_sub_cir.m_meas_qubit.insert(sub_cir.m_meas_qubit.begin(), sub_cir.m_meas_qubit.end());
		target_sub_cir.m_auxi_qubit_map.insert(sub_cir.m_auxi_qubit_map.begin(), sub_cir.m_auxi_qubit_map.end());
	}
	
	// update vertex-sub-cir info
	for (const auto& vertex_index : sub_cir.m_vertice) {
		m_vertex_sub_cir_info[vertex_index] = sub_cir_index;
	}
}

/** 该接口需要进一步优化：
* 第一次遍历，找入度为0的顶点，是个全遍历过程。
* 第二次遍历，从入度为0的节点的所有后项节点中查找，新产生的入度为0的顶点
* 后续遍历规则和第二次遍历规则相同。
* 直到顶点容器为空
*/
void CutQCircuit::generate_subcircuits(QuantumMachine *qvm)
{
	const auto& qv_map = m_src_prog_dag.m_qubits;
	std::vector<std::vector<uint32_t>> topo_seq;
	std::vector<std::vector<uint32_t>>::iterator cur_topo_seq_itr;
	const auto& all_vertice_vec = m_cut_prog_dag.get_vertex_c();
	
	for (auto& _sub_cir : m_cut_fragment_vec)
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
				while (cur_vertex_node.m_succ_edges.size() > 0) {
					const auto _e = *(cur_vertex_node.m_succ_edges.begin());
					m_cut_prog_dag.remove_edge(_e);
				}
			}

			topo_seq.emplace_back(_layer);
		}

		// build QCircuit by topo_seq
		std::map<uint32_t, Qubit*> vir_qubit_map = get_continue_qubit_map(_sub_cir, qvm);
		for (const auto& _layer : topo_seq)
		{
			for (const auto& _node_index : _layer)
			{
				auto _gate = remap_to_virtual_qubit(all_vertice_vec[_node_index].m_node, vir_qubit_map);
				_sub_cir.m_cir << _gate;
			}
		}

		m_vir_qubit_map_vec.emplace_back(vir_qubit_map);
		for (const auto& _q_map : vir_qubit_map)
		{
			auto _itr = _sub_cir.m_prep_qubit.find(_q_map.first);
			if (_sub_cir.m_prep_qubit.end() != _itr)
			{
				_sub_cir.m_prep_qubit.erase(_itr);
				_sub_cir.m_prep_qubit.emplace(_q_map.second->get_phy_addr());
			}

			_itr = _sub_cir.m_meas_qubit.find(_q_map.first);
			if (_sub_cir.m_meas_qubit.end() != _itr)
			{
				_sub_cir.m_meas_qubit.erase(_itr);
				_sub_cir.m_meas_qubit.emplace(_q_map.second->get_phy_addr());
			}
		}
	}
}

std::map<uint32_t, Qubit*> 
CutQCircuit::get_continue_qubit_map(const CutFragment& frag, QuantumMachine *qvm)
{
	std::vector<uint32_t> _qv;
	for (const auto& _q : frag.m_qubit) {
		_qv.emplace_back(_q);
	}
	sort(_qv.begin(), _qv.end(), [](const auto& a, const auto& b) { return a < b; });

	const auto qv_size = _qv.size();
	for (const auto& _auxi_q : frag.m_auxi_qubit_map)
	{
		for (auto _itr = _qv.begin(); _itr != _qv.end(); ++_itr)
		{
			if (_auxi_q.first == *_itr) {
				_qv.insert(_itr + 1, _auxi_q.second);
				break;
			}
		}
	}

	while (_qv.size() > qv_size)
	{
		_qv.pop_back();
	}

	std::map<uint32_t, Qubit*> vir_qubit_map;
	for (size_t _i = 0; _i < _qv.size(); ++_i) {
		vir_qubit_map.insert(std::make_pair(_qv[_i], qvm->allocateQubitThroughPhyAddress(_i)));
	}

	return vir_qubit_map;
}

/* 根据frag中的qubit信息，将gate_node 映射成连续qubit
*/
QGate CutQCircuit::remap_to_virtual_qubit(std::shared_ptr<QProgDAGNode> gate_node, const std::map<uint32_t, Qubit*>& vir_qubit_map)
{
	QVec new_qv;
	for (const auto& _q : gate_node->m_qubits_vec){
		new_qv.emplace_back(vir_qubit_map.at(_q->get_phy_addr()));
	}

	QGate ret_gate(std::dynamic_pointer_cast<AbstractQGateNode>(*(gate_node->m_itr)));
	ret_gate = deepCopy(ret_gate);
	ret_gate.remap(new_qv);

	return ret_gate;
}

int CutQCircuit::find_vertex_in_sub_cir(const uint32_t& vertex)
{
	for (size_t i = 0; i < m_cut_fragment_vec.size(); ++i)
	{
		const CutFragment& _sub_cir = m_cut_fragment_vec[i];
		if (_sub_cir.m_vertice.end() != _sub_cir.m_vertice.find(vertex)) {
			return i;
		}
	}

	return -1;
}

uint32_t CutQCircuit::get_target_qubit(const uint32_t& vertex_index, const uint32_t& q)
{
	const auto& all_vertice_vec = m_cut_prog_dag.get_vertex_c();
	const auto& _vertex_gate_node = all_vertice_vec[vertex_index].m_node;
	QVec _used_qv;
	std::dynamic_pointer_cast<AbstractQGateNode>(*(_vertex_gate_node->m_itr))->getQuBitVector(_used_qv);
	uint32_t _target_q = UINT32_MAX;
	for (auto i = 0; i < _used_qv.size(); ++i)
	{
		if (_used_qv[i]->get_phy_addr() == q) {
			_target_q = _vertex_gate_node->m_qubits_vec[i]->get_phy_addr();
		}
	}

	if (UINT32_MAX == _target_q) {
		QCERR_AND_THROW(run_fail, "Error: qubit error.");
	}

	return _target_q;
}

std::vector<StitchesInfo> CutQCircuit::get_stitches(const std::map<uint32_t, std::vector<uint32_t>>& cut_pos)
{
	const auto& qubit_vertices_map = m_cut_prog_dag.get_qubit_vertices_map();
	std::vector<StitchesInfo> all_stitches;
	const auto& all_vertice_vec = m_cut_prog_dag.get_vertex_c();

	auto _sort_fun = [](Qubit* q_0, Qubit* q_1) {return q_0->get_phy_addr() < q_1->get_phy_addr(); };
	for (const auto& _cut : cut_pos)
	{
		const auto& qubit_vertices = qubit_vertices_map.at(_cut.first);
		for (const auto& _cut_index : _cut.second)
		{
			all_stitches.emplace_back(StitchesInfo());
			//get measure qubit
			uint32_t _target_q = get_target_qubit(qubit_vertices[_cut_index], _cut.first);
			const auto& _meas_sub_cir_index = m_vertex_sub_cir_info[qubit_vertices[_cut_index]];
			if (_meas_sub_cir_index == -1){
				QCERR_AND_THROW(run_fail, "Error: error sub_cir_index.");
			}
			all_stitches.back().m_meas_qubit.first = _meas_sub_cir_index;
			all_stitches.back().m_meas_qubit.second = m_vir_qubit_map_vec[_meas_sub_cir_index].at(_target_q)->get_phy_addr();

			//get pre qubit
			_target_q = get_target_qubit(qubit_vertices[_cut_index + 1], _cut.first);

			const auto& _pre_sub_cir_vertex = m_vertex_sub_cir_info[qubit_vertices[_cut_index + 1]];
			if (_pre_sub_cir_vertex == -1) {
				QCERR_AND_THROW(run_fail, "Error: error sub_cir_index.");
			}

            all_stitches.back().m_prep_qubit.first = _pre_sub_cir_vertex;
			all_stitches.back().m_prep_qubit.second = m_vir_qubit_map_vec[_pre_sub_cir_vertex].at(_target_q)->get_phy_addr();
		}
	}

	return all_stitches;
}

const std::vector<SubCircuit>& CutQCircuit::get_cutted_sub_circuits(std::vector<uint32_t>& qubit_permutation)
{ 
	m_sub_cirs.resize(m_cut_fragment_vec.size());
	
	for (size_t i = 0; i < m_cut_fragment_vec.size(); ++i)
	{
		m_sub_cirs[i].m_cir = m_cut_fragment_vec[i].m_cir;

		QVec qv;
		get_all_used_qubits(m_sub_cirs[i].m_cir, qv);
		for (const auto& _q : qv)
		{
            if (m_cut_fragment_vec[i].m_prep_qubit.find(_q->get_phy_addr())
                != m_cut_fragment_vec[i].m_prep_qubit.end()) {
				m_sub_cirs[i].m_prep_qubit.emplace_back(_q);
            }

            if (m_cut_fragment_vec[i].m_meas_qubit.find(_q->get_phy_addr())
                != m_cut_fragment_vec[i].m_meas_qubit.end()) {
				m_sub_cirs[i].m_meas_qubit.emplace_back(_q);
			}
		}
	}

	qubit_permutation = get_qubit_permutation();
	return m_sub_cirs;
}

const std::vector<uint32_t>& CutQCircuit::get_qubit_permutation()
{
	if (m_sub_cirs.size() != m_cut_fragment_vec.size()){
		QCERR_AND_THROW(run_fail, "Error: Failed to get_qubit_permutation, sub_cir_qubit_seq is empty.");
	}

	std::vector<uint32_t> tmp_qubit_vec;
	for (size_t i = 0; i < m_cut_fragment_vec.size(); ++i)
	{
		tmp_qubit_vec.resize(m_vir_qubit_map_vec[i].size());

		std::map<uint32_t, uint32_t> reverse_auxi_qubit_map;
		for (const auto& _auxi_q : m_cut_fragment_vec[i].m_auxi_qubit_map){
			reverse_auxi_qubit_map.insert(std::make_pair(_auxi_q.second, _auxi_q.first));
		}

		for (const auto& _vir_qubit : m_vir_qubit_map_vec[i])
		{
			const auto _itr = reverse_auxi_qubit_map.find(_vir_qubit.first);
			if (_itr != reverse_auxi_qubit_map.end())
			{
				tmp_qubit_vec[_vir_qubit.second->get_phy_addr()] = _itr->second;
			}
			else
			{
				tmp_qubit_vec[_vir_qubit.second->get_phy_addr()] = _vir_qubit.first;
			}
		}

		const auto _sub_cir_measure_qubit = m_sub_cirs[i].m_meas_qubit;
		for (size_t j = 0; j < _sub_cir_measure_qubit.size(); ++j){
			tmp_qubit_vec[_sub_cir_measure_qubit[j]->get_phy_addr()] = UINT_MAX;
		}

		/* reverse traversal */
		for ( ; tmp_qubit_vec.size() > 0; ){
			if (UINT_MAX > tmp_qubit_vec.back()){
				m_qubit_permutation.emplace_back(tmp_qubit_vec.back());
			}
	
			tmp_qubit_vec.pop_back();
		}
	}

	return m_qubit_permutation;
}

/*******************************************************************
*                      class RecombineFragment
********************************************************************/
static const std::map<PrepState, QStat> prep_state_matrices_map =
{
	{ PrepState::S0, {1., 0., 0., 0.}},

	{ PrepState::S1, {1. / 3., std::sqrt(2.) / 3., std::sqrt(2.) / 3., 2. / 3.}},

	{ PrepState::S2, {1.0 / 3.0, std::exp(qcomplex_t(0, PI * (2. / 3.))) * std::sqrt(2.) * 1. / 3.,
			std::exp(qcomplex_t(0, -PI * (2. / 3.))) * std::sqrt(2.) * 1. / 3., 2. / 3.}},

	{ PrepState::S3, {1.0 / 3.0, std::exp(qcomplex_t(0, -PI * (2. / 3.))) * std::sqrt(2.) * 1. / 3.,
			std::exp(qcomplex_t(0, PI * (2. / 3.))) * std::sqrt(2.) * 1. / 3., 2. / 3.}}
};

static const std::map<MeasState, QStat> meas_state_matrices_map =
{
	{ MeasState::Xp, {.5, .5, .5, .5}},
	{ MeasState::Xm, {.5, -.5, -.5, .5}},

	{ MeasState::Yp, {.5, qcomplex_t(0 ,-.5), qcomplex_t(0 ,.5), .5}},
	{ MeasState::Ym, {.5, qcomplex_t(0 ,.5), qcomplex_t(0 ,-.5), .5}},

	{ MeasState::Zp, {1., 0., 0., 0.}},
	{ MeasState::Zm, {0., 0., 0.,1.}}
};

prob_vec RecombineFragment::state_to_probs(QStat & state)
{
	prob_vec probs;
	for (auto val : state)
	{
		probs.emplace_back(std::norm(val));
	}

	return probs;
}

QMatrixXcd RecombineFragment::target_labels_to_matrix(const ResultData& data)
{
	const auto& prep_labels = data.m_prep_labels;
	const auto& meas_labels = data.m_meas_labels;

	QMatrixXcd prep_matrix = QMatrixXcd::Identity(1, 1);
	QMatrixXcd meas_matrix = QMatrixXcd::Identity(1, 1);

	for (auto label : prep_labels)
	{
		QStat label_matrix = prep_state_matrices_map.find(label)->second;
		//PTrace("prep_labels-label_matrix" << label_matrix);

		prep_matrix = Eigen::kroneckerProduct(QMatrixXcd::Map(&label_matrix[0], 2, 2), prep_matrix).eval();
		//PTrace("prep_labels-prep_matrix kroneckerProduct\n" << prep_matrix);
	}

	for (auto label : meas_labels)
	{
		QStat label_matrix = meas_state_matrices_map.find(label)->second;
		//PTrace("meas_labels-label_matrix" << label_matrix);
		meas_matrix = Eigen::kroneckerProduct(QMatrixXcd::Map(&label_matrix[0], 2, 2), meas_matrix).eval();
		//PTrace("meas_matrix-meas_matrix kroneckerProduct\n" << meas_matrix);
	}

	return Eigen::kroneckerProduct(prep_matrix.transpose(), meas_matrix);
}

void RecombineFragment::tomography_meas_circuit(QCircuit& circuit, MeasBasis basis, Qubit* qubit)
{
	switch (basis)
	{
	case MeasBasis::BASIS_Z:
	{
		auto i_gate_node = std::dynamic_pointer_cast<QNode>(I(qubit).getImplementationPtr());
		circuit.insertQNode(circuit.getLastNodeIter()--, i_gate_node);
		break;
	}

	case MeasBasis::BASIS_X:
	{
		auto h_gate_node = std::dynamic_pointer_cast<QNode>(H(qubit).getImplementationPtr());
		circuit.insertQNode(circuit.getLastNodeIter()--, h_gate_node);
		break;
	}

	case MeasBasis::BASIS_Y:
	{
		auto s_gate_node = std::dynamic_pointer_cast<QNode>(S(qubit).dagger().getImplementationPtr());
		auto h_gate_node = std::dynamic_pointer_cast<QNode>(H(qubit).getImplementationPtr());

		circuit.insertQNode(circuit.getLastNodeIter()--, s_gate_node);
		circuit.insertQNode(circuit.getLastNodeIter()--, h_gate_node);
		break;
	}

	default:
		QCERR_AND_THROW(run_fail, "MeasBasis Error");
		break;
	}
}

void RecombineFragment::tomography_prep_circuit(QCircuit& circuit, PrepState state, Qubit* qubit)
{
	switch (state)
	{
	case PrepState::S0:
	{
		auto i_gate_node = std::dynamic_pointer_cast<QNode>(I(qubit).getImplementationPtr());
		circuit.insertQNode(circuit.getHeadNodeIter(), i_gate_node);
		break;
	}

	case PrepState::S1:
	{
		auto u3_gate_node = std::dynamic_pointer_cast<QNode>(U3(qubit, -1.9106332, PI, 0.).getImplementationPtr());
		circuit.insertQNode(circuit.getHeadNodeIter(), u3_gate_node);
		break;
	}

	case PrepState::S2:
	{
		auto u3_gate_node = std::dynamic_pointer_cast<QNode>(U3(qubit, -1.9106332, PI / 3, 0.).getImplementationPtr());
		circuit.insertQNode(circuit.getHeadNodeIter(), u3_gate_node);
		break;
	}

	case PrepState::S3:
	{
		auto u3_gate_node = std::dynamic_pointer_cast<QNode>(U3(qubit, -1.9106332, -PI / 3, 0.).getImplementationPtr());
		circuit.insertQNode(circuit.getHeadNodeIter(), u3_gate_node);
		break;
	}

	default:
		QCERR_AND_THROW(run_fail, "PrepState Error");
		break;
	}
}

void RecombineFragment::partial_tomography(QCirFragments& fragment, std::string backend /*= "CPU"*/)
{
	/** 初态比特数量与测量比特数量
	*/
	auto prep_qubits_num = fragment.m_prep_qubits.size();
	auto meas_qubits_num = fragment.m_meas_qubits.size();

	auto prep_count = (size_t)std::pow(4, prep_qubits_num);
	auto meas_count = (size_t)std::pow(3, meas_qubits_num);

	std::vector<MeasBasis> meas_basis = { MeasBasis::BASIS_Z, MeasBasis::BASIS_X, MeasBasis::BASIS_Y };
	std::vector<PrepState> prep_state = { PrepState::S0, PrepState::S1, PrepState::S2, PrepState::S3 };

	/** 一个子图对应的总线路数量3^M * 4^P
	*/
	size_t circuits_num = prep_count * meas_count;

    /** 拷贝初始线路，对应总线路数量
	*/
    std::vector<QCircuit> actual_circuits;
    for (auto i = 0; i < circuits_num; ++i)
    {
        actual_circuits.emplace_back(deepCopy(fragment.m_cir));
    }

	/** 这个大循环的作用是：
	* 1.对每个子线路施加测量和初态
	* 2.对每个子线路的每个比特的测量和初态信息添加标记
	*/
	for (auto i = 0; i < circuits_num; ++i)
	{
		auto prep_coeff = i / meas_count;
		auto meas_coeff = i % meas_count;

		for (int p = prep_qubits_num - 1; p >= 0; --p)
		{
			auto prep_index = (size_t)std::pow(4, p);
			auto prep_label = (prep_coeff / prep_index) % 4;

			fragment.m_fragment_results[i].set_prep_label(prep_qubits_num - 1 - p, static_cast<PrepState>(prep_label));

			tomography_prep_circuit(actual_circuits[i], prep_state[prep_label], fragment.m_prep_qubits[prep_qubits_num - 1 - p]);
		}

		for (int m = meas_qubits_num - 1; m >= 0; --m)
		{
			auto meas_index = (size_t)std::pow(3, m);
			auto meas_label = (meas_coeff / meas_index) % 3;

			fragment.m_fragment_results[i].set_meas_label(meas_qubits_num - 1 - m, static_cast<MeasBasis>(meas_label));

			tomography_meas_circuit(actual_circuits[i], meas_basis[meas_label], fragment.m_meas_qubits[meas_qubits_num - 1 - m]);
		}
	}

	/** 运行所有子线路，存储结果
	*/
	for (auto i = 0; i < circuits_num; ++i){
#if PRINT_TRACE
		//cout << "actual_circuits:" << i << actual_circuits[i] << endl;
#endif

        auto partial_result = run_circuits(actual_circuits[i], "CPU");
		fragment.m_fragment_results[i].m_result = partial_result;
	}
}

std::string convert_result_to_key(const ResultData& data)
{
    std::string key;

    for (auto prep_state : data.m_prep_labels)
    {
        key.append(to_string((int)(prep_state)));
    }

    key.append("|");

    for (auto meas_state : data.m_meas_labels)
    {
        key.append(to_string((int)(meas_state)));
    }

    return key;
}

ResultData convert_key_to_result(const string& str)
{
    auto iter = str.find("|");

    string prep_string = str.substr(0, iter);
    string meas_string = str.substr(iter + 1);

    std::vector<PrepState> prep_labels;
    for (auto val : prep_string)
    {
        auto prep_state = static_cast<PrepState>(val - '0');

        prep_labels.emplace_back(prep_state);
    }

    std::vector<MeasState> meas_labels;
    for (auto val : meas_string)
    {
        auto meas_state = static_cast<MeasState>(val - '0');

        meas_labels.emplace_back(meas_state);
    }

    return ResultData(prep_labels, meas_labels);
}

void RecombineFragment::organize_tomography(const QCirFragments& fragment, ResultDataMap& organize_data)
{
	/** 获取所有的测量比特和初态制备比特信息
	*/
	auto prep_qubits = get_qubits_addr(fragment.m_prep_qubits);
	auto meas_qubits = get_qubits_addr(fragment.m_meas_qubits);

	auto prep_qubit_num = prep_qubits.size();
	auto meas_qubit_num = meas_qubits.size();

    auto full_prep_num = std::pow(4, prep_qubit_num);
    auto full_meas_num = std::pow(6, meas_qubit_num);

	using TmpResultDataMap_ = std::map<std::string, std::map<std::string, size_t>>;
	TmpResultDataMap_ _tmp_organize_data; //临时数据结构

	/** 这个大循环的作用是：
	* 1.对每个子线路施加测量和初态
	* 2.对每个子线路的每个比特的测量和初态信息添加标记
	*/
	for (const auto& fragment_result : fragment.m_fragment_results)
	{
		/** 获取实际运行的子线路的每个比特的实际初态制备和测量信息
		*/
		auto full_prep_labels = fragment_result.m_prep_state;
		auto full_meas_labels = fragment_result.m_meas_basis;

		std::vector<PrepState> prep_labels;
		for (auto idx = 0; idx < prep_qubits.size(); ++idx)
		{
			prep_labels.emplace_back(full_prep_labels[idx]);
		}

		std::vector<MeasBasis> meas_labels;
		for (auto idx = 0; idx < meas_qubits.size(); ++idx)
		{
			meas_labels.emplace_back(full_meas_labels[idx]);
		}

		/** 这个大循环的作用是：
		* 1.遍历当前子线路的运行结果，筛选 mid_bits 和 fin_bits
		* 2.根据当前这一条结果的量子态（'00','01'...）确定测量态（'Zm','Zx'...）
		* 3.将初态制备与测量态作为 key 值，当前运行结果的次数作为 value ，插入数据结构
		*/
		std::string fin_bits;
		for (auto val : fragment_result.m_result)
		{
			auto bits = val.first;
			auto cnts = val.second;

			std::string mid_bits;
			for (auto qubit : meas_qubits)
			{
				mid_bits.push_back(bits[bits.size() - qubit - 1]);
			}

			fin_bits.clear();
			for (auto i = 0; i < bits.size(); ++i)
			{
				auto iter = std::find(meas_qubits.begin(), meas_qubits.end(), bits.size() - i - 1);
				QPANDA_OP(iter == meas_qubits.end(), fin_bits.push_back(bits[i]));
			}

			std::vector<MeasState> meas_states;
			for (auto i = 0; i < meas_qubits.size(); ++i)
			{
                auto meas_state = static_cast<MeasState>((size_t)meas_labels[i] * 2 + (size_t)(mid_bits[i] != '0'));
                meas_states.emplace_back(meas_state);
			}

			ResultData tmp(prep_labels, meas_states);
            auto result_key = convert_result_to_key(tmp);
            //organize_data[fin_bits][result_key] = cnts;
			_tmp_organize_data[fin_bits][result_key] = cnts;
		}
	}

	/** 补全多余的 key 值，value取0
	*/
    std::vector<PrepState> origin_prep_labels(prep_qubit_num, PrepState::S0);
    std::vector<MeasState> origin_meas_labels(meas_qubit_num, MeasState::Zm);

    std::vector<std::vector<PrepState>> full_prep_states(full_prep_num, origin_prep_labels);
    std::vector<std::vector<MeasState>> full_meas_states(full_meas_num, origin_meas_labels);

	for (auto i = 0; i < full_prep_num; ++i)
	{
		for (auto j = 0; j < prep_qubit_num; ++j)
		{
			auto coeff = (size_t)std::pow(4, j);
			full_prep_states[i][j] = static_cast<PrepState>((i / coeff) % 4);
		}
	}

	for (auto i = 0; i < full_meas_num; ++i)
	{
		for (auto j = 0; j < meas_qubit_num; ++j)
		{
			auto coeff = (size_t)std::pow(6, j);
			full_meas_states[i][j] = static_cast<MeasState>((i / coeff) % 6);
		}
	}

	std::vector<ResultData> full_count_labels;
	for (auto i = 0; i < full_prep_num; ++i)
	{
		for (auto j = 0; j < full_meas_num; ++j)
		{
			ResultData count_label(full_prep_states[i], full_meas_states[j]);
			full_count_labels.emplace_back(count_label);
		}
	}

	for (const auto& val : _tmp_organize_data)
	{
		QPANDA_OP(val.second.size() == full_count_labels.size(), continue);

		auto bits = val.first;
		auto bits_iter = _tmp_organize_data.find(bits);
		QPANDA_ASSERT(bits_iter == _tmp_organize_data.end(), "_tmp_organize_data.find(bits) ERROR");

		for (const auto& count_label : full_count_labels)
		{
            string count_key = convert_result_to_key(count_label);

            const auto& count_data = bits_iter->second;
            const auto& label_iter = count_data.find(count_key);

            QPANDA_OP(label_iter == count_data.end(), _tmp_organize_data[bits][count_key] = 0);
		}
	}

	for (const auto& _data : _tmp_organize_data) {
		std::vector<std::pair<std::string, size_t>> _tmp_vec;
		for (const auto& _val : _data.second)
		{
			if (0 == _val.second){
				_tmp_vec.emplace_back(std::make_pair(_val.first, _val.second));
			}
			else{
				organize_data[_data.first].emplace_back(std::make_pair(_val.first, _val.second));
			}
		}

		organize_data[_data.first].insert(organize_data[_data.first].end(), _tmp_vec.begin(), _tmp_vec.end());
	}

	return;
}

void RecombineFragment::direct_fragment_model(const std::vector<ResultDataMap>& organize_data_vec, std::vector<ChoiMatrices>& choi_states_vec)
{
	choi_states_vec.resize(organize_data_vec.size());
	uint32_t i = 0;
	for (const auto& _data: organize_data_vec)
	{
		get_choi_matrix(_data, choi_states_vec[i]);
		++i;
	}

    return;
}

void RecombineFragment::get_choi_matrix(const ResultDataMap& organize_data, ChoiMatrices& choi_states)
{
	for (const auto& data : organize_data)
	{
		auto final_bits = data.first;
		auto result_datas = data.second;

        std::vector<double> counts;

		std::vector<ResultData> prep_meas_states;

		for (const auto& val : result_datas)
		{
			counts.emplace_back(val.second);

            auto result_key = convert_key_to_result(val.first);

            prep_meas_states.emplace_back(result_key);
		}

		auto prep_qubit_num = prep_meas_states.front().m_prep_labels.size();
		auto meas_qubit_num = prep_meas_states.front().m_meas_labels.size();

		auto cut_qubit_num = prep_qubit_num + meas_qubit_num;
		auto choi_trace = std::accumulate(counts.begin(), counts.end(), 0) / ((1ull << prep_qubit_num) * std::pow(3, meas_qubit_num));
		counts.emplace_back(choi_trace);

        QMatrixXcd state_matrix = QMatrixXcd::Zero(result_datas.size() + 1, pow(2, 2*cut_qubit_num));

		for (auto i = 0; i < prep_meas_states.size(); ++i)
		{
			auto label_matrix = target_labels_to_matrix(prep_meas_states[i]);
			//PTrace("label_matrix \n" << label_matrix);

			state_matrix.row(i) = QVectorXcd::Map(label_matrix.data(), label_matrix.size());
		}

		QMatrixXcd identity_matrix = QMatrixXcd::Identity(1ull << cut_qubit_num, 1ull << cut_qubit_num);
		state_matrix.row(result_datas.size()) = QVectorXcd::Map(identity_matrix.data(), identity_matrix.size());


		QVectorXcd state_counts = QVectorXd::Map(counts.data(), counts.size()).cast<qcomplex_t>();
        /*PTrace("state_matrix \n" << state_matrix);
        PTrace("state_counts \n" << state_counts);*/

        PTrace("state_matrix rows" << state_matrix.rows());
        PTrace("state_matrix cols" << state_matrix.cols());

        PTrace("state_counts rows" << state_counts.rows());
        PTrace("state_counts cols" << state_counts.cols());
  

		//PTrace("On state_matrix.conjugate().bdcSvd.....");
		//QMatrixXcd choi_fit = state_matrix.conjugate().bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(state_counts);
        //QMatrixXcd choi_fit = state_matrix.conjugate().colPivHouseholderQr().solve(state_counts); /**< method 2 */
        //QMatrixXcd choi_fit = state_matrix.conjugate().householderQr().solve(state_counts); /**< method 3 */
        //QMatrixXcd choi_fit = state_matrix.conjugate().jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(state_counts); /**< method 4 */
		
		/* 速度较快的线性求解方法求解方法
		*/
		LeastSquaresConjugateGradient <QMatrixXcd> solver;
		//ConjugateGradient  <QMatrixXcd> solver;
		solver.compute(state_matrix.conjugate());
		QMatrixXcd choi_fit = solver.solve(state_counts);
 
		//PTrace("choi_fit" << choi_fit);
        //PTrace("choi_fit");

        Eigen::Map<QMatrixXcd> reshape_matrix(choi_fit.data(), 1ull << cut_qubit_num, 1ull << cut_qubit_num);
		//PTrace("final_bits -> " << final_bits);
		//PTrace("reshape_matrix -> " << reshape_matrix);

		choi_states[final_bits] = reshape_matrix;
	}

	return;
}

std::map<string, size_t> RecombineFragment::run_circuits(const QCircuit& circuit, std::string backend)
{
	QVec qv;
	auto qubit_num = circuit.get_used_qubits(qv);

	CPUQVM machine;
	machine.init();

	auto qlist = machine.qAllocMany(qubit_num);
	auto clist = machine.cAllocMany(qubit_num);

	QProg prog;
	prog << circuit << MeasureAll(qlist, clist);
	auto result = machine.runWithConfiguration(prog, clist, m_shots);

	machine.finalize();
	return result;
}

std::vector<RecombineFragment::ResultDataMap> RecombineFragment::collect_fragment_data()
{
	std::vector<ResultDataMap> frag_data;
	
	CutQCircuits fragments;
	uint32_t actual_cirs = 0;
	for (const auto& _sub_cir_val : m_sub_cir_vec) {
		fragments.emplace_back(QCirFragments(_sub_cir_val));
		actual_cirs += fragments.back().m_fragment_results.size();
	}

	m_shots = g_shot / actual_cirs;
	for (auto& fragment : fragments) {
		partial_tomography(fragment);
	}

	for (auto& fragment : fragments) {
		ResultDataMap tmp;
		organize_tomography(fragment, tmp);
		frag_data.emplace_back(tmp);
	}

	return frag_data;
}

void RecombineFragment::build_frag_labels(const std::string& base_label, 
	const std::vector<StitchesInfo>& stitches, std::vector<FragLabel>& frag_labels)
{
	std::map<uint32_t, std::map<uint32_t, string>> sub_circuit_pre_base_label; //子图：<qubit：标签>
	std::map<uint32_t, std::map<uint32_t, string>> sub_circuit_meas_base_label;

	auto get_direct_labels = [&base_label](std::map<uint32_t, std::map<uint32_t, string>>& sub_circuit_direct_label
		, const StitchesInfo::sub_cir_op_qubit_index& _sti_qubits, const string& base_label_string) {
		auto _itr = sub_circuit_direct_label.find(_sti_qubits.first);
		if (sub_circuit_direct_label.end() == _itr)
		{
			std::map<uint32_t, string> _qubit_label;
			_qubit_label.insert(std::make_pair(_sti_qubits.second, base_label_string));
			sub_circuit_direct_label.insert(std::make_pair(_sti_qubits.first, _qubit_label));
		}
		else{
			_itr->second.insert(std::make_pair(_sti_qubits.second, base_label_string));
		}
	};

	for (uint32_t _i = 0; _i < stitches.size(); ++_i) {
		const auto& _sti = stitches[_i];
		const string base_label_string = base_label.substr(_i, 1);
        get_direct_labels(sub_circuit_pre_base_label, _sti.m_prep_qubit, base_label_string);
        get_direct_labels(sub_circuit_meas_base_label, _sti.m_meas_qubit, base_label_string);
	}

	for (uint32_t _i = 0; _i < frag_labels.size(); ++_i)
	{
		auto& _frag_label = frag_labels[_i];
		if (sub_circuit_pre_base_label.end() != sub_circuit_pre_base_label.find(_i))
		{
			const std::map<uint32_t, string>& _pre_direct_label = sub_circuit_pre_base_label.at(_i);
			for (const auto _qubit_label : _pre_direct_label) {
				_frag_label.m_prep_label += _qubit_label.second;
			}
		}

		if (sub_circuit_meas_base_label.end() != sub_circuit_meas_base_label.find(_i))
		{
			const std::map<uint32_t, string>& _mea_base_label = sub_circuit_meas_base_label.at(_i);
			for (const auto _qubit_label : _mea_base_label) {
				_frag_label.m_meas_label += _qubit_label.second;
			}
		}
    }

	return ;
}

std::vector<QMatrixXcd> RecombineFragment::target_labels_to_matrix(const std::vector<FragLabel>& labels)
{
	auto string_to_matrix_fun = [&](const std::string& _str) ->QMatrixXcd {
		if (_str.size() == 0) {
			return QMatrixXcd::Identity(1, 1);
		}

		if (_str.compare("I") == 0){
			return QMatrixXcd::Identity(2, 2);
		}
		
		switch (_str.c_str()[0])
		{
		case 'X':
		{
			QStat label_matrix_xp = meas_state_matrices_map.find(MeasState::Xp)->second;
			QStat label_matrix_xm = meas_state_matrices_map.find(MeasState::Xm)->second;
			QStat label_matrix_x = label_matrix_xp - label_matrix_xm;

			return QMatrixXcd::Map(&label_matrix_x[0], 2, 2);
		}
		break;

		case 'Y':
		{
			QStat label_matrix_yp = meas_state_matrices_map.find(MeasState::Yp)->second;
			QStat label_matrix_ym = meas_state_matrices_map.find(MeasState::Ym)->second;
			QStat label_matrix_y = label_matrix_yp - label_matrix_ym;
			return QMatrixXcd::Map(&label_matrix_y[0], 2, 2);
		}
		break;

		case 'Z':
		{
			QStat label_matrix_zp = meas_state_matrices_map.find(MeasState::Zp)->second;
			QStat label_matrix_zm = meas_state_matrices_map.find(MeasState::Zm)->second;
			QStat label_matrix_z = label_matrix_zp - label_matrix_zm;
			return QMatrixXcd::Map(&label_matrix_z[0], 2, 2);
		}
		break;

		default:
			break;
		}

		QCERR_AND_THROW(run_fail, "Error: unknow label_string on target_labels_to_matrix.");
	};

	auto label_to_matrix_fun = [&](const std::string& _tmp_label) ->QMatrixXcd {
		QMatrixXcd _label_matrix = QMatrixXcd::Identity(1, 1);
		for (size_t _i = 0; _i < _tmp_label.length(); ++_i)
		{
			const std::string _str = _tmp_label.substr(_i, 1);
			_label_matrix = Eigen::kroneckerProduct(string_to_matrix_fun(_str), _label_matrix).eval();
			//PTrace("_label_matrix:\n" << _label_matrix);
		}

		//PTrace("-------------ret _label_matrix:\n" << _label_matrix);
		return _label_matrix;
	};

	std::vector<QMatrixXcd> frag_mats;
	for (const auto& _label : labels)
	{
        const auto prep_matrix = label_to_matrix_fun(_label.m_prep_label);
        const auto meas_matrix = label_to_matrix_fun(_label.m_meas_label);
		frag_mats.emplace_back(Eigen::kroneckerProduct(prep_matrix.transpose(), meas_matrix).eval());
	}

	return frag_mats;
}

std::vector<RecombineFragment::FinalQubitJoinedStr> 
RecombineFragment::get_final_qubit_combination(const std::vector<ChoiMatrices>& choi_states_vec)
{
	const auto sub_circuit_cnt = choi_states_vec.size();

	// get final_qubits_str
	std::vector<std::vector<string>> final_qubits_str(sub_circuit_cnt);
	size_t _sub_cir_index = 0;
	for (const auto& _sub_cir_choi : choi_states_vec)
	{
		for (const auto& _choi_state : _sub_cir_choi)
		{
			final_qubits_str[_sub_cir_index].emplace_back(_choi_state.first);
		}

		++_sub_cir_index;
	}

	std::vector<RecombineFragment::FinalQubitJoinedStr> final_qubit_combination;
	std::vector<RecombineFragment::FinalQubitJoinedStr> _tmp_combination;
	for (size_t i = 0; i < final_qubits_str[0].size(); ++i){
		_tmp_combination.emplace_back(std::vector<string>({ final_qubits_str[0][i] }));
	}

	for (size_t i = 1; i < final_qubits_str.size(); ++i)
	{
		for (size_t j = 0; j < _tmp_combination.size(); ++j)
		{
			for (size_t h = 0; h < final_qubits_str[i].size(); ++h)
			{
				// positive sequence
				final_qubit_combination.emplace_back(std::vector<string>({ _tmp_combination[j] }));
				final_qubit_combination.back().emplace_back(final_qubits_str[i][h]);

				//inverted order
				/*final_qubit_combination.emplace_back(std::vector<string>({ final_qubits_str[i][h] }));
				auto& _target_vec = final_qubit_combination.back();
				_target_vec.insert(_target_vec.end(), _tmp_combination[j].begin(), _tmp_combination[j].end());*/
			}
		}

		_tmp_combination.swap(final_qubit_combination);
		final_qubit_combination.clear();
	}

	final_qubit_combination.swap(_tmp_combination);
	return final_qubit_combination;
}

std::string RecombineFragment::correct_qubit_order(const std::string& src_qubit_str, 
	const std::vector<uint32_t>& qubit_permutation)
{
	if (qubit_permutation.size() != src_qubit_str.size()){
		QCERR_AND_THROW(run_fail, "Error: Failed to correct_qubit_order, the size of qubit_permutation is error");
	}

	string _ret_str = src_qubit_str;
	for (auto i = 0; i < qubit_permutation.size(); ++i) {
		_ret_str.replace(src_qubit_str.size() - 1 - qubit_permutation[i], 1, src_qubit_str.substr(i, 1));
	}

	return _ret_str;
}

std::map<std::string, double> RecombineFragment::recombine_using_insertions(const std::vector<ChoiMatrices>& choi_states_vec,
	const std::vector<StitchesInfo>& stitches, const std::vector<uint32_t>& qubit_permutation)
{
	/** 切割点数目
	*/
	const auto stitches_cnt = stitches.size();
	
	/** 根据入参repeat，计算meas_base的全排列信息
	*/
	const string meas_base("IZXY");
	auto itr_product = [&meas_base](const uint32_t repeat) ->std::vector<string> {
		const auto base_cnt = meas_base.length();
		std::vector<std::string> combine_lables;
		combine_lables.reserve(pow(base_cnt, repeat));
		for (size_t _b = 0; _b < base_cnt; ++_b){
			combine_lables.emplace_back(meas_base.substr(_b, 1));
		}
		
		uint32_t combine_index = 0;
		for (size_t i = repeat - 1; 0 < i ; --i)
		{
			const auto last_combine_lables = combine_lables;
			for (size_t _b_1 = 0; _b_1 < base_cnt; ++_b_1)
			{
				const auto _combine_lables_cnt = last_combine_lables.size();
				for (size_t _b_2 = 0; _b_2 < _combine_lables_cnt; ++_b_2)
				{
					const auto _cut_index = (_combine_lables_cnt*_b_1) + _b_2;
					if (_cut_index < _combine_lables_cnt){
						combine_lables[_cut_index] = meas_base[_b_1] + last_combine_lables[_b_2];
					}
					else{
						combine_lables.emplace_back(meas_base[_b_1] + last_combine_lables[_b_2]);
					}
				}
			}
		}

		if (combine_lables.size() != (size_t)(pow(base_cnt, repeat))){
			QCERR_AND_THROW(run_fail, "Error: itr_product lable error.");
		}

		return combine_lables;
	};

	std::map<std::string, double> combine_result;

	/** 获取所有标签的组合关系
	*/
	const auto combine_lables = itr_product(stitches_cnt);

	for (const auto& _lables : combine_lables)
	{
		/** 得到标签 
		* 根据全排列信息计算 frag_labels：每个子图的测量标签和制备标签
		*/
		std::vector<FragLabel> frag_labels(choi_states_vec.size());
		build_frag_labels(_lables, stitches, frag_labels);

		/** 根据标签得到子图矩阵 
		*/
		auto frag_mats = target_labels_to_matrix(frag_labels);
        
		/**
		* 拼接final-qubit，得到量子态字符串，然后计算各个量子态的值
		* 注意:处理bit序
		*/
		std::vector<RecombineFragment::FinalQubitJoinedStr> 
			joined_str_vec = get_final_qubit_combination(choi_states_vec); //final_bit_pieces

		for (const auto& _joined_str : joined_str_vec)
		{
			string quantum_state_str;
			std::vector<std::complex<double>> val_vec(_joined_str.size(), 0);
			for (size_t _i = 0; _i < _joined_str.size(); ++_i)
			{
				const auto _sub_cir_index = /*_joined_str.size() - 1 - */_i;
				quantum_state_str += _joined_str[_i];

				auto _t_frag_mats = frag_mats[_sub_cir_index].adjoint();
				const QVectorXcd _mat = QVectorXcd::Map(_t_frag_mats.eval().data(), _t_frag_mats.size());
				//PTrace("frag_mats: " << _sub_cir_index << ":\n" << _mat);

				const QVectorXcd _choi = QVectorXcd::Map(choi_states_vec[_sub_cir_index].at(_joined_str[_i]).data(),
					choi_states_vec[_sub_cir_index].at(_joined_str[_i]).size());
				//PTrace("_choi: " << _i << ", "<< _joined_str[_i] <<":\n" << _choi);

				const auto mat_size = choi_states_vec[_sub_cir_index].at(_joined_str[_i]).size();
				for (size_t _j = 0; _j < mat_size; ++_j){
		 			val_vec[_i] += (_mat[_j] * _choi[_j]);
				}
			}

			quantum_state_str = correct_qubit_order(quantum_state_str, qubit_permutation);

			std::complex<double> final_val = 1.;
			for (const auto& _d : val_vec){
				final_val *= _d;
			}

            combine_result[quantum_state_str] += final_val.real();
		}
	}

	/** 归一化处理
	*/
	double _sum = 0.;
	for (const auto& _result : combine_result){
		_sum += _result.second;
	}
	for (auto& _result : combine_result) {
		_result.second /= _sum;
	}

    return combine_result;
}

static QMatrixXcd array_outer(QVectorXcd choi_vector)
{
	auto dim = choi_vector.size();

	QMatrixXcd matrix = QMatrixXcd::Zero(dim, dim);

	for (auto i = 0; i < dim; ++i)
	{
		QVectorXcd outer_result = choi_vector.conjugate()[i] * choi_vector;

		for (auto j = 0; j < dim; ++j)
		{
			matrix(i, j) = outer_result[j];
		}
	}

	return matrix;
}

template <typename T>
static std::vector<int> argsort(const std::vector<T> &array)
{
	const int array_len(array.size());
	std::vector<int> array_index(array_len, 0);
	for (int i = 0; i < array_len; ++i)
		array_index[i] = i;

	std::sort(array_index.begin(), array_index.end(),
		[&](int pos1, int pos2) { return (array[pos1] < array[pos2]); });

	return array_index;
}

void RecombineFragment::maximum_likelihood_model(const std::vector<ChoiMatrices>& choi_states_vec, 
	std::vector<ChoiMatrices>& likely_choi_states_vec)
{
	for (auto choi_state : choi_states_vec)
	{
		std::map<string, QVectorXd> choi_eigs;
		std::map<string, QMatrixXcd> choi_vecs;
		for (auto val : choi_state)
		{
			auto final_bits = val.first;
			auto choi_state = val.second;

			QMatrixXcd choi_identity = QMatrixXcd::Identity(choi_state.rows(), choi_state.cols());

			//Eigen::ComplexEigenSolver<QMatrixXcd> ges;
			Eigen::GeneralizedSelfAdjointEigenSolver<QMatrixXcd> ges;

			ges.compute(choi_state, choi_identity);
			//ges.compute(choi_state, true);

			choi_eigs[final_bits] = ges.eigenvalues();
			choi_vecs[final_bits] = ges.eigenvectors();
		}

		prob_vec all_eigens;
		for (auto val : choi_eigs)
		{
			for (auto i = 0; i < val.second.size(); ++i)
			{
				all_eigens.emplace_back(val.second[i]);
			}
		}

        auto eigen_order = argsort(all_eigens);

        prob_vec sort_eigens = all_eigens;
		std::sort(sort_eigens.begin(), sort_eigens.end());

		auto dim = all_eigens.size();

		for (auto i = 0; i < dim; ++i)
		{
			auto val = sort_eigens[i];

			if ((val > 0.) && fabs(val - 0.) > 1e-6)
			{
				break;
			}

			sort_eigens[i] = 0.;

			for (auto j = i + 1; j < dim; ++j)
			{
                sort_eigens[j] += (val / (dim - i - 1.));
			}
		}

		auto order = argsort(eigen_order);

		prob_vec final_all_eigens;
		for (auto idx : order)
		{
			final_all_eigens.emplace_back(sort_eigens[idx]);
		}
		auto num_block = choi_eigs.size();
		auto block_size = dim / num_block;

		Eigen::Map<QMatrixXd> reshape_all_eigens(final_all_eigens.data(), num_block, block_size);

		std::map<string, QVectorXd> choi_eigen_values;

		size_t idx = 0;
		for (auto val : choi_eigs)
		{
			auto bits = val.first;

			QVectorXd zero_array = QVectorXd::Zero(val.second.size());

			QVectorXd choi_value = QVectorXd::Map(reshape_all_eigens.row(idx).data(), reshape_all_eigens.row(idx).size());

			QPANDA_OP(choi_value != zero_array, choi_eigen_values[bits] = choi_value);

			idx++;
		}

		ChoiMatrices result;
		for (auto items : choi_eigen_values)
		{
			auto choi_qubits = items.first;  //bits
			auto choi_values = items.second; //vals

			for (auto i = 0; i < choi_values.size(); ++i)
			{
				if (choi_values[i] > 0.) //val
				{
					QVectorXcd choi_vector = choi_vecs[choi_qubits].col(i); //choi_vecs[bits][:,idx]
					QMatrixXcd outer_result = array_outer(choi_vector); //to_projector result

					auto iter = result.find(choi_qubits);
					if (result.end() != iter)
					{
						auto choi_matrix = iter->second;
						choi_matrix += choi_values[i] * outer_result;

                        result[choi_qubits] = choi_matrix;
					}
					else
					{
						result[choi_qubits] = choi_values[i] * outer_result;
					}
				}
			}
		}

		likely_choi_states_vec.emplace_back(result);
	}
}

/*******************************************************************
*                      class QCircuitStripping
********************************************************************/
class QCircuitStripping : protected TraverseByNodeIter
{
public:
	QCircuitStripping() {}
	virtual ~QCircuitStripping() {}

	QCircuit get_stripped_cir(QProg prog) {
		traverse_qprog(prog);
		return m_stripped_cir;
	}

protected:
	void execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node,
		QCircuitParam& param, NodeIter& cur_iter) override {
		QVec used_qv;
		cur_node->getQuBitVector(used_qv);
		QVec ctrl_qv;
		cur_node->getQuBitVector(ctrl_qv);
		ctrl_qv += param.m_control_qubits;
		used_qv += ctrl_qv;
		const auto qv_size = used_qv.size();
		if (qv_size == 2)
		{
			m_stripped_cir << QGate(cur_node);
		}
		else if (qv_size > 2)
		{
			QCERR_AND_THROW(run_fail, "Error: unsupport multiple_control-gate.");
		}
	}

	void execute(std::shared_ptr<AbstractControlFlowNode>, std::shared_ptr<QNode>, QCircuitParam&, NodeIter&)override {
		QCERR_AND_THROW(run_fail, "Error: unsupport controlflow node.");
	}

private:
	QCircuit m_stripped_cir;
};

/*******************************************************************
*                      class DAGCutter
********************************************************************/
class DAGCutter 
{
public:
	DAGCutter(std::shared_ptr<QProgDAG> graph, uint32_t max_qubit) 
		: m_graph(graph), m_max_qubit(max_qubit)
	{};
	~DAGCutter() {};

	std::map<uint32_t, std::vector<uint32_t>> cutDAG(std::vector<std::vector<uint32_t>>& sub_graph_vertice) {
		auto vertices = m_graph->get_vertex();
		std::set<uint32_t> cut;
		sub_graph_vertice.clear(); /**< store每个切割得到的子图中包含的vertex信息 */
		std::map<uint32_t, std::set<uint32_t>> cut_pos_list;
		while (true) {
			auto edges = m_graph->get_edges();
			uint32_t edge_num = 0;
			for (auto& _e : edges) {
				if (cut.find(_e.m_from) == cut.end() && cut.find(_e.m_to) == cut.end()) {
					edge_num++;
				}
			}

			if (2 * (vertices.size() - cut.size()) - edge_num <= m_max_qubit) {
				sub_graph_vertice.emplace_back(std::vector<uint32_t>());
				for (uint32_t _i = 0; _i < vertices.size(); ++_i)
				{
					if (cut.end() == cut.find(_i)) {
						sub_graph_vertice.back().emplace_back(_i);
					}
				}

				break;
			}
			std::set<uint32_t> sub;
			std::map<std::set<uint32_t>, std::pair<uint32_t, std::map<uint32_t, std::set<uint32_t>>>> circuit_map;
			uint32_t i = 0;
			while (true) {
				if (cut.count(i) == 0) {
					sub.insert(i);
					recurCut(cut, sub, circuit_map);
					break;
				}
				i = (i + 1) % vertices.size();
			}

			uint32_t min_overhead = INT_MAX;
			uint32_t max_node = 0;
			for (auto& _set_overhead : circuit_map) {
				if (_set_overhead.second.first < min_overhead) min_overhead = _set_overhead.second.first;
			}
			for (auto& _set_overhead : circuit_map) {
				if (_set_overhead.second.first == min_overhead && _set_overhead.first.size() > max_node) {
					max_node = _set_overhead.first.size();
				}
			}
			for (auto& _set_overhead : circuit_map)
			{
				if (_set_overhead.second.first == min_overhead && _set_overhead.first.size() == max_node)
				{
					auto cut_list = _set_overhead.second.second;
					for (auto& _itr : cut_list)
					{
						if (cut_pos_list.find(_itr.first) == cut_pos_list.end())
						{
							cut_pos_list[_itr.first] = _itr.second;
						}
						else
						{
							for (auto& _cut : _itr.second) {
								cut_pos_list[_itr.first].insert(_cut);
							}
						}
					}
					sub_graph_vertice.emplace_back(std::vector<uint32_t>());
					for (auto& _cut_node : _set_overhead.first) {
						cut.insert(_cut_node);
						sub_graph_vertice.back().emplace_back(_cut_node);
					}
					break;
				}
			}
		}
		std::map<uint32_t, std::vector<uint32_t>> cut_pos_vec;
		for (auto& _itr : cut_pos_list) {
			cut_pos_vec[_itr.first].assign(_itr.second.begin(), _itr.second.end());
		}

		return cut_pos_vec;
	}

	void recurCut(std::set<uint32_t> cut, std::set<uint32_t>& sub,
		std::map<std::set<uint32_t>, std::pair<uint32_t, std::map<uint32_t, std::set<uint32_t>>>>& circuit_map) {
		auto vertices = m_graph->get_vertex();
		auto qubit_vertices_map = m_graph->get_qubit_vertices_map();
		std::set<uint32_t> T;
		uint32_t edge_num = 0;
		uint32_t overhead = 0;
		std::map<uint32_t, std::set<uint32_t>> cut_list;
		for (auto& _dag_node_itr : sub) {
			auto v = vertices[_dag_node_itr];
			for (auto& _e : v.m_succ_edges) {
				if (sub.find(_e.m_to) != sub.end()) {
					edge_num++;
				}
				else {
					overhead++;
					auto gate_list = qubit_vertices_map[_e.m_qubit];
					for (int i = 0; i < gate_list.size(); ++i) {
						if (gate_list[i] == _dag_node_itr) {
							cut_list[_e.m_qubit].insert(i);
						}
					}
				}
			}
			for (auto& _e : v.m_pre_edges) {
				if (sub.find(_e.m_from) == sub.end()) {
					overhead++;
					auto gate_list = qubit_vertices_map[_e.m_qubit];
					for (int i = 0; i < gate_list.size(); ++i) {
						if (gate_list[i] == _dag_node_itr) {
							cut_list[_e.m_qubit].insert(i - 1);
						}
					}
				}
			}
		}
		if (2 * sub.size() - edge_num > m_max_qubit) {
			return;
		}
		else if (2 * sub.size() - edge_num == m_max_qubit) {
			circuit_map[sub] = std::make_pair(overhead, cut_list);
		}
		for (auto& _dag_node_itr : sub) {
			auto v = vertices[_dag_node_itr];
			std::for_each(v.m_pre_node.begin(), v.m_pre_node.end(),
				[&](uint32_t n) {
				if (sub.find(n) == sub.end() && cut.find(n) == cut.end()) {
					T.insert(n);
				}
			});
			std::for_each(v.m_succ_node.begin(), v.m_succ_node.end(),
				[&](uint32_t n) {
				if (sub.find(n) == sub.end() && cut.find(n) == cut.end()) {
					T.insert(n);
				}
			});
		}
		if (T.empty()) {
			circuit_map[sub] = std::make_pair(overhead, cut_list);
			return;
		}
		for (auto& _neighbor : T) {
			sub.insert(_neighbor);
			if (circuit_map.find(sub) == circuit_map.end()) {
				recurCut(cut, sub, circuit_map);
			}
			sub.erase(_neighbor);
		}
	}

private:
	std::shared_ptr<QProgDAG> m_graph;
	uint32_t m_max_qubit;
};

/*******************************************************************
*                      public interface
********************************************************************/
QCircuit QPanda::circuit_stripping(QProg prog)
{
	return QCircuitStripping().get_stripped_cir(prog);
}

static std::map<uint32_t, uint32_t> get_vertex_map_from_striped_cir_to_src_cir(const QProgDAG& striped_prog_dag, 
	const QProgDAG& src_prog_dag)
{
	const auto& src_cir_qubit_vertices_map = src_prog_dag.get_qubit_vertices_map();
	const auto& striped_cir_qubit_vertices_map = striped_prog_dag.get_qubit_vertices_map();
	const std::vector<QProgDAGVertex>& src_dag_vertices = src_prog_dag.get_vertex_c();
	std::map<uint32_t, uint32_t> ret_map;

	for (const auto& _qubit_vertex_info : striped_cir_qubit_vertices_map)
	{
		const auto& _target_qubit = _qubit_vertex_info.first;
		if (src_cir_qubit_vertices_map.find(_target_qubit) == src_cir_qubit_vertices_map.end()){
			continue;
		}

		uint32_t _i = 0; /**< on double-QGate index */
		for (const auto& _src_vertex : src_cir_qubit_vertices_map.at(_target_qubit))
		{
			if (src_dag_vertices[_src_vertex].m_node->m_qubits_vec.size() > 1)
			{
				ret_map.insert(std::make_pair(_qubit_vertex_info.second[_i], _src_vertex));
				++_i;
			}
		}
	}

	return ret_map;
}

std::vector<std::vector<uint32_t>> 
QPanda::get_real_vertex_form_stripped_cir_vertex(const std::vector<std::vector<uint32_t>>& vertex_on_stripped_cir,
	const QProgDAG& striped_prog_dag, const QProgDAG& src_prog_dag)
{
	auto tmp_vertice = vertex_on_stripped_cir;
	std::map<uint32_t, uint32_t> vertex_map = get_vertex_map_from_striped_cir_to_src_cir(striped_prog_dag,
		src_prog_dag);
	for (auto& _sub_graph_vertice : tmp_vertice)
	{
		for (auto& _vertex : _sub_graph_vertice)
		{
			_vertex = vertex_map.at(_vertex);
		}
	}

	return tmp_vertice;
}

std::map<uint32_t, std::vector<uint32_t>> QPanda::get_real_cut_pos_from_stripped_cir_cut_pos(
	const std::map<uint32_t, std::vector<uint32_t>>& cut_pos, const QProgDAG& prog_dag)
{
	const auto& qubit_vertices_map = prog_dag.get_qubit_vertices_map();
	const std::vector<QProgDAGVertex>& dag_vertices = prog_dag.get_vertex_c();
	std::map<uint32_t, std::vector<uint32_t>> real_cut_pos;
	for (const auto& _cur : cut_pos)
	{
		real_cut_pos.insert(std::make_pair(_cur.first, std::vector<uint32_t>()));
		const auto& qubit_vertices = qubit_vertices_map.at(_cur.first);
		const auto qubit_vertices_size = qubit_vertices.size();	
		for (auto _itr = _cur.second.begin(); _itr != _cur.second.end(); ++_itr)
		{
			const auto& _vertex_index = *_itr;
			if (qubit_vertices_size <= _vertex_index + 1) {
				QCERR_AND_THROW(run_fail, "Error: cut-pos error; vertices.size() = "
					<< qubit_vertices_size << "_vertex_index= " << _vertex_index);
			}
			uint32_t _j = 0;
			for (uint32_t _i = 0; _i < qubit_vertices_size; ++_i)
			{
				if (dag_vertices[qubit_vertices[_i]].m_node->m_qubits_vec.size() > 1)
				{
					if (_vertex_index == _j)
					{
						real_cut_pos.at(_cur.first).emplace_back(_i);
					}
					++_j;
				}
			}
		}
	}

	return real_cut_pos;
}

std::vector<SubCircuit> QPanda::cut_circuit(const QProgDAG& prog_dag, const std::map<uint32_t, std::vector<uint32_t>>& cut_pos,
	std::vector<StitchesInfo>& stitches, QuantumMachine *qvm, const std::vector<std::vector<uint32_t>>& sub_graph_vertice,
	std::vector<uint32_t>& qubit_permutation)
{
	CutQCircuit cutter(prog_dag);
	cutter.cut_circuit(cut_pos, sub_graph_vertice, qvm);
	cutter.generate_subcircuits(qvm);
	stitches = cutter.get_stitches(cut_pos);
	return cutter.get_cutted_sub_circuits(qubit_permutation);
}

std::vector<SubCircuit> QPanda::cut_circuit(const QProg src_prog, QuantumMachine *qvm, const size_t& max_sub_cir_qubits,
	std::vector<StitchesInfo>& stitches, std::vector<uint32_t>& qubit_permutation)
{
	auto striped_cir = circuit_stripping(src_prog);
#if PRINT_TRACE
	cout << "striped_cir:" << striped_cir << endl;
#endif

	auto striped_prog = CreateEmptyQProg();
	striped_prog << striped_cir;

	//get cutting-point
	std::vector<std::vector<uint32_t>> sub_graph_vertice;
	auto striped_cir_dag = qprog_to_DAG(striped_prog);

	DAGCutter graph_cutter(striped_cir_dag, max_sub_cir_qubits);
	std::map<uint32_t, std::vector<uint32_t>> cut_pos_on_striped_cir = graph_cutter.cutDAG(sub_graph_vertice);

	uint32_t sum = 0;
	std::for_each(cut_pos_on_striped_cir.begin(), cut_pos_on_striped_cir.end(),
		[&](std::pair<uint32_t, std::vector<uint32_t>> p) {
		sum += p.second.size();
	});
	PTrace("The total overhead is " << sum);

	//run cut-circuit
	auto src_cir_dag = qprog_to_DAG(src_prog);
	std::map<uint32_t, std::vector<uint32_t>> real_cut_pos = get_real_cut_pos_from_stripped_cir_cut_pos(
		cut_pos_on_striped_cir, *src_cir_dag);

	std::vector<std::vector<uint32_t>> sub_graph_vertice_on_src_cir = get_real_vertex_form_stripped_cir_vertex(
		sub_graph_vertice, *striped_cir_dag, *src_cir_dag);

	std::vector<SubCircuit> sub_cir_info = cut_circuit(*src_cir_dag, real_cut_pos, stitches, qvm, 
		sub_graph_vertice_on_src_cir, qubit_permutation);
	return sub_cir_info;
}

std::map<std::string, double>
QPanda::recombine_sub_circuit_exec_data(const std::vector<SubCircuit>& sub_cir, const std::vector<StitchesInfo>& stitches,
	const std::vector<uint32_t>& qubit_permutation)
{
	//Execute each sub-circuit
	PTrace("On collect_fragment_data ....");
	RecombineFragment recombiner(sub_cir);
	std::vector<RecombineFragment::ResultDataMap> frag_data = recombiner.collect_fragment_data();

	//get choi-matrix
	PTrace("On direct_fragment_model to get choi-matrix ....");
	std::vector<RecombineFragment::ChoiMatrices> choi_states_vec;
	recombiner.direct_fragment_model(frag_data, choi_states_vec);

	//maximum_likelihood_model
	PTrace("On maximum_likelihood_model ....");
	std::vector<RecombineFragment::ChoiMatrices> likely_choi_states_vec; 
	recombiner.maximum_likelihood_model(choi_states_vec, likely_choi_states_vec);

	//recombine the execution data of sub-circuits 
	PTrace("On recombine_using_insertions ....");
	std::map<std::string, double> result = recombiner.recombine_using_insertions(likely_choi_states_vec, stitches, qubit_permutation);
	return result;
}

std::map<std::string, double> 
QPanda::exec_by_cutQC(QCircuit cir, QuantumMachine *qvm, const std::string& backend, const uint32_t& max_back_end_qubit)
{
	std::vector<StitchesInfo> stitches;
	std::vector<uint32_t> qubit_permutation;
	std::vector<SubCircuit> sub_cir_info = cut_circuit(cir, qvm, max_back_end_qubit, stitches, qubit_permutation);

#if PRINT_TRACE
	for (uint32_t i = 0; i < sub_cir_info.size(); ++i) {
		cout << "sub_cir_" << i << ":::::::::::::" << sub_cir_info[i].m_cir << endl;
	}
#endif

	return recombine_sub_circuit_exec_data(sub_cir_info, stitches, qubit_permutation);
}
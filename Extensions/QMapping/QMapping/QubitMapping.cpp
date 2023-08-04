#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include <iterator>
#include <memory>

#include "QubitMapping.h"

using namespace std;
using namespace QPanda;

const uint32_t QPanda::UNDEF_UINT32 = (std::numeric_limits<uint32_t>::max)();

InverseMap QPanda::InvertMapping(uint32_t archQ, Mapping mapping, bool fill)
{
    //uint32_t progQ = mapping.size();
    // 'archQ' is the number of qubits from the architecture.
    std::vector<uint32_t> inv(archQ, UNDEF_UINT32);

    // for 'u' in arch; and 'a' in prog:
    // if 'a' -> 'u', then 'u' -> 'a'
    for (uint32_t i = 0; i < mapping.size(); ++i)
        if (mapping[i] != UNDEF_UINT32)
            inv[mapping[i]] = i;

    if (fill) {
        // Fill the qubits in the architecture that were not mapped.
        Fill(mapping, inv);
    }

    return inv;
}

void QPanda::Fill(Mapping& mapping, InverseMap& inv)
{
    uint32_t progQ = mapping.size(), archQ = inv.size();
    uint32_t a = 0, u = 0;

    do {
        while (a < progQ && mapping[a] != UNDEF_UINT32) ++a;
        while (u < archQ && inv[u] != UNDEF_UINT32) ++u;

        if (u < archQ && a < progQ) {
            mapping[a] = u;
            inv[u] = a;
            ++u; ++a;
        } else {
            break;
        }
    } while (true);
}

void QPanda::Fill(uint32_t archQ, Mapping& mapping)
{
    auto inv = InvertMapping(archQ, mapping, false);
    Fill(mapping, inv);
}

Mapping QPanda::IdentityMapping(uint32_t progQ)
{
    Mapping mapping(progQ, UNDEF_UINT32);

    for (uint32_t i = 0; i < progQ; ++i) {
        mapping[i] = i;
    }

    return mapping;
}

std::string QPanda::MappingToString(Mapping m) {
    std::string s = "[";
    for (uint32_t i = 0, e = m.size(); i < e; ++i) {
        s = s + std::to_string(i) + " => ";
        if (m[i] == UNDEF_UINT32) s = s + "UNDEF_UINT32";
        else s = s + std::to_string(m[i]);
        s = s + ";";
        if (i != e - 1) s = s + " ";
    }
    s = s + "]";
    return s;
}

/*******************************************************************
*                      class AbstractQubitMapping
********************************************************************/
AbstractQubitMapping::AbstractQubitMapping(ArchGraph::sRef archGraph)
	: mArchGraph(archGraph), m_CX_cost(10), m_CZ_cost(10), m_u3_cost(1)
{
    mGateWeightMap = { {"U", 1}, {"CX", 10}, {"CZ", 10} };
}

uint32_t AbstractQubitMapping::get_CX_cost(uint32_t u, uint32_t v)
{
	if (mArchGraph->hasEdge(u, v)) return m_CX_cost;
	if (mArchGraph->hasEdge(v, u)) return m_CX_cost + (4 * m_u3_cost);
}

uint32_t AbstractQubitMapping::get_CZ_cost(uint32_t u, uint32_t v)
{
	if ((mArchGraph->hasEdge(u, v)) || (mArchGraph->hasEdge(v, u))) return m_CZ_cost;
}

uint32_t AbstractQubitMapping::getSwapCost(uint32_t u, uint32_t v) {
    uint32_t uvCost = get_CZ_cost(u, v);
    return uvCost * 3;
}

bool AbstractQubitMapping::run(QPanda::QProg prog, QuantumMachine *qvm)
{
    // Filling Qubit information.
	QVec used_qubits;
    mVQubits = get_all_used_qubits(prog, used_qubits);
    mPQubits = mArchGraph->size();

	if (mVQubits > mPQubits)
	{
		QCERR_AND_THROW(run_fail,
				"Error: The number of qubits used in target QPorg exceeds the number of qubits of physical chips. mVQubits: "s
				+ std::to_string(mVQubits) + ", mPQubits: " + std::to_string(mPQubits));
	}

	m_final_mapping = allocate(prog, qvm);
    return true;
}

/*******************************************************************
*                      class FrontLayer
********************************************************************/
void FrontLayer::remove_node(pPressedCirNode p_node)
{
	update_cur_layer_qubits(p_node);
	for (auto itr = m_front_layer_nodes.begin(); itr != m_front_layer_nodes.end(); ++itr) {
		if (*itr == p_node) {
			m_front_layer_nodes.erase(itr);
			return;
		}
	}

	QCERR_AND_THROW(run_fail, "unknow error on FrontLayer::remove_node.");
}

/*******************************************************************
*                      class DynamicQCircuitGraph
********************************************************************/
DynamicQCircuitGraph::DynamicQCircuitGraph(QProg prog)
	:m_prog(prog) {
	init();
}

DynamicQCircuitGraph::DynamicQCircuitGraph(const DynamicQCircuitGraph& c) 
{
	m_prog = c.m_prog;
	m_layer_info = c.m_layer_info;
	m_cur_layer_iter = m_layer_info.begin();
}

void DynamicQCircuitGraph::init() 
{
	auto start = chrono::system_clock::now();
	m_layer_info = get_pressed_layer(m_prog);
#if PRINT_TRACE
	auto end = chrono::system_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
	cout << "The layer takes "
		<< double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den
		<< "seconds. layers_cnt:" << m_layer_info.size() << endl;
	{
		size_t gate_cnt = 0;
		for (auto& layer : m_layer_info) {
			gate_cnt += layer.size();
			for (const auto& node : layer) {
				gate_cnt += node.first->m_relation_pre_nodes.size();
				gate_cnt += node.first->m_relation_successor_nodes.size();
			}
		}
		cout << "total gate_cnt=" << gate_cnt << endl;
		/*cout << "Press enter to continue." << endl;
		getchar();*/
	}
#endif
	m_cur_layer_iter = m_layer_info.begin();
}

FrontLayer& DynamicQCircuitGraph::get_front_layer()
{
	bool b_stay_cur_layer = false;
	if (m_cur_layer_iter != m_layer_info.end())
	{
		auto& cur_layer = *m_cur_layer_iter;
		for (auto node_iter = cur_layer.begin(); node_iter != cur_layer.end(); )
		{
			const auto tmp_node = node_iter->first->m_cur_node;
			/*if (BARRIER_GATE == tmp_node->m_gate_type)
			{
				m_front_layer.m_front_layer_nodes.push_back(node_iter->first);
				m_front_layer.m_cur_layer_qubits += (tmp_node->m_target_qubits + tmp_node->m_control_qubits);
				node_iter = cur_layer.erase(node_iter);
				continue;
			}*/
			auto node_qubits = tmp_node->m_target_qubits + tmp_node->m_control_qubits;
			auto q = node_qubits - m_front_layer.m_cur_layer_qubits;
			//bool b_qubit_multiplex = (q.size() != tmp_node->m_target_qubits.size());
			bool b_qubit_multiplex = (q.size() != node_qubits.size());
			b_stay_cur_layer = (b_stay_cur_layer || b_qubit_multiplex);
			if (!b_qubit_multiplex)
			{
				m_front_layer.m_front_layer_nodes.push_back(node_iter->first);
				//m_front_layer.m_cur_layer_qubits += tmp_node->m_target_qubits;
				m_front_layer.m_cur_layer_qubits += node_qubits;
				node_iter = cur_layer.erase(node_iter);
				continue;
			}

			++node_iter;
		}

		if (cur_layer.size() == 0) {
			++m_cur_layer_iter;
		}
	}

	return m_front_layer;
}

QMappingConfig::QMappingConfig(const std::map<size_t, Qnum>& config_arch_map)
{
    // Step 1: Determine the size of the adjacency matrix
    size_t max_index = 0;
    for (const auto& entry : config_arch_map) 
    {
        max_index = std::max(max_index, entry.first);
        for (size_t neighbor : entry.second) 
            max_index = std::max(max_index, neighbor);
    }

    // Step 2: Create an Eigen::MatrixXd object and initialize all elements to 0
    Eigen::MatrixXd adjacency_matrix = Eigen::MatrixXd::Zero(max_index + 1, max_index + 1);

    // Step 3: Set the corresponding matrix elements to 1 for each node and its neighbors
    for (const auto& entry : config_arch_map) 
    {
        size_t node = entry.first;
        for (size_t neighbor : entry.second) 
        {
            adjacency_matrix(node, neighbor) = 1; // Adjust indices to 0-based indexing
        }
    }

    // Step 4: Set diagonal elements to 0 (optional, depends on the adjacency matrix definition)
    adjacency_matrix.diagonal().setZero();

    initialize(adjacency_matrix);
}

static bool is_valid_adjacency_matrix(const Eigen::MatrixXd& matrix) 
{
    // Step 1: Check if the matrix is square
    if (matrix.rows() != matrix.cols())
        return false;

    // Step 2: Check if all elements are non-negative
    if ((matrix.array() < 0).any())
        return false;

    // Step 3: Check if diagonal elements are zero
    if ((matrix.diagonal().array() != 0).any())
        return false;

    // Step 4: Check if the matrix is symmetric
    if (!matrix.isApprox(matrix.transpose()))
        return false;

    // All checks passed, the matrix is a valid adjacency matrix
    return true;
}

void QMappingConfig::initialize(const dmatrix_t& arch_matrix)
{
    const uint32_t qubits = arch_matrix.rows();

    auto graph = ArchGraph::Create(qubits);

    std::string name = "quantum_chip";
    graph->putReg(name, std::to_string(qubits));

    for (int i = 0; i < arch_matrix.cols(); ++i)
    {
        for (int j = 0; j < arch_matrix.rows(); ++j)
        {
            double weight = arch_matrix(i, j);

            if (weight > (0.0 + 1e-6) && weight <= (1.0 + 1e-6))
                graph->putEdge(i, j, weight);
        }
    }

    m_arch_ptr = std::move(graph);
}

QMappingConfig::QMappingConfig(const dmatrix_t& arch_matrix)
{
    if (!is_valid_adjacency_matrix(arch_matrix))
        QCERR_AND_THROW(run_fail, "invalid adjacency matrix!");

    initialize(arch_matrix);
}

QMappingConfig::QMappingConfig(std::string_view config_data)
{
    if (config_data.length() < 6) 
        QCERR_AND_THROW(std::runtime_error, "config_data length error");

    string suffix = std::string(config_data.substr(config_data.length() - 5));
    transform(suffix.begin(), suffix.end(), suffix.begin(), ::tolower);
    if (0 == suffix.compare(".json"))
        m_arch_ptr =  JsonParser<ArchGraph>::ParseFile(config_data.data());
    else
        m_arch_ptr = JsonParser<ArchGraph>::ParseString(config_data.data());
}

QMappingConfig::QMappingConfig(const prob_vec& arch_matrix)
{
    size_t size = arch_matrix.size();
    size_t n = static_cast<size_t>(std::sqrt(size));

    if (n * n != size)
        QCERR_AND_THROW(run_fail, "arch_matrix error");

    Eigen::MatrixXd matrix(n, n);

    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            matrix(i, j) = arch_matrix[i * n + j];

    if (!is_valid_adjacency_matrix(matrix))
        QCERR_AND_THROW(run_fail, "invalid adjacency matrix!");

    initialize(matrix);
}

std::shared_ptr<ArchGraph> QMappingConfig::get_arch_config() const
{
    return  m_arch_ptr;
}

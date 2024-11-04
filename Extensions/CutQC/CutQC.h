#ifndef CUT_QC_H
#define CUT_QC_H

#include <map>
#include "Core/Utilities/Tools/Traversal.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"
#include "Core/Utilities/Tools/QPandaException.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/QProgDAG.h"
#include "Core/Utilities/Tools/QMatrixDef.h"

QPANDA_BEGIN

struct SubCircuit
{
    QCircuit m_cir;
    QVec m_prep_qubit; //preparation-qubit
    QVec m_meas_qubit; //measure-qubit
};

/**
* @brief Cutting point information
* Note: Each cut point corresponds to a measurement-qubit and a preparation-qubit of two sub-circuit respectively
*/
struct StitchesInfo
{
    using sub_cir_op_qubit_index = std::pair<uint32_t, uint32_t>; /**< sub_circuit_index : qubit_index */
    sub_cir_op_qubit_index m_meas_qubit;
    sub_cir_op_qubit_index m_prep_qubit;
};

class QuantumMachine;
class CutQCircuit
{
    struct CutFragment
    {
        std::set<uint32_t> m_vertice;
        std::set<uint32_t> m_qubit;
        std::set<uint32_t> m_prep_qubit; /**< preparation-qubit fragment */
        std::set<uint32_t> m_meas_qubit; /**< measure-qubit */
		std::map<uint32_t, uint32_t> m_auxi_qubit_map;
        QCircuit m_cir;
    };

public:
	CutQCircuit(const QProgDAG& prog_dag)
		:m_src_prog_dag(prog_dag), m_cut_prog_dag(prog_dag), m_max_qubit_num(0)
	{}
	virtual ~CutQCircuit() {}

	void cut_circuit(const std::map<uint32_t, std::vector<uint32_t>>& cut_pos, 
		const std::vector<std::vector<uint32_t>>& sub_graph_vertice, QuantumMachine *qvm);
	void generate_subcircuits(QuantumMachine *qvm);
	const std::vector<SubCircuit>& get_cutted_sub_circuits(std::vector<uint32_t>& qubit_permutation);

	std::vector<StitchesInfo> get_stitches(const std::map<uint32_t, std::vector<uint32_t>>& cut_pos);

protected:
    bool exist_edge_on_target_qubit(const uint32_t& target_qubit, const std::vector<QProgDAGEdge>& edges);
	int find_vertex_in_sub_cir(const uint32_t& vertex);
	void append_sub_cir(CutFragment& sub_cir, QuantumMachine *qvm);
	bool is_pre_qubit(const QProgDAGVertex& src_dag_node, const QProgDAGVertex& cut_dag_node, const uint32_t& target_qubit);
	bool is_mea_qubit(const QProgDAGVertex& src_dag_node, const QProgDAGVertex& cut_dag_node, const uint32_t& target_qubit);
	const std::vector<uint32_t>& get_qubit_permutation();
	QGate remap_to_virtual_qubit(std::shared_ptr<QProgDAGNode> gate_node, const std::map<uint32_t, Qubit*>& vir_qubit_map);
	std::map<uint32_t, Qubit*> get_continue_qubit_map(const CutFragment& frag, QuantumMachine *qvm);
	uint32_t get_target_qubit(const uint32_t& vertex_index, const uint32_t& q);

private:
	const QProgDAG& m_src_prog_dag;
	QProgDAG m_cut_prog_dag;
	std::vector<CutFragment> m_cut_fragment_vec;
	std::vector<int> m_vertex_sub_cir_info;
	std::vector<SubCircuit> m_sub_cirs;
	//std::vector<std::vector<uint32_t>> m_sub_cir_qubit_seq; /**< 每个子图对应的qubit序列，映射过的连续qubit */
	std::vector<uint32_t> m_qubit_permutation;
	uint32_t m_max_qubit_num;
	std::vector<std::map<uint32_t, Qubit*>> m_vir_qubit_map_vec;
};

/*******************************************************************
*                      class RecombineFragment
********************************************************************/
enum class MeasBasis
{
	BASIS_Z,
	BASIS_X,
	BASIS_Y
};

enum class MeasState
{
    Zp,
    Zm,
    Xp,
    Xm,
    Yp,
    Ym
};

enum class PrepState
{
	S0,
	S1,
	S2,
	S3
};

class FragmentResult
{
public:
	FragmentResult(size_t prep_num, size_t meas_num){
		m_prep_state.resize(prep_num, PrepState::S0);
		m_meas_basis.resize(meas_num, MeasBasis::BASIS_Z);
	}

	void set_meas_label(size_t meas_qubit, MeasBasis basis){
		m_meas_basis[meas_qubit] = basis;
	}

	void set_prep_label(size_t prep_qubit, PrepState state){
		m_prep_state[prep_qubit] = state;
	}

public:
	std::vector<PrepState> m_prep_state;
	std::vector<MeasBasis> m_meas_basis;
	std::map<std::string, size_t> m_result;
};

class QCirFragments
{
public:
	QCirFragments(const SubCircuit& frag) {
		m_cir = frag.m_cir;
        m_meas_qubits = frag.m_meas_qubit;
        m_prep_qubits = frag.m_prep_qubit;

        auto prep_num = frag.m_prep_qubit.size();
        auto meas_num = frag.m_meas_qubit.size();
        auto prep_count = (size_t)std::pow(4, prep_num);
		auto meas_count = (size_t)std::pow(3, meas_num);
		m_fragment_results.resize(prep_count * meas_count, FragmentResult(prep_num, meas_num));
	}

public:
	QCircuit m_cir;
	QVec m_prep_qubits;
	QVec m_meas_qubits;
	std::vector<FragmentResult> m_fragment_results;
};

class ResultData
{
public:
	ResultData(std::vector<PrepState> prep_labels, std::vector<MeasState> meas_labels) :
		m_prep_labels(prep_labels), m_meas_labels(meas_labels) {}

	std::vector<PrepState> m_prep_labels;
	std::vector<MeasState> m_meas_labels;

    bool operator < (const ResultData &data) const {
        return (m_prep_labels == data.m_prep_labels) && (m_meas_labels == data.m_meas_labels);
    }

private:
	ResultData() = delete;

};

struct FragLabel
{
    std::string m_prep_label;
    std::string m_meas_label;
};

class RecombineFragment
{
public:
	
    //using ResultDataMap = std::map<std::string, std::unordered_map<std::string, size_t>>;
	using ResultDataMap = std::map<std::string, std::vector<std::pair<std::string, size_t>>>;

    /** FragResultData: 所有切割得到的子图对应的所有测量数据，如果切割得到2个子线路，那么vector.size() == 2;
	*/
    using FragResultData = std::vector<ResultDataMap>;
    using ChoiMatrices = std::map<std::string, QMatrixXcd>;
	using CutQCircuits = std::vector<QCirFragments>;
	using FinalQubitJoinedStr = std::vector<std::string>;
	std::string correct_qubit_order(const std::string& src_qubit_str, const std::vector<uint32_t>& qubit_permutation);

public:
	RecombineFragment(const std::vector<SubCircuit>& sub_cir_info)
		:m_sub_cir_vec(sub_cir_info), m_shots(0)
	{}
	~RecombineFragment() {}

public:
	std::vector<ResultDataMap> collect_fragment_data();
	void direct_fragment_model(const std::vector<ResultDataMap>& organize_data_vec, std::vector<ChoiMatrices>& choi_states_vec);
	std::map<std::string, double> recombine_using_insertions(const std::vector<ChoiMatrices>& choi_states_vec, 
		const std::vector<StitchesInfo>& stitches, const std::vector<uint32_t>& qubit_permutation);
	void maximum_likelihood_model(const std::vector<ChoiMatrices>& choi_states_vec,
		std::vector<ChoiMatrices>& likely_choi_states_vec);

protected:
	prob_vec state_to_probs(QStat & state);
	QMatrixXcd target_labels_to_matrix(const ResultData& data);

	void tomography_prep_circuit(QCircuit& circuit, PrepState state, Qubit* qubit);
	void tomography_meas_circuit(QCircuit& circuit, MeasBasis basis, Qubit* qubit);
	void get_choi_matrix(const ResultDataMap& organize_data, ChoiMatrices& choi_states);
	void partial_tomography(QCirFragments& fragment, std::string backend = "CPU");
	void organize_tomography(const QCirFragments& fragment, ResultDataMap& organize_data);
	std::map<std::string, size_t> run_circuits(const QCircuit& circuit, std::string backend);

	void build_frag_labels(const std::string& base_label, 
		const std::vector<StitchesInfo>& stitches, std::vector<FragLabel>& frag_labels);
	std::vector<QMatrixXcd> target_labels_to_matrix(const std::vector<FragLabel>& labels);
	
	std::vector<FinalQubitJoinedStr> get_final_qubit_combination(const std::vector<ChoiMatrices>& choi_states_vec);

private:
	const std::vector<SubCircuit>& m_sub_cir_vec;
	size_t m_shots;
};

/*******************************************************************
*                      public interface
********************************************************************/
QCircuit circuit_stripping(QProg prog);

std::map<uint32_t, std::vector<uint32_t>> get_real_cut_pos_from_stripped_cir_cut_pos(
	const std::map<uint32_t, std::vector<uint32_t>>& cut_pos, const QProgDAG& prog_dag);

std::vector<std::vector<uint32_t>>
get_real_vertex_form_stripped_cir_vertex(const std::vector<std::vector<uint32_t>>& vertex_on_stripped_cir,
	const QProgDAG& striped_prog_dag, const QProgDAG& src_prog_dag);

std::vector<SubCircuit> cut_circuit(const QProgDAG& prog_dag, const std::map<uint32_t, std::vector<uint32_t>>& cut_pos,
	std::vector<StitchesInfo>& stitches, QuantumMachine *qvm, const std::vector<std::vector<uint32_t>>& sub_graph_vertice, 
	std::vector<uint32_t>& qubit_permutation);

/**
* @brief cut quantum circuit, The maximum number of qubits for every sub-circuit cannot exceed \p max_sub_cir_qubits.
         Note: The algorithm will try to get the smallest cut subgraph through the smallest cut point.
* @ingroup Utilities
* @param[in] const QProg The src quantum-prog
* @param[in] QuantumMachine *  quantum machine
* @param[in] const size_t&  The maximum number of qubits for every cut-circuit
* @param[out] std::vector<StitchesInfo>&  The cutting point information, @See StitchesInfo
* @param[out] std::vector<uint32_t>& Qubit-order of recombined quantum-state-string
* @return cutting-sub-circuit vector
*/
std::vector<SubCircuit> cut_circuit(const QProg src_prog, QuantumMachine *qvm, const size_t& max_sub_cir_qubits,
	std::vector<StitchesInfo>& stitches, std::vector<uint32_t>& qubit_permutation);

/**
* @brief Recombining the execution results of all sub-circuits in \p sub_cir
* @ingroup Utilities
* @param[in] const std::vector<SubCircuit>& cutting-sub-circuit vector
* @param[in] const std::vector<StitchesInfo>&  The cutting point information, @See StitchesInfo
* @param[in] const std::vector<uint32_t>& Qubit-order of recombined quantum-state-string
* @return std::map<std::string, double> The results of the complete quantum circuit got by recombine all sub-circuits
*/
std::map<std::string, double> 
recombine_sub_circuit_exec_data(const std::vector<SubCircuit>& sub_cir, const std::vector<StitchesInfo>& stitches, 
	const std::vector<uint32_t>& qubit_permutation);

/**
* @brief Running quantum circuits by cutqc algorithm
* @ingroup Utilities
* @param[in] QCircuit The target quantum circuit
* @param[in] QuantumMachine*  The cutting point information, @See StitchesInfo
* @param[in] const std::string& The target backend name
* @param[in] const uint32_t& The maximum number of qubits in the target backend
* @return std::map<std::string, double> The running-results of the target quantum circuit
*/
std::map<std::string, double> 
exec_by_cutQC(QCircuit cir, QuantumMachine* qvm, const std::string& backend, const uint32_t& max_back_end_qubit);

QPANDA_END
#endif

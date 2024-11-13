#ifndef QUBIT_MAPPING_H
#define QUBIT_MAPPING_H

#include "Core/Utilities/Tools/ArchGraph.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumMachine/QuantumMachineInterface.h"
#include "Core/Utilities/Tools/ProcessOnTraversing.h"
#include "Core/VirtualQuantumProcessor/DensityMatrix/VectorMatrix.h"

using QNodeRef = std::shared_ptr<QPanda::QNode>;

QPANDA_BEGIN

// Defines the type used for mapping the qubits.
typedef std::vector<uint32_t> Mapping;
typedef std::vector<uint32_t> InverseMap;

// Constant should be used as an undefined in a mapping.
extern const uint32_t UNDEF_UINT32;

// Struct used for representing a swap between two qubits;
struct Swap {
    uint32_t u;
    uint32_t v;
};

inline bool operator==(const Swap& lhs, const Swap& rhs) {
    return (lhs.u == rhs.u && lhs.v == rhs.v) ||
        (lhs.u == rhs.v && lhs.v == rhs.u);
}

inline bool operator!=(const Swap& lhs, const Swap& rhs) {
    return !(lhs == rhs);
}

typedef std::vector<Swap> SwapSeq;

using GateWeightMap = std::map<std::string, uint32_t>;

/**
* @brief Base abstract class that allocates the qbits used in the program to
   the qbits that are in the physical architecture.
*/
class AbstractQubitMapping
{
public:
    typedef AbstractQubitMapping* Ref;
    typedef std::unique_ptr<AbstractQubitMapping> uRef;

public:
    bool run(QPanda::QProg prog, QPanda::QuantumMachine *qvm);

    // Sets the weights to be used for each gate.
    void setGateWeightMap(const GateWeightMap& weightMap) { mGateWeightMap = weightMap; }

    const Mapping& get_final_mapping() const { return m_final_mapping; }
    const Mapping& get_init_mapping() const { return m_init_mapping; }
    QPanda::QProg get_mapped_prog() const { return m_mapped_prog; }

    /*virtual void set_specified_block(const std::map<double, std::vector<Mapping>, std::greater<double>>& specified_block)
    { m_specified_blocks = specified_block; }*/
    virtual void set_specified_block(const std::vector<uint32_t>& specified_block){
        m_specified_blocks = specified_block;
    }

    virtual void set_hops(uint32_t hops)
    { m_hops = hops; }

protected:
    QPanda::ArchGraph::sRef mArchGraph;
    GateWeightMap mGateWeightMap;

    uint32_t mVQubits{0};
    uint32_t mPQubits{0};
    QPanda::QProg m_mapped_prog;

    AbstractQubitMapping(QPanda::ArchGraph::sRef archGraph);

    // Executes the allocation algorithm after the preprocessing.
    virtual Mapping allocate(QPanda::QProg prog, QPanda::QuantumMachine *qvm) = 0;

    // Returns the cost of a  CNOT gate, based on the defined weights.
    uint32_t get_CX_cost(uint32_t u, uint32_t v);

    // Returns the cost of a CZ gate, based on the defined weights.
    uint32_t get_CZ_cost(uint32_t u, uint32_t v);

    // Returns the cost of a  SWAP gate, based on the defined weights.
    uint32_t getSwapCost(uint32_t u, uint32_t v);

    QCircuit _swap(Qubit* q_0, Qubit* q_1) {
        QCircuit cir;
#if 0
        cir << U3(q_1, PI / 2.0, 0, PI) << CZ(q_1, q_0) << U3(q_1, PI / 2.0, 0, PI)
            << U3(q_0, PI / 2.0, 0, PI) << CZ(q_0, q_1) << U3(q_0, PI / 2.0, 0, PI)
            << U3(q_1, PI / 2.0, 0, PI) << CZ(q_1, q_0) << U3(q_1, PI / 2.0, 0, PI);
#else
        cir << SWAP(q_1, q_0);
#endif
        return cir;
    }

protected:
    std::vector<uint32_t> m_specified_blocks;
    Mapping m_final_mapping;
    Mapping m_init_mapping;
    uint32_t m_hops = 1;
    uint32_t m_CX_cost;
    uint32_t m_CZ_cost;
    uint32_t m_u3_cost;
};

class FrontLayer
{
    friend class DynamicQCircuitGraph;

public:
    const pPressedCirNode& operator[](uint32_t i) const {
        if (m_front_layer_nodes.size() <= i)
        {
            QCERR_AND_THROW(run_fail, "Error: Array access violation on FrontLayer.");
        }

        return m_front_layer_nodes[i];
    }

    uint32_t size() const { return m_front_layer_nodes.size(); }

    const std::vector<pPressedCirNode>& get_front_layer_nodes() const {
        return m_front_layer_nodes;
    }

    void remove_node(pPressedCirNode p_node);

    // return next index
    uint32_t remove_node(uint32_t i) {
        update_cur_layer_qubits(m_front_layer_nodes[i]);
        m_front_layer_nodes.erase(m_front_layer_nodes.begin() + i);
        return i;
    }

protected:
    FrontLayer() {}
    void update_cur_layer_qubits(pPressedCirNode p_node) {
        m_cur_layer_qubits -= (p_node->m_cur_node->m_target_qubits + p_node->m_cur_node->m_control_qubits);
    }

private:
    std::vector<pPressedCirNode> m_front_layer_nodes;
    QVec m_cur_layer_qubits;
};

class DynamicQCircuitGraph
{
public:
    DynamicQCircuitGraph(QProg prog);
    DynamicQCircuitGraph(const DynamicQCircuitGraph& c);

    virtual ~DynamicQCircuitGraph() {};

    FrontLayer& get_front_layer();

    const FrontLayer& get_front_layer_c() const { return m_front_layer; }
    const PressedTopoSeq& get_layer_topo_seq() const { return m_layer_info; }

protected:
    void init();

private:
    QProg m_prog;
    PressedTopoSeq m_layer_info;
    PressedTopoSeq::iterator m_cur_layer_iter;
    FrontLayer m_front_layer;
};

/*******************************************************************
*                      class RemoveMeasureNode
********************************************************************/
class RemoveMeasureNode : public TraverseByNodeIter
{
public:
    RemoveMeasureNode() {}
    ~RemoveMeasureNode() {}

    template <typename _Ty>
    void remove_measure(_Ty &node)
    {
        /*execute(node.getImplementationPtr(), nullptr);*/
        TraverseByNodeIter::traverse_qprog(node);
    }

    std::vector<std::pair<uint32_t, CBit*>> get_measure_info() {
        std::vector<std::pair<uint32_t, CBit*>> measure_info;
        for (const auto& _m : m_measure_nodes) {
            measure_info.emplace_back(std::make_pair(_m->getQuBit()->get_phy_addr(), _m->getCBit()));
        }

        return measure_info;
    }

    std::vector<std::shared_ptr<AbstractQuantumMeasure>> get_measure_node() { return m_measure_nodes; }

protected:
    /*!
    * @brief  Execution traversal measure node
    * @param[in,out]  AbstractQuantumMeasure*  measure node
    * @param[in]  AbstractQGateNode*  quantum gate
    * @return     void
    */
    void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
        m_measure_nodes.emplace_back(cur_node);
        cur_node_iter = std::dynamic_pointer_cast<AbstractNodeManager>(parent_node)->deleteQNode(cur_node_iter);
    }

    /*!
   * @brief  Execution traversal reset node
   * @param[in,out]  AbstractQuantumReset*  reset node
   * @param[in]  AbstractQGateNode*  quantum gate
   * @return     void
   */
    void execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)override {
        QCERR_AND_THROW(run_fail, "Error: Unsupported reset node.");
    }

private:
    QProg m_measure_prog;
    std::vector<std::shared_ptr<AbstractQuantumMeasure>> m_measure_nodes;
};

/**
* @brief Generates an assignment mapping (maps the architecture's qubits
   to the logical ones) of size archQ.
*/
InverseMap InvertMapping(uint32_t archQ, Mapping mapping, bool fill = true);

// Fills the unmapped qubits with the ones missing.
void Fill(uint32_t archQ, Mapping& mapping);
void Fill(Mapping& mapping, InverseMap& inv);

// Returns an identity mapping.
Mapping IdentityMapping(uint32_t progQ);

// Prints the mapping \p m to a string and returns it.
std::string MappingToString(Mapping m);

class QMappingConfig
{
private:
    std::shared_ptr<ArchGraph> m_arch_ptr;

    void initialize(const dmatrix_t& arch_matrix);

public:
    QMappingConfig(const prob_vec& arch_matrix);
    QMappingConfig(const dmatrix_t& arch_matrix);
    QMappingConfig(std::string_view config_data = CONFIG_PATH);
    QMappingConfig(const std::map<size_t, Qnum>& config_arch_map);

    std::shared_ptr<ArchGraph> get_arch_config() const;
};

QPANDA_END
#endif

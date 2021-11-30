#include "Core/Utilities/QProgInfo/Visualization/DrawLatex.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"
#include "Core/QuantumCircuit/QuantumMeasure.h"
#include <sstream>
#include <memory>
#include <fstream>

namespace // namespace LATEX TOOL
{

    const std::string LATEX_QWIRE = "\\qw";
    const std::string LATEX_CWIRE = "\\cw";

    /* special gate symbol */
    const std::string LATEX_SWAP = "\\qswap";
    const std::string LATEX_CNOT = "\\targ";
    const std::string LATEX_MEASURE = "\\meter";
    const std::string LATEX_RESET = "\\gate{\\mathrm{\\left|0\\right\\rangle}}";

    const std::string LATEX_HEADER = "\\documentclass[border=2px]{standalone}\n"
                                     "\n"
                                     "\\usepackage[braket, qm]{qcircuit}\n"
                                     "\\usepackage{graphicx}\n"
                                     "\n"
                                     "\\begin{document}\n"
                                     "\\scalebox{1.0}{\n"
                                     "\\Qcircuit @C = 1.0em @R = 0.2em @!R{ \\\\\n";

    const std::string LATEX_FOOTER = "\\\\ }}\n"
                                     "\\end{document}\n";

    std::string latex_qubit(uint64_t qid)
    {
        /*
          latex syntax:
          \ghost command is used to get the spacing
          and connections right. \ghost behaves like an invisible
          gate that allows the quantum wires on either side of your
          multigate to connect correctly.
 
          In addition it is possible to use a classical input to a
          gate with \cghost, or no input with \nghost.
 
          The \lstick command is used
          for input labels (on the left of the diagram), and the
          \rstick command is used for output labels (on the right
          of the diagram)
 
          @see latex package qcircuit tutorial
        */
        std::stringstream ss;
        ss << "\\nghost{{q}_{" << qid << "}:  \\ket{0}} & \\lstick{{q}_{" << qid << "}:  \\ket{0}}";
        return std::move(ss.str());
    }

    std::string latex_cbit(uint64_t qid)
    {
        std::stringstream ss;
        ss << "\\nghost{\\mathrm{{c}_{" << qid << "} :  0}} & \\lstick{\\mathrm{{c}_{" << qid << "} :  0}}";
        return std::move(ss.str());
    }

    std::string latex_ctrl(uint64_t ctrl, uint64_t target)
    {
        std::stringstream ss;
        ss << "\\ctrl{" << (int)target - (int)ctrl << "}";
        return std::move(ss.str());
    }

    /**
     * @brief generate single bits gate latex str
     * 
     * @param[in] gate_name name of gate 
     * @param[in] qbits     not used
     * @param[in] param     gate param
     * @param[in] is_dagger gate is dagger
     * @return gate latex string
     */
    std::vector<std::string> latex_single_bit_gate(const std::string &gate_name,
                                                   std::vector<size_t> qbits,
                                                   const std::string &param = "",
                                                   bool is_dagger = false)
    {
        std::string gate_latex = "\\gate{\\mathrm{" + gate_name + "}" +
                                 (!param.empty() ? "\\,\\mathrm{" + param + "}" : "") +
                                 (is_dagger ? "^\\dagger" : "") + "}";
        return {gate_latex};
    }

    /**
     * @brief generate mulit bits gate latex str
     * 
     * @param[in] gate_name name of gate 
     * @param[in] qbits     gate qbits id
     * @param[in] param     gate param
     * @param[in] is_dagger gate is dagger
     * @return gate latex string, matchs qbits 
     */
    std::vector<std::string> latex_multi_bits_gate(const std::string &gate_name,
                                                   std::vector<size_t> qbits,
                                                   const std::string &param = "",
                                                   bool is_dagger = false)
    {

        QPANDA_ASSERT(2 > qbits.size(), "Mulit bits gate have at least two target bits");

        std::vector<std::string> gate_latex;

        int row_span_offset = int(qbits.back()) - int(qbits.front());
        std::stringstream ss;

        ss << "\\multigate{" << row_span_offset << "}"
           << "{\\mathrm{" + gate_name + "}" << (!param.empty() ? "\\,(\\mathrm{" + param + "})}" : "}")
           << "_<<<{" << qbits.front() << "}";
        gate_latex.push_back(ss.str());

        for (size_t i = 1; i < qbits.size(); i++)
        {
            std::stringstream ss;
            ss << "\\ghost{\\mathrm{" + gate_name + "}" << (!param.empty() ? "\\,(\\mathrm{" + param + "})}" : "}")
               << "_<<<{" << qbits.at(i) << "}";
            gate_latex.push_back(ss.str());
        }
        return gate_latex;
    }

    /**
     * @brief generate ctrl gate latex str
     * 
     * @param[in] gate_name name of gate 
     * @param[in] qbits     gate qbits id
     * @param[in] param     gate param
     * @param[in] is_dagger gate is dagger
     * @return gate latex string, matchs qbits 
     */
    std::vector<std::string> latex_ctrl_gate(const std::string &gate_name,
                                             std::vector<size_t> qbits,
                                             const std::string &param = "",
                                             bool is_dagger = false)
    {
        QPANDA_ASSERT(2 != qbits.size(), "Ctrl gate should have two target bits");

        std::vector<std::string> gate_latex;
        /* qbits.front() is control bit, qbits.back() is target bit */
        uint64_t ctrl_bit = qbits.front();
        uint64_t target_bit = qbits.back();
        std::string ctrl_latex = latex_ctrl(ctrl_bit, target_bit);
        if ("CNOT" == gate_name)
        {
            gate_latex.emplace_back(ctrl_latex);
            gate_latex.emplace_back(LATEX_CNOT);
        }
        else
        {
            std::string gate = "\\gate{\\mathrm{" + gate_name + "}" +
                               (!param.empty() ? "\\,\\mathrm{" + param + "}" : "") +
                               (is_dagger ? "^\\dagger" : "") +
                               "}";
            gate_latex.emplace_back(ctrl_latex);
            gate_latex.emplace_back(gate);
        }
        return gate_latex;
    }

    /**
     * @brief generate swap gate latex str
     * 
     * @param[in] gate_name name of gate 
     * @param[in] qbits     gate qbits id
     * @param[in] param     gate param
     * @param[in] is_dagger gate is dagger
     * @return gate latex string, matchs qbits 
     */
    std::vector<std::string> latex_swap_gate(const std::string &gate_name,
                                             std::vector<size_t> qbits,
                                             const std::string &param = "",
                                             bool is_dagger = false)
    {
        QPANDA_ASSERT(2 != qbits.size(), "Swap gate should have two target bits");

        std::vector<std::string> gate_latex;
        uint64_t qbit1 = qbits.front();
        uint64_t qbit2 = qbits.back();

        int offset = (int)qbit1 - (int)qbit2;
        std::stringstream ss;
        ss << "\\qwx[" << offset << "]";
        std::string swap_qwx = LATEX_SWAP + ss.str();
        if (offset > 0)
        {
            gate_latex.emplace_back(swap_qwx);
            gate_latex.emplace_back(LATEX_SWAP);
        }
        else if (offset < 0)
        {
            gate_latex.emplace_back(LATEX_SWAP);
            gate_latex.emplace_back(swap_qwx);
        }
        else
        {
            QCERR_AND_THROW(std::runtime_error, "Swap self");
        }

        return gate_latex;
    }

    /**
     * @brief get measure to cbit latex statement
     * 
     * @param[in] cbit classic bit id
     * @param[in] qbit quantum bit id
     * @param[in] total_qbit_size total quantum bit size
     * @return measure to cbit latex statement
     */
    std::string latex_measure_to(uint64_t cbit, uint64_t qbit, size_t total_qbit_size)
    {
        std::stringstream ss;
        ss << "\\dstick{_{_{\\hspace{0.0em}" << cbit << "}}} \\cw \\ar @{<=} ["
           << (int)qbit - (int)cbit - (int)total_qbit_size << ", 0]";
        return std::move(ss.str());
    }

    std::string latex_barrier(size_t span_start, size_t span_end)
    {
        std::stringstream ss;
        ss << "\\barrier[0em]{" << span_end - span_start << "}";
        return std::move(ss.str());
    }

} // namespace LATEX TOOL

namespace // namespace utils
{
    auto compare_int_min = [](const int &q1, const int &q2)
    { return q1 < q2; };
} // namespace utils

QPANDA_BEGIN

DrawLatex::DrawLatex(const QProg &prog, LayeredTopoSeq &layer_info, uint32_t length)
    : AbstractDraw(prog, layer_info, length),
      m_latex_qwire(LATEX_QWIRE),
      m_latex_cwire(LATEX_CWIRE),
      m_latex_time_seq("")
{
}

void DrawLatex::init(std::vector<int> &qbits, std::vector<int> &cbits)
{
    /* insert qbits and cbits to latex matrix first col */
    std::vector<int>::iterator max_qid_it = std::max_element(qbits.begin(), qbits.end());
    std::vector<int>::iterator max_cid_it = std::max_element(cbits.begin(), cbits.end());
    int max_qid = -1;
    int max_cid = -1;
    if (max_qid_it != qbits.end())
        max_qid = *max_qid_it;
    if (max_cid_it != cbits.end())
        max_cid = *max_cid_it;

    for (int i = 0; i <= max_qid; i++)
    {
        m_latex_qwire.insert(i, 0, latex_qubit(i));
    }

    for (int i = 0; i <= max_cid; i++)
    {
        m_latex_cwire.insert(i, 0, latex_cbit(i));
    }
}

void DrawLatex::draw_by_layer()
{
    const auto &layer_info = m_layer_info;

    uint32_t layer_id = 0;
    for (auto seq_item_itr = layer_info.begin(); seq_item_itr != layer_info.end(); ++seq_item_itr, ++layer_id)
    {

        for (auto &seq_node_item : (*seq_item_itr))
        {
            auto opt_node_info = seq_node_item.first;
            append_node((DAGNodeType)(opt_node_info->m_type), opt_node_info, layer_id);
        }
    }
}

void DrawLatex::append_node(DAGNodeType t, pOptimizerNodeInfo &node_info, uint64_t layer_id)
{
    if (DAGNodeType::NUKNOW_SEQ_NODE_TYPE < t && t <= DAGNodeType::MAX_GATE_TYPE)
    {
        append_gate(node_info, layer_id);
    }
    else if (DAGNodeType::MEASURE == t)
    {
        append_measure(node_info, layer_id);
    }
    else if (DAGNodeType::RESET == t)
    {
        append_reset(node_info, layer_id);
    }
    else if (DAGNodeType::QUBIT == t)
    {
        QCERR_AND_THROW(std::runtime_error, "OptimizerNodeInfo shuould not contain qubits");
    }
    else
    {
        QCERR_AND_THROW(std::runtime_error, "OptimizerNodeInfo contains uknown nodes");
    }
}

int DrawLatex::update_layer_time_seq(int time_seq)
{
    m_layer_max_time_seq = std::max(m_layer_max_time_seq, time_seq);
    return m_layer_max_time_seq;
}

void DrawLatex::append_gate(pOptimizerNodeInfo &node_info, uint64_t layer_id)
{
    GateType gate_type = (GateType)(node_info->m_type);

    if (BARRIER_GATE == gate_type)
    {
        /* 
          in qpanda barrier is a special single bit gate
          we process barrier in different way
          and barrier won't comsume time
        */
        return append_barrier(node_info, layer_id);
    }

    QVec qubits_vec = node_info->m_target_qubits;
    QVec ctrl_vec = node_info->m_control_qubits;

    std::shared_ptr<AbstractQGateNode> p_gate = std::dynamic_pointer_cast<AbstractQGateNode>(*(node_info->m_iter));

    /* get gate name */
    std::string gate_name = TransformQGateType::getInstance()[gate_type];

    /* get gate parameter */
    std::string gate_param;
    get_gate_parameter(p_gate, gate_param);

    /* get dagger */
    bool is_dagger = check_dagger(p_gate, p_gate->isDagger());

    int gate_time_seq = 0;
    std::vector<std::string> gate_latex_str;

    /* get qbits id */
    std::vector<size_t> qubits_id;
    std::for_each(qubits_vec.begin(), qubits_vec.end(), [&qubits_id](const Qubit *qbit)
                  { qubits_id.emplace_back(qbit->getPhysicalQubitPtr()->getQubitAddr()); });

    /* get gate span rows */
    std::vector<size_t> used_bits(qubits_id);
    for (auto q_ptr : ctrl_vec)
    {
        used_bits.emplace_back(q_ptr->getPhysicalQubitPtr()->getQubitAddr());
    }
    auto target_span = std::minmax_element(used_bits.begin(), used_bits.end(), compare_int_min);
    size_t span_start = *target_span.first;
    size_t span_end = *target_span.second;

    /* get gate latex matrix dst col */
    size_t gate_col = get_dst_col(layer_id, span_start, span_end);

    /* trick for put gate span qwires placeholder in matrix, then next gate will not appear at occupied qwires */
    for (size_t r = span_start; r <= span_end; r++)
    {
        /* 
          if qubit is not used, m_latex_qwire[r][0] should be default
          then not marked m_latex_qwire[r][gate_col] as span row
        */
        // if (!m_latex_qwire.is_empty(r, 0))
        // {
        m_latex_qwire.insert(r, gate_col, LATEX_QWIRE);
        // }
    }

    switch (gate_type)
    {
    case ISWAP_THETA_GATE:
    case ISWAP_GATE:
    case SQISWAP_GATE:
    case SWAP_GATE:
    {
        gate_latex_str = latex_swap_gate(gate_name, qubits_id, gate_param, is_dagger);
        /*
          record layer time sequnce
          FIXME:this time calcutaion code is from DraPicture, I can't explain why,
          why add control bits size multiplied with swap gate time? 
        */
        gate_time_seq = m_time_sequence_conf.get_swap_gate_time_sequence() * (ctrl_vec.size() + 1);
        break;
    }
    case CU_GATE:
    case CNOT_GATE:
    case CZ_GATE:
    case CP_GATE:
    case CPHASE_GATE:
    {
        if (CPHASE_GATE == gate_type)
        {
            gate_name = "CR";
        }
        gate_latex_str = latex_ctrl_gate(gate_name, qubits_id, gate_param, is_dagger);
        gate_time_seq = m_time_sequence_conf.get_ctrl_node_time_sequence() * (ctrl_vec.size() + 1);
        break;
    }
    case TWO_QUBIT_GATE:
    case TOFFOLI_GATE:
    case ORACLE_GATE:
    {
        gate_latex_str = latex_multi_bits_gate(gate_name, qubits_id, gate_param, is_dagger);
        break;
    }
    case BARRIER_GATE:
    {
        QCERR_AND_THROW(std::runtime_error, "BARRIER_GATE should be processd in another way");
        break;
    }
        /* 
          single target bit gate:
          P0_GATE,             
          P1_GATE,           
          PAULI_X_GATE,  
          PAULI_Y_GATE,  
          PAULI_Z_GATE,  
          X_HALF_PI,     
          Y_HALF_PI,     
          Z_HALF_PI,     
          P_GATE,        
          HADAMARD_GATE, 
          T_GATE,        
          S_GATE,        
          RX_GATE,       
          RY_GATE,       
          RZ_GATE,       
          RPHI_GATE,
          U1_GATE,          
          U2_GATE,          
          U3_GATE,          
          U4_GATE,          
          P00_GATE,
          P11_GATE,
          I_GATE,
          ECHO_GATE,   
        */
    default:
        gate_latex_str = latex_single_bit_gate(gate_name, qubits_id, gate_param, is_dagger);
        gate_time_seq = ctrl_vec.size() > 0 ? (m_time_sequence_conf.get_ctrl_node_time_sequence() * ctrl_vec.size()) : m_time_sequence_conf.get_single_gate_time_sequence();
        break;
    }

    /* insert gate latex statement to matrix, row = qibit id */
    for (size_t i = 0; i < qubits_id.size(); ++i)
    {
        size_t gate_row = qubits_id.at(i);
        /* gate_latex_str matchs qubits_id sequnce */
        m_latex_qwire.insert(gate_row, gate_col, gate_latex_str.at(i));
    }

    /* insert crtl latex statement to matrix */
    size_t ctrl_col = gate_col;
    size_t target_qubit_id = qubits_vec.front()->getPhysicalQubitPtr()->getQubitAddr();
    for (const auto qbit : ctrl_vec)
    {
        size_t ctrl_qbit_id = qbit->getPhysicalQubitPtr()->getQubitAddr();
        std::string gate_ctrl_latex_str = latex_ctrl(ctrl_qbit_id, target_qubit_id);
        uint64_t ctrl_row = ctrl_qbit_id;
        m_latex_qwire.insert(ctrl_row, ctrl_col, gate_ctrl_latex_str);
    }

    /* update curent layer latex matrix col range */
    m_layer_col_range[layer_id] = std::max(gate_col, m_layer_col_range[layer_id]);
    /* record layer time sequnce*/
    update_layer_time_seq(gate_time_seq);
}

void DrawLatex::append_barrier(pOptimizerNodeInfo &node_info, uint64_t layer_id)
{
    /* get target bits */
    QVec qubits_vec = node_info->m_target_qubits;
    QPANDA_ASSERT(qubits_vec.size() != 1, "barrier should only have one target bit");

    /* get control info */
    QVec ctrl_vec = node_info->m_control_qubits;

    /* barrier is special single bit gate in qpanda, qubits_vec and ctrl_vec contains all qibits be barriered */
    /* gather barrier sapn qubit rows */
    std::vector<size_t> span_id;
    std::for_each(qubits_vec.begin(), qubits_vec.end(), [&span_id](const Qubit *qbit)
                  { span_id.push_back(qbit->getPhysicalQubitPtr()->getQubitAddr()); });
    std::for_each(ctrl_vec.begin(), ctrl_vec.end(), [&span_id](const Qubit *qbit)
                  { span_id.push_back(qbit->getPhysicalQubitPtr()->getQubitAddr()); });
    auto target_span = std::minmax_element(span_id.begin(), span_id.end(), compare_int_min);
    size_t span_start = *target_span.first;
    size_t span_end = *target_span.second;

    size_t barrier_col = get_dst_col(layer_id, span_start, span_end);

    /* 
      barrier is special in latex.
      for current col barrier, it's latex statment "\barrier" is append to gate or qwire last col
      like "\qw \barrier[0em]{1}"
      barrier always append to latex content before current col, so minus 1
    */
    barrier_col -= 1;

    size_t barrier_row = qubits_vec.front()->getPhysicalQubitPtr()->getQubitAddr();

    std::string barrier_latex = latex_barrier(span_start, span_end);

    /* insert barrier to matrix */
    if (m_latex_qwire.is_empty(barrier_row, barrier_col))
    {
        barrier_latex = LATEX_QWIRE + " " + barrier_latex;
    }
    else
    {
        barrier_latex = m_latex_qwire[barrier_row][barrier_col] + " " + barrier_latex;
    }
    m_latex_qwire.insert(barrier_row, barrier_col, barrier_latex);

    /* 
      record curent layer end at latex matrix col, 
      barrier should occupy barrier_col + 1 for barrirer_col had minused 1
    */
    m_layer_col_range[layer_id] = std::max(barrier_col + 1, m_layer_col_range[layer_id]);
}

void DrawLatex::append_measure(pOptimizerNodeInfo &node_info, uint64_t layer_id)
{
    std::shared_ptr<AbstractQuantumMeasure> p_measure = std::dynamic_pointer_cast<AbstractQuantumMeasure>(*(node_info->m_iter));

    size_t qbit_id = p_measure->getQuBit()->getPhysicalQubitPtr()->getQubitAddr();
    size_t cbit_id = p_measure->getCBit()->get_addr();

    size_t qbits_row = m_latex_qwire.max_row();
    size_t total_qbits_size = qbits_row ? qbits_row + 1 : 0;

    size_t meas_col = get_dst_col(layer_id, qbit_id, qbits_row);

    m_latex_qwire.insert(qbit_id, meas_col, LATEX_MEASURE);

    /* trick for put measure span qwires placeholder in matrix */
    for (size_t r = qbit_id + 1; r <= qbits_row; r++)
    {
        /* 
          if qubit is not used, m_latex_qwire[r][0] should be default
          then not marked m_latex_qwire[r][gate_col] as span row
        */
        // if (!m_latex_qwire.is_empty(r, 0))
        // {
        m_latex_qwire.insert(r, meas_col, LATEX_QWIRE);
        // }
    }

    m_latex_cwire.insert(cbit_id, meas_col, latex_measure_to(cbit_id, qbit_id, total_qbits_size));

    /* record curent layer end at latex matrix col */
    m_layer_col_range[layer_id] = std::max(meas_col, m_layer_col_range[layer_id]);

    update_layer_time_seq(m_time_sequence_conf.get_measure_time_sequence());
}

void DrawLatex::append_reset(pOptimizerNodeInfo &node_info, uint64_t layer_id)
{
    std::shared_ptr<AbstractQuantumReset> p_reset = std::dynamic_pointer_cast<AbstractQuantumReset>(*(node_info->m_iter));

    int qubit_index = p_reset->getQuBit()->getPhysicalQubitPtr()->getQubitAddr();

    size_t gate_col = get_dst_col(layer_id, qubit_index, qubit_index);

    m_latex_qwire.insert(qubit_index, gate_col, LATEX_RESET);

    /* record curent layer end at latex matrix col */
    m_layer_col_range[layer_id] = std::max(gate_col, m_layer_col_range[layer_id]);

    update_layer_time_seq(m_time_sequence_conf.get_reset_time_sequence());
}

void DrawLatex::draw_by_time_sequence(const std::string config_data /*= CONFIG_PATH*/)
{
    m_time_sequence_conf.load_config(config_data);

    const auto &layer_info = m_layer_info;

    int time_seq = 0;
    m_latex_time_seq.insert(0, 0, "\\nghost{\\mathrm{time :}}&\\lstick{\\mathrm{time :}}");
    size_t qbit_max_row = m_latex_qwire.max_row();
    std::stringstream ss;
    std::string time_seq_str;

    uint32_t layer_id = 0;
    for (auto seq_item_itr = layer_info.begin(); seq_item_itr != layer_info.end(); ++seq_item_itr, ++layer_id)
    {
        m_layer_max_time_seq = 0;
        for (auto &seq_node_item : (*seq_item_itr))
        {
            auto opt_node_info = seq_node_item.first;
            append_node((DAGNodeType)(opt_node_info->m_type), opt_node_info, layer_id);
        }
        time_seq += m_layer_max_time_seq;
        size_t time_col = m_layer_col_range.at(layer_id);
        ss.clear();
        ss << time_seq;
        time_seq_str.clear();
        ss >> time_seq_str;
        m_latex_time_seq.insert(0, time_col, time_seq_str);
    }
}

std::string DrawLatex::present(const std::string &file_name)
{
    align_matrix_col();

    /* add two empty wire to extend right border, beautfier latex output */
    m_latex_qwire.max_col() += 2;
    m_latex_cwire.max_col() += 2;

    std::string out_str(LATEX_HEADER);

    for (const auto &row : m_latex_qwire)
    {
        for (const auto &elem : row)
        {
            /* add "&" seperate matrix element to format latex array */
            out_str += elem + "&";
        }
        /* elemiate last '&' */
        out_str.pop_back();
        out_str += "\\\\\n";
    }

    for (auto row : m_latex_cwire)
    {
        for (auto elem : row)
        {
            out_str += elem + "&";
        }
        /* elemiate last '&' */
        out_str.pop_back();
        out_str += "\\\\\n";
    }

    /* if draw_by_time_sequnce not be called, first element should be empty */
    if (!m_latex_time_seq.is_empty(0, 0))
    {
        for (auto row : m_latex_time_seq)
        {
            for (auto elem : row)
            {
                out_str += elem + "&";
            }
            /* elemiate last '&' */
            out_str.pop_back();
            out_str += "\\\\\n";
        }
    }

    out_str += LATEX_FOOTER;

    std::fstream f(file_name, std::ios_base::out);
    f << out_str;
    f.close();

    return out_str;
}

size_t DrawLatex::get_dst_col(size_t layer_id, size_t span_start, size_t span_end)
{
    /* 
      trans layer to destination latex matrix col
      if layer == 0, col = layer + 1 for the first latex matrix col contains qubits symbol.
      else gates in current layer start col after last layer col
    */
    uint64_t gate_col = layer_id == 0 ? layer_id + 1 : m_layer_col_range.at(layer_id - 1) + 1;
    /* 
      as gates in same layer may cross much col,
      for gate with ctrl bits may cross whole qwires, other gate in same layer can't be placed at same col
      gates in same layer may spread at multi cols,
      so find valid zone to put gate, return the col num
    */
    return find_valid_matrix_col(span_start, span_end, gate_col);
}

void DrawLatex::align_matrix_col()
{
    size_t &qwire_col = m_latex_qwire.max_col();
    size_t &cwire_col = m_latex_cwire.max_col();
    size_t max_col = std::max(qwire_col, cwire_col);
    qwire_col = max_col;
    cwire_col = max_col;
    m_latex_time_seq.max_col() = max_col;
}

size_t DrawLatex::find_valid_matrix_col(size_t span_start, size_t span_end, size_t col)
{
    for (size_t r = span_start; r <= span_end; ++r)
    {
        if (!m_latex_qwire.is_empty(r, col))
        {
            return find_valid_matrix_col(span_start, span_end, ++col);
        }
    }
    return col;
}

QPANDA_END
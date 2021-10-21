#include "Core/Utilities/QProgInfo/Visualization/DrawLatex.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"
#include "Core/QuantumCircuit/QuantumMeasure.h"
#include <sstream>
#include <memory>
#include <fstream>

namespace
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

    std::string latex_gate_ctrl(uint64_t ctrl, uint64_t target)
    {
        std::stringstream ss;
        ss << "\\ctrl{" << (int)target - (int)ctrl << "}";
        return std::move(ss.str());
    }

    /**
     * @brief generate mulit/single bits gate latex str
     * 
     * @param[in] gate_name name of gate 
     * @param[in] qbits     gate ops gbits ids
     * @param[in] param     gate param
     * @param[in] is_dagger gate is dagger
     * @return gate latex string, matchs qbits 
     */
    std::vector<std::string> latex_gate(const std::string &gate_name,
                                        std::vector<uint64_t> qbits,
                                        const std::string &param = "",
                                        bool is_dagger = false)
    {
        std::vector<std::string> gate_latex;
        std::string gate;

        if (1 == qbits.size())
        {
            gate = "\\gate{\\mathrm{" + gate_name + "}" +
                   (!param.empty() ? "\\,\\mathrm{" + param + "}" : "") +
                   (is_dagger ? "^\\dagger" : "") +
                   "}";
            gate_latex.emplace_back(gate);
        }
        else if (2 == qbits.size())
        {
            /* qbits.front() is control bit, qbits.back() is target bit */
            uint64_t ctrl_bit = qbits.front();
            uint64_t target_bit = qbits.back();
            std::string ctrl_latex = latex_gate_ctrl(ctrl_bit, target_bit);
            if ("CNOT" == gate_name)
            {
                gate_latex.emplace_back(ctrl_latex);
                gate_latex.emplace_back(LATEX_CNOT);
            }
            else if ("SWAP" == gate_name)
            {
                int offset = (int)ctrl_bit - (int)target_bit;
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
            }
            else
            {
                gate = "\\gate{\\mathrm{" + gate_name + "}" +
                       (!param.empty() ? "\\,\\mathrm{" + param + "}" : "") +
                       (is_dagger ? "^\\dagger" : "") +
                       "}";
                gate_latex.emplace_back(ctrl_latex);
                gate_latex.emplace_back(gate);
            }
        }
        else
        {
            QCERR_AND_THROW(std::runtime_error, "Not implemented for gate operator on more than 2 qubit");
        }

        return gate_latex;
    }

    std::string latex_measure(uint64_t cbit, uint64_t qbit, size_t qbit_size)
    {
        std::stringstream ss;
        ss << "\\dstick{_{_{\\hspace{0.0em}" << cbit << "}}} \\cw \\ar @{<=} ["
           << (int)qbit - (int)cbit - (int)qbit_size << ", 0]";
        return std::move(ss.str());
    }

    std::string latex_barrier(size_t span_start, size_t span_end)
    {
        std::stringstream ss;
        ss << "\\barrier[0em]{" << span_end - span_start << "}";
        return std::move(ss.str());
    }

} // namespace LATEX TOOL

namespace
{
    auto compare_qbitid_min = [](const int &q1, const int &q2)
    { return q1 < q2; };
} // namespace

QPANDA_BEGIN

DrawLatex::DrawLatex(const QProg &prog, LayeredTopoSeq &layer_info, uint32_t length)
    : AbstractDraw(prog, layer_info, length),
      m_latex_qwire(LATEX_QWIRE),
      m_latex_cwire(LATEX_CWIRE)
{
}

void DrawLatex::init(std::vector<int> &quBits, std::vector<int> &clBits)
{
    for (auto i : quBits)
    {
        m_latex_qwire.insert(i, 0, latex_qubit(i));
    }

    for (auto i : clBits)
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
        QCERR_AND_THROW(std::runtime_error, "OptimizerNodeInfo contains uknown types");
    }
}

void DrawLatex::append_gate(pOptimizerNodeInfo &node_info, uint64_t layer_id)
{
    GateType gate_type = (GateType)(node_info->m_type);
    /* in qpanda barrier is a special gate */
    if (GateType::BARRIER_GATE == gate_type)
    {
        return append_barrier(node_info, layer_id);
    }

    std::shared_ptr<AbstractQGateNode> p_gate = std::dynamic_pointer_cast<AbstractQGateNode>(*(node_info->m_iter));

    /* get target bits */
    QVec qubits_vec = node_info->m_target_qubits;

    /* get control info */
    QVec cbits_vec = node_info->m_control_qubits;

    /* get gate name */
    std::string gate_name = TransformQGateType::getInstance()[gate_type];

    if (GateType::CPHASE_GATE == gate_type)
    {
        gate_name = "CR";
    }

    /* get gate parameter */
    std::string gate_param;
    get_gate_parameter(p_gate, gate_param);

    /* get dagger */
    bool is_dagger = check_dagger(p_gate, node_info->m_is_dagger);

    /* get gate latex */
    std::vector<uint64_t> qubits_id;
    for (auto q_ptr : qubits_vec)
    {
        qubits_id.emplace_back(q_ptr->getPhysicalQubitPtr()->getQubitAddr());
    }
    std::vector<std::string> gate_latex_str = latex_gate(gate_name, qubits_id, gate_param, is_dagger);

    /* get gate span rows */
    std::vector<uint64_t> used_bits(qubits_id);
    for (auto q_ptr : cbits_vec)
    {
        used_bits.emplace_back(q_ptr->getPhysicalQubitPtr()->getQubitAddr());
    }
    auto target_span = std::minmax_element(used_bits.begin(), used_bits.end(), compare_qbitid_min);
    size_t span_start = *target_span.first;
    size_t span_end = *target_span.second;

    /* layer start form 0, col = layer level add 1 for the first col is qubits. */
    uint64_t gate_col = layer_id + 1;
    gate_col = find_valid_matrix_col(span_start, span_end, gate_col);

    /* trick for put placeholder in matrix for gate span qwires */
    for (size_t r = span_start; r <= span_end; r++)
    {
        m_latex_qwire.insert(r, gate_col, LATEX_QWIRE);
    }

    for (size_t i = 0; i < qubits_id.size(); ++i)
    {
        uint64_t gate_row = qubits_id.at(i);
        m_latex_qwire.insert(gate_row, gate_col, gate_latex_str.at(i));
    }

    /* insert crtl */
    uint64_t ctrl_col = gate_col;
    size_t target_qubit_id = qubits_vec.front()->getPhysicalQubitPtr()->getQubitAddr();
    for (const auto qbit : cbits_vec)
    {
        size_t ctrl_qbit_id = qbit->getPhysicalQubitPtr()->getQubitAddr();
        std::string gate_ctrl_latex_str = latex_gate_ctrl(ctrl_qbit_id, target_qubit_id);
        uint64_t ctrl_row = ctrl_qbit_id;
        m_latex_qwire.insert(ctrl_row, ctrl_col, gate_ctrl_latex_str);
    }
}

void DrawLatex::append_measure(pOptimizerNodeInfo &node_info, uint64_t layer_id)
{
    std::shared_ptr<AbstractQuantumMeasure> p_measure = std::dynamic_pointer_cast<AbstractQuantumMeasure>(*(node_info->m_iter));

    int qbit_id = p_measure->getQuBit()->getPhysicalQubitPtr()->getQubitAddr();
    int cbit_id = p_measure->getCBit()->get_addr();

    size_t qbits_row = m_latex_qwire.max_row();
    size_t qbits_size = qbits_row ? qbits_row + 1 : 0;

    uint64_t meas_col = layer_id + 1;
    meas_col = find_valid_matrix_col(qbit_id, qbits_row, meas_col);

    m_latex_qwire.insert(qbit_id, meas_col, LATEX_MEASURE);

    /* trick for put placeholder in matrix for measure span qwires */
    for (size_t r = qbit_id + 1; r <= qbits_row; r++)
    {
        m_latex_qwire.insert(r, meas_col, LATEX_QWIRE);
    }

    m_latex_cwire.insert(cbit_id, meas_col, latex_measure(cbit_id, qbit_id, qbits_size));
}

void DrawLatex::append_reset(pOptimizerNodeInfo &node_info, uint64_t layer_id)
{
    std::shared_ptr<AbstractQuantumReset> p_reset = std::dynamic_pointer_cast<AbstractQuantumReset>(*(node_info->m_iter));

    int qubit_index = p_reset->getQuBit()->getPhysicalQubitPtr()->getQubitAddr();
    /* layer start form 0, col = layer level add 1 for the first col is qubits. */
    uint64_t gate_col = layer_id + 1;
    gate_col = find_valid_matrix_col(qubit_index, qubit_index, gate_col);

    m_latex_qwire.insert(qubit_index, gate_col, LATEX_RESET);
}

void DrawLatex::append_barrier(pOptimizerNodeInfo &node_info, uint64_t layer_id)
{
    /* get target bits */
    QVec qubits_vec = node_info->m_target_qubits;
    QPANDA_ASSERT(qubits_vec.size() != 1, "measure should only have one target bit");

    /* get control info */
    QVec cbits_vec = node_info->m_control_qubits;

    // std::vector<size_t> qbits_id, cbits_id;

    /* 
      barrier is specil in latex, it is append to last gate or qwire
    */
    std::vector<size_t> span_id;
    std::for_each(qubits_vec.begin(), qubits_vec.end(), [&span_id](const Qubit *qbit)
                  { span_id.push_back(qbit->getPhysicalQubitPtr()->getQubitAddr()); });
    std::for_each(cbits_vec.begin(), cbits_vec.end(), [&span_id](const Qubit *qbit)
                  { span_id.push_back(qbit->getPhysicalQubitPtr()->getQubitAddr()); });
    // std::for_each(cbits_id.begin(), cbits_id.end(), [&span_id](const size_t &id)
    //               { span_id.push_back(id); });
    auto target_span = std::minmax_element(span_id.begin(), span_id.end(), compare_qbitid_min);
    size_t span_start = *target_span.first;
    size_t span_end = *target_span.second;

    uint64_t barrier_col = layer_id + 1;
    barrier_col = find_valid_matrix_col(span_start, span_end, barrier_col);
    /* 
     barrier always append to latex content before current 
     like "\qw \barrier[0em]{1}"
    */
    barrier_col -= 1;

    size_t barrier_row = qubits_vec.front()->getPhysicalQubitPtr()->getQubitAddr();

    std::string barrier_latex = latex_barrier(span_start, span_end);
    if (m_latex_qwire.is_empty(barrier_row, barrier_col))
    {
        barrier_latex = LATEX_QWIRE + " " + barrier_latex;
    }
    else
    {
        barrier_latex = m_latex_qwire[barrier_row][barrier_col] + " " + barrier_latex;
    }

    m_latex_qwire.insert(barrier_row, barrier_col, barrier_latex);
}

void DrawLatex::draw_by_time_sequence(const std::string config_data)
{
    QCERR_AND_THROW(std::runtime_error, "Not implemented yet");
}

std::string DrawLatex::present(const std::string &file_name)
{
    align_matrix_col();
    std::string out_str(LATEX_HEADER);

    for (auto &row : m_latex_qwire)
    {
        for (const auto &elem : row)
        {
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

    out_str += LATEX_FOOTER;

    std::fstream f(file_name, std::ios_base::out);
    f << out_str;
    f.close();

    return out_str;
}

void DrawLatex::align_matrix_col()
{
    size_t &qwire_col = m_latex_qwire.max_col();
    size_t &cwire_col = m_latex_cwire.max_col();
    size_t max_col = (qwire_col > cwire_col) ? qwire_col : cwire_col;
    /* add two empty wire to extend right border, beautfier latex output */
    max_col += 2;
    qwire_col = max_col;
    cwire_col = max_col;
}

size_t DrawLatex::find_valid_matrix_col(size_t span_start, size_t span_end, size_t col)
{
    for (size_t r = span_start; r <= span_end; ++r)
    {
        if (!m_latex_qwire.is_empty(r, col))
        {
            ++col;
            return find_valid_matrix_col(span_start, span_end, col);
        }
    }
    return col;
}

QPANDA_END
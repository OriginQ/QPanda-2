#include "Core/Utilities/QProgInfo/Visualization/DrawLatex.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"
#include "Core/QuantumCircuit/QuantumMeasure.h"
#include <sstream>
#include <memory>
#include <fstream>

namespace LATEX_SYNTAX
{
    /*
      latex syntax based on qcircuit package
      see qcircuit tutorial [https://physics.unm.edu/CQuIC/Qcircuit/Qtutorial.pdf]
    */
    const std::string LATEX_QWIRE = "\\qw";
    const std::string LATEX_CWIRE = "\\cw";

    /* special gate symbol */
    const std::string LATEX_SWAP = "\\qswap";
    const std::string LATEX_CNOT = "\\targ";
    const std::string LATEX_MEASURE = "\\meter";
    const std::string LATEX_RESET = "\\gate{\\mathrm{\\left|0\\right\\rangle}}";

    const std::string LATEX_FOOTER = "\\\\ }}\n"
                                     "\\end{document}\n";

    std::string latex_header(const std::string &logo)
    {
        return "\\documentclass[border=2px]{standalone}\n"
               "\n"
               "\\usepackage[braket, qm]{qcircuit}\n"
               "\\usepackage{graphicx}\n"
               "\n"
               "\\begin{document}\n" +
               (logo.empty() ? "" : logo + "\\\\\n\\\\\n\\\\\n\\\\\n") +
               "\\scalebox{1.0}{\n"
               "\\Qcircuit @C = 1.0em @R = 0.2em @!R{ \\\\\n";
    }

    std::string latex_qwire_head_label(const std::string &label)
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

          see latex package qcircuit tutorial [https://physics.unm.edu/CQuIC/Qcircuit/Qtutorial.pdf]
        */
        std::string s = "\\nghost{" + label + "  \\ket{0}}" +
                        " & " +
                        "\\lstick{" + label + "  \\ket{0}}";

        return s;
    }

    std::string latex_cwire_head_label(const std::string &label)
    {
        std::string s = "\\nghost{" + label + "  0}" +
                        " & " +
                        "\\lstick{\\mathrm{" + label + "  0}}";
        return s;
    }

    std::string latex_time_head_label(const std::string &label)
    {
        std::string s = "\\nghost{" + label + "}" +
                        " & " +
                        "\\lstick{\\mathrm{" + label + "}}";
        return s;
    }

    std::string latex_qwire_tail_label(const std::string &label, const std::string &sub_symbol = "")
    {
        std::string s = "\\rstick{" + label + "}\\qw" +
                        " & " +
                        "\\nghost{" + label + "}";
        return s;
    }

    std::string latex_cwire_tail_label(const std::string &main_symbol, const std::string &sub_symbol = "")
    {
        std::string s = "\\rstick{\\mathrm{" + main_symbol + "}}\\cw" +
                        " & " +
                        "\\nghost{" + main_symbol + "}";
        return s;
    }

    std::string latex_time_tail_label(const std::string &label)
    {
        std::string s = "\\rstick{\\mathrm{" + label + "}}" +
                        " & " +
                        "\\nghost{" + label + "}";
        return s;
    }

    std::string latex_ctrl(uint64_t ctrl_row, uint64_t target_row)
    {
        std::stringstream ss;
        ss << "\\ctrl{" << (int)target_row - (int)ctrl_row << "}";
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
    std::unordered_map<uint64_t, std::string> latex_single_bit_gate(const std::string &gate_name,
                                                                    std::set<uint64_t> tag_rows,
                                                                    std::set<uint64_t> ctrl_rows,
                                                                    const std::string &param = "",
                                                                    bool is_dagger = false)
    {
        assert(tag_rows.size() == 1);
        std::unordered_map<uint64_t, std::string> gate_latex;
        uint64_t tag_row = *tag_rows.begin();
        std::string tag_latex = "\\gate{\\mathrm{" + gate_name + "}" +
                                (!param.empty() ? "\\,\\mathrm{" + param + "}" : "") +
                                (is_dagger ? "^\\dagger" : "") + "}";
        gate_latex[tag_row] = tag_latex;

        for (auto row : ctrl_rows)
        {
            gate_latex[row] = latex_ctrl(row, tag_row);
        }

        return gate_latex;
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
    std::unordered_map<uint64_t, std::string> latex_multi_bits_gate(const std::string &gate_name,
                                                                    std::set<uint64_t> tag_rows,
                                                                    std::set<uint64_t> ctrl_rows,
                                                                    const std::string &param = "",
                                                                    bool is_dagger = false)
    {

        assert(tag_rows.size() > 1);

        std::unordered_map<uint64_t, std::string> gate_latex;

        uint64_t tag_first = *tag_rows.begin();
        uint64_t tag_last = *tag_rows.rbegin();
        int tag_span_offset = int(tag_last) - int(tag_first);

        for (uint64_t row = tag_first; row <= tag_last; row++)
        {
            std::stringstream ss;
            if (row == tag_first)
            {
                ss << "\\multigate{" << tag_span_offset << "}"
                   << "{\\mathrm{" + gate_name + "}"
                   << (!param.empty() ? "\\,(\\mathrm{" + param + "})" : "")
                   << (is_dagger ? "^\\dagger" : "")
                   << "}"
                   /* label row id at multigate input qwire */
                   << "_<<<{" << row << "}";
                gate_latex[row] = ss.str();
            }
            else
            {
                ss << "\\ghost{\\mathrm{" + gate_name + "}"
                   << (!param.empty() ? "\\,(\\mathrm{" + param + "})" : "")
                   << (is_dagger ? "^\\dagger" : "")
                   << "}";
                if (tag_rows.count(row))
                {
                    ss << "_<<<{" << row << "}";
                }

                gate_latex[row] = ss.str();
            }
        }

        for (auto row : ctrl_rows)
        {
            gate_latex[row] = latex_ctrl(row, tag_first);
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
    std::unordered_map<uint64_t, std::string> latex_cnot_gate(const std::string &gate_name,
                                                              std::set<uint64_t> tag_rows,
                                                              std::set<uint64_t> ctrl_rows,
                                                              const std::string &param = "",
                                                              bool is_dagger = false)
    {
        assert(tag_rows.size() == 1);

        std::unordered_map<uint64_t, std::string> gate_latex;
        uint64_t tag_row = *tag_rows.begin();

        gate_latex[tag_row] = LATEX_CNOT;

        for (auto row : ctrl_rows)
        {
            gate_latex[row] = latex_ctrl(row, tag_row);
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
    std::unordered_map<uint64_t, std::string> latex_swap_gate(const std::string &gate_name,
                                                              std::set<uint64_t> tag_rows,
                                                              std::set<uint64_t> ctrl_rows,
                                                              const std::string &param = "",
                                                              bool is_dagger = false)
    {
        assert(tag_rows.size() == 2);

        std::unordered_map<uint64_t, std::string> gate_latex;
        uint64_t tag_row_1 = *tag_rows.begin();
        uint64_t tag_row_2 = *tag_rows.rbegin();

        int offset = (int)tag_row_1 - (int)tag_row_2;
        assert(offset < 0);
        std::stringstream ss;
        ss << "\\qwx[" << offset << "]";
        std::string swap_qwx = LATEX_SWAP + ss.str();
        gate_latex[tag_row_1] = LATEX_SWAP;
        gate_latex[tag_row_2] = swap_qwx;

        for (auto row : ctrl_rows)
        {
            gate_latex[row] = latex_ctrl(row, tag_row_1);
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
    std::string latex_measure_to(uint64_t c_row, uint64_t q_row, uint64_t row_size)
    {
        std::stringstream ss;
        ss << "\\dstick{_{_{\\hspace{0.0em}" << c_row << "}}} \\cw \\ar @{<=} ["
           << (int)q_row - (int)row_size - (int)c_row << ", 0]";
        return std::move(ss.str());
    }

    std::string latex_barrier(size_t row_start, size_t row_end)
    {
        std::stringstream ss;
        ss << "\\barrier[0em]{" << row_end - row_start << "}";
        return std::move(ss.str());
    }

} // namespace

namespace // namespace utils
{
    std::vector<std::set<uint64_t>> sliceToContinuousSeq(std::set<uint64_t> vec)
    {
        std::vector<std::set<uint64_t>> sliced_seq;
        for (auto it : vec)
        {
            if (sliced_seq.empty())
            {
                sliced_seq.emplace_back(std::set<uint64_t>{it});
            }
            else
            {
                bool continus = (1 == it - *sliced_seq.back().rbegin());
                if (continus)
                {
                    sliced_seq.back().insert(it);
                }
                else
                {
                    sliced_seq.emplace_back(std::set<uint64_t>{it});
                }
            }
        }
        return sliced_seq;
    }
} // namespace utils
/*---------------------------------------------------------------------------*/
QPANDA_BEGIN

LatexMatrix::LatexMatrix()
    : m_latex_qwire(LATEX_SYNTAX::LATEX_QWIRE),
      m_latex_cwire(LATEX_SYNTAX::LATEX_CWIRE),
      m_latex_time_seq(""),
      m_latex_qwire_head(LATEX_SYNTAX::latex_qwire_head_label("")),
      m_latex_cwire_head(LATEX_SYNTAX::latex_cwire_head_label("")),
      m_latex_time_seq_head(LATEX_SYNTAX::latex_time_head_label("time")),
      m_latex_qwire_tail(LATEX_SYNTAX::latex_qwire_tail_label("")),
      m_latex_cwire_tail(LATEX_SYNTAX::latex_cwire_tail_label("")),
      m_latex_time_seq_tail(LATEX_SYNTAX::latex_time_tail_label("")),
      m_barrier_mark_head(""),
      m_barrier_mark_qwire("")
{
}

void LatexMatrix::setLabel(const Label &qubit_label, const Label &cbit_label /* ={}*/, const TimeSeqLabel &time_seq_label /* = ""*/, bool head /*=true*/)
{

    for (auto it : qubit_label)
    {
        Row row = it.first;
        auto &label = it.second;
        if (head)
        {
            m_latex_qwire_head.insert(row, 0, LATEX_SYNTAX::latex_qwire_head_label(label));
        }
        else
        {
            m_latex_qwire_tail.insert(row, 0, LATEX_SYNTAX::latex_qwire_tail_label(label));
        }
    }

    for (auto it : cbit_label)
    {
        Row row = it.first;
        auto &label = it.second;
        if (head)
        {
            m_latex_cwire_head.insert(row, 0, LATEX_SYNTAX::latex_cwire_head_label(label));
        }
        else
        {
            m_latex_cwire_tail.insert(row, 0, LATEX_SYNTAX::latex_cwire_tail_label(label));
        }
    }

    if (!time_seq_label.empty())
    {
        if (head)
        {
            m_latex_time_seq_head.insert(0, 0, LATEX_SYNTAX::latex_time_head_label(time_seq_label));
        }
        else
        {
            m_latex_time_seq_tail.insert(0, 0, LATEX_SYNTAX::latex_time_tail_label(time_seq_label));
        }
    }
}

void LatexMatrix::setLogo(const std::string &logo)
{
    m_logo = logo;
}

LatexMatrix::Col LatexMatrix::insertGate(const std::set<Row> &target_rows,
                                         const std::set<Row> &ctrl_rows,
                                         Col from_col,
                                         LATEX_GATE_TYPE type,
                                         const std::string &gate_name,
                                         bool dagger /*=false*/,
                                         const std::string &param /*= ""*/)
{
    /* get gate latex matrix dst row */
    assert(target_rows.size() > 0);
    Row tag_row = *target_rows.begin();

    /* get gate latex matrix dst col */
    auto row_range = rowRange(target_rows, ctrl_rows);
    Row span_start = row_range.first;
    Row span_end = row_range.second;
    Col tag_col = validColForRowRange(span_start, span_end, from_col);

    std::unordered_map<uint64_t, std::string> gate_latex_str;
    switch (type)
    {
    case LATEX_GATE_TYPE::GENERAL_GATE:
        if (1 == target_rows.size())
        {
            gate_latex_str = LATEX_SYNTAX::latex_single_bit_gate(gate_name, target_rows, ctrl_rows, param, dagger);
        }
        else
        {
            gate_latex_str = LATEX_SYNTAX::latex_multi_bits_gate(gate_name, target_rows, ctrl_rows, param, dagger);
        }
        break;
    case LATEX_GATE_TYPE::CNOT:
        gate_latex_str = LATEX_SYNTAX::latex_cnot_gate(gate_name, target_rows, ctrl_rows, param, dagger);
        break;
    case LATEX_GATE_TYPE::SWAP:
        gate_latex_str = LATEX_SYNTAX::latex_swap_gate(gate_name, target_rows, ctrl_rows, param, dagger);
        break;
    default:
        QCERR_AND_THROW(std::runtime_error, "Unknown gate");
        break;
    }

    for (Row row = span_start; row <= span_end; row++)
    {
        /* insert gate latex statement to matrix along target qbit and control qbit */
        if (gate_latex_str.count(row))
        {
            m_latex_qwire.insert(row, tag_col, gate_latex_str.at(row));
        }
        /* else marked gate span zone with \qw */
        else
        {
            m_latex_qwire.insert(row, tag_col, LATEX_SYNTAX::LATEX_QWIRE);
        }
    }

    return tag_col;
}

LatexMatrix::Col LatexMatrix::insertBarrier(const std::set<uint64_t> &rows, uint64_t from_col)
{
    Col dst_col = 0;
    auto continus_rows = sliceToContinuousSeq(rows);
    for (auto rows : continus_rows)
    {
        auto row_range = rowRange(rows, {});
        Row span_start = row_range.first;
        Row span_end = row_range.second;
        Col barrier_col = validColForRowRange(span_start, span_end, from_col);

        /*
          barrier is special in latex.
          for current col barrier, it's latex statment "\barrier" is append to gate or qwire last col
          like "\qw \barrier[0em]{1}" instead of "\qw & \barrier[0em]{1}"
          barrier always append to latex content before current col,
          so we just record gate dest position and append latex code later
          mark barrier zone with \qw as placeholder to forbiden place other gates
        */
        Row barrier_row = span_start;

        std::string barrier_latex = LATEX_SYNTAX::latex_barrier(span_start, span_end);

        if (0 == barrier_col)
        {
            m_barrier_mark_head.insert(barrier_row, 0, barrier_latex);
        }
        else
        {
            m_barrier_mark_qwire.insert(barrier_row, barrier_col - 1, barrier_latex);
        }

        for (size_t row = span_start; row <= span_end; row++)
        {
            m_latex_qwire.insert(row, barrier_col, LATEX_SYNTAX::LATEX_QWIRE);
        }

        dst_col = std::max(dst_col, barrier_col);
    }
    return dst_col;
}

LatexMatrix::Col LatexMatrix::insertMeasure(Row q_row, Row c_row, Col from_col)
{
    Row row_size = m_latex_qwire.row();
    Row span_end = row_size - 1;

    Col measure_col = validColForRowRange(q_row, span_end, from_col);

    m_latex_qwire.insert(q_row, measure_col, LATEX_SYNTAX::LATEX_MEASURE);

    /* mark all measure span qwires with \qw as placeholder */
    for (Row r = q_row + 1; r < row_size; r++)
    {
        m_latex_qwire.insert(r, measure_col, LATEX_SYNTAX::LATEX_QWIRE);
    }

    m_latex_cwire.insert(c_row, measure_col, LATEX_SYNTAX::latex_measure_to(c_row, q_row, row_size));

    return measure_col;
}

LatexMatrix::Col LatexMatrix::insertReset(Row q_row, Col from_col)
{
    Col q_col = validColForRowRange(q_row, q_row, from_col);
    m_latex_qwire.insert(q_row, q_col, LATEX_SYNTAX::LATEX_RESET);
    return q_col;
}

void LatexMatrix::insertTimeSeq(Col t_col, uint64_t time_seq)
{
    std::stringstream ss;
    ss << time_seq;
    m_latex_time_seq.insert(0, t_col, ss.str());
}

std::pair<LatexMatrix::Row, LatexMatrix::Row> LatexMatrix::rowRange(const std::set<Row> &row1, const std::set<Row> &row2)
{
    assert(!row1.empty() | !row2.empty());
    if (row1.empty())
    {
        return {*row2.begin(), *row2.rbegin()};
    }
    else if (row2.empty())
    {
        return {*row1.begin(), *row1.rbegin()};
    }
    else
    {
        Row start_row = (std::min)(*row1.begin(), *row2.begin());
        Row end_row = (std::max)(*row1.rbegin(), *row2.rbegin());
        return {start_row, end_row};
    }
}

LatexMatrix::Col LatexMatrix::validColForRowRange(uint64_t start_row, uint64_t end_row, uint64_t from_col)
{
    for (uint64_t row = start_row; row <= end_row; row++)
    {
        if (m_latex_qwire.isOccupied(row, from_col))
        {
            return validColForRowRange(start_row, end_row, ++from_col);
        }
    }
    return from_col;
}

void LatexMatrix::align_matrix(bool with_time /* = false*/)
{
    /* align row of head, wire and tail */
    decltype(m_latex_qwire)::Row q_row = 0;
    q_row = (std::max)({m_latex_qwire.row(), m_latex_qwire_head.row(), m_latex_qwire_tail.row()});
    m_latex_qwire.row() = q_row;
    m_latex_qwire_head.row() = q_row;
    m_latex_qwire_tail.row() = q_row;

    decltype(m_latex_qwire)::Row c_row = 0;
    c_row = (std::max)({m_latex_cwire.row(), m_latex_cwire_head.row(), m_latex_cwire_tail.row()});
    m_latex_cwire.row() = c_row;
    m_latex_cwire_head.row() = c_row;
    m_latex_cwire_tail.row() = c_row;

    /* align col of qwire and cwire */
    decltype(m_latex_qwire)::Col col = 0;
    if (with_time)
    {
        col = (std::max)({m_latex_qwire.col(), m_latex_cwire.col(), m_latex_time_seq.col()});
        m_latex_time_seq.col() = col;
    }
    else
    {
        col = (std::max)({m_latex_qwire.col(), m_latex_cwire.col()});
    }

    m_latex_cwire.col() = col;
    m_latex_qwire.col() = col;
}

std::string LatexMatrix::str(bool with_time /* = false */)
{
    align_matrix(with_time);

    std::string out_str(LATEX_SYNTAX::latex_header(m_logo));

    for (Row row = 0; row < m_latex_qwire.row(); row++)
    {
        for (Col col = 0; col < m_latex_qwire.col(); col++)
        {
            if (0 == col)
            {
                /* add head and barrier(if it is marked) here, so we can call str() without change anything */
                out_str += m_latex_qwire_head[row][0];
                if (m_barrier_mark_head.isOccupied(row, 0))
                {
                    out_str += " " + m_barrier_mark_head[row][0];
                }
                /* add "&" seperate matrix element to format latex array */
                out_str += "&";
            }

            out_str += m_latex_qwire[row][col];
            if (m_barrier_mark_qwire.isOccupied(row, col))
            {
                out_str += " " + m_barrier_mark_qwire[row][col];
            }
            out_str += "&";

            if (m_latex_qwire.col() == col + 1)
            {
                out_str += m_latex_qwire_tail[row][0] + "&";
            }
        }

        /* array row finished, pop out last redundancy '&', change to '\n' */
        out_str.pop_back();
        out_str += "\\\\\n";
    }

    for (Row row = 0; row < m_latex_cwire.row(); row++)
    {
        for (Col col = 0; col < m_latex_cwire.col(); col++)
        {
            if (0 == col)
            {
                out_str += m_latex_cwire_head[row][0] + "&";
            }
            out_str += m_latex_cwire[row][col] + "&";

            if (m_latex_cwire.col() == col + 1)
            {
                out_str += m_latex_cwire_tail[row][0] + "&";
            }
        }
        out_str.pop_back();
        out_str += "\\\\\n";
    }

    if (with_time)
    {
        out_str += m_latex_time_seq_head[0][0] + "&";
        for (Col col = 0; col < m_latex_time_seq.col(); col++)
        {
            out_str += m_latex_time_seq[0][col] + "&";
        }
        out_str += m_latex_time_seq_tail[0][0] + "\\\\\n";
    }

    out_str += LATEX_SYNTAX::LATEX_FOOTER;
    return out_str;
}

/*---------------------------------------------------------------------------*/

DrawLatex::DrawLatex(const QProg &prog, LayeredTopoSeq &layer_info, uint32_t length)
    : AbstractDraw(prog, layer_info, length),
      m_logo("OriginQ")
{
}

void DrawLatex::init(std::vector<int> &qbits, std::vector<int> &cbits)
{
    LatexMatrix::Label q_label, c_label;
    for (size_t i = 0; i < qbits.size(); i++)
    {
        m_qid_row[qbits[i]] = i;

        std::stringstream ss;
        ss << "q_{" << qbits[i] << "}";
        ss >> q_label[i];
    }

    for (size_t i = 0; i < cbits.size(); i++)
    {
        m_cid_row[cbits[i]] = i;
        std::stringstream ss;
        ss << "c_{" << cbits[i] << "}";
        ss >> c_label[i];
    }

    m_latex_matrix.setLabel(q_label, c_label);
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

void DrawLatex::draw_by_time_sequence(const std::string config_data /*= CONFIG_PATH*/)
{
    m_output_time = true;
    m_time_sequence_conf.load_config(config_data);

    const auto &layer_info = m_layer_info;

    int time_seq = 0;

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
        m_latex_matrix.insertTimeSeq(time_col, time_seq);
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

    QVec tag_vec = node_info->m_target_qubits;
    QVec ctrl_vec = node_info->m_control_qubits;

    std::set<uint64_t> tag_rows = qvecRows(tag_vec);
    std::set<uint64_t> ctrl_rows = qvecRows(ctrl_vec);

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

    /* get gate latex matrix dst col */
    uint64_t layer_col = layer_start_col(layer_id);
    uint64_t gate_col = 0;

    switch (gate_type)
    {
    case ISWAP_THETA_GATE:
    case ISWAP_GATE:
    case SQISWAP_GATE:
    case SWAP_GATE:
    {
        gate_col = m_latex_matrix.insertGate(tag_rows, ctrl_rows, layer_col, LATEX_GATE_TYPE::SWAP, gate_name, is_dagger, gate_param);
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
        /* in QPanda, control gate is special designed. last qid of tag_vec is target, others is ctrol */
        tag_rows = {qidRow(tag_vec.back()->get_phy_addr())};
        QVec real_ctrl_vec(tag_vec);
        real_ctrl_vec.pop_back();
        real_ctrl_vec += ctrl_vec;
        ctrl_rows = qvecRows(real_ctrl_vec);

        if (CPHASE_GATE == gate_type)
        {
            gate_name = "CR";
        }

        if (CNOT_GATE == gate_type)
        {

            gate_col = m_latex_matrix.insertGate(tag_rows, ctrl_rows, layer_col, LATEX_GATE_TYPE::CNOT, gate_name, is_dagger, gate_param);
        }
        else
        {
            gate_col = m_latex_matrix.insertGate(tag_rows, ctrl_rows, layer_col, LATEX_GATE_TYPE::GENERAL_GATE, gate_name, is_dagger, gate_param);
        }

        gate_time_seq = m_time_sequence_conf.get_ctrl_node_time_sequence() * (ctrl_vec.size() + 1);
        break;
    }
    case TWO_QUBIT_GATE:
    case TOFFOLI_GATE:
    case ORACLE_GATE:
    {
        gate_col = m_latex_matrix.insertGate(tag_rows, ctrl_rows, layer_col, LATEX_GATE_TYPE::GENERAL_GATE, gate_name, is_dagger, gate_param);
        /* TODO: how to calculate multi bits gate gate_time_seq */
        break;
    }
    case BARRIER_GATE:
    {
        QCERR_AND_THROW(std::runtime_error, "BARRIER_GATE should be processd in another way");
        break;
    }
        /*
          single target bit gate:

          TWO_QUBIT_GATE,
          TOFFOLI_GATE,
          ORACLE_GATE,
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
        gate_col = m_latex_matrix.insertGate(tag_rows, ctrl_rows, layer_col, LATEX_GATE_TYPE::GENERAL_GATE, gate_name, is_dagger, gate_param);
        gate_time_seq = ctrl_vec.size() > 0 ? (m_time_sequence_conf.get_ctrl_node_time_sequence() * ctrl_vec.size()) : m_time_sequence_conf.get_single_gate_time_sequence();
        break;
    }

    /* update curent layer latex matrix col range */
    m_layer_col_range[layer_id] = std::max(gate_col, m_layer_col_range[layer_id]);
    /* record layer time sequnce*/
    update_layer_time_seq(gate_time_seq);
}

void DrawLatex::append_barrier(pOptimizerNodeInfo &node_info, uint64_t layer_id)
{
    /* get target bits */
    QVec target_vec = node_info->m_target_qubits;
    QPANDA_ASSERT(target_vec.size() != 1, "barrier should only have one target bit");

    /* get control info */
    QVec ctrl_vec = node_info->m_control_qubits;

    /*
      barrier is special single bit gate in qpanda, qubits_vec and ctrl_vec contains all qibits are barriered.
      and all bits may not be continuous nor in ascending order
      so we first sort all bits then slice continuous sequence, finally write to latex matrix
    */
    auto all_rows = qvecRows(target_vec + ctrl_vec);

    uint64_t try_col = layer_start_col(layer_id);

    auto barrier_col = m_latex_matrix.insertBarrier(all_rows, try_col);
    m_layer_col_range[layer_id] = std::max(barrier_col, m_layer_col_range[layer_id]);
}

void DrawLatex::append_measure(pOptimizerNodeInfo &node_info, uint64_t layer_id)
{
    std::shared_ptr<AbstractQuantumMeasure> p_measure = std::dynamic_pointer_cast<AbstractQuantumMeasure>(*(node_info->m_iter));

    size_t qbit_id = p_measure->getQuBit()->getPhysicalQubitPtr()->getQubitAddr();
    size_t cbit_id = p_measure->getCBit()->get_addr();

    size_t layer_col = layer_start_col(layer_id);

    auto q_row = qidRow(qbit_id);
    auto c_row = cidRow(cbit_id);

    auto meas_col = m_latex_matrix.insertMeasure(q_row, c_row, layer_col);

    /* record curent layer end at latex matrix col */
    m_layer_col_range[layer_id] = std::max(meas_col, m_layer_col_range[layer_id]);

    update_layer_time_seq(m_time_sequence_conf.get_measure_time_sequence());
}

void DrawLatex::append_reset(pOptimizerNodeInfo &node_info, uint64_t layer_id)
{
    std::shared_ptr<AbstractQuantumReset> p_reset = std::dynamic_pointer_cast<AbstractQuantumReset>(*(node_info->m_iter));

    int qubit_index = p_reset->getQuBit()->getPhysicalQubitPtr()->getQubitAddr();

    uint64_t q_row = qidRow(qubit_index);

    uint64_t from_col = layer_start_col(layer_id);

    auto q_col = m_latex_matrix.insertReset(q_row, from_col);

    /* record curent layer end at latex matrix col */
    m_layer_col_range[layer_id] = std::max(q_col, m_layer_col_range[layer_id]);

    update_layer_time_seq(m_time_sequence_conf.get_reset_time_sequence());
}

std::string DrawLatex::present(const std::string &file_name)
{
    auto latex_src = m_latex_matrix.str(m_output_time);

    std::fstream f(file_name, std::ios_base::out);
    f << latex_src;
    f.close();

    return latex_src;
}

uint64_t DrawLatex::layer_start_col(size_t layer_id /*, size_t span_start, size_t span_end*/)
{
    uint64_t gate_col = layer_id == 0 ? layer_id : m_layer_col_range.at(layer_id - 1);
    return gate_col;
}

void DrawLatex::setLogo(const std::string &logo /* = ""*/)
{
    m_latex_matrix.setLogo(logo.empty() ? m_logo : logo);
}

std::set<uint64_t> DrawLatex::qvecRows(QVec qbits)
{
    std::set<uint64_t> rows;
    std::for_each(qbits.begin(), qbits.end(),
                  [&](const Qubit *qbit)
                  {
                      int qid = qbit->get_phy_addr();
                      rows.insert(m_qid_row.at(qid));
                  });
    return rows;
}

uint64_t DrawLatex::qidRow(int qid)
{
    return m_qid_row.at(qid);
}

uint64_t DrawLatex::cidRow(int cid)
{
    return m_cid_row.at(cid);
}

QPANDA_END
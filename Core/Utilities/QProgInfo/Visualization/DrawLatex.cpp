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

    // qcircuit
    const std::string k_circuit_qcircuit = "Qciruit";

    // wire
    const std::string k_wire_qw = "\\qw";
    const std::string k_wire_cw = "\\cw";
    const std::string k_wire_qw_x = "\\qwx";
    const std::string k_wire_cw_x = "\\cwx";

    // gate
    const std::string k_gate_gate = "\\gate";
    const std::string k_gate_ghost = "\\ghost";
    const std::string k_gate_nghost = "\\nghost";
    const std::string k_gate_multigate = "\\multigate";
    const std::string k_gate_swap = "\\qswap";
    const std::string k_gate_x = "\\gate{\\mathrm{X}}";
    const std::string k_gate_z = "\\gate{\\mathrm{Z}}";
    const std::string k_gate_barrier = "\\barrier[0em]";
    const std::string k_gate_cx = "\\targ";
    const std::string k_gate_dagger = "^\\dagger";
    const std::string k_gate_measure = "\\meter";
    const std::string k_gate_reset = "\\gate{\\mathrm{\\left|0\\right\\rangle}}";

    // control
    const std::string k_control_ctrl = "\\ctrl";
    const std::string k_control_control = "\\control";

    // lable
    const std::string k_lable_ket = "\\ket";
    const std::string k_lable_lstick = "\\lstick";
    const std::string k_lable_rstick = "\\rstick";
    const std::string k_lable_mathrm = "\\mathrm";
    const std::string k_lable_hspace = "\\hspace";
    const std::string k_lable_dstick = "\\dstick";

    // general
    const std::string k_symbol_slash = "/";
    const std::string k_symbol_brace_left = "{";
    const std::string k_symbol_brace_right = "}";
    const std::string k_symbol_bracket_left = "[";
    const std::string k_symbol_bracket_right = "]";
    const std::string k_symbol_parenthesis_left = "(";
    const std::string k_symbol_parenthesis_right = ")";

    const std::string k_symbol_underline = "_";
    const std::string k_symbol_zero = "0";
    const std::string k_symbol_and = "&";
    const std::string k_symbol_space = " ";
    const std::string k_symbol_backslash = "\\";
    const std::string k_symbol_comma = ",";
    const std::string k_symbol_enter = "\\\\\n";
    const std::string k_symbol_quotation = "\"";
    const std::string k_symbol_spacil = "_<<<";
    const std::string k_symbol_footer = "\\\\ }}\n"
                                        "\\end{document}\n";
    const std::string k_symbol_em = "0em";
    const std::string k_symbol_q = "q";
    const std::string k_symbol_c = "c";
    const std::string k_symbol_colon = ":";

    static std::string latex_header(const std::string &logo);
    static std::string latex_qwire_head_label(const std::string &label);
    static std::string latex_cwire_head_label(const std::string &label);
    static std::string latex_time_head_label(const std::string &label);
    static std::string latex_qwire_tail_label(const std::string &label, const std::string &sub_symbol = "");
    static std::string latex_cwire_tail_label(const std::string &main_symbol, const std::string &sub_symbol = "");
    static std::string latex_time_tail_label(const std::string &label);
    static void add_ctrl(std::unordered_map<uint64_t, std::string> &latex, uint64_t tag_min, uint64_t tag_max, std::vector<uint64_t> &ctrls);


    static std::unordered_map<uint64_t, std::string> latex_single_bit_gate(const std::string &gate_name,
                                                                 std::vector<uint64_t> tag_rows,
                                                                 std::vector<uint64_t> ctrl_rows,
                                                                 const std::string &param = "",
                                                                 bool is_dagger = false);

    static std::unordered_map<uint64_t, std::string> latex_multi_bits_gate(const std::string &gate_name,
                                                                 std::vector<uint64_t> tag_rows,
                                                                 std::vector<uint64_t> ctrl_rows,
                                                                 const std::string &param = "",
                                                                 bool is_dagger = false);

    static std::unordered_map<uint64_t, std::string> latex_x_gate(std::vector<uint64_t> tag_rows,
                                                        std::vector<uint64_t> ctrl_rows);
    static std::unordered_map<uint64_t, std::string> latex_swap_gate(std::vector<uint64_t> tag_rows,
                                                           std::vector<uint64_t> ctrl_rows);
    static std::unordered_map<uint64_t, std::string> latex_z_gate(std::vector<uint64_t> tag_rows,
                                                        std::vector<uint64_t> ctrl_rows);
    static std::string latex_measure_to(uint64_t c_row, uint64_t q_row, uint64_t row_size);
    static std::string latex_barrier(size_t row_start, size_t row_end);



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
               "\\Qcircuit @C = 1.0em @R = 0.5em @!R{ \\\\\n";
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

        std::string head_wire = k_gate_nghost + k_symbol_brace_left + label +
                k_lable_ket + k_symbol_brace_left + k_symbol_zero +
                k_symbol_brace_right + k_symbol_brace_right +
                k_symbol_and +
                k_lable_lstick + k_symbol_brace_left +
                label + k_lable_ket + k_symbol_brace_left +
                k_symbol_zero + k_symbol_brace_right + k_symbol_brace_right;
        return head_wire;
    }

    std::string latex_cwire_head_label(const std::string &label)
    {
        std::string cwire_head = k_gate_nghost + k_symbol_brace_left +
                label + k_symbol_brace_right +
                k_symbol_and +
                k_lable_lstick + k_symbol_brace_left + k_lable_mathrm +
                k_symbol_brace_left + k_symbol_c + k_symbol_colon + k_symbol_slash +
                k_symbol_brace_right + k_symbol_brace_right;
        return cwire_head;
    }

    std::string latex_time_head_label(const std::string &label)
    {
        std::string time_head = k_gate_nghost + k_symbol_brace_left +
                label + k_symbol_brace_right +
                k_symbol_and +
                k_lable_lstick + k_symbol_brace_left + k_lable_mathrm +
                k_symbol_brace_left + label + k_symbol_brace_right +
                k_symbol_brace_right;
        return time_head;
    }

    std::string latex_qwire_tail_label(const std::string &label, const std::string &sub_symbol)
    {
        std::string qwire_tail = k_lable_rstick + k_symbol_brace_left + label +
                k_symbol_brace_right + k_wire_qw +
                k_symbol_and +
                k_gate_nghost + k_symbol_brace_left + label + k_symbol_brace_right;
        return qwire_tail;
    }

    std::string latex_cwire_tail_label(const std::string &main_symbol, const std::string &sub_symbol)
    {
        std::string cwire_tail = k_lable_rstick + k_symbol_brace_left + k_lable_mathrm +
                k_symbol_brace_left + main_symbol + k_symbol_brace_right + k_symbol_brace_right +
                k_wire_cw +
                k_symbol_and +
                k_gate_nghost + k_symbol_brace_left + main_symbol + k_symbol_brace_right;
        return cwire_tail;
    }

    std::string latex_time_tail_label(const std::string &label)
    {
        std::string time_tail = k_lable_rstick + k_symbol_brace_left + k_lable_mathrm + k_symbol_brace_left +
                label + k_symbol_brace_right + k_symbol_brace_right +
                k_symbol_and +
                k_gate_nghost + k_symbol_brace_left + label + k_symbol_brace_right;
        return time_tail;
    }


    static void add_ctrl(std::unordered_map<uint64_t, std::string> &latex, uint64_t tag_min, uint64_t tag_max, std::vector<uint64_t> &ctrls)
    {
        std::sort(ctrls.begin(), ctrls.end());
        ctrls.erase(unique(ctrls.begin(), ctrls.end()), ctrls.end());

        for (auto &ctrl : ctrls)
        {
            uint64_t tag = tag_max;
            if (ctrl < tag_min)
            {
                tag = tag_min;
            }

            auto offset = static_cast<int>(tag) - static_cast<int>(ctrl);
            std::string string_ctrl = k_control_ctrl + k_symbol_brace_left +
                std::to_string(offset) + k_symbol_brace_right;
            latex.insert({ ctrl, string_ctrl });
        }

        return;
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
                                                                    std::vector<uint64_t> tag_rows,
                                                                    std::vector<uint64_t> ctrl_rows,
                                                                    const std::string &param,
                                                                    bool is_dagger)
    {
        assert(tag_rows.size() == 1);
        std::unordered_map<uint64_t, std::string> gate_latex;
        uint64_t tag_row = *tag_rows.begin();
        std::string lable_param = !param.empty() ? (k_symbol_backslash + k_symbol_comma + k_lable_mathrm +
                                                    k_symbol_brace_left + param + k_symbol_brace_right) :
                                                    std::string();
        std::string lable_dagger = is_dagger ? k_gate_dagger : std::string();
        std::string tag_latex = k_gate_gate + k_symbol_brace_left + k_lable_mathrm + 
            k_symbol_brace_left + gate_name + k_symbol_brace_right + lable_param + lable_dagger + k_symbol_brace_right;

        gate_latex.insert({tag_row, tag_latex});
        add_ctrl(gate_latex, tag_row, tag_row, ctrl_rows);
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
                                                          std::vector<uint64_t> tag_rows,
                                                          std::vector<uint64_t> ctrl_rows,
                                                          const std::string &param,
                                                          bool is_dagger)
    {
        assert(tag_rows.size() > 1);
        std::unordered_map<uint64_t, std::string> gate_latex;
        std::string lable_param = !param.empty() ? (k_symbol_backslash + k_symbol_comma + k_lable_mathrm +
                                                    k_symbol_brace_left + param + k_symbol_brace_right ) :
                                                    std::string();
        std::string lable_dagger = is_dagger ? k_gate_dagger : std::string();
        auto min = *std::min_element(tag_rows.begin(), tag_rows.end());
        auto max = *std::max_element(tag_rows.begin(), tag_rows.end());
        auto offset = max - min;

        for (size_t tag = min; tag <= max; tag++)
        {
            std::string str;
            auto iter = std::find(tag_rows.begin(), tag_rows.end(), tag);
            if (tag_rows.end() == iter)
            {
                str = k_gate_ghost + k_symbol_brace_left + k_lable_mathrm + k_symbol_brace_left +
                      gate_name + k_symbol_brace_right + lable_param + lable_dagger + k_symbol_brace_right;
            }
            else
            {
                auto index = std::distance(tag_rows.begin(), iter);
                if (*iter == min)
                {
                    str = k_gate_multigate + k_symbol_brace_left + std::to_string(offset) + k_symbol_brace_right +
                            k_symbol_brace_left + k_lable_mathrm + k_symbol_brace_left + gate_name + k_symbol_brace_right +
                            lable_param + lable_dagger + k_symbol_brace_right +
                            k_symbol_spacil + k_symbol_brace_left + std::to_string(index) + k_symbol_brace_right;
                    gate_latex.insert({tag, str});
                }
                else
                {
                    str = k_gate_ghost + k_symbol_brace_left + k_lable_mathrm + k_symbol_brace_left +
                          gate_name + k_symbol_brace_right + lable_param + lable_dagger + k_symbol_brace_right;
                    str +=  k_symbol_spacil + k_symbol_brace_left + std::to_string(index) + k_symbol_brace_right;
                }
            }
            gate_latex.insert({tag, str});
        }

        add_ctrl(gate_latex, min, max, ctrl_rows);
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
    std::unordered_map<uint64_t, std::string> latex_x_gate(std::vector<uint64_t> tag_rows,
                                                           std::vector<uint64_t> ctrl_rows)
    {
        assert(tag_rows.size() == 1);
        std::unordered_map<uint64_t, std::string> gate_latex;
        std::string tag_latex;

        if (0 == ctrl_rows.size())
        {
            tag_latex = k_gate_x + k_wire_qw;
        }
        else
        {
            tag_latex = k_gate_cx + k_wire_qw;
        }

        uint64_t tag_row = *tag_rows.begin();
        gate_latex.insert({tag_row, tag_latex});
        add_ctrl(gate_latex, tag_row, tag_row, ctrl_rows);
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
    std::unordered_map<uint64_t, std::string> latex_swap_gate(std::vector<uint64_t> tag_rows,
                                                    std::vector<uint64_t> ctrl_rows)
    {
        assert(tag_rows.size() == 2);

        std::unordered_map<uint64_t, std::string> gate_latex;
        auto min = *std::min_element(tag_rows.begin(), tag_rows.end());
        auto max = *std::max_element(tag_rows.begin(), tag_rows.end());
        auto offset = static_cast<int>(max) - static_cast<int>(min);

        std::string swap_qwx = k_gate_swap + k_wire_qw_x + k_symbol_bracket_left +
                std::to_string(offset) + k_symbol_bracket_right;
        gate_latex.insert({max, k_gate_swap});
        gate_latex.insert({min, swap_qwx});
        add_ctrl(gate_latex, min, max, ctrl_rows);
        return gate_latex;
    }

    /**
     * @brief generate cz gate latex str
     *
     * @param[in] gate_name name of gate
     * @param[in] qbits     gate qbits id
     * @param[in] param     gate param
     * @param[in] is_dagger gate is dagger
     * @return gate latex string, matchs qbits
     */
    std::unordered_map<uint64_t, std::string> latex_z_gate(std::vector<uint64_t> tag_rows,
                                                           std::vector<uint64_t> ctrl_rows)
    {
        assert(tag_rows.size() == 1);
        std::unordered_map<uint64_t, std::string> gate_latex;
        std::string tag_latex;

        if (0 == ctrl_rows.size())
        {
            tag_latex = k_gate_z + k_wire_qw;
        }
        else
        {
            tag_latex = k_control_control + k_wire_qw;
        }

        uint64_t tag_row = *tag_rows.begin();
        gate_latex.insert({tag_row, tag_latex});
        add_ctrl(gate_latex, tag_row, tag_row, ctrl_rows);
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
    std::string latex_measure_to(uint64_t c_row, uint64_t q_row, uint64_t row_size, uint64_t cbit_id)
    {
        int offset = static_cast<int>(q_row) - static_cast<int>(row_size) - static_cast<int>(c_row);
        std::string str = k_lable_dstick + k_symbol_brace_left + k_symbol_underline +
                k_symbol_brace_left + k_symbol_underline + k_symbol_brace_left + std::to_string(cbit_id) + k_symbol_brace_right +
                k_symbol_brace_left +
                k_symbol_brace_right + k_symbol_brace_right + k_symbol_brace_right +
                k_wire_cw + " \\ar @{<=} " + k_symbol_bracket_left +
                std::to_string(offset) + k_symbol_comma + k_symbol_zero +
                k_symbol_bracket_right;

        return str;
    }

    std::string latex_barrier(size_t row_start, size_t row_end)
    {
        int offset = static_cast<int>(row_end) - static_cast<int>(row_start);
        std::string barrier = k_gate_barrier + k_symbol_brace_left +
                std::to_string(offset) + k_symbol_brace_right;
        return barrier;
    }

} // namespace

namespace // namespace utils
{
    std::vector<std::vector<uint64_t>> slice_to_continuous_seq(std::vector<uint64_t> vec)
    {
        std::vector<std::vector<uint64_t>> sliced_seq;
        for (auto it : vec)
        {
            if (sliced_seq.empty())
            {
                sliced_seq.emplace_back(std::vector<uint64_t>{it});
            }
            else
            {
                bool continus = (1 == it - *sliced_seq.back().rbegin());
                if (continus)
                {
                    sliced_seq.back().push_back(it);
                }
                else
                {
                    sliced_seq.emplace_back(std::vector<uint64_t>{it});
                }
            }
        }
        return sliced_seq;
    }
} // namespace utils
/*---------------------------------------------------------------------------*/
QPANDA_BEGIN

LatexMatrix::LatexMatrix()
    : m_latex_qwire(LATEX_SYNTAX::k_wire_qw),
      m_latex_cwire(LATEX_SYNTAX::k_wire_cw),
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

void LatexMatrix::set_row(uint64_t row_qubit, uint64_t row_cbit)
{
    m_row_qubit = row_qubit;
    m_row_cbit = row_cbit;
    return;
}

uint64_t LatexMatrix::get_qubit_row()
{
    return m_row_qubit;
}
uint64_t LatexMatrix::get_cbit_row()
{
    return m_row_cbit;
}

void LatexMatrix::set_label(const Label &qubit_label, const Label &cbit_label /* ={}*/, const TimeSeqLabel &time_seq_label /* = ""*/, bool head /*=true*/)
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

void LatexMatrix::set_logo(const std::string &logo)
{
    m_logo = logo;
}

LatexMatrix::Col LatexMatrix::insert_gate(const std::vector<Row> &target_rows,
                                         const std::vector<Row> &ctrl_rows,
                                         Col from_col,
                                         LATEX_GATE_TYPE type,
                                         const std::string &gate_name,
                                         bool dagger /*=false*/,
                                         const std::string &param /*= ""*/)
{
    /* get gate latex matrix dst row */
    assert(target_rows.size() > 0);

    /* get gate latex matrix dst col */
    auto row_of_range = row_range(target_rows, ctrl_rows);
    Row span_start = row_of_range.first;
    Row span_end = row_of_range.second;
    Col tag_col = valid_col_for_row_range(span_start, span_end, from_col);


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
    case LATEX_GATE_TYPE::X:
        gate_latex_str = LATEX_SYNTAX::latex_x_gate(target_rows, ctrl_rows);
        break;
    case LATEX_GATE_TYPE::Z:
        gate_latex_str = LATEX_SYNTAX::latex_z_gate(target_rows, ctrl_rows);
        break;
    case LATEX_GATE_TYPE::SWAP:
        gate_latex_str = LATEX_SYNTAX::latex_swap_gate(target_rows, ctrl_rows);
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
            m_latex_qwire.insert(row, tag_col, LATEX_SYNTAX::k_wire_qw);
        }
    }

    return tag_col;
}


LatexMatrix::Col LatexMatrix::insert_barrier(const std::vector<Row> &rows, Col from_col)
{
    Col dst_col = valid_col_for_row_range(rows.front(), rows.back(), from_col);
    auto continus_rows = slice_to_continuous_seq(rows);

    for (size_t i = 0; i < continus_rows.size(); i++)
    {
        Row span_start = continus_rows[i].front();
        Row span_end = i < continus_rows.size() - 1 ? continus_rows[i + 1].front() : continus_rows[i].back();
        std::string barrier_latex = LATEX_SYNTAX::latex_barrier(continus_rows[i].front(), continus_rows[i].back());

        if (0 == dst_col)
        {
            m_barrier_mark_head.insert(span_start, 0, barrier_latex);
        }
        else
        {
            m_barrier_mark_qwire.insert(span_start, dst_col - 1, barrier_latex);
        }

        for (size_t row = span_start; row <= span_end; row++)
        {
            m_latex_qwire.insert(row, dst_col, LATEX_SYNTAX::k_wire_qw);
        }
    }

    return dst_col;
}


LatexMatrix::Col LatexMatrix::insert_measure(Row q_row, Row c_row, Col from_col, uint64_t cbit_id)
{
    Row span_end = m_row_qubit - 1;
    Col measure_col = valid_col_for_row_range(q_row, span_end, from_col);
    m_latex_qwire.insert(q_row, measure_col, LATEX_SYNTAX::k_gate_measure);

    /* mark all measure span qwires with \qw as placeholder */
    for (Row r = q_row + 1; r < m_row_qubit; r++)
    {
        m_latex_qwire.insert(r, measure_col, LATEX_SYNTAX::k_wire_qw);
    }

    m_latex_cwire.insert(c_row, measure_col, LATEX_SYNTAX::latex_measure_to(c_row, q_row, m_row_qubit,cbit_id));
    return measure_col;
}

LatexMatrix::Col LatexMatrix::insert_reset(Row q_row, Col from_col)
{
    Col q_col = valid_col_for_row_range(q_row, q_row, from_col);
    m_latex_qwire.insert(q_row, q_col, LATEX_SYNTAX::k_gate_reset);
    return q_col;
}

void LatexMatrix::insert_time_seq(Col t_col, uint64_t time_seq)
{
    std::string str = std::to_string(time_seq);
    m_latex_time_seq.insert(0, t_col, str);
}

std::pair<LatexMatrix::Row, LatexMatrix::Row> LatexMatrix::row_range(const std::vector<Row> &row1, const std::vector<Row> &row2)
{
    if (row1.empty() && row2.empty())
    {
        throw std::invalid_argument("Error: row_range");
    }

    Row row1_min = 0, row1_max = 0, row2_min = 0, row2_max = 0;
    if (row1.empty())
    {
        row2_min = *std::min_element(row2.begin(), row2.end());
        row2_max = *std::max_element(row2.begin(), row2.end());
        return { row2_min , row2_max };
    }
    else if (row2.empty())
    {
        row1_min = *std::min_element(row1.begin(), row1.end());
        row1_max = *std::max_element(row1.begin(), row1.end());
        return { row1_min , row1_max };
    }
    else
    {
        row1_min = *std::min_element(row1.begin(), row1.end());
        row1_max = *std::max_element(row1.begin(), row1.end());
        row2_min = *std::min_element(row2.begin(), row2.end());
        row2_max = *std::max_element(row2.begin(), row2.end());
        return { std::min(row1_min, row2_min), std::max(row1_max, row2_max) };
    }
}

LatexMatrix::Col LatexMatrix::valid_col_for_row_range(Row start_row, Row end_row, Col from_col)
{
    for (uint64_t row = start_row; row <= end_row; row++)
    {
        if (m_latex_qwire.isOccupied(row, from_col))
        {
            return valid_col_for_row_range(start_row, end_row, ++from_col);
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
                    out_str += LATEX_SYNTAX::k_symbol_space + m_barrier_mark_head[row][0];
                }
                /* add "&" seperate matrix element to format latex array */
                out_str += LATEX_SYNTAX::k_symbol_and;
            }

            out_str += m_latex_qwire[row][col];
            if (m_barrier_mark_qwire.isOccupied(row, col))
            {
                out_str += LATEX_SYNTAX::k_symbol_space + m_barrier_mark_qwire[row][col];
            }
            out_str += LATEX_SYNTAX::k_symbol_and;

            if (m_latex_qwire.col() == col + 1)
            {
                out_str += m_latex_qwire_tail[row][0] + LATEX_SYNTAX::k_symbol_and;
            }
        }

        /* array row finished, pop out last redundancy '&', change to '\n' */
        out_str.pop_back();
        out_str += LATEX_SYNTAX::k_symbol_enter;
    }

    for (Row row = 0; row < m_latex_cwire.row(); row++)
    {
        for (Col col = 0; col < m_latex_cwire.col(); col++)
        {
            if (0 == col)
            {
                out_str += m_latex_cwire_head[row][0] + LATEX_SYNTAX::k_symbol_and;
            }
            out_str += m_latex_cwire[row][col] + LATEX_SYNTAX::k_symbol_and;

            if (m_latex_cwire.col() == col + 1)
            {
                out_str += m_latex_cwire_tail[row][0] + LATEX_SYNTAX::k_symbol_and;
            }
        }
        out_str.pop_back();
        out_str += LATEX_SYNTAX::k_symbol_enter;
    }

    if (with_time)
    {
        out_str += m_latex_time_seq_head[0][0] + LATEX_SYNTAX::k_symbol_and;
        for (Col col = 0; col < m_latex_time_seq.col(); col++)
        {
            out_str += m_latex_time_seq[0][col] + LATEX_SYNTAX::k_symbol_and;
        }
        out_str += m_latex_time_seq_tail[0][0] + LATEX_SYNTAX::k_symbol_enter;
    }

    out_str += LATEX_SYNTAX::k_symbol_footer;
    return out_str;
}

/*---------------------------------------------------------------------------*/

DrawLatex::DrawLatex(const QProg& prog, LayeredTopoSeq& layer_info, uint32_t length,bool b_with_gate_params)
    : AbstractDraw(prog, layer_info, length,b_with_gate_params),
    m_logo("OriginQ")
{
}

void DrawLatex::init(std::vector<int> &qbits, std::vector<int> &cbits)
{
    LatexMatrix::Label q_label, c_label;
    for (size_t i = 0; i < qbits.size(); i++)
    {
        m_qid_row[qbits[i]] = i;
        q_label[i] = LATEX_SYNTAX::k_symbol_q + LATEX_SYNTAX::k_symbol_underline +
                 LATEX_SYNTAX::k_symbol_brace_left + std::to_string(qbits[i]) + LATEX_SYNTAX::k_symbol_brace_right;
    }

    if (cbits.size() != 0) {
        m_cid_row[cbits[0]] = 0;
        std::stringstream ss;
        ss << "c_{" << cbits[0] << "}";
        ss >> c_label[0];
    }
 /*   for (size_t i = 0; i < cbits.size(); i++)
    {
        m_cid_row[cbits[i]] = i;
        std::stringstream ss;
        ss << "c_{" << cbits[i] << "}";
        ss >> c_label[i];
    }*/

    m_latex_matrix.set_row(qbits.size(), cbits.size());
    m_latex_matrix.set_label(q_label, c_label);
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
        m_latex_matrix.insert_time_seq(time_col, time_seq);
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
    QVec tag_vec;
    QVec ctrl_vec = node_info->m_control_qubits;

    if (CZ_GATE == gate_type || CNOT_GATE == gate_type ||
        CU_GATE == gate_type || CP_GATE == gate_type ||
        CPHASE_GATE == gate_type)
    {
        tag_vec = node_info->m_target_qubits.back();
        ctrl_vec.push_back(node_info->m_target_qubits.front());
    }
    else
    {
         tag_vec = node_info->m_target_qubits;
    }

    std::vector<uint64_t> tag_rows = qvec_rows(tag_vec);
    std::vector<uint64_t> ctrl_rows = qvec_rows(ctrl_vec);

    std::shared_ptr<AbstractQGateNode> p_gate = std::dynamic_pointer_cast<AbstractQGateNode>(*(node_info->m_iter));
    /* get gate parameter */
    std::string gate_param;
    if (m_draw_with_gate_params) {
        get_gate_parameter(p_gate, gate_param);
    }

    /* get dagger */
    bool is_dagger = check_dagger(p_gate, p_gate->isDagger());


    int gate_time_seq = get_time_sequence(gate_type, ctrl_vec, tag_vec);
    /* get gate name */
    std::string gate_name = get_gate_name(gate_type);
    std::vector<std::string> gate_latex_str;

    /* get gate latex matrix dst col */
    uint64_t layer_col = layer_start_col(layer_id);
    uint64_t gate_col = 0;

    switch (gate_type)
    {
    case SWAP_GATE:
        gate_col = m_latex_matrix.insert_gate(tag_rows, ctrl_rows, layer_col, LATEX_GATE_TYPE::SWAP, gate_name, is_dagger, gate_param);
        break;
    case PAULI_Z_GATE:
    case CZ_GATE:
        gate_col = m_latex_matrix.insert_gate(tag_rows, ctrl_rows, layer_col, LATEX_GATE_TYPE::Z, gate_name, is_dagger, gate_param);
        break;
    case PAULI_X_GATE:
    case CNOT_GATE:
        gate_col = m_latex_matrix.insert_gate(tag_rows, ctrl_rows, layer_col, LATEX_GATE_TYPE::X, gate_name, is_dagger, gate_param);
        break;
    case BARRIER_GATE:
    {
        std::vector<uint64_t> all_qubits(tag_rows);
        all_qubits.insert(all_qubits.end(), ctrl_rows.begin(), ctrl_rows.end());
        std::sort(all_qubits.begin(), all_qubits.end());
        gate_col = m_latex_matrix.insert_barrier(all_qubits, layer_col);
    }
        break;
    default:
        gate_col = m_latex_matrix.insert_gate(tag_rows, ctrl_rows, layer_col, LATEX_GATE_TYPE::GENERAL_GATE, gate_name, is_dagger, gate_param);
        gate_time_seq = ctrl_vec.size() > 0 ? (m_time_sequence_conf.get_ctrl_node_time_sequence() * ctrl_vec.size()) : m_time_sequence_conf.get_single_gate_time_sequence();
        break;
    }

    /* update curent layer latex matrix col range */
    m_layer_col_range[layer_id] = std::max(gate_col, m_layer_col_range[layer_id]);
    /* record layer time sequnce*/
    update_layer_time_seq(gate_time_seq);
}



size_t DrawLatex::get_time_sequence(GateType type, const QVec &ctrls, const QVec &tags)
{
    size_t time_seq = 0;
    switch (type) {
    case SWAP_GATE:
    case ISWAP_GATE:
    case SQISWAP_GATE:
    case ISWAP_THETA_GATE:
    case MS_GATE:
    case TWO_QUBIT_GATE:
    case ORACLE_GATE:
        time_seq = m_time_sequence_conf.get_swap_gate_time_sequence() * (ctrls.size() + 1);
        break;
    case CZ_GATE:
    case CNOT_GATE:
    case CP_GATE:
    case CPHASE_GATE:
    case CU_GATE:
    case TOFFOLI_GATE:
        time_seq = m_time_sequence_conf.get_ctrl_node_time_sequence() * (ctrls.size() + 1);
        break;
    default:
        time_seq = 0;
        break;
    }

    return time_seq;
}

std::string DrawLatex::get_gate_name(GateType type)
{
    std::string gate_name;
    switch (type) {
    case CZ_GATE:
        gate_name = TransformQGateType::getInstance()[GateType::PAULI_Z_GATE];
        break;
    case CNOT_GATE:
    case TOFFOLI_GATE:
        gate_name = TransformQGateType::getInstance()[GateType::PAULI_X_GATE];
        break;
    case CP_GATE:
        gate_name = TransformQGateType::getInstance()[GateType::P_GATE];
        break;
    case CPHASE_GATE:
        gate_name = "Phase";
        break;
    case CU_GATE:
        gate_name = TransformQGateType::getInstance()[GateType::U1_GATE];
        break;
    case ORACLE_GATE:
    case CORACLE_GATE:
        gate_name = "Unitary";
        break;
    default:
        gate_name = TransformQGateType::getInstance()[type];
        break;
    }

    return gate_name;
}

void DrawLatex::append_measure(pOptimizerNodeInfo &node_info, uint64_t layer_id)
{
    std::shared_ptr<AbstractQuantumMeasure> p_measure = std::dynamic_pointer_cast<AbstractQuantumMeasure>(*(node_info->m_iter));
    size_t qbit_id = p_measure->getQuBit()->getPhysicalQubitPtr()->getQubitAddr();
    size_t cbit_id = p_measure->getCBit()->get_addr();

    size_t layer_col = layer_start_col(layer_id);
    auto q_row = qid_row(qbit_id);

    auto c_row = cid_row(m_cid_row.begin()->first);
    auto meas_col = m_latex_matrix.insert_measure(q_row, c_row, layer_col,cbit_id);

    /* record curent layer end at latex matrix col */
    m_layer_col_range[layer_id] = std::max(meas_col, m_layer_col_range[layer_id]);

    update_layer_time_seq(m_time_sequence_conf.get_measure_time_sequence());
}

void DrawLatex::append_reset(pOptimizerNodeInfo &node_info, uint64_t layer_id)
{
    std::shared_ptr<AbstractQuantumReset> p_reset = std::dynamic_pointer_cast<AbstractQuantumReset>(*(node_info->m_iter));

    int qubit_index = p_reset->getQuBit()->getPhysicalQubitPtr()->getQubitAddr();

    uint64_t q_row = qid_row(qubit_index);

    uint64_t from_col = layer_start_col(layer_id);

    auto q_col = m_latex_matrix.insert_reset(q_row, from_col);

    /* record curent layer end at latex matrix col */
    m_layer_col_range[layer_id] = std::max(q_col, m_layer_col_range[layer_id]);

    update_layer_time_seq(m_time_sequence_conf.get_reset_time_sequence());
}

std::string DrawLatex::present(const std::string &file_name)
{
    auto latex_src = m_latex_matrix.str(m_output_time);

    if (file_name.length() > 0) {
        std::fstream f(file_name, std::ios_base::out);
        f << latex_src;
        f.close();
    }


    return latex_src;
}

uint64_t DrawLatex::layer_start_col(size_t layer_id /*, size_t span_start, size_t span_end*/)
{
    uint64_t gate_col = layer_id == 0 ? layer_id : m_layer_col_range.at(layer_id - 1);
    return gate_col;
}

void DrawLatex::set_logo(const std::string &logo /* = ""*/)
{
    m_latex_matrix.set_logo(logo.empty() ? m_logo : logo);
}

std::vector<uint64_t> DrawLatex::qvec_rows(QVec qbits)
{
    std::vector<uint64_t> rows;
    std::for_each(qbits.begin(), qbits.end(),
                  [&](const Qubit *qbit)
                  {
                      int qid = qbit->get_phy_addr();
                      rows.push_back(m_qid_row.at(qid));
                  });
    return rows;
}

uint64_t DrawLatex::qid_row(int qid)
{
    return m_qid_row.at(qid);
}

uint64_t DrawLatex::cid_row(int cid)
{

    return m_cid_row.at(cid);
}

QPANDA_END

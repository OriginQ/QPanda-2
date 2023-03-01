#ifndef _DRAW_LATEX_H_
#define _DRAW_LATEX_H_


#include "Core/Utilities/QPandaNamespace.h"
#include "Core/Utilities/QProgInfo/Visualization/AbstractDraw.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/QProgDAG.h"
#include <unordered_map>
#include <string>
#include <memory>

/**
 * @brief spare matrix like class, only contain nodefault value element
 * only insert() can add 'nonempty' element, whatever content it is, even it's value == default value
 * get others element will return default value. silimar as real spare matrix, those element treated as 'empty' element
 *
 * @tparam Elem_t
 */
template <typename Elem_t>
class SpareMatrix
{
public:
	using Row = uint64_t;
	using Col = uint64_t;
	class ElemView;
	class RowView;

	SpareMatrix(const Elem_t &default_value)
		: m_default_value(default_value) {}
	~SpareMatrix() {}

	const RowView begin() const
	{
		return RowView(*this);
	}

	const RowView end() const
	{
		return RowView(*this);
	}

	/**
	 * @brief insert matrix element
	 *
	 * @note allow over write
	 */
	void insert(Row row, Col col, const Elem_t &elem, bool overwrite = true)
	{
		m_row_size = row >= m_row_size ? row + 1 : m_row_size;
		m_col_size = col >= m_col_size ? col + 1 : m_col_size;
		if (!overwrite && m_matrix.count(row) && m_matrix.at(row).count(col))
		{
			return ;
		}
        else
        {
           m_matrix[row][col] = elem;
        }
	}

	/**
	 * @brief return row size
	 *
	 * @return Row&
	 */
	Row &row()
	{
		return m_row_size;
	}

	/**
	 * @brief return col size
	 *
	 * @return Col&
	 */
	Col &col()
	{
		return m_col_size;
	}

	const RowView operator[](Row row) const
	{
		return RowView(*this, row);
	}

	/**
	 * @brief check matrix position at row:col had been occupied. element iserted treated as occupied
	 *
	 * @return true if element had been inserted
	 * @note even inserted element value == default value, return false
	 */
	bool isOccupied(Row row, Col col)
	{
		if (m_matrix.count(row) && m_matrix.at(row).count(col))
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	friend class RowView;
	friend class ElemView;

private:
	Row m_row_size{0};
	Col m_col_size{0};
	Elem_t m_default_value;
	std::unordered_map<Row, std::unordered_map<Col, std::string>> m_matrix;

public:
	class ElemView
	{
	public:
		ElemView(const SpareMatrix<Elem_t> &m, Row cur_row)
			: m_spare_matrix(m),
			  m_cur_row(cur_row)
		{
		}

		ElemView &operator++()
		{
			m_cur_col += 1;
			return *this;
		}

		bool operator!=(const ElemView &lhs)
		{
			/*
			  for iterator compare with RowView.end() in range for

			  if current hit matrix col bottom, return true, means reach matrix last col
			*/
			bool col_valid = (m_cur_col < m_spare_matrix.m_col_size);
			return col_valid;
		}

		const Elem_t &operator*() const
		{
			if (m_spare_matrix.m_matrix.count(m_cur_row) && m_spare_matrix.m_matrix.at(m_cur_row).count(m_cur_col))
			{
				return m_spare_matrix.m_matrix.at(m_cur_row).at(m_cur_col);
			}
			else
			{
				return m_spare_matrix.m_default_value;
			}
		}

	private:
		Row m_cur_row{0};
		Col m_cur_col{0};
		const SpareMatrix<Elem_t> &m_spare_matrix;
	};

	class RowView
	{
	public:
		RowView(const SpareMatrix<Elem_t> &m) : m_spare_matrix(m) {}
		RowView(const SpareMatrix<Elem_t> &m, Row row) : m_spare_matrix(m), m_cur_row(row) {}
		~RowView() {}

		RowView &operator++()
		{
			m_cur_row += 1;
			return *this;
		}

		bool operator!=(const RowView &lhs)
		{
			/*
			  for iterator compare with SpareMatrix.end() in range for

			  if current row hit matrix row bottom, return true, means reach the matrix bottom
			*/
			bool row_valid = ( m_cur_row < m_spare_matrix.m_row_size);
			return row_valid;
		}

		/*
		  dummy iterator for sapre matrix
		*/
		const RowView &operator*() const
		{
			return *this;
		}

		const ElemView begin() const
		{
			return ElemView(m_spare_matrix, m_cur_row);
		}

		const ElemView end() const
		{
			return ElemView(m_spare_matrix, m_cur_row);
		}

		/*
		  get non-writable element,
		  for writable reference may cause member 'm_default_value' be modified
		*/
		const Elem_t &operator[](Col col) const
		{
			if (m_spare_matrix.m_matrix.count(m_cur_row) && m_spare_matrix.m_matrix.at(m_cur_row).count(col))
			{
				return m_spare_matrix.m_matrix.at(m_cur_row).at(col);
			}
			else
			{
				return m_spare_matrix.m_default_value;
			}
		}

	private:
		Row m_cur_row{0};
		const SpareMatrix<Elem_t> &m_spare_matrix;
	};
};

QPANDA_BEGIN

enum class LATEX_GATE_TYPE
{
	GENERAL_GATE,
    X,
    Z,
    SWAP
};

/**
 * @brief generate quantum circuits latex src code can be compiled on latex package 'qcircuit'
 * circuits element treated as matrix element in latex syntax
 * 
 * qcircuit tutorial [https://physics.unm.edu/CQuIC/Qcircuit/Qtutorial.pdf]
 * 
 */
class LatexMatrix
{
public:
	using Row = uint64_t;
	using Col = uint64_t;
	using Label = std::unordered_map<Row, std::string>;
	using TimeSeqLabel = std::string;

	LatexMatrix();
	~LatexMatrix() = default;

	/**
	 * @brief Set Label at left most head col or right most tail col
	 * label can be reseted at any time
	 *
	 * @param qubit_label label for qwire left most head lebel, at row, in latex syntax. not given row will keep empty
	 * @param cbit_label
	 * @param time_seq_label
	 * @param head if true, label append head; false, append at tail
	 */
    void set_row(uint64_t row_qubit, uint64_t row_cbit);
    uint64_t get_qubit_row();
    uint64_t get_cbit_row();

    void set_label(const Label &qubit_label, const Label &cbit_label = {}, const TimeSeqLabel &time_seq_label = "", bool head = true);
    void set_logo(const std::string &logo);

	/**
	 * @brief  
	 * 
	 * @param target_rows gate targets row of latex matrix
	 * @param ctrl_rows 
	 * @param from_col 	  gate wanted col pos, but there may be not enough zone to put gate
	 * @param type 
	 * @param gate_name 
	 * @param dagger 
	 * @param param 
	 * @return if there is no enough zone to put gate at 'from_col', we will find suitable col to put gate after 'from_col',
	 * 		   the real col placed the gate will be return
	 */
    Col insert_gate(const std::vector<Row> &target_rows,
                    const std::vector<Row> &ctrl_rows,
                    Col from_col,
                    LATEX_GATE_TYPE type,
                    const std::string &gate_name,
                    bool dagger = false,
                    const std::string &param = "");

    /**
     * @brief
     *
     * @param qubits gate targets row of latex matrix
     * @param from_col 	  gate wanted col pos, but there may be not enough zone to put gate
     * @param type
     * @return if there is no enough zone to put gate at 'from_col', we will find suitable col to put gate after 'from_col',
     * 		   the real col placed the gate will be return
     */
    Col insert_barrier(const std::vector<Row> &rows,
                       Col from_col);

    Col insert_measure(Row q_row, Row c_row, Col from_col);

    Col insert_reset(Row q_row, Col from_col);

	/**
	 * @note we do not check col num, may cause overwrite. user must take care col num self.
	 */
    void insert_time_seq(Col t_col, uint64_t time_seq);
	
	/**
	 * @brief return final latex source code, can be called at any time
	 * 
	 * @param with_time output with or not time sequence
	 * @return std::string 
	 */
	std::string str(bool with_time = false);
private:
	/**
	 * @brief find valid col to put gate from start_row row to end_row row
	 *
	 * @param span_start gate start row
	 * @param span_end 	 gate end row
	 * @param from_col   try destiny col
	 * @return size_t return col of valid zone can place whole gate
	 */
    Col valid_col_for_row_range(Row start_row, Row end_row, Col from_col);
	
	/**
	 * @brief 
	 * 
	 * @param row1 
	 * @param row2 
	 * @return return rows span lowest and highest row of row1 and row2
	 */
    std::pair<Row, Row> row_range(const std::vector<Row> &row1, const std::vector<Row> &row2);

	/**
	 * @brief align inner latex matrix row and col
	 * 
	 * @param with_time weather align with time seq matrix
	 */
	void align_matrix(bool with_time = false);

private:
	SpareMatrix<std::string> m_latex_qwire; /**< latex quantum circuit formarted as matrix, we only save code except for wires */
	SpareMatrix<std::string> m_latex_cwire; /**< cbits and qubits all start from 0, better slipt two matrix */
	SpareMatrix<std::string> m_latex_time_seq;

	SpareMatrix<std::string> m_latex_qwire_head; /**< latex quantum circuit left most label */
	SpareMatrix<std::string> m_latex_cwire_head; /**< cbits left most label */
	SpareMatrix<std::string> m_latex_time_seq_head;

	SpareMatrix<std::string> m_latex_qwire_tail; /**< latex quantum circuit right most label */
	SpareMatrix<std::string> m_latex_cwire_tail;
	SpareMatrix<std::string> m_latex_time_seq_tail;

	std::string m_logo;

	SpareMatrix<std::string> m_barrier_mark_head; /**< marker to mark right place put barrier latex code, case it's synatex odd */
	SpareMatrix<std::string> m_barrier_mark_qwire;
    uint64_t m_row_qubit{0};
    uint64_t m_row_cbit{0};
};

/**
 * @brief ouput qprog circute to latex source file
 *
 * @note qprog circute is represented by LayeredTopoSeq,
 * 		 in which there is a conception 'layer'.
 *       each layer has a layer id, contains a bunch of OptimizerNodeInfo from qprog
 *
 * @see LayeredTopoSeq
 * @see SeqLayer
 * @see OptimizerNodeInfo
 */
class DrawLatex : public AbstractDraw
{
public:
	DrawLatex(const QProg &prog, LayeredTopoSeq &layer_info, uint32_t length);
	virtual ~DrawLatex() {}

	/**
	 * @brief initialize
	 *
	 * @param[in] qbits std::vector<int>& used qubits
	 * @param[in] cbits std::vector<int>& used class bits
	 */
	virtual void init(std::vector<int> &qbits, std::vector<int> &cbits) override;

	/**
	 * @brief draw latex-picture by layer
	 *
	 */
	virtual void draw_by_layer() override;

	/**
	 * @brief draw latex-picture by time sequence
	 *
	 * @param[in] config_data const std::string It can be configuration file or configuration data,
							  which can be distinguished by file suffix,
							  so the configuration file must be end with ".json", default is CONFIG_PATH
	 */
	virtual void draw_by_time_sequence(const std::string config_data /*= CONFIG_PATH*/) override;

	/**
	 * @brief display and return the target string
	 *
	 * @param[in] file_name output latex source file
	 * @return std::string 	latex source code
	 */
	virtual std::string present(const std::string &file_name) override;

    void set_logo(const std::string &logo = "");

	/**
	 * @brief return layer start col position
	 * 
	 */
	uint64_t layer_start_col(size_t layer_id /*, size_t span_start, size_t span_end*/);

	/**
	 * @brief return qid mapped latex matrix row
	 * 
	 * @param qbits 
	 * @return std::set<uint64_t> 
	 */
    std::vector<uint64_t> qvec_rows(QVec qbits);
    uint64_t qid_row(int qid);
    uint64_t cid_row(int cid);

private:
	void append_node(DAGNodeType t, pOptimizerNodeInfo &node_info, uint64_t layer_id);
	void append_gate(pOptimizerNodeInfo &node_info, uint64_t layer_id);
	void append_measure(pOptimizerNodeInfo &node_info, uint64_t layer_id);
	void append_reset(pOptimizerNodeInfo &node_info, uint64_t layer_id);

    size_t get_time_sequence(GateType type, const QVec &ctrls, const QVec &tags);
    std::string get_gate_name(GateType type);
	int update_layer_time_seq(int time_seq);

	std::unordered_map<uint64_t, uint64_t> m_qid_row; /**< qubit id map to latex matrix row number */
	std::unordered_map<uint64_t, uint64_t> m_cid_row; /**< cbit id map to latex matrix row number */
	LatexMatrix m_latex_matrix;
	TimeSequenceConfig m_time_sequence_conf;
	std::unordered_map<uint64_t, uint64_t> m_layer_col_range; /**< layer id map to latex matrix cols span range(only save layer end col for short) */
	bool m_output_time{false};
	int m_layer_max_time_seq{0};
	const std::string m_logo;
};

QPANDA_END


#endif

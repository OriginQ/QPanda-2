#pragma once
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/Utilities/QProgInfo/Visualization/Draw.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/QProgDAG.h"
#include <unordered_map>
#include <string>
#include <memory>

template <typename Elem_t>
class SpareMatrix
{
public:
	using row_t = size_t;
	using col_t = size_t;

	class ElemView
	{
	public:
		ElemView(const SpareMatrix<Elem_t> &m, row_t cur_row)
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
			 for iterator compare with end()
			 
			 if matrix is empty OR current hit matrix col bottom, return true
			*/
			bool col_valid = (!m_spare_matrix.m_matrix.empty() && m_cur_col <= m_spare_matrix.m_max_col);
			return col_valid;
		}

		const Elem_t& operator*()
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
		row_t m_cur_row{0};
		col_t m_cur_col{0};
		const SpareMatrix<Elem_t> &m_spare_matrix;
	};

	class RowView
	{
	public:
		RowView(const SpareMatrix<Elem_t> &m) : m_spare_matrix(m) {}
		RowView(const SpareMatrix<Elem_t> &m, row_t row) : m_spare_matrix(m), m_cur_row(row) {}
		~RowView() {}

		RowView &operator++()
		{
			m_cur_row += 1;
			return *this;
		}

		bool operator!=(const RowView &lhs)
		{
			/*
			 for iterator compare with end()
			 
			 if matrix is empty OR current row hit matrix row bottom, return true
			*/
			bool row_valid = (!m_spare_matrix.m_matrix.empty() && m_cur_row <= m_spare_matrix.m_max_row);
			return row_valid;
		}

		RowView &operator*()
		{
			return *this;
		}

		const ElemView begin()
		{
			return ElemView(m_spare_matrix, m_cur_row);
		}

		const ElemView end()
		{
			return ElemView(m_spare_matrix, m_cur_row);
		}

		Elem_t operator[](col_t col)
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
		row_t m_cur_row{0};
		const SpareMatrix<Elem_t> &m_spare_matrix;
	};

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

	void insert(row_t row, col_t col, const Elem_t &elem)
	{
		m_max_row = row > m_max_row ? row : m_max_row;
		m_max_col = col > m_max_col ? col : m_max_col;
		m_matrix[row][col] = elem;
	}

	friend class RowView;
	friend class ElemView;

	row_t &max_row()
	{
		return m_max_row;
	}

	col_t &max_col()
	{
		return m_max_col;
	}

	RowView operator[](row_t row)
	{
		return RowView(*this, row);
	}

	/**
	 * @brief check element at row:col is or not default value
	 * 
	 * @return true if element is default value 
	 */
	bool is_empty(row_t row, col_t col)
	{
		if (m_matrix.count(row) && m_matrix.at(row).count(col))
		{
			return false;
		}
		else
		{
			return true;
		}
	}

private:
	row_t m_max_row{0};
	col_t m_max_col{0};
	Elem_t m_default_value;
	std::unordered_map<row_t, std::unordered_map<col_t, std::string>> m_matrix;
};

QPANDA_BEGIN

class DrawLatex : public AbstractDraw
{
public:
	DrawLatex(const QProg &prog, LayeredTopoSeq &layer_info, uint32_t length);
	virtual ~DrawLatex() {}

	/**
	 * @brief initialize
	 * 
	 * @param[in] quBits std::vector<int>& used qubits
	 * @param[in] clBits std::vector<int>& used class bits
	 */
	virtual void init(std::vector<int> &quBits, std::vector<int> &clBits) override;

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
			 				  so the configuration file must be end with ".json", default is CONFIG_PAT
	 */
	virtual void draw_by_time_sequence(const std::string config_data /*= CONFIG_PATH*/) override;

	/**
	 * @brief display and return the target string
	 * 
	 * @param[in] file_name 
	 * @return std::string 
	 */
	virtual std::string present(const std::string &file_name) override;

private:
	void append_node(DAGNodeType t, pOptimizerNodeInfo &node_info, uint64_t layer_id);
	void append_gate(pOptimizerNodeInfo &node_info, uint64_t layer_id);
	void append_measure(pOptimizerNodeInfo &node_info, uint64_t layer_id);
	void append_reset(pOptimizerNodeInfo &node_info, uint64_t layer_id);
	void append_barrier(pOptimizerNodeInfo &node_info, uint64_t layer_id);

	/**
	 * @brief find valid col to put gate from span_start row to span_end
	 * 
	 * @param span_start gate start row
	 * @param span_end 	 gate end row
	 * @param col 		 try destiny col
	 * @return size_t return valid zone col can put whole gate
	 */
	size_t find_valid_matrix_col(size_t span_start, size_t span_end, size_t col);
	void align_matrix_col();

	SpareMatrix<std::string> m_latex_qwire; /**< latex quantum circuit formarted as matrix, we only save code except for wires */
	SpareMatrix<std::string> m_latex_cwire; /**< cbits and qubits all start from 0, better slipt two matrix */
};

QPANDA_END
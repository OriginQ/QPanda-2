#pragma once
#include "Core/Utilities/QPandaNamespace.h"
#include <string>
#include <list>
#include <vector>
#include <fstream>
#include "Core/Utilities/QProgTransform/QProgToDAG/GraphMatch.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/QProgDAG.h"

QPANDA_BEGIN

enum TEXT_PIC_TYPE
{
	LAYER = 0,
	TIME_SEQUENCE
};

class DrawBox
{
public:
	DrawBox(const std::string &top_format_str, const std::string &mid_format_str, const std::string &bot_format_str)
		:m_top_format(top_format_str), m_mid_format(mid_format_str), m_bot_format(bot_format_str)
	{}

	virtual const std::string& getTopStr() const {
		return m_top_format;
	}

	virtual const std::string& getMidStr() const {
		return m_mid_format;
	}

	virtual const std::string& getBotStr() const {
		return m_bot_format;
	}

	virtual void set_top_connected() {}
	virtual void set_bot_connected() {}

	virtual int getLen() const = 0;

protected:
	std::string m_top_format;
	std::string m_mid_format;
	std::string m_bot_format;
};

class Wire
{
public:
	Wire(const std::string& connect_str)
		:m_connect_str(connect_str), m_cur_len(0), m_b_merged_bot_line(false), m_time_sequence(0)
	{}
	virtual ~Wire() {}

	virtual void setName(const std::string& name, size_t nameLen) {
		for (size_t i = 0; i < nameLen; i++)
		{
			m_top_line.append(" ");
			m_bot_line.append(" ");
		}
		m_mid_line.append(name);
		m_cur_len = nameLen;
	}

	virtual int append(const DrawBox& box, const int box_pos) {
		if (box_pos > m_cur_len)
		{
			for (size_t i = m_cur_len; i < box_pos; i++)
			{
				m_top_line.append(" ");
				m_mid_line.append(m_connect_str);
				m_bot_line.append(" ");
				++m_cur_len;
			}
		}

		m_top_line.append(box.getTopStr());
		m_mid_line.append(box.getMidStr());
		m_bot_line.append(box.getBotStr());

		m_cur_len += box.getLen();
		return (box_pos + box.getLen());
	}

	virtual int getWireLength() { return m_cur_len; }

	virtual std::string draw(std::ofstream& fd, int srartRow) {
		std::string outputStr;
		fd << (m_top_line.append("\n"));
		outputStr.append(m_top_line);

		fd << (m_mid_line.append("\n"));
		outputStr.append(m_mid_line);

		if (!m_b_merged_bot_line)
		{
			fd << (m_bot_line.append("\n"));
			outputStr.append(m_bot_line);
		}

		return outputStr;
	}

	virtual void updateWireLen(const int len) {
		for (size_t i = m_cur_len; i < len; i++)
		{
			m_top_line.append(" ");
			m_mid_line.append(m_connect_str);
			m_bot_line.append(" ");
		}

		m_cur_len = len;
	}

	virtual void setMergedFlag(bool b) { m_b_merged_bot_line = b; }

	virtual const std::string& getTopLine() const { return m_top_line; }

	virtual const std::string& getMidLine() const { return m_mid_line; }

	virtual const std::string& getBotLine() const { return m_bot_line; }

	int update_time_sequence(unsigned int increase_time_sequence) { return (m_time_sequence += increase_time_sequence); }
	int get_time_sequence() { return m_time_sequence; }

protected:
	const std::string m_connect_str;
	std::string m_top_line;
	std::string m_mid_line;
	std::string m_bot_line;
	int m_cur_len;
	bool m_b_merged_bot_line;
	int m_time_sequence;
};

class TimeSequenceConfig;
class DrawPicture
{
	using wireElement = std::pair<int, std::shared_ptr<Wire>>;
	using wireIter = std::map<int, std::shared_ptr<Wire>>::iterator;

public:
	DrawPicture(QProg &prog);
	~DrawPicture() {}

	void init(std::vector<int>& quBits, std::vector<int>& clBits);
	std::string present();
	void mergeLine();
	int getMaxQuWireLength(wireIter start_quBit_wire, wireIter end_quBit_wire);
	void updateTextPicLen();
	unsigned long getWideCharVal(const unsigned char* wide_char) {
		if (nullptr == wide_char)
		{
			return 0;
		}
		return (((unsigned long)(wide_char[0]) << 16) | ((unsigned long)(wide_char[1]) << 8) | (((unsigned long)(wide_char[2]))));
	}

	void check_time_sequence(std::vector<SequenceLayer>::iterator cur_layer_iter);
	void update_time_sequence(std::shared_ptr<Wire> p_wire, int increased_time_sequence);
	void append_time_sequence_line();
	void draw_by_layer();
	void draw_by_time_sequence();

protected:
	void appendMeasure(std::shared_ptr<AbstractQuantumMeasure> pMeasure);
	void append_ctrl_gate(std::string gate_name, const int terget_qubit, QVec &self_control_qubits_vec, QVec &circuit_control_qubits_vec);
	void append_swap_gate(std::string gate_name, QVec &qubits_vector, QVec &circuit_control_qubits_vec);
	void append_single_gate(std::string gate_name, QVec &qubits_vector, QVec &circuit_control_qubits_vec);
	void merge(const std::string& up_wire, std::string& down_wire);
	void fit_to_gbk(std::string &utf8_str);
	int get_wide_char_pos(const std::string &str, int start_pos);
	void set_connect_direction(const int& qubit, const std::vector<int>& vec_qubits, DrawBox& box);
	void append_ctrl_line(int line_start, int line_end, int pos);
	bool check_time_sequence_one_qubit(wireIter qu_wire_itr, std::vector<SequenceLayer>::iterator cur_layer_iter);
	bool is_qubit_in_vec(const int qubit, const QVec& vec);
	void layer();
	bool append_node_to_cur_time_sequence(wireIter qu_wire_itr, SequenceLayer::iterator seq_iter, SequenceLayer& node_vec);
	void append_gate_param(std::string &gate_name, std::shared_ptr<AbstractQGateNode> p_gate, GateType type);
	bool check_ctrl_gate_time_sequence_conflicting(const QVec &control_qubits_vec, const QVec &qubits_vector);
	void fill_layer(TopologicalSequence::iterator lay_iter);
	void get_gate_from_next_layer(TopologicalSequence::iterator to_fill_lay_iter, QVec &unused_qubits_vec, TopologicalSequence::iterator next_lay_iter);

	int get_measure_time_sequence();
	int get_ctrl_node_time_sequence();
	int get_swap_gate_time_sequence();
	int get_single_gate_time_sequence();

private:
	std::map<int, std::shared_ptr<Wire>> m_quantum_bit_wires;
	std::map<int, std::shared_ptr<Wire>> m_class_bit_wires;
	TopologicalSequence m_layer_info;
	std::shared_ptr<GraphMatch> m_p_grapth_dag;
	int m_text_len;
	int m_max_time_sequence;
	QProg &m_prog;
	QProg m_tmp_remain_prog;
	QVec m_quantum_bits_in_use;
	TimeSequenceConfig& m_time_sequence_conf;
};

QPANDA_END
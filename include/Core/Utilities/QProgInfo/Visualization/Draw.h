#pragma once
#include "Core/Utilities/QPandaNamespace.h"
#include <string>
#include <list>
#include <vector>
#include <fstream>
#include "Core/Utilities/QProgTransform/QProgToDAG/GraphMatch.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/QProgDAG.h"
#include "Core/Utilities/Tools/ProcessOnTraversing.h"

QPANDA_BEGIN

namespace DRAW_TEXT_PIC
{
	using TopoSeq = LayeredTopoSeq;
	using TopoSeqIter = LayeredTopoSeq::iterator;
	using TopoSeqLayer = SeqLayer<pOptimizerNodeInfo>;
	using TopoSeqLayerIter = SeqLayer<pOptimizerNodeInfo>::iterator;

	enum TEXT_PIC_TYPE
	{
		LAYER = 0,      /**< draw text-picture by layer */
		TIME_SEQUENCE   /**< draw text-picture by time sequence */
	};

	/**
	* @brief Draw text-picture box
	* @ingroup QProgInfo
	*/
	class DrawBox
	{
	public:
		/**
		* @brief  Constructor of DrawBox
		*/
		DrawBox(const std::string &top_format_str, const std::string &mid_format_str, const std::string &bot_format_str)
			:m_top_format(top_format_str), m_mid_format(mid_format_str), m_bot_format(bot_format_str)
		{}
		virtual ~DrawBox() {}

		/**
		* @brief get top string of box
		*/
		virtual const std::string& getTopStr() const {
			return m_top_format;
		}

		/**
		* @brief get middle string of box
		*/
		virtual const std::string& getMidStr() const {
			return m_mid_format;
		}

		/**
		* @brief get bottom string of box
		*/
		virtual const std::string& getBotStr() const {
			return m_bot_format;
		}

		/**
		* @brief set connected str on the top of box
		*/
		virtual void set_top_connected() {}

		/**
		* @brief set connected str on the bottom of box
		*/
		virtual void set_bot_connected() {}

		/**
		* @brief get box len
		* @return int the length of box
		*/
		virtual int getLen() const = 0;

	protected:
		std::string m_top_format; /**< the top string of box */
		std::string m_mid_format; /**< the middle string of box */
		std::string m_bot_format; /**< the bottom string of box */
	};

	/**
	* @brief the wire of text-picture
	* @ingroup QProgInfo
	*/
	class Wire
	{
	public:
		/**
		* @brief  Constructor of DrawBox
		*/
		Wire(const std::string& connect_str)
			:m_connect_str(connect_str), m_cur_len(0), m_b_merged_bot_line(false), m_time_sequence(0)
		{}
		virtual ~Wire() {}

		/**
		* @brief set the name of wire
		* @param[in] std::string&  name
		* @param[in] size_t  name length
		*/
		virtual void setName(const std::string& name, size_t nameLen) {
			for (size_t i = 0; i < nameLen; i++)
			{
				m_top_line.append(" ");
				m_bot_line.append(" ");
			}
			m_mid_line.append(name);
			m_cur_len = nameLen;
		}

		/**
		* @brief append a box to current wire
		* @param[in] DrawBox&  box
		* @param[in] int  append postion
		* @return int the length of current wire
		*/
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

		/**
		* @brief get the length of current wire
		* @return int the length of current wire
		*/
		virtual int getWireLength() { return m_cur_len; }

		/**
		* @brief conver current wire to string and save to file
		* @return std::string
		*/
		virtual std::string draw() {
			std::string outputStr;
			m_top_line.append("\n");
			outputStr.append(m_top_line);

			m_mid_line.append("\n");
			outputStr.append(m_mid_line);

			if (!m_b_merged_bot_line)
			{
				m_bot_line.append("\n");
				outputStr.append(m_bot_line);
			}

			return outputStr;
		}

		/**
		* @brief update current wire length
		* @param[in] int the new length
		*/
		virtual void updateWireLen(const int len) {
			for (size_t i = m_cur_len; i < len; i++)
			{
				m_top_line.append(" ");
				m_mid_line.append(m_connect_str);
				m_bot_line.append(" ");
			}

			m_cur_len = len;
		}

		/**
		* @brief set whether to merge wire
		* @param[in] bool
		*/
		virtual void setMergedFlag(bool b) { m_b_merged_bot_line = b; }

		/**
		* @brief get top line string
		* @return std::string
		*/
		virtual const std::string& getTopLine() const { return m_top_line; }

		/**
		* @brief get middle line string
		* @return std::string
		*/
		virtual const std::string& getMidLine() const { return m_mid_line; }

		/**
		* @brief get bottom line string
		* @return std::string
		*/
		virtual const std::string& getBotLine() const { return m_bot_line; }

		/**
		* @brief update current wire time sequence
		* @param[in] int the increased time sequence
		*/
		int update_time_sequence(unsigned int increase_time_sequence) { return (m_time_sequence += increase_time_sequence); }

		/**
		* @brief get current wire time sequence
		* @return int
		*/
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

	
	class DrawByLayer;
	class GetUsedQubits;
	class FillLayerByNextLayerNodes;
	class TryToMergeTimeSequence;
	class TryToMergeTimeSequence;
	/**
	* @brief draw text-picture
	* @ingroup QProgInfo
	*/
	class DrawPicture
	{
		using wireElement = std::pair<int, std::shared_ptr<Wire>>;
		using WireIter = std::map<int, std::shared_ptr<Wire>>::iterator;
		
		friend class DrawByLayer;
		friend class GetUsedQubits;
		friend class FillLayerByNextLayerNodes;
		friend class TryToMergeTimeSequence;
		friend class TryToMergeTimeSequence;

	public:
		/**
		* @brief  Constructor of DrawPicture
		*/
		DrawPicture(QProg prog, LayeredTopoSeq& layer_info);
		~DrawPicture() {}

		/**
		* @brief initialize
		* @param[in] std::vector<int>& used qubits
		* @param[in] std::vector<int>& used class bits
		*/
		void init(std::vector<int>& quBits, std::vector<int>& clBits);

		/**
		* @brief display and return the target string
		* @return std::string
		*/
		std::string present();

		/**
		* @brief merge wire line
		*/
		void mergeLine();

		/**
		* @brief get the max length of quantum wire between start_quBit_wire and end_quBit_wire
		* @param[in] WireIter start quBit wire
		* @param[in] WireIter end quBit wire
		* @return int the max length
		*/
		int getMaxQuWireLength(WireIter start_quBit_wire, WireIter end_quBit_wire);

		/**
		* @brief update TextPic length
		*/
		void updateTextPicLen();

		/**
		* @brief get val of wide char
		* @param[in] "unsigned char*" the target wide char
		* @return the val of the wide char
		*/
		unsigned long getWideCharVal(const unsigned char* wide_char) {
			if (nullptr == wide_char)
			{
				return 0;
			}
			return (((unsigned long)(wide_char[0]) << 16) | ((unsigned long)(wide_char[1]) << 8) | (((unsigned long)(wide_char[2]))));
		}

		/**
		* @brief check the target wire time sequence
		* @param[in] TopoSeqIter the target layer
		*/
		void check_time_sequence(TopoSeqIter cur_layer_iter);

		/**
		* @brief update the target wire time sequence
		* @param[in] std::shared_ptr<Wire> the target wire
		* @param[in] int increased time sequence
		*/
		void update_time_sequence(std::shared_ptr<Wire> p_wire, int increased_time_sequence);

		/**
		* @brief append time sequence line to text-picture
		*/
		void append_time_sequence_line();

		/**
		* @brief append layer line to text-picture
		*/
		void append_layer_line();

		/**
		* @brief draw text-picture by layer
		*/
		void draw_by_layer();

		/**
		* @brief draw text-picture by time sequence
		* @param[in] const std::string It can be configuration file or configuration data, which can be distinguished by file suffix,
			 so the configuration file must be end with ".json", default is CONFIG_PATH
		*/
		void draw_by_time_sequence(const std::string config_data = CONFIG_PATH);

		/**
		* @brief get the difference of two QVecs
		* @return QVec vec1 - vec2
		*/
		QVec get_qvec_difference(QVec &vec1, QVec &vec2);

	private:
		void appendMeasure(std::shared_ptr<AbstractQuantumMeasure> pMeasure);
		void append_reset(std::shared_ptr<AbstractQuantumReset> pReset);
		void append_ctrl_gate(std::string gate_name, const int terget_qubit, QVec &self_control_qubits_vec, QVec &circuit_control_qubits_vec);
		void append_barrier_line(int line_start, int line_end, int pos);
		void append_swap_gate(std::string gate_name, QVec &qubits_vector, QVec &circuit_control_qubits_vec);
		void append_single_gate(std::string gate_name, QVec &qubits_vector, QVec &circuit_control_qubits_vec);
		void merge(const std::string& up_wire, std::string& down_wire);
		void set_connect_direction(const int& qubit, const std::vector<int>& vec_qubits, DrawBox& box);
		void append_ctrl_line(int line_start, int line_end, int pos);
		bool check_time_sequence_one_qubit(WireIter qu_wire_itr, TopoSeqIter cur_layer_iter);
		bool is_qubit_in_vec(const int qubit, const QVec& vec);
		void append_gate_param(std::string &gate_name, pOptimizerNodeInfo node_info);
		bool check_ctrl_gate_time_sequence_conflicting(const QVec &control_qubits_vec, const QVec &qubits_vector);
		void fill_layer(TopoSeqIter lay_iter);
		void get_gate_from_next_layer(TopoSeqIter to_fill_lay_iter, QVec &unused_qubits_vec, TopoSeqIter next_lay_iter);
		NodeType sequence_node_type_to_node_type(SequenceNodeType sequence_node_type);

		int get_measure_time_sequence();
		int get_ctrl_node_time_sequence();
		int get_swap_gate_time_sequence();
		int get_single_gate_time_sequence();
		int get_reset_time_sequence();

	private:
		std::map<int, std::shared_ptr<Wire>> m_quantum_bit_wires;
		std::map<int, std::shared_ptr<Wire>> m_class_bit_wires;
		LayeredTopoSeq& m_layer_info;
		int m_text_len;
		int m_max_time_sequence;
		QProg m_prog;
		QProg m_tmp_remain_prog;
		QVec m_quantum_bits_in_use;
		TimeSequenceConfig m_time_sequence_conf;
	};

	/**
	* @brief node handle
	* @ingroup QProgInfo
	*/
	template<typename... Args >
	class AbstractHandleNodes
	{
	public:
		/**
		* @brief handle measure node
		*/
		virtual void handle_measure_node(Args&& ... func_args) = 0;

		/**
		* @brief handle reset node
		*/
		virtual void handle_reset_node(Args&& ... func_args) = 0;

		/**
		* @brief handle gate node
		*/
		virtual void handle_gate_node(Args&& ... func_args) = 0;

		/**
		* @brief handle work run
		*/
		virtual void handle_work(const NodeType node_t, Args&& ... func_args) {
			switch (node_t)
			{
			case GATE_NODE:
				handle_gate_node(std::forward<Args>(func_args)...);
				break;

			case MEASURE_GATE:
				handle_measure_node(std::forward<Args>(func_args)...);
				break;

			case RESET_NODE:
				handle_reset_node(std::forward<Args>(func_args)...);
				break;

			default:
				break;
			}
		}
	};

	/**
	* @brief draw layer nodes
	* @ingroup QProgInfo
	*/
	class DrawByLayer : public AbstractHandleNodes<std::shared_ptr<QNode>&, pOptimizerNodeInfo&>
	{
	public:
		DrawByLayer(DrawPicture& parent)
			:m_parent(parent)
		{}
		~DrawByLayer() {}

		void handle_measure_node(std::shared_ptr<QNode>& p_node, pOptimizerNodeInfo& p_node_info) override;
		void handle_reset_node(std::shared_ptr<QNode>& p_node, pOptimizerNodeInfo& p_node_info) override;
		void handle_gate_node(std::shared_ptr<QNode>& p_node, pOptimizerNodeInfo& p_node_info) override;

	private:
		DrawPicture& m_parent;
	};

	/**
	* @brief get all used qubits
	* @ingroup QProgInfo
	*/
	class GetUsedQubits : public AbstractHandleNodes<std::shared_ptr<QNode>&>
	{
	public:
		GetUsedQubits(DrawPicture& parent, QVec &vec)
			:m_parent(parent), m_qubits_vec(vec)
		{}
		~GetUsedQubits() {}

		void handle_measure_node(std::shared_ptr<QNode>& p_node) override;
		void handle_reset_node(std::shared_ptr<QNode>& p_node) override;
		void handle_gate_node(std::shared_ptr<QNode>& p_node) override;

	private:
		DrawPicture& m_parent;
		QVec &m_qubits_vec;
	};

	/**
	* @brief Fill layer by next layer nodes
	* @ingroup QProgInfo
	*/
	class FillLayerByNextLayerNodes : public AbstractHandleNodes<TopoSeqLayerIter&>
	{
	public:
		FillLayerByNextLayerNodes(DrawPicture& parent, QVec &unused_qubits_vec, TopoSeqLayer &target_layer, TopoSeqLayer &next_layer)
			:m_parent(parent), m_unused_qubits_vec(unused_qubits_vec), m_target_layer(target_layer), m_next_layer(next_layer)
			, m_b_got_available_node(false)
		{}
		~FillLayerByNextLayerNodes() {}

		void handle_measure_node(TopoSeqLayerIter& itr_on_next_layer) override;
		void handle_reset_node(TopoSeqLayerIter& itr_on_next_layer) override;
		void handle_gate_node(TopoSeqLayerIter& itr_on_next_layer) override;

		/**
		* @brief judge whether get available node
		* @return bool if got available node, return true, or else return false
		*/
		bool have_got_available_node() {
			if (m_b_got_available_node)
			{
				m_b_got_available_node = false;
				return true;
			}

			return false;
		}

	private:
		template<typename Node>
		void handle_single_qubit_node(Node &tag_node, TopoSeqLayerIter& itr_on_next_layer) {
			for (auto qubit_iter = m_unused_qubits_vec.begin(); qubit_iter != m_unused_qubits_vec.end(); ++qubit_iter)
			{
				auto& tmp_qubit = *qubit_iter;
				if (tag_node.getQuBit() == tmp_qubit)
				{
					m_target_layer.push_back(*itr_on_next_layer);
					m_unused_qubits_vec.erase(qubit_iter);
					itr_on_next_layer = m_next_layer.erase(itr_on_next_layer);
					m_b_got_available_node = true;
					break;
				}
			}
		}

	private:
		DrawPicture& m_parent;
		QVec &m_unused_qubits_vec;
		TopoSeqLayer &m_target_layer;
		TopoSeqLayer &m_next_layer;
		bool m_b_got_available_node;
	};

	/**
	* @brief Try to merge time sequence
	* @ingroup QProgInfo
	*/
	class TryToMergeTimeSequence : public AbstractHandleNodes<DrawPicture::WireIter&, TopoSeqLayerIter&, bool&>
	{
	public:
		TryToMergeTimeSequence(DrawPicture& parent, TopoSeqLayer &next_layer)
			:m_parent(parent), m_next_layer(next_layer)
			, m_b_continue_recurse(true)
		{}
		~TryToMergeTimeSequence() {}

		void handle_measure_node(DrawPicture::WireIter& cur_qu_wire, TopoSeqLayerIter& itr_on_next_layer, bool &b_found_node_on_cur_qu_wire) override;
		void handle_reset_node(DrawPicture::WireIter& cur_qu_wire, TopoSeqLayerIter& itr_on_next_layer, bool &b_found_node_on_cur_qu_wire) override;
		void handle_gate_node(DrawPicture::WireIter& cur_qu_wire, TopoSeqLayerIter& itr_on_next_layer, bool &b_found_node_on_cur_qu_wire) override;

		/**
		* @brief judge whether to continue recurse
		* @return bool if could to continue recurse, return true, or else return false
		*/
		bool could_continue_merge() { return m_b_continue_recurse; }

	private:
		void try_to_append_gate_to_cur_qu_wire(DrawPicture::WireIter &qu_wire_itr, TopoSeqLayerIter& seq_iter, TopoSeqLayer& node_vec);

	private:
		DrawPicture& m_parent;
		TopoSeqLayer &m_next_layer;
		bool m_b_continue_recurse;
	};
}
QPANDA_END
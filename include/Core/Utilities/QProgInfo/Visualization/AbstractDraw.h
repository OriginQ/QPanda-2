#pragma once
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/Utilities/Tools/ProcessOnTraversing.h"

QPANDA_BEGIN

enum PIC_TYPE
{
	TEXT,
	LATEX
};

class AbstractDraw
{
public:
	/*AbstractDraw(const QProg &prog, LayeredTopoSeq &layer_info, uint32_t length)
		: m_prog(prog),
		  m_layer_info(layer_info),
		  m_wire_length(length) {}*/
	AbstractDraw(const QProg& prog, LayeredTopoSeq& layer_info, uint32_t length,bool b_with_gate_params)
		: m_prog(prog),
		m_layer_info(layer_info),
		m_wire_length(length),
		m_draw_with_gate_params(b_with_gate_params)
	{}
	virtual ~AbstractDraw() {}

	/**
	 * @brief initialize
	 * 
	 * @param[in] quBits std::vector<int>& used qubits
	 * @param[in] clBits std::vector<int>& used class bits
	 */
	virtual void init(std::vector<int> &quBits, std::vector<int> &clBits) = 0;

	/**
	 * @brief draw picture by layer
	 * 
	 */
	virtual void draw_by_layer() = 0;

	/**
	 * @brief draw picture by time sequence
	 * 
	 * @param[in] config_data const std::string It can be configuration file or configuration data, 
	 						  which can be distinguished by file suffix,
			 				  so the configuration file must be end with ".json", default is CONFIG_PATH
	 */
	virtual void draw_by_time_sequence(const std::string config_data = CONFIG_PATH) = 0;

	/**
	 * @brief display and return the target string
	 * 
	 * @param[in] file_name 
	 * @return std::string 
	 */
	virtual std::string present(const std::string &file_name) = 0;

protected:
	QProg m_prog;
	LayeredTopoSeq &m_layer_info;
	uint32_t m_wire_length;
	bool m_draw_with_gate_params;
};

QPANDA_END
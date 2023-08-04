#pragma once
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/GraphMatch.h"
#include "Core/Utilities/QProgInfo/Visualization/AbstractDraw.h"
#include "Core/Utilities/QProgInfo/Visualization/DrawTextPic.h"
#include "Core/Utilities/QProgInfo/Visualization/DrawLatex.h"

QPANDA_BEGIN

namespace DRAW_TEXT_PIC
{
	enum LAYER_TYPE
	{
		LAYER = 0,	  /**< draw text-picture by layer */
		TIME_SEQUENCE /**< draw text-picture by time sequence */
	};
	/**
    * @brief draw Qprog by text
    * @ingroup Utilities
    */
	class DrawQProg
	{
	public:
		/**
		* @brief  Constructor of DrawQProg
		*/
		DrawQProg(QProg& prg, const NodeIter node_itr_start, const NodeIter node_itr_end, bool b_draw_with_gate_params, const std::string& output_file = "");
		~DrawQProg();

		/**
		* @brief Draw text-picture
		* @param[in] LAYER_TYPE draw type
		* @param[in] PIC_TYPE draw text pic or latex
		* @param[in] const std::string It can be configuration file or configuration data, which can be distinguished by file suffix,
			 so the configuration file must be end with ".json", default is CONFIG_PATH
		* @return std::string the text-picture
		* @see LAYER_TYPE
		* @see PIC_TYPE
		*/
		std::string textDraw(LAYER_TYPE t, PIC_TYPE p = PIC_TYPE::TEXT, bool with_logo = false , uint32_t length = 100, const std::string config_data = CONFIG_PATH);

	private:
		QProg m_prog;
		AbstractDraw *m_drawer;
		std::vector<int> m_quantum_bits_in_use;
		std::vector<int> m_class_bits_in_use;
		LayeredTopoSeq m_layer_info;
		const std::string &m_output_file;
		bool m_draw_with_gate_params;
	};
}

QPANDA_END
#pragma once
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/GraphMatch.h"
#include "Core/Utilities/QProgInfo/Visualization/Draw.h"

QPANDA_BEGIN

namespace DRAW_TEXT_PIC
{ 
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
		DrawQProg(QProg &prg, const NodeIter node_itr_start, const NodeIter node_itr_end);
		~DrawQProg();

		/**
		* @brief Draw text-picture
		* @param[in] TEXT_PIC_TYPE draw type
		* @param[in] const std::string It can be configuration file or configuration data, which can be distinguished by file suffix,
			 so the configuration file must be end with ".json", default is CONFIG_PATH
		* @return std::string the text-picture
		* @see TEXT_PIC_TYPE
		*/
		std::string textDraw(TEXT_PIC_TYPE t, const std::string config_data = CONFIG_PATH);

	private:
		QProg m_prog;
		DrawPicture* m_p_text;
		std::vector<int> m_quantum_bits_in_use;
		std::vector<int> m_class_bits_in_use;
		LayeredTopoSeq m_layer_info;
	};
}

QPANDA_END
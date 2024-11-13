#ifndef _QVISUALIZATION_H_
#define _QVISUALIZATION_H_

#include "DrawQProg.h"
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/Utilities/QProgInfo/Visualization/CharsTransform.h"

QPANDA_BEGIN

/**
 * @brief output a quantum prog/circuit to console by text-pic(UTF-8 code),
   and will save the text-pic in file named QCircuitTextPic.txt in the same time in current path.
 * @ingroup Utilities
 * @param[in] prog the source prog
 * @param[in] uint32_t The max length of text-pic, will auto-wrap
 * @param[in] const std::string& The output-file, if output-file is empty, do not output to file.
 * @param[in] itr_start The start pos, default is the first node of the prog
 * @param[in] itr_end The end pos, default is the end node of the prog
 * @param[in] with_gate_params If it's true, draw gates with gates'params. If it's false, draw gates without gates'params. It's default is false
 * @return the output string
 * @note All the output characters are UTF-8 encoded.
 */

std::string draw_qprog(QProg prog, PIC_TYPE p = PIC_TYPE::TEXT, bool with_logo = false, bool with_gate_params=false, uint32_t length = 100, const std::string& output_file = "",
		const NodeIter itr_start = NodeIter(), const NodeIter itr_end = NodeIter());

std::string draw_qprog(QProg prog, LayeredTopoSeq& m_layer_info, PIC_TYPE p = PIC_TYPE::TEXT , bool with_logo = false , bool with_gate_params =false ,uint32_t length = 100,
	const std::string& output_file = "");

/**
 * @brief output a quantum prog/circuit by time sequence to console by text-pic(UTF-8 code),
   and will save the text-pic in file named QCircuitTextPic.txt in the same time in current path.
 * @ingroup Utilities
 * @param[in] prog  the source prog
 * @param[in] const std::string It can be configuration file or configuration data, which can be distinguished by file suffix,
 			 so the configuration file must be end with ".json", default is CONFIG_PATH
 * @param[in] itr_start The start pos, default is the first node of the prog
 * @param[in] itr_end The end pos, default is the end node of the prog
 * @param[in] with_gate_params If it's true, draw gates with gates'params. If it's false, draw gates without gates'params. It's default is false
 * @return the output string
 * @note All the output characters are GBK encoded on windows,  UTF-8 encoded on other OS.
 */

std::string draw_qprog_with_clock(QProg prog, PIC_TYPE p = PIC_TYPE::TEXT, const std::string config_data = CONFIG_PATH, bool with_logo = false,
	bool with_gate_params = false, uint32_t length = 100, const std::string& output_file = "", const NodeIter itr_start = NodeIter(), const NodeIter itr_end = NodeIter());

/**
 * @brief Overload operator <<
 * @ingroup Utilities
 * @param[in] std::ostream&  ostream
 * @param[in] QProg quantum program
 * @return std::ostream 
 */
inline std::ostream &operator<<(std::ostream &out, QProg prog)
{
	auto text_pic_str = draw_qprog(prog,QPanda::TEXT,false,true,100U,"");

	if (&out == &std::cout)
	{
#if defined(WIN32) || defined(_WIN32)
		text_pic_str = fit_to_gbk(text_pic_str);
		text_pic_str = Utf8ToGbkOnWin32(text_pic_str.c_str());
#endif
	}

	out << text_pic_str << std::endl;
	return out;
}

QPANDA_END

#endif

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
* @param[in] prog  the source prog
* @param[in] itr_start The start pos, default is the first node of the prog
* @param[in] itr_end The end pos, default is the end node of the prog
* @return the output string
* @note All the output characters are UTF-8 encoded.
*/
std::string draw_qprog(QProg prog, uint32_t length = 100, bool b_out_put_to_file = false, const NodeIter itr_start = NodeIter(), const NodeIter itr_end = NodeIter());
std::string draw_qprog(QProg prog, LayeredTopoSeq& m_layer_info, uint32_t length = 100, bool b_out_put_to_file = false);

/**
* @brief output a quantum prog/circuit by time sequence to console by text-pic(UTF-8 code),
  and will save the text-pic in file named QCircuitTextPic.txt in the same time in current path.
* @ingroup Utilities
* @param[in] prog  the source prog
* @param[in] const std::string It can be configuration file or configuration data, which can be distinguished by file suffix,
			 so the configuration file must be end with ".json", default is CONFIG_PATH
* @param[in] itr_start The start pos, default is the first node of the prog
* @param[in] itr_end The end pos, default is the end node of the prog
* @return the output string
* @note All the output characters are GBK encoded on windows,  UTF-8 encoded on other OS.
*/
std::string draw_qprog_with_clock(QProg prog, const std::string config_data = CONFIG_PATH, uint32_t length = 100, bool b_out_put_to_file = false, const NodeIter itr_start = NodeIter(), const NodeIter itr_end = NodeIter());

/**
 * @brief Overload operator <<
 * @ingroup Utilities
 * @param[in] std::ostream&  ostream
 * @param[in] QProg quantum program
 * @return std::ostream 
 */
inline std::ostream  &operator<<(std::ostream &out, QProg prog) {
	auto text_pic_str = draw_qprog(prog);

	if (&out == &std::cout){
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

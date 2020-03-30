#pragma once
#include "DrawQProg.h"
#include "Core/Utilities/QPandaNamespace.h"

QPANDA_BEGIN

/**
* @brief output a quantum prog/circuit to console by text-pic(UTF-8 code),
  and will save the text-pic in file named QCircuitTextPic.txt in the same time in current path.
* @ingroup Utilities
* @param[in] prog  the source prog
* @param[in] itr_start The start pos, default is the first node of the prog
* @param[in] itr_end The end pos, default is the end node of the prog
* @return the output string
* @note All the output characters are UTF-8 code.
*/
std::string draw_qprog(QProg prog, const NodeIter itr_start = NodeIter(), const NodeIter itr_end = NodeIter());

/**
* @brief output a quantum prog/circuit by time sequence to console by text-pic(UTF-8 code),
  and will save the text-pic in file named QCircuitTextPic.txt in the same time in current path.
* @ingroup Utilities
* @param[in] prog  the source prog
* @param[in] itr_start The start pos, default is the first node of the prog
* @param[in] itr_end The end pos, default is the end node of the prog
* @return the output string
* @note All the output characters are UTF-8 code.
*/
std::string draw_qprog_with_clock(QProg prog, const NodeIter itr_start = NodeIter(), const NodeIter itr_end = NodeIter());

/**
 * @brief Overload operator <<
 * @ingroup Utilities
 * @param[in] std::ostream&  ostream
 * @param[in] QProg quantum program
 * @return std::ostream 
 */
inline std::ostream  &operator<<(std::ostream &out, QProg prog) {
	std::cout << draw_qprog(prog);
	return std::cout;
}

QPANDA_END
#ifndef PYQPANDA_MAXCUTGENERATOR_H
#define PYQPANDA_MAXCUTGENERATOR_H

#include <vector>
#include <map>
#include "Core/Utilities/QPandaNamespace.h"
QPANDA_BEGIN

/**
* @brief vector dot product
* @ingroup MaxCutProblemGenerator
* @param[in] std::vector<double>& vector x, x will be clear
* @param[in] std::vector<double>& vector y, y will be clear
* @return double the dot product result of the two input vectors
*/
double vector_dot(std::vector<double> &x, std::vector<double> &y);

/**
* @brief all cut of graph
* @ingroup MaxCutProblemGenerator
* @param[in] std::vector<std::vector<double>> the adjacent matrix
* @param[out] std::vector<double>& all cut list
* @param[out] std::vector<size_t>& target value list
* @return double the max  cut value
*/
double all_cut_of_graph(std::vector<std::vector<double>> adjacent_matrix,
    std::vector<double> & all_cut_list, 
    std::vector<size_t> & target_value_list);
QPANDA_END

#endif // ! PYQPANDA_MAXCUTGENERATOR_H
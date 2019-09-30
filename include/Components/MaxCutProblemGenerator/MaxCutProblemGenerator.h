#ifndef PYQPANDA_MAXCUTGENERATOR_H
#define PYQPANDA_MAXCUTGENERATOR_H

#include <vector>
#include <map>
#include "Core/Utilities/QPandaNamespace.h"
QPANDA_BEGIN
//namespace py = pybind11;

double vector_dot(std::vector<double> &x, std::vector<double> &y);
double all_cut_of_graph(std::vector<std::vector<double>> adjacent_matrix,
    std::vector<double> & all_cut_list, 
    std::vector<size_t> & target_value_list);
QPANDA_END

#endif // ! PYQPANDA_MAXCUTGENERATOR_H
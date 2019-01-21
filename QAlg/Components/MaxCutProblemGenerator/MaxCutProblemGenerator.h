#ifndef PYQPANDA_MAXCUTGENERATOR_H
#define PYQPANDA_MAXCUTGENERATOR_H

//#include "pybind11/pybind11.h"
//#include "pybind11/stl.h"
#include "vector"
#include <map>

using std::vector;
using std::map;
//namespace py = pybind11;

double vector_dot(vector<double> &x, vector<double> &y);
double all_cut_of_graph(vector<vector<double>> adjacent_matrix,
    vector<double> & all_cut_list, 
    vector<size_t> & target_value_list);

#endif // ! PYQPANDA_MAXCUTGENERATOR_H
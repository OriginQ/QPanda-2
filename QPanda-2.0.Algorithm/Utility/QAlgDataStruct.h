/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

QAlgDataStruct.h

Author: LiYe
Created in 2018-09-06


*/
#ifndef QALGDATASTRUCT_H
#define QALGDATASTRUCT_H

#include <map>
#include <vector>
#include <functional>
#include <complex>

namespace QPanda {

#define DEF_WARING                    ("Warning: ")
#define DEF_ITERATIONS                ("Iterations")
#define DEF_EVALUATIONS               ("Function evaluations")
#define DEF_VALUE                     ("Value")
#define DEF_KEY                       ("Key")
#define DEF_MESSAGE                   ("Message")
#define DEF_OPTI_STATUS_PARA_ERROR    ("Optimizer parameter setting error.")
#define DEF_OPTI_STATUS_CALCULATING   ("Calculating")
#define DEF_OPTI_STATUS_SUCCESS       ("Optimization terminated successfully.")
#define DEF_OPTI_STATUS_MAX_FEV       ("Maximum number of function evaluations" \
                                       " has been exceeded.")
#define DEF_OPTI_STATUS_MAX_ITER      ("Maximum number of iterations has been" \
                                       " exceeded.")
#ifndef DEF_UNINIT_INT
#define DEF_UNINIT_INT                (-1234567)
#endif

#ifndef DEF_UNINIT_FLOAT
#define DEF_UNINIT_FLOAT              (-1234567.0)
#endif

#define Q_PI       3.14159265358979323846   // pi
#define Q_PI_2     1.57079632679489661923   // pi/2

struct QGraphItem
{
    size_t first;
    size_t second;
    double weight;
    QGraphItem():
        first(0),
        second(0),
        weight(0.0)
    {}

    QGraphItem(size_t first, size_t second, double weight):
        first(first),
        second(second),
        weight(weight)
    {}
};

using complex_d = std::complex<double>;
using vector_d = std::vector<double>;
using vector_s = std::vector<std::string>;
using QGraph = std::vector<QGraphItem>;
using OptimizationResult = std::map<std::string, std::string>;
using QResultPair = std::pair<std::string, double>;
using QFunc = std::function<QResultPair(vector_d)>;
using QUserDefinedFunc = std::function<double(const std::string &)>;

/*
Note:
	The QTerm value char only will be 'X','Y','Z'.
	If QTerm is empty it reperents 'I'.
*/
using QTerm = std::map<size_t, char>;
using QTermPair = std::pair<size_t, char>;
using QHamiltonianItem = std::pair<QTerm, double>;
using QHamiltonian = std::vector<QHamiltonianItem>;
using QPauliPair = std::pair<QTerm, std::string>;
using QPauliItem = std::pair<QPauliPair, complex_d>;
using QPauli = std::vector<QPauliItem>;
using QPauliMap = std::map<std::string, complex_d>;
using QIndexMap = std::map<size_t, size_t>;

enum Optimizer
{
    NELDER_MEAD
};

}

#endif // QALGDATASTRUCT_H

/*
Copyright (c) 2017-2023 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

AbstractOptimizer.h

Author: LiYe
Created in 2021-5-10

*/

#pragma once

#include<string>
#include<vector>
#include<functional>
#include "Core/Utilities/QPandaNamespace.h"
#include "OptimizerMacro.h"

QPANDA_BEGIN

/**
* @brief optimization result structure
*/
struct QOptimizationResult
{
    std::string message;
    size_t iters;           /**< iteration count. */
    size_t fcalls;          /**< function call count. */
    std::string key;        /**< problem solution. */
    double fun_val;         /**< minimun value of the problem. */
    std::vector<double> para;          /**< optimized para */
};

/**
* @brief Optimizer Type
*/
enum class OptimizerType
{
    NELDER_MEAD,
    POWELL,
    COBYLA,
    GRADIENT,
    GRAD_DIRCTION,
    L_BFGS_B,
    SLSQP
};

using QResultPair = std::pair<std::string, double>;
using QOptFunc = std::function <QResultPair(const std::vector<double> &x,
                                            std::vector<double>& grad,
                                            size_t, size_t)>;

extern bool operator < (const QResultPair &p1, const QResultPair &p2);
extern bool operator < (const QResultPair &p1, const double &coef);
extern bool operator <= (const QResultPair &p1, const QResultPair &p2);
extern bool operator <= (const QResultPair &p1, const double &coef);
extern bool operator > (const QResultPair &p1, const QResultPair &p2);
extern bool operator >= (const QResultPair &p1, const QResultPair &p2);
extern double operator - (const QResultPair &p1, const QResultPair &p2);
extern double operator + (const QResultPair &p1, const QResultPair &p2);
extern double operator * (const double &coef, const QResultPair &p);
extern double operator * (const QResultPair &p, const double &coef);

QPANDA_END

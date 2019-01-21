/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

AbstractOptimizer.h

Author: LiYe
Created in 2018-09-06


*/

#ifndef ABSTRACTOPTIMIZER_H
#define ABSTRACTOPTIMIZER_H

#include <vector>
#include <functional>
#include "QAlg/DataStruct.h"

namespace QPanda
{

    /*

    Abstract Optimizer

    */
    class AbstractOptimizer
    {
    public:
        AbstractOptimizer();
        AbstractOptimizer(const AbstractOptimizer &) = delete;
        AbstractOptimizer& operator = (const AbstractOptimizer &) = delete;
        virtual ~AbstractOptimizer();

        void registerFunc(const QFunc &func, const vector_d &func_para)
        {
            m_func = func;
            m_optimized_para = func_para;
        }

        void setDisp(bool disp)
        {
            m_disp = disp;
        }

        void setAdaptive(bool adaptive)
        {
            m_adaptive = adaptive;
        }

        void setXatol(double xatol)
        {
            m_xatol = xatol;
        }

        void setFatol(double fatol)
        {
            m_fatol = fatol;
        }

        void setMaxFCalls(size_t max_fcalls)
        {
            m_max_fcalls = max_fcalls;
        }

        void setMaxIter(size_t max_iter)
        {
            m_max_iter = max_iter;
        }

        // only for test
        void setTestValueAndParaFile(double test_value, const std::string &filename)
        {
            m_test_value = test_value;
            m_para_file = filename;
        }

        virtual void exec() = 0;

        virtual QOptimizationResult getResult()
        {
            return m_result;
        }

    protected:
        QFunc m_func;

        vector_d m_optimized_para;

        bool m_disp;        // Whether to print the log to the terminal
        bool m_adaptive;    // Para of Nelder-Mead.
                            // Adapt algorithm parameters to dimensionality
                            // of problem.Useful for high-dimensional
                            // minimization. [Optional]

        double m_xatol;     // Absolute error in xopt between iterations that is
                            // acceptable for convergence. [Optional]

        double m_fatol;     // Absolute error in func(xopt) between iterations that is
                            // acceptable for convergence. [Optional]

        double m_test_value;// user test value
        std::string m_para_file;

        size_t m_max_fcalls;// Maximum allowed number of function evaluations
        size_t m_max_iter;  // Maximum allowed number of iterations

        QOptimizationResult m_result;
    };

}

#endif // ABSTRACTOPTIMIZER_H

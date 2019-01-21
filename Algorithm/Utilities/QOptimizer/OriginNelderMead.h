/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

OriginNelderMead.h

Author: LiYe
Created in 2018-09-06


*/

#ifndef ORIGINNELDERMEAD_H
#define ORIGINNELDERMEAD_H

#include "Eigen/Dense"
#include "AbstractQOptimizer.h"

namespace QPanda
{

    /*

    Minimization of scalar function of one or more variables using the
    Nelder-Mead algorithm.

    */
    class OriginNelderMead : public AbstractQOptimizer
    {
    public:
        OriginNelderMead();
        OriginNelderMead(const OriginNelderMead &) = delete;
        OriginNelderMead& operator = (const OriginNelderMead &) = delete;

        virtual void exec();

        /*

        Get the result.
        return:
            OptimizationResult

        Note:
            The returned result contains the following information.

            [
                key: DEF_MESSAGE ("Message")
                value: DEF_OPTI_STATUS_SUCCESS
                    or DEF_OPTI_STATUS_MAX_FEV
                    or DEF_OPTI_STATUS_MAX_ITER
            ]

            [
                key: DEF_VALUE
                value: m_fsim[0], minimum value of the graph.
            ]

            [
                key: DEF_KEY
                value: m_key[0], problem solution.
            ]

            [
                key: DEF_ITERATIONS
                value: m_iter, iteration count.
            ]

            [
                key: DEF_EVALUATIONS
                value: m_fcalls, function call count.
            ]
         */
        virtual OptimizationResult getResult();
    private:
        bool init();
        void adaptFourPara();
        void adaptTerminationPara();
        void initialSimplex();

        QResultPair callFunc(const Eigen::VectorXd &para);
        bool testTermination();
        void calcCentroid();
        void sortData();
        std::vector<size_t> sortVector(Eigen::VectorXd& vec);
        void dispResult();
    private:
        double m_rho;       // | ρ <=> α | para of nelder-mead
        double m_chi;       // | χ <=> γ | para of nelder-mead
        double m_psi;       // | ψ <=> β | para of nelder-mead
        double m_sigma;     // | σ <=> δ | para of nelder-mead

        double m_nonzdelt;
        double m_zdelt;

        size_t m_fcalls;
        size_t m_iter;
        size_t m_n;

        Eigen::VectorXd m_x0;
        Eigen::VectorXd m_centroid;
        Eigen::VectorXd m_fsim;
        Eigen::MatrixXd m_sim;

        vector_s m_key;
    };
}

#endif // ORIGINNELDERMEAD_H

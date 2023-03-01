/*
Copyright (c) 2017-2023 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

OriginNelderMead.h

Author: LiYe
Created in 2021-01-02


*/

#ifndef ORIGINGRADIENT_H
#define ORIGINGRADIENT_H

#include "Components/Optimizer/AbstractOptimizer.h"
#include "Core/Utilities/QPandaNamespace.h"
#include "Eigen/Dense"

QPANDA_BEGIN

    /**
    * @brief  Minimization of Gradient Descent algorithm.
    * @ingroup Optimizer
    */
    class OriginGradient : public AbstractOptimizer
    {
    public:
        /**
        * @brief  Constructor of OriginNelderMead
        */
        OriginGradient();
        OriginGradient(const OriginGradient&) = delete;
        OriginGradient& operator = (const OriginGradient&) = delete;

        virtual void exec();
        virtual QOptimizationResult getResult();
    protected:
        virtual void init();
        bool testTermination();
        void dispResult();
        bool saveParaToCache();
        bool restoreParaFromCache();
    private:
        size_t m_fcalls;
        size_t m_iter;
        double m_learning_rate;
        Eigen::VectorXd m_fsim;
        Eigen::MatrixXd m_sim;
        std::vector<double> m_gradient;
    };

QPANDA_END

#endif // ORIGINGRADIENT_H

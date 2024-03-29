/*
Copyright (c) 2017-2023 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

OriginNelderMead.h

Author: LiYe
Created in 2018-09-06


*/

#ifndef ORIGINNELDERMEAD_H
#define ORIGINNELDERMEAD_H

#include "ThirdParty/Eigen/Dense"
#include "Components/Optimizer/AbstractOptimizer.h"

namespace QPanda
{
	/**
	* @brief  Minimization of scalar function of one or more variables using the
              Nelder-Mead algorithm.
    * @ingroup Optimizer
	*/
    class OriginNelderMead : public AbstractOptimizer
    {
    public:
		/**
		* @brief  Constructor of OriginNelderMead
		*/
        OriginNelderMead();
        OriginNelderMead(const OriginNelderMead &) = delete;
        OriginNelderMead& operator = (const OriginNelderMead &) = delete;

        virtual void exec();
        virtual QOptimizationResult getResult();
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
        void writeToFile();
        void saveParaToCache();
        bool restoreParaFromCache();
    private:
        double m_rho;       /**< | ρ <=> α | para of nelder-mead */
        double m_chi;       /**< | χ <=> γ | para of nelder-mead */
        double m_psi;       /**< | ψ <=> β | para of nelder-mead */
        double m_sigma;     /**< | σ <=> δ | para of nelder-mead */

        double m_nonzdelt;
        double m_zdelt;

        size_t m_fcalls;
        size_t m_iter;
        size_t m_n;

        Eigen::VectorXd m_x0;
        Eigen::VectorXd m_centroid;
        Eigen::VectorXd m_fsim;
        Eigen::MatrixXd m_sim;

        std::vector<std::string> m_key;
    };
}

#endif // ORIGINNELDERMEAD_H

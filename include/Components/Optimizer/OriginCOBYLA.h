/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

OriginCOBYLA.h

Author: LiuYan
Created in 2020/11/27


*/

#ifndef ORIGINCOBYLA_H
#define ORIGINCOBYLA_H

#include "ThirdParty/nlopt/nlopt-internal.h"
#include "Components/Optimizer/AbstractOptimizer.h"

#define INF HUGE_VAL

namespace QPanda
{
	/**
	* @brief  Minimization of scalar function of one or more variables using the
              COBYLA algorithm.
    * @ingroup Optimizer
	*/
    class OriginCOBYLA : public AbstractOptimizer
    {
    public:
		/**
		* @brief  Constructor of OriginCOBYLA
		*/
        OriginCOBYLA();
        OriginCOBYLA(const OriginCOBYLA&) = delete;
        OriginCOBYLA& operator = (const OriginCOBYLA&) = delete;

        virtual void exec();
        void set_lower_and_upper_bounds(vector_d& lower_bound, vector_d& upper);
        void add_equality_constraint(QOptFunc func);
        void add_inequality_constraint(QOptFunc func);
    private:
        void dispResult();
        void init();
        void outputResult();

    private:
        size_t m_dimension;
        size_t m_fcalls;
        size_t m_iter;
        double f_min;
        double* x;

        nlopt_opt_s opter;

    };
}

#endif // ORIGINNELDERMEAD_H

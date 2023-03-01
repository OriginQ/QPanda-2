/*
Copyright (c) 2017-2023 Origin Quantum Computing. All Right Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef ORIGINBASICOPTNL_H
#define ORIGINBASICOPTNL_H

#include "ThirdParty/nlopt/include/nlopt-internal.h"
#include "Components/Optimizer/AbstractOptimizer.h"
#include "Components/Optimizer/OptimizerFactory.h"

#define INF HUGE_VAL

namespace QPanda
{
    /**
    * @brief  Minimization of scalar function of one or more variables using the
              COBYLA/LBFGSB/SLSQP algorithm.
    * @ingroup Optimizer
    */
    class OriginBasicOptNL : public AbstractOptimizer
    {
    public:
        /**
        * @brief  Constructor of OriginBasicOptNL
        */
        OriginBasicOptNL(OptimizerType opt_type);
        OriginBasicOptNL(const OriginBasicOptNL&) = delete;
        OriginBasicOptNL& operator = (const OriginBasicOptNL&) = delete;

        virtual void exec();
        void registerFunc(const QOptFunc& func, const std::vector<double>& optimized_para);
        void set_lower_and_upper_bounds(std::vector<double>& lower_bound, std::vector<double>& upper_bound);
        void add_equality_constraint(QOptFunc func);
        void add_inequality_constraint(QOptFunc func);
        void setXatol(double xatol);
        void setXrtol(double xrtol);
        void setFatol(double fatol);
        void setFrtol(double frtol);
        void setMaxFCalls(size_t max_fcalls);
        void setMaxIter(size_t max_iter);
    private:
        void dispResult();
        void init();
        void outputResult();
        nlopt_func function_transform(QOptFunc func);

    private:
        OptimizerType opt_type;
        size_t m_dimension;
        size_t m_fcalls;
        size_t m_iter;
        double f_min, m_xrtol, m_frtol;
        std::vector<double> lb, ub;

        nlopt_func obj_func, inequality_constraint, equality_constraint;
        nlopt_opt_s opter;

    };
};

#endif // ORIGINNELDERMEAD_H
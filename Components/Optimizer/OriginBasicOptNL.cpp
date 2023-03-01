#include "Components/Optimizer/OriginBasicOptNL.h"
#include "Core/Utilities/Tools/OriginCollection.h"
#include "Core/Utilities/Tools/QString.h"

const std::string COBYLA_CACHE_HEADER = "slsqp_break_point.json";

std::map<QPanda::OptimizerType, nlopt_algorithm> algorithm_map = {
    {QPanda::OptimizerType::COBYLA, nlopt_algorithm::NLOPT_LN_COBYLA},
    {QPanda::OptimizerType::L_BFGS_B, nlopt_algorithm::NLOPT_LD_LBFGSB},
    {QPanda::OptimizerType::SLSQP, nlopt_algorithm::NLOPT_LD_SLSQP}
};

std::map<nlopt_result, std::string> result_map = {
    {nlopt_result::NLOPT_FAILURE, "NLOPT_FAILURE"},
    {nlopt_result::NLOPT_INVALID_ARGS, "NLOPT_INVALID_ARGS"},
    {nlopt_result::NLOPT_OUT_OF_MEMORY, "NLOPT_OUT_OF_MEMORY"},
    {nlopt_result::NLOPT_ROUNDOFF_LIMITED, "NLOPT_ROUNDOFF_LIMITED"},
    {nlopt_result::NLOPT_FORCED_STOP, "NLOPT_FORCED_STOP"},
    {nlopt_result::NLOPT_SUCCESS, "NLOPT_SUCCESS"},
    {nlopt_result::NLOPT_STOPVAL_REACHED, "NLOPT_STOPVAL_REACHED"},
    {nlopt_result::NLOPT_FTOL_REACHED, "NLOPT_FTOL_REACHED"},
    {nlopt_result::NLOPT_XTOL_REACHED, "NLOPT_XTOL_REACHED"},
    {nlopt_result::NLOPT_MAXEVAL_REACHED, "NLOPT_MAXEVAL_REACHED"},
    {nlopt_result::NLOPT_MAXTIME_REACHED, "NLOPT_MAXTIME_REACHED"},
    {nlopt_result::NLOPT_MAXITER_REACHED, "NLOPT_MAXITER_REACHED"},
};

namespace QPanda
{
    OriginBasicOptNL::OriginBasicOptNL(OptimizerType opt_type)
    {
        this->opt_type = opt_type;
        m_dimension = 0;
        m_fcalls = 0;
        m_iter = 0;
        f_min = 0;
        m_fatol = 1e-4;
        m_xatol = 1e-4;
        m_frtol = 0;
        m_xrtol = 0;
    }

    nlopt_func OriginBasicOptNL::function_transform(QOptFunc func)
    {
        nlopt_func a([func](unsigned n, const double* x, double* gradient, void* func_data, int n_iters, int n_evals)
            {
                std::vector<double> x_data(x, x + n), grad_data;
                if (gradient) grad_data = std::vector<double>(gradient, gradient + n);
                double f = func(x_data, grad_data, n_iters, n_evals).second;
                if (gradient) memcpy(gradient, grad_data.data(), grad_data.size() * sizeof(double));
                return f;
            });
        return a;
    }

    void OriginBasicOptNL::registerFunc(const QOptFunc& func, const std::vector<double>& optimized_para)
    {
        m_func = func;
        m_optimized_para = optimized_para;
        obj_func = function_transform(m_func);
        m_dimension = m_optimized_para.size();
        
    }

    void OriginBasicOptNL::set_lower_and_upper_bounds(std::vector<double>& lower_bound, std::vector<double>& upper_bound)
    {
        lb = lower_bound;
        ub = upper_bound;
    }

    void OriginBasicOptNL::add_equality_constraint(QOptFunc func)
    {
        equality_constraint = function_transform(func);
    }

    void OriginBasicOptNL::add_inequality_constraint(QOptFunc func)
    {
        inequality_constraint = function_transform(func);
    }

    void OriginBasicOptNL::setXatol(double xatol)
    {
        m_xatol = xatol;
    }

    void OriginBasicOptNL::setXrtol(double xrtol)
    {
        m_xrtol = xrtol;
    }

    void OriginBasicOptNL::setFatol(double fatol)
    {
        m_fatol = fatol;
    }

    void OriginBasicOptNL::setFrtol(double frtol)
    {
        m_frtol = frtol;
    }

    void OriginBasicOptNL::setMaxFCalls(size_t max_fcalls)
    {
        m_max_fcalls = max_fcalls;
    }

    void OriginBasicOptNL::setMaxIter(size_t max_iter)
    {
        m_max_iter = max_iter;
    }

    void OriginBasicOptNL::init() {
        opter = nlopt_create(algorithm_map[opt_type], m_dimension);
        nlopt_set_min_objective(&opter, obj_func, NULL);
        if (equality_constraint)
        {
            nlopt_add_equality_constraint(&opter, equality_constraint, NULL, m_fatol);
        }
        if (inequality_constraint)
        {
            nlopt_add_inequality_constraint(&opter, inequality_constraint, NULL, m_fatol);
        }
        
        nlopt_set_xtol_abs1(&opter, m_xatol);
        nlopt_set_xtol_rel(&opter, m_xrtol);
        nlopt_set_xtol_abs1(&opter, m_fatol);
        nlopt_set_ftol_rel(&opter, m_frtol);
        nlopt_set_maxeval(&opter, m_max_fcalls);
        nlopt_set_maxiter(&opter, m_max_iter);

        if (!lb.empty())
        {
            nlopt_set_lower_bounds(&opter, lb.data());
        }
        if (!ub.empty())
        {
            nlopt_set_upper_bounds(&opter, ub.data());
        }
    }

    void OriginBasicOptNL::exec()
    {
        init();
        nlopt_result result = nlopt_optimize(&opter, m_optimized_para.data(), &f_min, m_restore_from_cache_file, m_cache_file);
        m_iter = nlopt_get_numiters(&opter);
        m_fcalls = nlopt_get_numevals(&opter);
        m_dimension = nlopt_get_dimension(&opter);
        m_result.message = nlopt_get_errmsg(&opter) == NULL ? "No Error: " + result_map[result] : nlopt_get_errmsg(&opter);
        outputResult();
    }

    void OriginBasicOptNL::outputResult()
    {
        //m_result.key = m_key[0];
        m_result.fun_val = f_min;
        m_result.fcalls = m_fcalls;
        m_result.iters = m_iter;
        m_result.para.resize(m_dimension);
        for (int i = 0; i < m_dimension; i++)
        {
            m_result.para[i] = m_optimized_para[i];
        }
        if (m_fcalls >= m_max_fcalls)
        {
            m_result.message = DEF_OPTI_STATUS_MAX_FEV;
            std::cout << DEF_WARING + m_result.message
                << std::endl;
        }
        else if (m_iter >= m_max_iter)
        {
            m_result.message = DEF_OPTI_STATUS_MAX_ITER;
            std::cout << DEF_WARING + m_result.message
                << std::endl;
        }
        else if (m_result.message.size() < 1)
        {
            m_result.message = DEF_OPTI_STATUS_SUCCESS;
            dispResult();
        }
        else
        {
            dispResult();
        }
        return;
    }

    void OriginBasicOptNL::dispResult()
    {
        if (m_disp)
        {
            std::cout << m_result.message << std::endl;
            std::cout << "         Current function value: "
                << f_min << std::endl;
            std::cout << "         Iterations: "
                << m_iter << std::endl;
            std::cout << "         Function evaluations: "
                << m_fcalls << std::endl;
            std::cout << "         Optimized para: " << std::endl;
            for (int i = 0; i < m_dimension; i++)
            {
                std::cout << "             " << m_result.para[i] << std::endl;
            }
        }
    }
}
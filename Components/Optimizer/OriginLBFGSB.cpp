#include "Components/Optimizer/OriginLBFGSB.h"
#include "Core/Utilities/Tools/OriginCollection.h"
#include "Core/Utilities/Tools/QString.h"

const std::string COBYLA_CACHE_HEADER = "lbfgsb_break_point.json";

namespace QPanda
{
    OriginLBFGSB::OriginLBFGSB()
    {
        m_dimension = 0;
        m_fcalls = 0;
        m_iter = 0;
        f_min = 0;
        x = (double*)calloc(0, sizeof(double));
    }

    void OriginLBFGSB::set_lower_and_upper_bounds(vector_d& lower_bound, vector_d& upper)
    {
        nlopt_set_lower_bounds(&opter, lower_bound.data());
        nlopt_set_upper_bounds(&opter, upper.data());
    }

    void OriginLBFGSB::add_equality_constraint(QOptFunc func)
    {
        nlopt_func a([func](unsigned n, const double* x, double* gradient, void* func_data)
            {
                vector_d x_data(x, x + n), grad_data(gradient, gradient + n);
                double f = func(x_data, grad_data, 0, 0).second;
                return f;
            });
        nlopt_add_equality_constraint(&opter, a, NULL, m_fatol);
    }

    void OriginLBFGSB::add_inequality_constraint(QOptFunc func)
    {
        nlopt_func a([func](unsigned n, const double* x, double* gradient, void* func_data)
            {
                vector_d x_data(x, x + n), grad_data(gradient, gradient + n);
                double f = func(x_data, grad_data, 0, 0).second;
                return f;
            });
        nlopt_add_inequality_constraint(&opter, a, NULL, m_fatol);
    }

    void OriginLBFGSB::init() {
        m_dimension = m_optimized_para.size();
        x = (double*)calloc(m_dimension, sizeof(double));
        if (x) {
            for (int i = 0; i < m_dimension; i++)
            {
                x[i] = m_optimized_para[i];
            }
        }
        opter = nlopt_create(nlopt_algorithm::NLOPT_LD_LBFGSB, m_dimension);
        nlopt_func a([&](unsigned n, const double* x, double* gradient, void* func_data)
            {
                vector_d grad;
                if (gradient)
                {
                    grad = vector_d(gradient, gradient + n);
                }
                vector_d x_data(x, x + n);
                double f = m_func(x_data, grad, 0, 0).second;
                if (gradient)
                {
                    for (int i = 0; i < n; i++) {
                        gradient[i] = grad[i];
                    }
                }
                return f;
            });
        nlopt_set_min_objective(&opter, a, NULL);
        nlopt_set_xtol_rel(&opter, m_xatol);
        nlopt_set_ftol_rel(&opter, m_fatol);
        nlopt_set_maxeval(&opter, m_max_fcalls);
        nlopt_set_maxiter(&opter, m_max_iter);

    }

    void OriginLBFGSB::exec()
    {
        init();
        nlopt_result result = nlopt_optimize(&opter, x, &f_min, m_restore_from_cache_file, m_cache_file);
        m_iter = nlopt_get_numiters(&opter);
        m_fcalls = nlopt_get_numevals(&opter);
        m_dimension = nlopt_get_dimension(&opter);
        m_result.message = nlopt_get_errmsg(&opter) == NULL ? "No Error" : nlopt_get_errmsg(&opter);
        outputResult();
    }

    void OriginLBFGSB::outputResult()
    {
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
        else
        {
            m_result.message = DEF_OPTI_STATUS_SUCCESS;
            dispResult();
        }
        //m_result.key = m_key[0];
        m_result.fun_val = f_min;
        m_result.fcalls = m_fcalls;
        m_result.iters = m_iter;
        m_result.para.resize(m_dimension);
        for (int i = 0; i < m_dimension; i++)
        {
            m_result.para[i] = x[i];
        }
        return;
    }

    void OriginLBFGSB::dispResult()
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
                std::cout << x[i] << std::endl;
            }
        }
    }
}
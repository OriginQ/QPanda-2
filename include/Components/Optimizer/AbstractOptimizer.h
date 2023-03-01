/*
Copyright (c) 2017-2023 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

AbstractOptimizer.h

Author: LiYe
Created in 2018-09-06

Alter by LiYe in 2021-1-21

*/

#ifndef ABSTRACTOPTIMIZER_H
#define ABSTRACTOPTIMIZER_H

#include <vector>
#include <functional>
#include "OptimizerDataStruct.h"

namespace QPanda
{
	/**
    * @brief Abstract Optimizer
    * @ingroup Optimizer
    */
    class AbstractOptimizer
    {
    public:
		/**
	    * @brief  Constructor of AbstractOptimizer class
	    */
        AbstractOptimizer();
        AbstractOptimizer(const AbstractOptimizer &) = delete;
        AbstractOptimizer& operator = (const AbstractOptimizer &) = delete;
        virtual ~AbstractOptimizer();

		/**
	    * @brief register a user defined function and set Optimizer parameters
	    * @param[in] QFunc& user defined function
		* @param[in] vector_d& Optimizer parameters
	    */
        virtual void registerFunc(
            const QOptFunc &func, 
            const std::vector<double> &optimized_para)
        {
            m_func = func;
            m_optimized_para = optimized_para;
        }

		/**
		* @brief whether or not display the log info
		* @param[in] bool
		*/
        virtual void setDisp(bool disp)
        {
            m_disp = disp;
        }

		/**
		* @brief whether or not use Para of Nelder-Mead
		* @param[in] bool
		*/
        virtual void setAdaptive(bool adaptive)
        {
            m_adaptive = adaptive;
        }

		/**
		* @brief set absolute error in xopt between iterations 
        *        that is acceptable for convergence
		* @param[in] double
		*/
        virtual void setXatol(double xatol)
        {
            m_xatol = xatol;
        }

		/**
		* @brief set Absolute error in func(xopt) between iterations 
                 that is acceptable for convergence
		* @param[in] double
		*/
        virtual void setFatol(double fatol)
        {
            m_fatol = fatol;
        }

		/**
		* @brief set the max call times
		* @param[in] size_t
		*/
        virtual void setMaxFCalls(size_t max_fcalls)
        {
            m_max_fcalls = max_fcalls;
        }

		/**
		* @brief set the max iter times
		* @param[in] size_t max iter
		*/
        virtual void setMaxIter(size_t max_iter)
        {
            m_max_iter = max_iter;
        }

        /**
        * @brief set optional paraments
        * @param[in] std::map<std::string, std::string> optional paraments
        */
        virtual void setOptionalPara(const std::map<std::string, std::string>&
                                     optional_para)
        {
            m_optional_para = optional_para;
        }

		/**
		* @brief set whether or not restore from cache file
		* @param[in] bool restore flag
		*/
        virtual void setRestoreFromCacheFile(bool restore)
        {
            m_restore_from_cache_file = restore;
        }

		/**
		* @brief set cache file
		* @param[in] std::string& cache file name
		*/
        virtual void setCacheFile(const std::string& cache_file)
        {
            m_cache_file = cache_file;
        }

		/**
		* @brief  execute optimization
		*/
        virtual void exec() = 0;

		/**
		* @brief get optimization result
		* @return QOptimizationResult optimization result
		*/
        virtual QOptimizationResult getResult()
        {
            return m_result;
        }

    protected:
        QOptFunc m_func; /**< user defined loss function */

        std::vector<double> m_optimized_para; /**< optimized parameter */

        bool m_disp;        /**< Whether to print the log to the terminal */
        bool m_adaptive;    /**< Para of Nelder-Mead.
                             Adapt algorithm parameters to dimensionality
                             of problem.Useful for high-dimensional
                             minimization. [Optional]*/

        double m_xatol;     /**< Absolute error in xopt between iterations that 
                            is acceptable for convergence. [Optional] */

        double m_fatol;     /**< Absolute error in func(xopt) between 
                            iterations that is acceptable for convergence. 
                            [Optional]*/

        double m_test_value;/**<  user test value*/
        std::string m_para_file; /**< parameter file */

        size_t m_max_fcalls;/**<  Maximum allowed number 
                                  of function evaluations*/
        size_t m_max_iter;  /**<  Maximum allowed number of iterations*/

        std::map<std::string, std::string> m_optional_para;/**< Optional paraments*/

        bool m_restore_from_cache_file; /**< Whether to restore data 
                                             from cache file */
        std::string m_cache_file; /**< cache file */

        QOptimizationResult m_result; /**< optimization result */
    };

}

#endif // ABSTRACTOPTIMIZER_H

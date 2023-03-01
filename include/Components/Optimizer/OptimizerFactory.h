/*
Copyright (c) 2017-2023 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

OptimizerFactor.h

Author: LiYe
Created in 2018-09-06


*/

#ifndef OPTIMIZERFACTOR_H
#define OPTIMIZERFACTOR_H

#include <memory>
#include "OptimizerDataStruct.h"

namespace QPanda
{
    class AbstractOptimizer;
	
	/**
    * @brief Class of Optimizer factory.
	* @ingroup Optimizer
    */
    class OptimizerFactory
    {
    public:
		/**
	    * @brief  Constructor of OptimizerFactory
	    */
        OptimizerFactory();

		/**
		* @brief create a Optimizer object by OptimizerType
		* @param[in] OptimizerType  Optimizer Type
		* @return std::unique_ptr<AbstractOptimizer>
		*/
        static std::unique_ptr<AbstractOptimizer>
            makeOptimizer(const OptimizerType &optimizer);

		/**
		* @brief create a Optimizer object by OptimizerType string
		* @param[in] std::string&  Optimizer Type string
		* @return std::unique_ptr<AbstractOptimizer>
		*/
        static std::unique_ptr<AbstractOptimizer>
            makeOptimizer(const std::string &optimizer);
    };

}

#endif // OPTIMIZERFACTOR_H

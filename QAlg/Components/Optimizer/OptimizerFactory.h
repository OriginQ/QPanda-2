/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

OptimizerFactor.h

Author: LiYe
Created in 2018-09-06


*/

#ifndef OPTIMIZERFACTOR_H
#define OPTIMIZERFACTOR_H

#include <memory>
#include "QAlg/DataStruct.h"

namespace QPanda
{
    /*
    Abstract class of Optimizer.
    */
    class AbstractOptimizer;
    class OptimizerFactory
    {
    public:
        OptimizerFactory();

        static std::unique_ptr<AbstractOptimizer>
            makeOptimizer(const OptimizerType &optimizer);
        static std::unique_ptr<AbstractOptimizer>
            makeOptimizer(const std::string &optimizer);
    };

}

#endif // OPTIMIZERFACTOR_H

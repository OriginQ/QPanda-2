/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

QOptimizerFactor.h

Author: LiYe
Created in 2018-09-06


*/

#ifndef QOPTIMIZERFACTOR_H
#define QOPTIMIZERFACTOR_H

#include <memory>
#include "Utilities/QAlgDataStruct.h"

namespace QPanda
{
    /*
    Abstract class of QOptimizer.
    */
    class AbstractQOptimizer;
    class QOptimizerFactor
    {
    public:
        QOptimizerFactor();

        static std::unique_ptr<AbstractQOptimizer>
            makeQOptimizer(Optimizer optimizer);
    };

}

#endif // QOPTIMIZERFACTOR_H

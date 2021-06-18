#include <memory>
#include "Components/Optimizer/OptimizerFactory.h"
#include "Components/Optimizer/OriginNelderMead.h"
#include "Components/Optimizer/OriginPowell.h"
#include "Components/Optimizer/OriginCOBYLA.h"
#include "Components/Optimizer/OriginLBFGSB.h"
#include "Components/Optimizer/OriginSLSQP.h"
#include "Core/Utilities/QPandaNamespace.h"
using std::invalid_argument;

namespace QPanda
{
    OptimizerFactory::OptimizerFactory()
    { }

    std::unique_ptr<AbstractOptimizer>
        OptimizerFactory::makeOptimizer(const OptimizerType &optimizer)
    {
        switch (optimizer)
        {
        case OptimizerType::NELDER_MEAD:
            return std::unique_ptr<AbstractOptimizer>(new OriginNelderMead);
        case OptimizerType::POWELL:
            return std::unique_ptr<AbstractOptimizer>(new OriginPowell);
        case OptimizerType::COBYLA:
            return std::unique_ptr<AbstractOptimizer>(new OriginCOBYLA);
        case OptimizerType::LBFGSB:
            return std::unique_ptr<AbstractOptimizer>(new OriginLBFGSB);
        case OptimizerType::SLSQP:
            return std::unique_ptr<AbstractOptimizer>(new OriginSLSQP);
        default:
            return std::unique_ptr<AbstractOptimizer>(nullptr);
        }
    }

    std::unique_ptr<AbstractOptimizer>
        OptimizerFactory::makeOptimizer(const std::string &optimizer)
    {
        if ("Nelder-Mead" == optimizer)
        {
            return std::unique_ptr<AbstractOptimizer>(new OriginNelderMead);
        }
        else if ("Powell" == optimizer)
        {
            return std::unique_ptr<AbstractOptimizer>(new OriginPowell);
        }
        else if ("COBYLA" == optimizer)
        {
            return std::unique_ptr<AbstractOptimizer>(new OriginCOBYLA);
        }
        else if ("LBFGSB" == optimizer)
        {
            return std::unique_ptr<AbstractOptimizer>(new OriginLBFGSB);
        }
        else if ("SLSQP" == optimizer)
        {
            return std::unique_ptr<AbstractOptimizer>(new OriginSLSQP);
        }
        else
        {
            return std::unique_ptr<AbstractOptimizer>(new OriginNelderMead);
        }
    }

}

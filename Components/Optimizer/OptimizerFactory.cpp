#include <memory>
#include "Components/Optimizer/OptimizerFactory.h"
#include "Components/Optimizer/OriginNelderMead.h"
#include "Components/Optimizer/OriginPowell.h"
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
        else
        {
            return std::unique_ptr<AbstractOptimizer>(new OriginNelderMead);
        }
    }

}

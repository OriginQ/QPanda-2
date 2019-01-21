#include <memory>
#include "OptimizerFactory.h"
#include "OriginNelderMead.h"
#include "OriginPowell.h"
#include "QPandaNamespace.h"
using std::invalid_argument;

namespace QPanda
{
    OptimizerFactory::OptimizerFactory()
    { }

    std::unique_ptr<AbstractOptimizer>
        OptimizerFactory::makeQOptimizer(const OptimizerType &optimizer)
    {
        switch (optimizer)
        {
        case OptimizerType::NELDER_MEAD:
            return std::unique_ptr<AbstractOptimizer>(new OriginNelderMead);
        case OptimizerType::POWELL:
            return std::unique_ptr<AbstractOptimizer>(new OriginPowell);
        default:
            return  std::unique_ptr<AbstractOptimizer>(nullptr);
        }
    }

    std::unique_ptr<AbstractOptimizer>
        OptimizerFactory::makeQOptimizer(const std::string &optimizer)
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

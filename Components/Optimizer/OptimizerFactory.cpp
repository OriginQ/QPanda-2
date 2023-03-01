#include <memory>
#include "Components/Optimizer/OptimizerFactory.h"
#include "Components/Optimizer/OriginNelderMead.h"
#include "Components/Optimizer/OriginPowell.h"
#include "Components/Optimizer/OriginBasicOptNL.h"
#include "Components/Optimizer/OriginGradient.h"
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
            return std::unique_ptr<AbstractOptimizer>(new OriginBasicOptNL(OptimizerType::COBYLA));
        case OptimizerType::L_BFGS_B:
            return std::unique_ptr<AbstractOptimizer>(new OriginBasicOptNL(OptimizerType::L_BFGS_B));
        case OptimizerType::SLSQP:
            return std::unique_ptr<AbstractOptimizer>(new OriginBasicOptNL(OptimizerType::SLSQP));
        case OptimizerType::GRADIENT:
            return std::unique_ptr<AbstractOptimizer>(new OriginGradient);
        default:
            QCERR_AND_THROW_ERRSTR(
                std::runtime_error, 
                "Unrecognized optimizer type");
        }
    }

    std::unique_ptr<AbstractOptimizer>
        OptimizerFactory::makeOptimizer(const std::string &optimizer)
    {
        if (DEF_NELDER_MEAD == optimizer)
        {
            return std::unique_ptr<AbstractOptimizer>(new OriginNelderMead);
        }
        else if (DEF_POWELL == optimizer)
        {
            return std::unique_ptr<AbstractOptimizer>(new OriginPowell);
        }
        else if (DEF_COBYLA == optimizer)
        {
            return std::unique_ptr<AbstractOptimizer>(new OriginBasicOptNL(OptimizerType::COBYLA));
        }
        else if (DEF_LBFGSB == optimizer)
        {
            return std::unique_ptr<AbstractOptimizer>(new OriginBasicOptNL(OptimizerType::L_BFGS_B));
        }
        else if (DEF_SLSQP == optimizer)
        {
            return std::unique_ptr<AbstractOptimizer>(new OriginBasicOptNL(OptimizerType::SLSQP));
        }
        else if (DEF_GRADIENT_DESCENT == optimizer)
        {
            return std::unique_ptr<AbstractOptimizer>(new OriginGradient);
        }
        else
        {
            QCERR_AND_THROW_ERRSTR(
                std::runtime_error,
                "Unrecognized optimizer type");
        }
    }

}

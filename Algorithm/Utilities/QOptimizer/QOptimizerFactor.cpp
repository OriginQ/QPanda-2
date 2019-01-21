#include "QOptimizerFactor.h"
#include "OriginNelderMead.h"

namespace QPanda
{

    QOptimizerFactor::QOptimizerFactor()
    {

    }

    std::unique_ptr<AbstractQOptimizer>
        QOptimizerFactor::makeQOptimizer(Optimizer optimizer)
    {
        switch (optimizer)
        {
        case NELDER_MEAD:
            return std::unique_ptr<AbstractQOptimizer>(new OriginNelderMead);
        default:
            return std::unique_ptr<AbstractQOptimizer>(new OriginNelderMead);
        }
    }

}

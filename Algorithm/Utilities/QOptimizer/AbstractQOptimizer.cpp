#include "AbstractQOptimizer.h"

namespace QPanda
{

    AbstractQOptimizer::AbstractQOptimizer() :
        m_optimized_para(2, 0),
        m_disp(false),
        m_adaptive(false),
        m_xatol(1e-4),
        m_fatol(1e-4),
        m_max_fcalls(DEF_UNINIT_INT),
        m_max_iter(DEF_UNINIT_INT)
    {
        m_result[DEF_MESSAGE] = "No exec.";
    }

}
